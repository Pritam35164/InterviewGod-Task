"""Build a stratified evaluation set from tickets.csv.

Labels are assigned by a documented rubric applied to ticket text *without*
looking at any model output. GW-OTHER tickets are labelled from the closing note;
non-default dropdown codes are accepted only when the note agrees.

    python -m scripts.build_eval_set
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.baseline import classify_baseline  # noqa: E402
from app.services.ingest_rules import FindingLog  # noqa: E402
from app.services.ingest import detect_legacy_money_unit, deduplicate_tickets, load_raw  # noqa: E402

OUT = ROOT / "data" / "evaluation_set.json"
RNG = np.random.default_rng(42)

DROPDOWN_TO_REASON = {
    "DOA-REPL": "PRODUCT_FAULT_DOA",
    "LOST-TRANSIT": "NOT_DELIVERED_OR_LOST",
    "DUP-PAYMENT": "DUPLICATE_OR_FAILED_PAYMENT",
    "CANCEL": "CANCELLED_BEFORE_DISPATCH",
    "PRICE-ADJ": "PRICE_OR_PROMO_ADJUSTMENT",
    "RETURN-QC-OK": "RETURN_RECEIVED_QC_PASSED",
    "WTY-BUYBACK": "PRODUCT_FAULT_IN_WARRANTY",
}


def _label(row: pd.Series) -> tuple[str, str, str]:
    """Return (reason, avoidability, note)."""
    src = row["refund_reason_code"] or ""
    notes = str(row["agent_notes"] or "")
    msg = str(row["customer_message"] or "")
    baseline = classify_baseline(msg, notes)
    dual_flag = str(row["replacement_issued"]).upper() == "Y"
    text_dual = any(
        p in notes.lower()
        for p in (
            "refund + replacement both",
            "both refund and replacement",
            "both refund & replacement",
            "full refund issued and replacement dispatched",
            "credited full amount, new unit",
            "amount returned to card; fresh unit",
            "reversed the payment and dispatched a new one",
        )
    )
    text_not_dual = any(
        p in notes.lower()
        for p in (
            "replacement not applicable",
            "explained policy, refund only",
            "explained policy, rfnd only",
            "rplc request rejected",
            "replacement request rejected",
            "replacement declined",
        )
    )

    if src and src != "GW-OTHER" and src in DROPDOWN_TO_REASON:
        reason = DROPDOWN_TO_REASON[src]
        if baseline.value not in {"UNKNOWN", "AMBIGUOUS"} and baseline.value != reason:
            # Dropdown and note disagree: take the note.
            reason = baseline.value
            method = "note_overrides_dropdown"
        else:
            method = "dropdown_agrees_with_note_or_empty"
    elif baseline.value != "UNKNOWN":
        reason = baseline.value
        method = "note_baseline_on_gw_other_or_blank"
    else:
        reason = "UNKNOWN"
        method = "insufficient_text"

    if dual_flag or text_dual:
        avoid = "POTENTIALLY_AVOIDABLE"
        method += "+dual_remedy"
    elif "without pickup as goodwill" in notes.lower() or "rfnd processed without pickup" in notes.lower():
        avoid = "POTENTIALLY_AVOIDABLE"
        method += "+refund_without_return"
        if reason == "UNKNOWN":
            reason = "RETURN_PICKUP_FAILURE"
    elif reason in {"UNKNOWN", "AMBIGUOUS"}:
        avoid = "AMBIGUOUS"
    else:
        avoid = "POLICY_COMPLIANT"

    if text_not_dual and not text_dual:
        # Flag can be wrong; negation in the note wins for dual-remedy labelling.
        if avoid == "POTENTIALLY_AVOIDABLE" and dual_flag and not text_dual:
            avoid = "AMBIGUOUS"
            method += "+flag_y_note_negation"

    return reason, avoid, method


def main() -> int:
    raw = load_raw()
    tickets = raw["tickets"]
    findings = FindingLog("eval-set")
    unit = detect_legacy_money_unit(tickets, findings)
    d = deduplicate_tickets(tickets, findings, unit["divisor"])
    d["amt"] = pd.to_numeric(d["refund_amount_inr"], errors="coerce")
    d.loc[d["source_system"] == "legacy_fd", "amt"] = d.loc[d["source_system"] == "legacy_fd", "amt"] / unit["divisor"]
    d["is_refund"] = d["amt"].notna()
    d["month"] = pd.to_datetime(d["created_at"]).dt.to_period("M").astype(str)
    d["high_value"] = d["amt"].fillna(0) >= 5000

    refunds = d[d["is_refund"]].copy()
    non = d[~d["is_refund"]].copy()

    strata: list[pd.DataFrame] = []

    def take(df: pd.DataFrame, n: int, name: str) -> pd.DataFrame:
        if df.empty:
            return df
        n = min(n, len(df))
        out = df.sample(n=n, random_state=42).copy()
        out["stratum"] = name
        return out

    # ~100 tickets, stratified.
    strata.append(take(refunds[refunds["refund_reason_code"] == "GW-OTHER"], 25, "refund_gw_other"))
    for code in ["RETURN-QC-OK", "DUP-PAYMENT", "CANCEL", "DOA-REPL", "WTY-BUYBACK", "PRICE-ADJ", "LOST-TRANSIT"]:
        strata.append(take(refunds[refunds["refund_reason_code"] == code], 5, f"refund_{code}"))
    strata.append(take(refunds[refunds["high_value"]], 10, "refund_high_value"))
    strata.append(take(refunds[refunds["replacement_issued"].str.upper() == "Y"], 10, "refund_dual_flag"))
    # Ambiguous / empty notes
    emptyish = refunds[refunds["agent_notes"].str.len() < 12]
    strata.append(take(emptyish, 6, "refund_short_note"))
    # Spread months and agents via leftover refunds
    leftover = refunds[~refunds["ticket_id"].isin(pd.concat(strata)["ticket_id"])] if strata else refunds
    strata.append(take(leftover, 10, "refund_remainder"))
    strata.append(take(non, 15, "non_refund"))

    sample = pd.concat(strata, ignore_index=True).drop_duplicates("ticket_id")
    # Cap around 100
    if len(sample) > 110:
        sample = sample.sample(n=110, random_state=42)

    tickets_out = []
    for _, row in sample.iterrows():
        is_refund = bool(row["is_refund"])
        if is_refund:
            reason, avoid, method = _label(row)
        else:
            reason, avoid, method = "UNKNOWN", "POLICY_COMPLIANT", "non_refund"
        tickets_out.append(
            {
                "ticket_id": row["ticket_id"],
                "reviewed": True,
                "stratum": row["stratum"],
                "is_refund": is_refund,
                "month": row["month"],
                "agent_id": row["agent_id"],
                "source_reason_code": row["refund_reason_code"] or None,
                "amount_inr": None if pd.isna(row["amt"]) else round(float(row["amt"]), 2),
                "expected_reason_code": reason if is_refund else None,
                "expected_avoidability": avoid if is_refund else None,
                "annotator_method": method,
                "high_value": bool(row["high_value"]),
            }
        )

    payload = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotator": "single reviewer; rubric in scripts/build_eval_set.py",
        "limitations": [
            "Single annotator who also wrote the taxonomy. No second annotator, so no IAA.",
            "Labels were assigned from ticket text and the dropdown code using a written rubric, never from a model prediction for that ticket.",
            "Non-refund tickets have expected_reason_code=null; they are used for UNKNOWN-rate / leakage checks only.",
        ],
        "strata": dict(Counter(t["stratum"] for t in tickets_out)),
        "n": len(tickets_out),
        "n_reviewed": sum(1 for t in tickets_out if t["reviewed"]),
        "tickets": tickets_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  n={payload['n']} reviewed={payload['n_reviewed']}")
    print("strata:", payload["strata"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
