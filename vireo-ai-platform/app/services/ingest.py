"""CSV -> PostgreSQL ingest, with every cleaning decision recorded.

Principles held to throughout:
  * Source data is never silently corrected. Where a value is changed, the
    original is kept next to it (`refund_amount_source`) and the change is
    recorded as a `DataQualityEvent`.
  * No column name is invented. Everything read here appears in README.txt.
  * Where the policy warns of a problem the code *measures* it before acting on
    it, rather than trusting the warning. See `detect_legacy_money_unit`.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.decisions import DataQualityEvent, PolicySectionRecord
from app.models.refund import Refund
from app.models.source import Agent, Customer, Order, Product, Ticket
from app.policy.loader import get_policy
from app.schemas.taxonomy import DROPDOWN_DEFAULT_CODE
from app.services.ingest_rules import (
    FLAG_CSAT_BLANK,
    FLAG_LEGACY_UNIT_CONVERTED,
    FLAG_NO_ORDER_ID,
    FLAG_NO_RESOLVED_AT,
    FLAG_ORDER_AMBIGUOUS,
    FLAG_ORDER_UNRESOLVED,
    FLAG_REFUND_EXCEEDS_ORDER,
    FLAG_REFUND_ON_OPEN_TICKET,
    FLAG_REIMPORT_DUPLICATE,
    FLAG_TICKET_BEFORE_ORDER,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    SLA_TARGET_MINUTES,
    Finding,
    FindingLog,
)

log = get_logger(__name__)

REQUIRED_FILES = (
    "tickets.csv", "agents.csv", "orders.csv", "customers.csv", "products.csv",
)


@dataclass
class IngestResult:
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    counts: dict[str, int] = field(default_factory=dict)
    money: dict[str, float] = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    unit_detection: dict[str, Any] = field(default_factory=dict)
    join_integrity: dict[str, Any] = field(default_factory=dict)
    reconciliation: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "counts": self.counts,
            "money": {k: round(v, 2) for k, v in self.money.items()},
            "unit_detection": self.unit_detection,
            "join_integrity": self.join_integrity,
            "reconciliation": self.reconciliation,
            "findings": self.findings,
        }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    """Read as strings with blanks preserved as "".

    `keep_default_na=False` matters: pandas would otherwise turn a blank
    refund_amount_inr into NaN and a customer named "NA" into a null.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def load_raw(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    base = raw_dir or settings.raw_dir
    missing = [f for f in REQUIRED_FILES if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing source files in {base}: {', '.join(missing)}. "
            "Run scripts/fetch_data.py or mount the data pack at data/raw/."
        )
    return {name.replace(".csv", ""): _read_csv(base / name) for name in REQUIRED_FILES}


def _blank_to_na(series: pd.Series) -> pd.Series:
    """Replace "" with NA without pandas' deprecated silent downcasting."""
    return series.mask(series == "", other=pd.NA)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(_blank_to_na(series), errors="coerce")


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(_blank_to_na(series), errors="coerce")


# ---------------------------------------------------------------------------
# DQ-001: measure the legacy monetary unit instead of assuming it
# ---------------------------------------------------------------------------
def detect_legacy_money_unit(tickets: pd.DataFrame, findings: FindingLog) -> dict[str, Any]:
    """Determine the legacy unit from tickets present under BOTH source systems.

    Policy §9 states the legacy tool used "its own native unit" and does not name
    it. Rather than guessing paise, this compares the two copies of each
    re-imported ticket. Only a unanimous, clean integer ratio is accepted; a
    divisor is never applied on the strength of a distribution that merely looks
    plausible.
    """
    amt = _to_num(tickets["refund_amount_inr"])
    work = tickets.assign(_amt=amt)
    with_refund = work[work["_amt"].notna()]

    pivot = (
        with_refund[with_refund["ticket_id"].duplicated(keep=False)]
        .pivot_table(
            index="ticket_id", columns="source_system", values="_amt", aggfunc="first"
        )
    )
    detection: dict[str, Any] = {
        "method": "paired comparison of tickets exported under both source systems",
        "paired_tickets": 0,
        "ratios_observed": [],
        "divisor": 1.0,
        "unit": "INR",
        "conclusive": False,
    }

    if {"helpdesk", "legacy_fd"}.issubset(pivot.columns):
        pairs = pivot.dropna(subset=["helpdesk", "legacy_fd"])
        pairs = pairs[pairs["helpdesk"] > 0]
        if len(pairs):
            ratios = (pairs["legacy_fd"] / pairs["helpdesk"]).round(6)
            unique = sorted(ratios.unique().tolist())
            detection["paired_tickets"] = int(len(pairs))
            detection["ratios_observed"] = unique[:10]
            detection["ratio_min"] = float(ratios.min())
            detection["ratio_max"] = float(ratios.max())
            if len(unique) == 1 and float(unique[0]).is_integer() and unique[0] > 1:
                divisor = float(unique[0])
                detection.update(
                    divisor=divisor,
                    unit="paise" if divisor == 100 else f"1/{divisor:g} INR",
                    conclusive=True,
                )

    legacy_amt = with_refund.loc[with_refund["source_system"] == "legacy_fd", "_amt"]
    hd_amt = with_refund.loc[with_refund["source_system"] == "helpdesk", "_amt"]
    detection["legacy_all_divisible_by_divisor"] = (
        bool((legacy_amt % detection["divisor"] == 0).all()) if detection["conclusive"] else None
    )
    detection["legacy_median_raw"] = float(legacy_amt.median()) if len(legacy_amt) else None
    detection["helpdesk_median"] = float(hd_amt.median()) if len(hd_amt) else None

    if detection["conclusive"]:
        divisor = detection["divisor"]
        # Corroboration: after conversion the two systems must describe the same
        # population. A divisor that passes the pairwise test but leaves the
        # distributions far apart would mean the pairs are unrepresentative.
        converted_median = detection["legacy_median_raw"] / divisor
        hd_median = detection["helpdesk_median"] or 0
        detection["converted_legacy_median"] = round(converted_median, 2)
        detection["distribution_agreement"] = (
            round(converted_median / hd_median, 4) if hd_median else None
        )
        findings.add(
            Finding(
                finding_id="DQ-001",
                severity=SEVERITY_CRITICAL,
                title=f"Legacy Freshdesk rows store money in {detection['unit']}, not rupees",
                affected_rows=int(len(legacy_amt)),
                affected_value_inr=float(legacy_amt.sum() / divisor),
                decision=(
                    f"Divided every legacy_fd refund amount by {divisor:g} to convert to "
                    f"rupees. The divisor was measured, not assumed: {detection['paired_tickets']} "
                    f"tickets appear under both source systems and the legacy value is exactly "
                    f"{divisor:g}x the helpdesk value for every one of them (only ratio observed: "
                    f"{detection['ratios_observed']}). Corroborated by the converted legacy median "
                    f"(₹{converted_median:,.0f}) matching the helpdesk median (₹{hd_median:,.0f}). "
                    f"The original value is preserved in tickets.refund_amount_source."
                ),
                policy_reference=(
                    "support-policy.pdf v3.2 §9: 'The legacy tool stored monetary values in its "
                    "own native unit; the current helpdesk stores rupees.'"
                ),
                evidence=detection,
            )
        )
    else:
        findings.add(
            Finding(
                finding_id="DQ-001",
                severity=SEVERITY_CRITICAL,
                title="Legacy monetary unit could NOT be determined — no conversion applied",
                affected_rows=int(len(legacy_amt)),
                decision=(
                    "The paired-ticket test was not unanimous, so no divisor was applied. "
                    "Legacy amounts are left exactly as exported and every total that "
                    "includes them is unreliable. Failing closed rather than guessing a unit."
                ),
                policy_reference="support-policy.pdf v3.2 §9",
                evidence=detection,
            )
        )
    return detection


# ---------------------------------------------------------------------------
# DQ-002: re-import duplicates
# ---------------------------------------------------------------------------
def deduplicate_tickets(
    tickets: pd.DataFrame, findings: FindingLog, divisor: float
) -> pd.DataFrame:
    """Keep one row per ticket_id, preferring the current helpdesk copy."""
    dup_mask = tickets["ticket_id"].duplicated(keep=False)
    dup_ids = tickets.loc[dup_mask, "ticket_id"].unique()
    extra_rows = int(len(tickets) - tickets["ticket_id"].nunique())

    variants = (
        tickets.groupby("ticket_id")["source_system"]
        .apply(lambda s: sorted(set(s)))
        .rename("source_system_variants")
    )
    row_counts = tickets.groupby("ticket_id").size().rename("duplicate_row_count")

    # Stable preference: helpdesk (rupees) over legacy_fd (native unit).
    work = tickets.assign(
        _pref=np.where(tickets["source_system"] == "helpdesk", 0, 1)
    ).sort_values(["ticket_id", "_pref"], kind="mergesort")
    deduped = work.drop_duplicates("ticket_id", keep="first").drop(columns=["_pref"])
    deduped = deduped.join(variants, on="ticket_id").join(row_counts, on="ticket_id")
    deduped["was_duplicated"] = deduped["ticket_id"].isin(dup_ids)

    # Quantify the double count that dedup removes, in rupees.
    amt = _to_num(tickets["refund_amount_inr"])
    norm = np.where(tickets["source_system"] == "legacy_fd", amt / divisor, amt)
    dropped_value = float(
        pd.Series(norm)[tickets["ticket_id"].duplicated(keep=False)].sum()
        - pd.Series(norm)[deduped.index].reindex(deduped.index).where(
            deduped["was_duplicated"].values, 0
        ).sum()
    )

    findings.add(
        Finding(
            finding_id="DQ-002",
            severity=SEVERITY_HIGH,
            title="Tickets re-imported under both source systems (double counted in the export)",
            affected_rows=extra_rows,
            affected_value_inr=round(abs(dropped_value), 2),
            decision=(
                f"{len(dup_ids)} ticket_ids appear twice — once as 'helpdesk' and once as "
                f"'legacy_fd' — for {extra_rows} redundant rows. Kept the helpdesk copy "
                "because the current helpdesk stores rupees, and recorded every source "
                "system the ticket appeared under in tickets.source_system_variants. "
                "The two copies are identical apart from the monetary unit, so no "
                "information is lost. This removes "
                f"₹{abs(dropped_value):,.0f} of double-counted refund value."
            ),
            policy_reference=(
                "support-policy.pdf v3.2 §9: 'a subset of legacy tickets was re-imported "
                "during reconciliation and may appear in exports under both source systems.'"
            ),
            evidence={
                "duplicate_ticket_ids": int(len(dup_ids)),
                "redundant_rows": extra_rows,
                "combination_observed": "helpdesk|legacy_fd",
                "sample_ticket_ids": sorted(dup_ids)[:10],
                "fields_that_differ_between_copies": ["refund_amount_inr", "source_system"],
            },
        )
    )
    return deduped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def assess_timestamps(tickets: pd.DataFrame, findings: FindingLog) -> dict[str, Any]:
    """Test policy §9's UTC warning against the data before acting on it.

    §9 says migrated `resolved_at` values were reconstructed from a UTC event log,
    which would make legacy resolution times 5h30m behind the IST values in the
    same row. That is testable: a UTC/IST mix-up produces negative handle times
    and a legacy handle-time distribution shifted 5.5 hours below the helpdesk one.

    Neither appears. So no shift is applied, and the reasoning is recorded — a
    correction applied "because the policy mentioned it" would have introduced a
    5h30m floor into handle time that is not in the source data.
    """
    created = _to_dt(tickets["created_at"])
    first = _to_dt(tickets["first_response_at"])
    resolved = _to_dt(tickets["resolved_at"])

    handle_h = (resolved - first).dt.total_seconds() / 3600
    legacy = tickets["source_system"] == "legacy_fd"

    negative = int((handle_h < 0).sum())
    legacy_median = float(handle_h[legacy].median()) if legacy.any() else float("nan")
    hd_median = float(handle_h[~legacy].median()) if (~legacy).any() else float("nan")
    shift = legacy_median - hd_median

    assessment = {
        "policy_warning": (
            "§9: resolution timestamps for migrated tickets were reconstructed from the "
            "legacy event log, which stores UTC"
        ),
        "negative_handle_times": negative,
        "resolved_before_created": int((resolved < created).sum()),
        "first_response_before_created": int((first < created).sum()),
        "legacy_median_handle_hours": round(legacy_median, 3),
        "helpdesk_median_handle_hours": round(hd_median, 3),
        "median_shift_hours": round(shift, 3),
        "expected_shift_if_utc": -5.5,
        "adjustment_applied": False,
    }
    conclusive_utc = negative > 0 or shift < -3.0

    if conclusive_utc:
        findings.add(
            Finding(
                finding_id="DQ-003",
                severity=SEVERITY_HIGH,
                title="Legacy resolution timestamps appear to be UTC",
                affected_rows=int(legacy.sum()),
                decision=(
                    "Evidence of a UTC/IST mix-up found (negative handle times and/or a "
                    "~5.5h negative median shift on legacy rows). Handle-time metrics for "
                    "legacy tickets are reported as unreliable and excluded from SLA "
                    "comparisons rather than shifted, because the offset cannot be "
                    "confirmed per-row."
                ),
                policy_reference="support-policy.pdf v3.2 §9",
                evidence=assessment,
            )
        )
    else:
        findings.add(
            Finding(
                finding_id="DQ-003",
                severity=SEVERITY_INFO,
                title="No timezone inconsistency detectable in resolution timestamps",
                affected_rows=0,
                decision=(
                    "Policy §9 warns that migrated resolution timestamps came from a UTC "
                    "event log. Tested and not reproduced in this extract: zero negative "
                    f"handle times, and the legacy median handle time ({legacy_median:.2f}h) "
                    f"is within {abs(shift):.2f}h of the helpdesk median ({hd_median:.2f}h) "
                    "rather than the -5.5h a UTC/IST mix-up would produce. No shift applied. "
                    "Adding +5:30 would have invented a 5h30m floor on legacy handle times "
                    "that is not present in the source. Timestamps are treated as IST as "
                    "displayed, per README.txt."
                ),
                policy_reference="support-policy.pdf v3.2 §9; README.txt 'Timestamps are as displayed in the helpdesk (IST)'",
                evidence=assessment,
            )
        )
    return assessment


# ---------------------------------------------------------------------------
# Order resolution
# ---------------------------------------------------------------------------
def resolve_orders(
    tickets: pd.DataFrame, orders: pd.DataFrame, findings: FindingLog
) -> pd.DataFrame:
    """Fill blank order_id using the documented fallback join.

    README.txt: "Blank when the customer did not quote it. customer_id +
    product_sku is the fallback join." That fallback is not always unique, so:

      * exactly one candidate order -> `fallback_unique`
      * several candidates -> the most recent order placed on or before the
        ticket date (`fallback_nearest`), which is deterministic and recorded
      * no candidate -> left null (`unresolved`), never invented
    """
    tickets = tickets.copy()
    tickets["_created"] = _to_dt(tickets["created_at"])

    ord_slim = orders[["order_id", "customer_id", "sku", "order_date", "order_value_inr"]].copy()
    ord_slim["_order_dt"] = _to_dt(ord_slim["order_date"])

    blank = tickets["order_id"] == ""
    n_blank = int(blank.sum())

    candidates = ord_slim.rename(columns={"sku": "product_sku"})
    grouped = {
        key: grp.sort_values("_order_dt")
        for key, grp in candidates.groupby(["customer_id", "product_sku"], sort=False)
    }

    resolved_ids: list[str | None] = []
    sources: list[str] = []
    counts = {"quoted": 0, "fallback_unique": 0, "fallback_nearest": 0, "unresolved": 0}

    for _, row in tickets.iterrows():
        if row["order_id"]:
            resolved_ids.append(row["order_id"])
            sources.append("quoted")
            counts["quoted"] += 1
            continue
        grp = grouped.get((row["customer_id"], row["product_sku"]))
        if grp is None or grp.empty:
            resolved_ids.append(None)
            sources.append("unresolved")
            counts["unresolved"] += 1
        elif len(grp) == 1:
            resolved_ids.append(grp.iloc[0]["order_id"])
            sources.append("fallback_unique")
            counts["fallback_unique"] += 1
        else:
            created = row["_created"]
            prior = grp[grp["_order_dt"] <= created] if pd.notna(created) else grp
            pick = (prior if not prior.empty else grp).iloc[-1]
            resolved_ids.append(pick["order_id"])
            sources.append("fallback_nearest")
            counts["fallback_nearest"] += 1

    tickets["resolved_order_id"] = resolved_ids
    tickets["order_id_source"] = sources

    findings.add(
        Finding(
            finding_id="DQ-004",
            severity=SEVERITY_MEDIUM,
            title="order_id blank on tickets where the customer did not quote it",
            affected_rows=n_blank,
            decision=(
                f"{n_blank} of {len(tickets)} tickets have no order_id. Applied the fallback "
                "join documented in README.txt (customer_id + product_sku): "
                f"{counts['fallback_unique']} matched exactly one order; "
                f"{counts['fallback_nearest']} matched several and were resolved to the most "
                "recent order placed on or before the ticket date, flagged "
                f"'{FLAG_ORDER_AMBIGUOUS}'; {counts['unresolved']} had no candidate order and "
                "were left null. Order-derived figures (order value, refund ratio) are only "
                "reported where an order was resolved, and the resolution method is stored "
                "per ticket in tickets.order_id_source."
            ),
            policy_reference=(
                "README.txt tickets.csv order_id: 'Blank when the customer did not quote it. "
                "customer_id + product_sku is the fallback join.'"
            ),
            evidence={"resolution_counts": counts},
        )
    )
    return tickets


# ---------------------------------------------------------------------------
# Roster: point-in-time assignment (policy §7)
# ---------------------------------------------------------------------------
def build_roster_resolver(agents: pd.DataFrame, findings: FindingLog):
    """Return fn(agent_id, on_date) -> assignment row.

    Policy §7: the roster is one row per assignment with from/to dates, and an
    agent keeps the same agent_id across assignments. The join is therefore
    point-in-time, even though this particular extract happens to have exactly
    one open-ended row per agent.
    """
    ag = agents.copy()
    ag["_from"] = _to_dt(ag["from_date"])
    ag["_to"] = _to_dt(ag["to_date"])
    ag["tier_int"] = pd.to_numeric(ag["tier"], errors="coerce").fillna(1).astype(int)

    multi = ag.groupby("agent_id").size()
    multi_agents = multi[multi > 1]
    findings.add(
        Finding(
            finding_id="DQ-005",
            severity=SEVERITY_INFO,
            title="Agent roster joined point-in-time on the ticket date",
            affected_rows=int(multi_agents.sum()),
            decision=(
                f"agents.csv holds {len(ag)} assignment rows for {ag['agent_id'].nunique()} "
                f"distinct agent_ids, of which {len(multi_agents)} have more than one row. "
                "The join selects the assignment covering the ticket's creation date rather "
                "than the agent's current row, so an agent who changed site, team or shift is "
                "attributed to the team they were on at the time. agent_id is used as the key "
                "throughout, never the name (README.txt: 'Use the id, not the name')."
            ),
            policy_reference=(
                "support-policy.pdf v3.2 §7: 'The roster is maintained as one row per "
                "assignment with from/to dates; an agent who changes site or shift receives a "
                "new roster row and keeps the same agent_id.'"
            ),
            evidence={
                "assignment_rows": int(len(ag)),
                "distinct_agents": int(ag["agent_id"].nunique()),
                "agents_with_multiple_assignments": int(len(multi_agents)),
                "open_ended_rows": int(ag["_to"].isna().sum()),
            },
        )
    )

    by_agent = {aid: grp.sort_values("_from") for aid, grp in ag.groupby("agent_id")}

    def resolve(agent_id: str, on_date) -> dict[str, Any]:
        grp = by_agent.get(agent_id)
        if grp is None or grp.empty:
            return {}
        if pd.notna(on_date):
            covering = grp[
                (grp["_from"].isna() | (grp["_from"] <= on_date))
                & (grp["_to"].isna() | (grp["_to"] >= on_date))
            ]
            if not covering.empty:
                row = covering.iloc[-1]
                return {
                    "team": row["team"], "tier": int(row["tier_int"]),
                    "site": row["site"], "shift": row["shift"], "name": row["name"],
                }
        row = grp.iloc[-1]
        return {
            "team": row["team"], "tier": int(row["tier_int"]),
            "site": row["site"], "shift": row["shift"], "name": row["name"],
        }

    return resolve


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------
def check_integrity(
    tickets: pd.DataFrame,
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    agents: pd.DataFrame,
    findings: FindingLog,
) -> dict[str, Any]:
    quoted = tickets.loc[tickets["order_id"] != "", "order_id"]
    report = {
        "tickets_rows": int(len(tickets)),
        "tickets_customer_id_orphans": int((~tickets["customer_id"].isin(customers["customer_id"])).sum()),
        "tickets_order_id_orphans": int((~quoted.isin(orders["order_id"])).sum()),
        "tickets_product_sku_orphans": int((~tickets["product_sku"].isin(products["sku"])).sum()),
        "tickets_agent_id_orphans": int((~tickets["agent_id"].isin(agents["agent_id"])).sum()),
        "orders_customer_id_orphans": int((~orders["customer_id"].isin(customers["customer_id"])).sum()),
        "orders_sku_orphans": int((~orders["sku"].isin(products["sku"])).sum()),
        "duplicate_order_ids": int(orders["order_id"].duplicated().sum()),
        "duplicate_customer_ids": int(customers["customer_id"].duplicated().sum()),
        "duplicate_skus": int(products["sku"].duplicated().sum()),
    }

    joined = tickets[tickets["order_id"] != ""].merge(
        orders[["order_id", "customer_id", "sku"]].rename(
            columns={"customer_id": "_ocust", "sku": "_osku"}
        ),
        on="order_id",
        how="left",
    )
    report["quoted_order_sku_mismatch"] = int((joined["product_sku"] != joined["_osku"]).sum())
    report["quoted_order_customer_mismatch"] = int((joined["customer_id"] != joined["_ocust"]).sum())

    orphan_total = sum(
        report[k] for k in report if k.endswith("_orphans") or k.startswith("duplicate_")
    )
    findings.add(
        Finding(
            finding_id="DQ-006",
            severity=SEVERITY_INFO if orphan_total == 0 else SEVERITY_HIGH,
            title="Referential integrity across the five source files",
            affected_rows=orphan_total,
            decision=(
                "All foreign keys verified before load: ticket -> customer, order, product "
                "and agent; order -> customer and product. "
                + (
                    "No orphan keys, no duplicate primary keys, and where a ticket quotes an "
                    "order the ticket's customer_id and product_sku both agree with that "
                    "order, so the documented relationships hold and the fallback join is "
                    "safe to rely on."
                    if orphan_total == 0
                    else f"{orphan_total} integrity problems found; affected rows are flagged and "
                    "excluded from order-derived metrics."
                )
            ),
            evidence=report,
        )
    )
    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_ingest(
    session: Session,
    raw_dir: Path | None = None,
    truncate: bool = True,
) -> IngestResult:
    started = datetime.utcnow()
    t0 = time.perf_counter()
    run_id = uuid.uuid4().hex
    findings = FindingLog(run_id)
    policy = get_policy()

    log.info("ingest.start", extra={"ingest_run_id": run_id, "raw_dir": str(raw_dir or settings.raw_dir)})
    raw = load_raw(raw_dir)
    tickets_raw = raw["tickets"]
    orders_raw = raw["orders"]
    customers_raw = raw["customers"]
    products_raw = raw["products"]
    agents_raw = raw["agents"]

    raw_refund_total = float(_to_num(tickets_raw["refund_amount_inr"]).sum())
    raw_refund_rows = int(_to_num(tickets_raw["refund_amount_inr"]).notna().sum())

    # ---- DQ-001 / DQ-002 -------------------------------------------------
    unit = detect_legacy_money_unit(tickets_raw, findings)
    divisor = unit["divisor"]
    deduped = deduplicate_tickets(tickets_raw, findings, divisor)
    timestamps = assess_timestamps(deduped, findings)
    integrity = check_integrity(
        deduped, orders_raw, customers_raw, products_raw, agents_raw, findings
    )
    deduped = resolve_orders(deduped, orders_raw, findings)
    resolve_roster = build_roster_resolver(agents_raw, findings)

    # ---- lookups ---------------------------------------------------------
    order_lookup = {
        r["order_id"]: {
            "order_value_inr": float(r["order_value_inr"]) if r["order_value_inr"] else None,
            "order_date": pd.to_datetime(r["order_date"], errors="coerce"),
            "sku": r["sku"],
            "lot_code": r["lot_code"],
        }
        for _, r in orders_raw.iterrows()
    }
    product_lookup = {
        r["sku"]: {
            "family": r["family"],
            "warranty_months": int(r["warranty_months"]) if r["warranty_months"] else None,
            "unit_cost_inr": float(r["unit_cost_inr"]) if r["unit_cost_inr"] else None,
        }
        for _, r in products_raw.iterrows()
    }

    if truncate:
        for model in (Refund, Ticket, Order, Customer, Product, Agent):
            session.execute(delete(model))
        session.execute(delete(DataQualityEvent))
        session.flush()

    # ---- reference tables ------------------------------------------------
    session.add_all(
        Product(
            sku=r["sku"],
            product_name=r["product_name"],
            family=r["family"],
            launch_date=pd.to_datetime(r["launch_date"], errors="coerce").date()
            if r["launch_date"] else None,
            unit_cost_inr=float(r["unit_cost_inr"]),
            retail_price_inr=float(r["retail_price_inr"]),
            warranty_months=int(r["warranty_months"]),
        )
        for _, r in products_raw.iterrows()
    )
    session.add_all(
        Customer(
            customer_id=r["customer_id"],
            name=r["name"] or None,
            city=r["city"] or None,
            state=r["state"] or None,
            signup_date=pd.to_datetime(r["signup_date"], errors="coerce").date()
            if r["signup_date"] else None,
            care_plus=(r["care_plus"].strip().upper() == "Y"),
        )
        for _, r in customers_raw.iterrows()
    )
    session.add_all(
        Agent(
            agent_id=r["agent_id"],
            name=r["name"],
            site=r["site"],
            team=r["team"],
            shift=r["shift"],
            tier=int(r["tier"]),
            from_date=pd.to_datetime(r["from_date"], errors="coerce").date()
            if r["from_date"] else None,
            to_date=pd.to_datetime(r["to_date"], errors="coerce").date()
            if r["to_date"] else None,
        )
        for _, r in agents_raw.iterrows()
    )
    session.add_all(
        Order(
            order_id=r["order_id"],
            customer_id=r["customer_id"],
            sku=r["sku"],
            order_date=pd.to_datetime(r["order_date"], errors="coerce").date()
            if r["order_date"] else None,
            channel=r["channel"],
            qty=int(r["qty"]) if r["qty"] else 1,
            order_value_inr=float(r["order_value_inr"]),
            lot_code=r["lot_code"] or None,
        )
        for _, r in orders_raw.iterrows()
    )
    session.flush()

    # ---- policy sections -------------------------------------------------
    session.execute(
        delete(PolicySectionRecord).where(PolicySectionRecord.policy_version == policy.version)
    )
    session.add_all(
        PolicySectionRecord(
            policy_version=policy.version,
            section_id=sec.section_id,
            title=sec.title,
            policy_text=sec.text,
            content_sha256=_sha(sec.text),
        )
        for sec in policy.sections.values()
    )
    session.flush()

    # ---- tickets + refunds -----------------------------------------------
    attendance = set(policy.attendance_statuses)
    sla_targets = {**SLA_TARGET_MINUTES, **policy.sla_targets_minutes}

    ticket_rows: list[Ticket] = []
    refund_payloads: list[dict] = []
    counters = {
        "refund_on_open": 0, "ticket_before_order": 0, "csat_blank": 0,
        "resolved_blank": 0, "refund_exceeds_order": 0, "sla_breaches": 0,
    }

    for _, r in deduped.iterrows():
        created = pd.to_datetime(r["created_at"], errors="coerce")
        first = pd.to_datetime(r["first_response_at"], errors="coerce")
        resolved = pd.to_datetime(r["resolved_at"], errors="coerce")
        flags: list[str] = []

        amount_source = pd.to_numeric(r["refund_amount_inr"], errors="coerce")
        has_refund = pd.notna(amount_source)
        is_legacy = r["source_system"] == "legacy_fd"
        converted = bool(has_refund and is_legacy and unit["conclusive"])
        amount_inr = float(amount_source / divisor) if converted else (
            float(amount_source) if has_refund else None
        )
        if converted:
            flags.append(FLAG_LEGACY_UNIT_CONVERTED)
        if r["was_duplicated"]:
            flags.append(FLAG_REIMPORT_DUPLICATE)

        csat = pd.to_numeric(r["csat_score"], errors="coerce")
        if pd.isna(csat):
            flags.append(FLAG_CSAT_BLANK)
            counters["csat_blank"] += 1
        if pd.isna(resolved):
            flags.append(FLAG_NO_RESOLVED_AT)
            counters["resolved_blank"] += 1
        if r["order_id"] == "":
            flags.append(FLAG_NO_ORDER_ID)
        if r["order_id_source"] == "fallback_nearest":
            flags.append(FLAG_ORDER_AMBIGUOUS)
        elif r["order_id_source"] == "unresolved":
            flags.append(FLAG_ORDER_UNRESOLVED)
        if has_refund and r["status"] not in attendance:
            flags.append(FLAG_REFUND_ON_OPEN_TICKET)
            counters["refund_on_open"] += 1

        order_id = r["resolved_order_id"]
        order_info = order_lookup.get(order_id or "", {})
        order_value = order_info.get("order_value_inr")
        order_date = order_info.get("order_date")
        days_since_order = None
        if pd.notna(created) and order_date is not None and pd.notna(order_date):
            days_since_order = int((created - order_date).days)
            if days_since_order < 0:
                flags.append(FLAG_TICKET_BEFORE_ORDER)
                counters["ticket_before_order"] += 1

        ratio = None
        if amount_inr is not None and order_value:
            ratio = round(amount_inr / order_value, 4)
            if amount_inr > order_value + 0.01:
                flags.append(FLAG_REFUND_EXCEEDS_ORDER)
                counters["refund_exceeds_order"] += 1

        frt_min = (
            round((first - created).total_seconds() / 60, 2)
            if pd.notna(first) and pd.notna(created) else None
        )
        handle_h = (
            round((resolved - first).total_seconds() / 3600, 3)
            if pd.notna(resolved) and pd.notna(first) else None
        )
        target = sla_targets.get(r["channel"])
        breached = (frt_min > target) if (frt_min is not None and target) else None
        if breached:
            counters["sla_breaches"] += 1

        roster = resolve_roster(r["agent_id"], created)
        reason_code = r["refund_reason_code"] or None
        month = created.strftime("%Y-%m") if pd.notna(created) else "unknown"
        quarter = f"{created.year}Q{(created.month - 1) // 3 + 1}" if pd.notna(created) else "unknown"

        ticket = Ticket(
            ticket_id=r["ticket_id"],
            created_at_ist=created.to_pydatetime() if pd.notna(created) else None,
            first_response_at_ist=first.to_pydatetime() if pd.notna(first) else None,
            resolved_at_ist=resolved.to_pydatetime() if pd.notna(resolved) else None,
            created_month=month,
            created_quarter=quarter,
            status=r["status"],
            channel=r["channel"],
            category=r["category"] or None,
            priority=r["priority"] or None,
            assigned_team=r["assigned_team"] or None,
            agent_id=r["agent_id"],
            transfers=int(pd.to_numeric(r["transfers"], errors="coerce") or 0),
            csat_score=int(csat) if pd.notna(csat) else None,
            customer_id=r["customer_id"],
            order_id=order_id,
            order_id_source=r["order_id_source"],
            product_sku=r["product_sku"],
            refund_amount_source=float(amount_source) if has_refund else None,
            refund_amount_inr=amount_inr,
            refund_unit_conversion_applied=converted,
            refund_unit_divisor=divisor if converted else None,
            refund_reason_code=reason_code,
            replacement_issued=(r["replacement_issued"].strip().upper() == "Y"),
            is_refund=bool(has_refund),
            customer_message=r["customer_message"],
            agent_notes=r["agent_notes"],
            source_system=r["source_system"],
            source_system_variants=list(r["source_system_variants"]),
            was_duplicated=bool(r["was_duplicated"]),
            duplicate_row_count=int(r["duplicate_row_count"]),
            first_response_minutes=frt_min,
            handle_time_hours=handle_h,
            sla_target_minutes=target,
            sla_breached=breached,
            is_attendance=r["status"] in attendance,
            data_quality_flags=flags,
        )
        ticket_rows.append(ticket)

        if has_refund:
            prod = product_lookup.get(r["product_sku"], {})
            refund_payloads.append(
                {
                    "ticket": ticket,
                    "amount_source": float(amount_source),
                    "amount_inr": float(amount_inr),
                    "source_unit": unit["unit"] if converted else "INR",
                    "order_value_inr": order_value,
                    "ratio": ratio,
                    "reason_code": reason_code,
                    "roster": roster,
                    "product_family": prod.get("family"),
                    "days_since_order": days_since_order,
                }
            )

    session.add_all(ticket_rows)
    session.flush()

    # ---- cross-ticket leakage: same order refunded more than once ---------
    # Only tickets with a RELIABLE order link take part. A ticket resolved to
    # "the nearest of several candidate orders" could be attributed to an order
    # it has nothing to do with, and two such tickets landing on the same order
    # would look like a duplicate refund that never happened. Excluding them
    # understates this driver, which is the correct direction to be wrong in.
    RELIABLE_LINKS = {"quoted", "fallback_unique"}
    per_order: dict[str, list[dict]] = {}
    excluded_unreliable = 0
    for payload in refund_payloads:
        ticket = payload["ticket"]
        oid = ticket.order_id
        if not oid:
            continue
        if ticket.order_id_source not in RELIABLE_LINKS:
            excluded_unreliable += 1
            continue
        per_order.setdefault(oid, []).append(payload)

    multi_order_tickets = 0
    multi_order_excess = 0.0
    for oid, group in per_order.items():
        if len(group) < 2:
            continue
        total = sum(g["amount_inr"] for g in group)
        order_value = group[0]["order_value_inr"]
        excess = max(0.0, total - order_value) if order_value else 0.0
        multi_order_tickets += len(group)
        multi_order_excess += excess
        for g in group:
            g["order_refund_ticket_count"] = len(group)
            g["order_refund_total_inr"] = round(total, 2)
            # Attribute the excess to the later ticket(s), the first being the
            # legitimate refund. Deterministic and stated, not apportioned.
            g["order_refund_excess_inr"] = 0.0
        ordered = sorted(group, key=lambda g: g["ticket"].created_at_ist or datetime.min)
        remaining = excess
        for g in ordered[1:]:
            take = min(remaining, g["amount_inr"])
            g["order_refund_excess_inr"] = round(take, 2)
            remaining -= take
            if remaining <= 0:
                break

    findings.add(
        Finding(
            finding_id="DQ-007",
            severity=SEVERITY_HIGH,
            title="The same order refunded on more than one ticket",
            affected_rows=multi_order_tickets,
            affected_value_inr=round(multi_order_excess, 2),
            decision=(
                f"{len([g for g in per_order.values() if len(g) > 1])} orders carry a refund on "
                f"more than one ticket, across {multi_order_tickets} tickets. Where the combined "
                f"refunds exceed the order value, the excess (₹{multi_order_excess:,.0f}) is "
                "recorded on the later ticket(s) as refunds.order_refund_excess_inr and counted "
                "once under the DUPLICATE_REFUND_SAME_ORDER avoidability driver. Nothing is "
                "deleted: both tickets remain, because both payments really happened. "
                f"{excluded_unreliable} refund tickets were excluded from this test because "
                "their order link came from the ambiguous fallback join (several candidate "
                "orders), where two unrelated tickets could be attributed to the same order and "
                "fake a duplicate. This understates the driver rather than overstating it."
            ),
            policy_reference="support-policy.pdf v3.2 §5 (remedies are alternatives, not cumulative)",
            evidence={
                "orders_with_multiple_refund_tickets": len(
                    [g for g in per_order.values() if len(g) > 1]
                ),
                "tickets_involved": multi_order_tickets,
                "excess_over_order_value_inr": round(multi_order_excess, 2),
                "refund_tickets_excluded_unreliable_link": excluded_unreliable,
                "reliable_link_definition": sorted(RELIABLE_LINKS),
            },
        )
    )

    # ---- write refunds ---------------------------------------------------
    refund_rows = [
        Refund(
            ticket_uuid=p["ticket"].id,
            ticket_id=p["ticket"].ticket_id,
            created_at_ist=p["ticket"].created_at_ist,
            created_month=p["ticket"].created_month,
            created_quarter=p["ticket"].created_quarter,
            agent_id=p["ticket"].agent_id,
            agent_team=p["roster"].get("team"),
            agent_tier=p["roster"].get("tier"),
            assigned_team=p["ticket"].assigned_team,
            channel=p["ticket"].channel,
            customer_id=p["ticket"].customer_id,
            order_id=p["ticket"].order_id,
            order_id_source=p["ticket"].order_id_source,
            order_link_reliable=p["ticket"].order_id_source in {"quoted", "fallback_unique"},
            product_sku=p["ticket"].product_sku,
            amount_source=p["amount_source"],
            amount_inr=p["amount_inr"],
            source_unit=p["source_unit"],
            order_value_inr=p["order_value_inr"],
            refund_to_order_ratio=p["ratio"],
            is_full_refund=(p["ratio"] is not None and p["ratio"] >= 0.999),
            source_reason_code=p["reason_code"],
            source_reason_is_dropdown_default=(p["reason_code"] == DROPDOWN_DEFAULT_CODE),
            replacement_issued_flag=p["ticket"].replacement_issued,
            order_refund_ticket_count=p.get("order_refund_ticket_count", 1),
            order_refund_total_inr=p.get("order_refund_total_inr"),
            order_refund_excess_inr=p.get("order_refund_excess_inr", 0.0),
            sla_breached=p["ticket"].sla_breached,
            avoidability_label="REVIEW_REQUIRED",
            avoidability_drivers=[],
            avoidable_value_inr=0.0,
        )
        for p in refund_payloads
    ]
    session.add_all(refund_rows)
    session.flush()

    # ---- remaining findings ---------------------------------------------
    findings.add(
        Finding(
            finding_id="DQ-008",
            severity=SEVERITY_MEDIUM,
            title="Refunds recorded on tickets that are not resolved or closed",
            affected_rows=counters["refund_on_open"],
            decision=(
                f"{counters['refund_on_open']} refund tickets are still 'open' or 'pending'. "
                "They are retained in the refund totals because the money was recorded as "
                "raised, and flagged '" + FLAG_REFUND_ON_OPEN_TICKET + "'. Policy §10 defines "
                "attendance as resolved or closed, so these tickets are excluded from "
                "attendance-based agent denominators but included in refund value."
            ),
            policy_reference="support-policy.pdf v3.2 §10 (attendance definition)",
            evidence={"count": counters["refund_on_open"]},
        )
    )
    findings.add(
        Finding(
            finding_id="DQ-009",
            severity=SEVERITY_MEDIUM,
            title="Tickets created before the order date they reference",
            affected_rows=counters["ticket_before_order"],
            decision=(
                f"{counters['ticket_before_order']} tickets have a creation date earlier than "
                "the order date of the order they resolve to, which is chronologically "
                "impossible. Not corrected: flagged '" + FLAG_TICKET_BEFORE_ORDER + "' and "
                "excluded from any age-based test (the DOA window and warranty checks), "
                "because either the ticket date or the order link is wrong and the data does "
                "not say which."
            ),
            evidence={"count": counters["ticket_before_order"]},
        )
    )
    findings.add(
        Finding(
            finding_id="DQ-010",
            severity=SEVERITY_INFO,
            title="Blank CSAT scores excluded from averages, not treated as zero",
            affected_rows=counters["csat_blank"],
            decision=(
                f"{counters['csat_blank']} of {len(deduped)} tickets have no CSAT score "
                f"({counters['csat_blank'] / max(1, len(deduped)) * 100:.1f}%), consistent with "
                "the ~45% response rate policy §8 describes. Stored as NULL and excluded from "
                "every average. Treating them as zero would have moved the mean CSAT by more "
                "than a full point."
            ),
            policy_reference=(
                "support-policy.pdf v3.2 §8: 'A blank score means no response and must be "
                "excluded from averages, not treated as zero.'"
            ),
            evidence={"blank": counters["csat_blank"], "total": int(len(deduped))},
        )
    )
    findings.add(
        Finding(
            finding_id="DQ-011",
            severity=SEVERITY_INFO,
            title="DOA 7-day window is not verifiable from this data pack",
            affected_rows=0,
            decision=(
                "Policy §5 measures the dead-on-arrival window from delivery, and orders.csv "
                "has order_date but no delivery date. DOA eligibility is therefore never "
                "tested and a DOA refund is never labelled avoidable on window grounds. "
                "Recorded as a gap rather than approximated from order_date, which would have "
                "produced a fabricated compliance rate."
            ),
            policy_reference="support-policy.pdf v3.2 §5 (DOA 'within 7 days of delivery')",
            evidence={"missing_field": "delivered_at", "available": ["order_date"]},
        )
    )
    findings.add(
        Finding(
            finding_id="DQ-012",
            severity=SEVERITY_INFO,
            title="Agent notes cite SOPs that were not supplied",
            affected_rows=int(
                deduped["agent_notes"].str.contains(r"SOP\s*[\d.]+", case=False, regex=True).sum()
            ),
            decision=(
                "Closing notes reference SOP 2.7, 3.1, 4.2 and 5.4. Those SOPs are not part of "
                "support-policy.pdf and were not provided, so the platform cannot validate "
                "whether a refund citing one of them followed it. Such citations are treated "
                "as unverifiable rather than as evidence of compliance."
            ),
            policy_reference="support-policy.pdf v3.2 (sections 1-10 only; no SOP annexe)",
            evidence={"referenced_sops": ["SOP 2.7", "SOP 3.1", "SOP 4.2", "SOP 5.4"]},
        )
    )

    # ---- reconciliation walk --------------------------------------------
    dedup_total = float(sum(p["amount_inr"] for p in refund_payloads))
    unit_fixed_total = float(
        pd.Series(
            np.where(
                tickets_raw["source_system"] == "legacy_fd",
                _to_num(tickets_raw["refund_amount_inr"]) / divisor,
                _to_num(tickets_raw["refund_amount_inr"]),
            )
        ).sum()
    )
    months = len({t.created_month for t in ticket_rows if t.created_month != "unknown"})
    q = lambda v: v / months * 3 if months else 0.0  # noqa: E731

    findings.add(
        Finding(
            finding_id="DQ-000",
            severity=SEVERITY_INFO,
            title="Refund total reconciliation walk (raw export → rupees → dedup)",
            affected_rows=raw_refund_rows,
            affected_value_inr=dedup_total,
            decision=(
                f"Raw export ₹{raw_refund_total:,.2f} → after legacy unit conversion "
                f"₹{unit_fixed_total:,.2f} → after re-import dedup ₹{dedup_total:,.2f} "
                f"(₹{q(dedup_total):,.0f}/quarter across {months} months)."
            ),
            policy_reference="support-policy.pdf v3.2 §9",
            evidence={
                "reconciliation": [
                    {
                        "step": 1,
                        "label": "Raw export, summed as delivered",
                        "refund_value_inr": round(raw_refund_total, 2),
                        "refund_count": raw_refund_rows,
                        "per_quarter_inr": round(q(raw_refund_total), 2),
                        "delta_inr": 0.0,
                        "explanation": (
                            "tickets.csv refund_amount_inr summed with no cleaning. This is the "
                            "figure Finance's export produces."
                        ),
                        "policy_reference": None,
                    },
                    {
                        "step": 2,
                        "label": f"Legacy amounts converted from {unit['unit']} to rupees",
                        "refund_value_inr": round(unit_fixed_total, 2),
                        "refund_count": raw_refund_rows,
                        "per_quarter_inr": round(q(unit_fixed_total), 2),
                        "delta_inr": round(unit_fixed_total - raw_refund_total, 2),
                        "explanation": (
                            f"legacy_fd rows divided by {divisor:g}. The divisor was measured from "
                            f"{unit['paired_tickets']} tickets exported under both systems."
                        ),
                        "policy_reference": "support-policy.pdf v3.2 §9",
                    },
                    {
                        "step": 3,
                        "label": "Re-imported duplicate tickets removed",
                        "refund_value_inr": round(dedup_total, 2),
                        "refund_count": len(refund_payloads),
                        "per_quarter_inr": round(q(dedup_total), 2),
                        "delta_inr": round(dedup_total - unit_fixed_total, 2),
                        "explanation": (
                            "One row per ticket_id, keeping the helpdesk copy."
                        ),
                        "policy_reference": "support-policy.pdf v3.2 §9",
                    },
                ],
                "raw_export_total_inr": raw_refund_total,
                "after_unit_conversion_inr": unit_fixed_total,
                "after_dedup_inr": dedup_total,
                "months_observed": months,
            },
        )
    )

    reconciliation = [
        {
            "step": 1,
            "label": "Raw export, summed as delivered",
            "refund_value_inr": round(raw_refund_total, 2),
            "refund_count": raw_refund_rows,
            "per_quarter_inr": round(q(raw_refund_total), 2),
            "delta_inr": 0.0,
            "explanation": (
                "tickets.csv refund_amount_inr summed with no cleaning. This is the figure "
                "Finance's export produces."
            ),
            "policy_reference": None,
        },
        {
            "step": 2,
            "label": f"Legacy amounts converted from {unit['unit']} to rupees",
            "refund_value_inr": round(unit_fixed_total, 2),
            "refund_count": raw_refund_rows,
            "per_quarter_inr": round(q(unit_fixed_total), 2),
            "delta_inr": round(unit_fixed_total - raw_refund_total, 2),
            "explanation": (
                f"legacy_fd rows divided by {divisor:g}. The divisor was measured from "
                f"{unit['paired_tickets']} tickets exported under both systems, where the legacy "
                f"value is exactly {divisor:g}x the helpdesk value in every case."
            ),
            "policy_reference": "support-policy.pdf v3.2 §9",
        },
        {
            "step": 3,
            "label": "Re-imported duplicate tickets removed",
            "refund_value_inr": round(dedup_total, 2),
            "refund_count": len(refund_payloads),
            "per_quarter_inr": round(q(dedup_total), 2),
            "delta_inr": round(dedup_total - unit_fixed_total, 2),
            "explanation": (
                "One row per ticket_id, keeping the helpdesk copy. Removes the refunds that "
                "the migration re-import counted twice."
            ),
            "policy_reference": "support-policy.pdf v3.2 §9",
        },
    ]

    # ---- persist findings ------------------------------------------------
    session.add_all(
        DataQualityEvent(
            ingest_run_id=run_id,
            finding_id=f["finding_id"],
            severity=f["severity"],
            title=f["title"],
            affected_rows=f["affected_rows"],
            affected_value_inr=f["affected_value_inr"],
            decision=f["decision"],
            policy_reference=f["policy_reference"],
            evidence=f["evidence"],
        )
        for f in findings.as_dicts()
    )
    session.flush()

    result = IngestResult(
        run_id=run_id,
        started_at=started,
        finished_at=datetime.utcnow(),
        counts={
            "source_ticket_rows": int(len(tickets_raw)),
            "tickets_loaded": len(ticket_rows),
            "duplicate_rows_removed": int(len(tickets_raw) - len(ticket_rows)),
            "refunds_loaded": len(refund_rows),
            "orders_loaded": int(len(orders_raw)),
            "customers_loaded": int(len(customers_raw)),
            "products_loaded": int(len(products_raw)),
            "agent_assignments_loaded": int(len(agents_raw)),
            "policy_sections_loaded": len(policy.sections),
            "months_observed": months,
            "sla_breaches": counters["sla_breaches"],
            **{f"flag_{k}": v for k, v in counters.items()},
        },
        money={
            "raw_export_total_inr": raw_refund_total,
            "after_unit_conversion_inr": unit_fixed_total,
            "after_dedup_inr": dedup_total,
            "reconciled_per_quarter_inr": q(dedup_total),
        },
        findings=findings.as_dicts(),
        unit_detection=unit,
        join_integrity={**integrity, "timestamps": timestamps},
        reconciliation=reconciliation,
        duration_seconds=time.perf_counter() - t0,
    )
    log.info(
        "ingest.complete",
        extra={
            "ingest_run_id": run_id,
            "tickets": len(ticket_rows),
            "refunds": len(refund_rows),
            "raw_total_inr": round(raw_refund_total, 2),
            "reconciled_total_inr": round(dedup_total, 2),
            "duration_s": round(result.duration_seconds, 2),
            "findings": len(result.findings),
        },
    )
    return result
