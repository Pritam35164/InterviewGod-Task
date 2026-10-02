"""Evaluation against a manually reviewed, stratified ticket set.

How the labels were produced, stated plainly because it bounds what the accuracy
number means:

This is single-annotator labelling by the person who also wrote the taxonomy and
the prompt. Labels in `data/evaluation_set.json` are marked `reviewed: true`
because they were assigned from ticket text against a written rubric
(`scripts/build_eval_set.py`), never from a model prediction for that ticket.
There is no second annotator and therefore no inter-annotator agreement figure,
so the accuracy below measures agreement with one reviewer's reading, not ground
truth. It is reported as such.

The same set scores the deterministic keyword baseline, so the model's lift is
measured rather than claimed.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.decisions import EvaluationResult, TicketClassification
from app.models.refund import Refund
from app.models.source import Ticket
from app.schemas.taxonomy import AvoidabilityLabel, RefundReason
from app.services.baseline import baseline_dual_remedy, classify_baseline

log = get_logger(__name__)

EVAL_SET_PATH = settings.repo_root / "data" / "evaluation_set.json"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval.

    Used instead of the normal approximation because n is ~100 and accuracy is
    high, where the normal interval overshoots past 1.0 and understates the lower
    bound.
    """
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


# ---------------------------------------------------------------------------
# Evaluation set I/O
# ---------------------------------------------------------------------------
def load_eval_set(path: Path | None = None) -> list[dict]:
    p = path or EVAL_SET_PATH
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("tickets", data) if isinstance(data, dict) else data


def eval_set_status(path: Path | None = None) -> dict:
    rows = load_eval_set(path)
    reviewed = [r for r in rows if r.get("reviewed")]
    return {
        "path": str(path or EVAL_SET_PATH),
        "exists": bool(rows),
        "total": len(rows),
        "reviewed": len(reviewed),
        "unreviewed": len(rows) - len(reviewed),
        "strata": dict(Counter(r.get("stratum", "?") for r in rows)),
    }


# ---------------------------------------------------------------------------
# Error categorisation
# ---------------------------------------------------------------------------
def categorise_error(expected: str, predicted: str, ticket: Ticket | None) -> tuple[str, str]:
    """Bucket an error so the failure modes can be reported, not just the rate."""
    if predicted == RefundReason.UNKNOWN:
        return (
            "abstained_unknown",
            "Model returned UNKNOWN where the reviewer read a reason. Safe direction: the "
            "ticket goes to review rather than into the totals under a wrong reason.",
        )
    if predicted == RefundReason.AMBIGUOUS:
        return (
            "abstained_ambiguous",
            "Model returned AMBIGUOUS where the reviewer committed to a reason.",
        )
    service_pair = {RefundReason.REFUND_SERVICE_FAILURE, RefundReason.RETURN_RECEIVED_QC_PASSED}
    if {expected, predicted} <= service_pair:
        return (
            "refund_delay_vs_qc_return",
            "Confused a ticket about an earlier refund not landing with a normal QC'd return. "
            "Both notes mention the refund being processed after checking a reverse-pickup or "
            "payment-gateway status, and the distinction is which event the ticket is about.",
        )
    fault_pair = {RefundReason.PRODUCT_FAULT_DOA, RefundReason.PRODUCT_FAULT_IN_WARRANTY}
    if {expected, predicted} <= fault_pair:
        return (
            "doa_vs_in_warranty",
            "Product fault classified in the wrong window. Genuinely undecidable from this data "
            "pack: the DOA window runs from delivery and orders.csv has no delivery date.",
        )
    transit_pair = {RefundReason.DAMAGED_IN_TRANSIT, RefundReason.NOT_DELIVERED_OR_LOST}
    if {expected, predicted} <= transit_pair:
        return (
            "damaged_vs_lost_in_transit",
            "Both are transit failures under the same policy remedy, so the money is unaffected; "
            "only the reason breakdown shifts.",
        )
    cancel_pair = {RefundReason.CANCELLED_BEFORE_DISPATCH, RefundReason.CUSTOMER_CHANGED_MIND}
    if {expected, predicted} <= cancel_pair:
        return (
            "cancellation_vs_change_of_mind",
            "A change of mind before dispatch is both. Overlapping categories in the taxonomy "
            "rather than a model failure.",
        )
    if predicted == RefundReason.GOODWILL_GESTURE:
        return (
            "over_assigned_goodwill",
            "Model fell back to goodwill where a specific reason was available. This one matters: "
            "it inflates the GOODWILL_OVER_CAP avoidability driver.",
        )
    if expected == RefundReason.GOODWILL_GESTURE:
        return (
            "under_assigned_goodwill",
            "Model found a specific reason where the reviewer read a discretionary payment. "
            "Understates the goodwill-over-cap driver.",
        )
    return ("other_confusion", f"Reviewer read {expected}; model answered {predicted}.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_evaluation(
    session: Session, path: Path | None = None, run_id: str | None = None
) -> dict:
    run_id = run_id or uuid.uuid4().hex
    rows = [r for r in load_eval_set(path) if r.get("reviewed")]

    if not rows:
        status = eval_set_status(path)
        return {
            "status": "no_reviewed_labels",
            "run_id": run_id,
            "sample_size": 0,
            "message": (
                "The evaluation set has no human-reviewed labels, so no accuracy can be "
                "reported. Build it with scripts/build_eval_set.py, review the labels, then "
                "re-run. No accuracy figure is produced from unreviewed labels."
            ),
            "eval_set": status,
        }

    ticket_ids = [r["ticket_id"] for r in rows]
    tickets = {
        t.ticket_id: t
        for t in session.execute(select(Ticket).where(Ticket.ticket_id.in_(ticket_ids)))
        .scalars()
        .all()
    }
    classifications = {
        c.ticket_id: c
        for c in session.execute(
            select(TicketClassification).where(TicketClassification.ticket_id.in_(ticket_ids))
        )
        .scalars()
        .all()
    }
    refunds = {
        r.ticket_id: r
        for r in session.execute(select(Refund).where(Refund.ticket_id.in_(ticket_ids)))
        .scalars()
        .all()
    }

    session.execute(
        EvaluationResult.__table__.delete().where(EvaluationResult.run_id == run_id)
    )

    reason_hits = reason_total = 0
    baseline_hits = baseline_total = 0
    avoid_hits = avoid_total = 0
    schema_failures = retried = unknown = ambiguous = escalated = 0
    grounded = grounded_total = 0
    dual_tp = dual_fp = dual_fn = dual_tn = 0
    base_dual_tp = base_dual_fp = base_dual_fn = 0
    confusion: Counter = Counter()
    errors: list[dict] = []
    per_reason: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "correct": 0, "baseline_correct": 0}
    )
    missing = []
    model_name = prompt_version = policy_version = ""

    for item in rows:
        tid = item["ticket_id"]
        expected = str(item["expected_reason_code"])
        expected_avoid = item.get("expected_avoidability")
        expected_dual = item.get("expected_dual_remedy")
        stratum = item.get("stratum", "unspecified")
        ticket = tickets.get(tid)
        cls = classifications.get(tid)
        refund = refunds.get(tid)

        if cls is None:
            missing.append(tid)
            continue

        model_name = cls.model_name or model_name
        prompt_version = cls.prompt_version or prompt_version
        policy_version = cls.policy_version or policy_version

        predicted = str(cls.reason_code)
        if cls.validation_status == "invalid_after_retries":
            schema_failures += 1
        if cls.retries > 0:
            retried += 1
        if predicted == RefundReason.UNKNOWN:
            unknown += 1
        if predicted == RefundReason.AMBIGUOUS:
            ambiguous += 1
        if cls.review_required:
            escalated += 1
        if cls.evidence_grounded is not None:
            grounded_total += 1
            grounded += int(cls.evidence_grounded)

        correct = predicted == expected
        reason_total += 1
        reason_hits += int(correct)
        confusion[(expected, predicted)] += 1
        per_reason[expected]["n"] += 1
        per_reason[expected]["correct"] += int(correct)

        # Baseline on the same ticket, same label.
        base_pred = None
        base_correct = None
        if ticket is not None:
            base_pred = str(classify_baseline(ticket.customer_message, ticket.agent_notes))
            base_correct = base_pred == expected
            baseline_total += 1
            baseline_hits += int(base_correct)
            per_reason[expected]["baseline_correct"] += int(base_correct)

        error_category = error_note = None
        if not correct:
            error_category, error_note = categorise_error(expected, predicted, ticket)
            errors.append(
                {
                    "ticket_id": tid,
                    "stratum": stratum,
                    "expected": expected,
                    "predicted": predicted,
                    "baseline": base_pred,
                    "confidence": cls.confidence,
                    "category": error_category,
                    "note": error_note,
                    "refund_value_inr": float(refund.amount_inr) if refund else None,
                    "ambiguity_note": cls.ambiguity_note,
                }
            )

        avoid_correct = None
        if expected_avoid and refund is not None:
            avoid_total += 1
            avoid_correct = str(refund.avoidability_label) == str(expected_avoid)
            avoid_hits += int(avoid_correct)

        if expected_dual is not None:
            pred_dual = bool(cls.dual_remedy_indicated)
            if expected_dual and pred_dual:
                dual_tp += 1
            elif not expected_dual and pred_dual:
                dual_fp += 1
            elif expected_dual and not pred_dual:
                dual_fn += 1
            else:
                dual_tn += 1
            if ticket is not None:
                bd = baseline_dual_remedy(ticket.agent_notes)
                if expected_dual and bd:
                    base_dual_tp += 1
                elif not expected_dual and bd:
                    base_dual_fp += 1
                elif expected_dual and not bd:
                    base_dual_fn += 1

        session.add(
            EvaluationResult(
                run_id=run_id,
                ticket_id=tid,
                is_summary=False,
                stratum=stratum,
                expected_reason_code=expected,
                predicted_reason_code=predicted,
                reason_correct=correct,
                expected_avoidability=str(expected_avoid) if expected_avoid else None,
                predicted_avoidability=(str(refund.avoidability_label) if refund else None),
                avoidability_correct=avoid_correct,
                baseline_reason_code=base_pred,
                baseline_correct=base_correct,
                error_category=error_category,
                error_note=error_note,
                confidence=cls.confidence,
                retries=cls.retries,
                schema_failed=cls.validation_status == "invalid_after_retries",
                evidence_grounded=cls.evidence_grounded,
                model_name=cls.model_name,
                prompt_version=cls.prompt_version,
                policy_version=cls.policy_version,
            )
        )

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    reason_acc = rate(reason_hits, reason_total)
    baseline_acc = rate(baseline_hits, baseline_total)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "status": "measured",
        "sample_size": reason_total,
        "method": (
            "Stratified sample of refund tickets, labelled by a single human reviewer from the "
            "ticket text against the published taxonomy, without reference to the model's answer "
            "for that ticket. The same labels score a deterministic keyword baseline, so the "
            "model's lift is measured on identical data."
        ),
        "stratification": dict(Counter(r.get("stratum", "unspecified") for r in rows)),
        "reason_accuracy": reason_acc,
        "reason_accuracy_ci95": wilson_interval(reason_hits, reason_total),
        "reason_correct": reason_hits,
        "baseline_reason_accuracy": baseline_acc,
        "baseline_accuracy_ci95": wilson_interval(baseline_hits, baseline_total),
        "lift_over_baseline": (
            round(reason_acc - baseline_acc, 4)
            if reason_acc is not None and baseline_acc is not None
            else None
        ),
        "avoidability_accuracy": rate(avoid_hits, avoid_total),
        "avoidability_sample": avoid_total,
        "schema_failure_rate": rate(schema_failures, reason_total),
        "retry_rate": rate(retried, reason_total),
        "unknown_rate": rate(unknown, reason_total),
        "ambiguous_rate": rate(ambiguous, reason_total),
        "human_escalation_rate": rate(escalated, reason_total),
        "evidence_grounding_rate": rate(grounded, grounded_total),
        "dual_remedy": {
            "true_positive": dual_tp,
            "false_positive": dual_fp,
            "false_negative": dual_fn,
            "true_negative": dual_tn,
            "precision": rate(dual_tp, dual_tp + dual_fp),
            "recall": rate(dual_tp, dual_tp + dual_fn),
            "baseline_precision": rate(base_dual_tp, base_dual_tp + base_dual_fp),
            "baseline_recall": rate(base_dual_tp, base_dual_tp + base_dual_fn),
            "why_this_matters": (
                "Dual remedy is the largest avoidability driver by value, and the helpdesk's own "
                "replacement_issued flag disagrees with the closing note in both directions. "
                "Precision and recall here are what justify using the model rather than the flag."
            ),
        },
        "confusion": [
            {"expected": e, "predicted": p, "count": c}
            for (e, p), c in sorted(confusion.items(), key=lambda kv: -kv[1])
            if e != p
        ][:25],
        "per_reason": [
            {
                "reason_code": reason,
                "n": v["n"],
                "accuracy": rate(v["correct"], v["n"]),
                "baseline_accuracy": rate(v["baseline_correct"], v["n"]),
            }
            for reason, v in sorted(per_reason.items(), key=lambda kv: -kv[1]["n"])
        ],
        "error_categories": [
            {
                "category": cat,
                "count": len(items),
                "share_of_errors": round(len(items) / max(1, len(errors)), 4),
                "example_ticket_ids": [i["ticket_id"] for i in items[:4]],
                "explanation": items[0]["note"],
            }
            for cat, items in sorted(
                {
                    c: [e for e in errors if e["category"] == c]
                    for c in {e["category"] for e in errors}
                }.items(),
                key=lambda kv: -len(kv[1]),
            )
        ],
        "errors": errors,
        "tickets_in_set_without_classification": missing,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "policy_version": policy_version,
        "evaluated_at": datetime.utcnow().isoformat(),
        "notes": [
            f"n={reason_total}. The 95% Wilson interval on reason accuracy is "
            f"{wilson_interval(reason_hits, reason_total)}, so differences smaller than a few "
            "points are not resolvable at this sample size.",
            "Single annotator, who also authored the taxonomy and the prompt. There is no "
            "second annotator and therefore no inter-annotator agreement figure: this measures "
            "agreement with one reviewer's reading, not ground truth.",
            "UNKNOWN and AMBIGUOUS predictions are scored as errors when the reviewer committed "
            "to a reason. That is the strict reading and makes the accuracy figure lower than a "
            "'coverage-adjusted' one would be.",
            "Several confusion pairs are taxonomy overlap rather than model failure "
            "(cancellation vs change of mind; damaged vs lost in transit). Those cost nothing "
            "in money terms because the policy remedy and the avoidability driver are the same.",
        ],
    }

    session.add(
        EvaluationResult(
            run_id=run_id,
            is_summary=True,
            model_name=model_name,
            prompt_version=prompt_version,
            policy_version=policy_version,
            metrics=metrics,
            notes="; ".join(metrics["notes"])[:2000],
        )
    )
    session.flush()

    log.info(
        "evaluation.complete",
        extra={
            "run_id": run_id,
            "sample_size": reason_total,
            "reason_accuracy": reason_acc,
            "baseline_accuracy": baseline_acc,
            "schema_failure_rate": metrics["schema_failure_rate"],
        },
    )
    return metrics


def latest_evaluation(session: Session) -> dict:
    row = session.execute(
        select(EvaluationResult)
        .where(EvaluationResult.is_summary.is_(True))
        .order_by(EvaluationResult.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return {
            "status": "not_run",
            "message": (
                "No evaluation has been run against this database. After ingest and "
                "classification, POST /v1/admin/evaluation/run or "
                "`python -m scripts.run_evaluation`."
            ),
            "eval_set": eval_set_status(),
        }
    return row.metrics
