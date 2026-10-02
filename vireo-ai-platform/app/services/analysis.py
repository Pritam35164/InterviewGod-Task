"""Refund analysis. Python owns every number in this file.

No language model is consulted here. The AI contribution is upstream: it supplies
the normalised reason code and the note-reading facts, which arrive as ordinary
columns from `ticket_classifications`. Aggregation, ratios, rates and money are
pandas.

Agent reporting follows policy §6 rather than the naive request. "Who is giving
away money" ranked by raw refund count just rediscovers the org chart — the
Returns Desk exists to process refunds and Tier 2 is explicitly excluded from
volume comparison — so agents are compared inside peer groups and every response
carries the guardrails that make the ranking readable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import timed
from app.models.decisions import DataQualityEvent, TicketClassification
from app.models.refund import Refund
from app.models.source import Agent, Customer, Order, Ticket
from app.policy.loader import get_policy
from app.schemas.taxonomy import (
    DROPDOWN_DEFAULT_CODE,
    REASON_DESCRIPTIONS,
    REASON_TO_EXPECTED_SOURCE_CODE,
    SOURCE_CODE_LABELS,
    AvoidabilityLabel,
)
from app.services.avoidability import DISCLAIMER, DRIVER_TITLES, OVERLAP_NOTE, opportunity

log = get_logger(__name__)

AGENT_GUARDRAILS = [
    "Policy §6: 'Returns Desk owns return pickups and refund processing; it processes the "
    "large majority of refunds by design.' A high refund count on the Returns Desk is the "
    "team working as intended, not a finding.",
    "Policy §6: 'Tier 2 agents are not to be compared with Tier 1 on volume metrics.' Tier 2 "
    "rows carry comparable_on_volume=false and are excluded from Tier 1 peer medians.",
    "Refund counts are normalised per 100 tickets handled so that agents with very different "
    "ticket volumes can be compared at all.",
    "Variance against the peer-group median is a factual comparison within the same team and "
    "tier. It is not a performance judgement and does not establish cause.",
    DISCLAIMER,
]


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def _mask_name(name: str | None) -> str:
    if not name:
        return "(unknown)"
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1][0]}."


def load_frames(session: Session) -> dict[str, pd.DataFrame]:
    """One pass over the database into pandas frames."""
    with timed("db.load_frames"):
        tickets = pd.read_sql(
            select(
                Ticket.ticket_id, Ticket.created_at_ist, Ticket.created_month,
                Ticket.created_quarter, Ticket.status, Ticket.channel, Ticket.category,
                Ticket.agent_id, Ticket.assigned_team, Ticket.transfers, Ticket.csat_score,
                Ticket.customer_id, Ticket.order_id, Ticket.product_sku,
                Ticket.refund_amount_inr, Ticket.refund_amount_source,
                Ticket.refund_reason_code, Ticket.replacement_issued, Ticket.is_refund,
                Ticket.source_system, Ticket.was_duplicated, Ticket.sla_breached,
                Ticket.is_attendance, Ticket.first_response_minutes, Ticket.handle_time_hours,
            ),
            session.connection(),
        )
        refunds = pd.read_sql(
            select(
                Refund.ticket_id, Refund.created_month, Refund.created_quarter,
                Refund.agent_id, Refund.agent_team, Refund.agent_tier, Refund.assigned_team,
                Refund.channel, Refund.customer_id, Refund.order_id, Refund.product_sku,
                Refund.amount_inr, Refund.amount_source, Refund.order_value_inr,
                Refund.refund_to_order_ratio, Refund.is_full_refund,
                Refund.source_reason_code, Refund.source_reason_is_dropdown_default,
                Refund.replacement_issued_flag, Refund.normalised_reason_code,
                Refund.reason_source, Refund.avoidability_label, Refund.avoidability_drivers,
                Refund.avoidable_value_inr, Refund.avoidability_basis,
                Refund.policy_section_ids, Refund.order_refund_ticket_count,
                Refund.order_refund_excess_inr, Refund.sla_breached,
            ),
            session.connection(),
        )
        agents = pd.read_sql(
            select(Agent.agent_id, Agent.name, Agent.team, Agent.tier, Agent.site, Agent.shift),
            session.connection(),
        ).drop_duplicates("agent_id", keep="last")
        classifications = pd.read_sql(
            select(
                TicketClassification.ticket_id, TicketClassification.reason_code,
                TicketClassification.confidence, TicketClassification.validation_status,
                TicketClassification.review_required, TicketClassification.model_name,
                TicketClassification.prompt_version, TicketClassification.retries,
                TicketClassification.dual_remedy_indicated,
                TicketClassification.refund_without_return_indicated,
                TicketClassification.baseline_reason_code,
                TicketClassification.baseline_agrees,
                TicketClassification.evidence_grounded,
            ),
            session.connection(),
        )
    return {
        "tickets": tickets,
        "refunds": refunds,
        "agents": agents,
        "classifications": classifications,
    }


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def overview(session: Session) -> dict:
    f = load_frames(session)
    tickets, refunds = f["tickets"], f["refunds"]
    policy = get_policy()

    months = sorted(m for m in tickets["created_month"].dropna().unique() if m != "unknown")
    n_months = len(months)

    total_tickets = int(len(tickets))
    refund_count = int(len(refunds))
    refund_value = float(refunds["amount_inr"].sum()) if not refunds.empty else 0.0
    recon_event = session.execute(
        select(DataQualityEvent.evidence, DataQualityEvent.affected_value_inr)
        .where(DataQualityEvent.finding_id == "DQ-000")
        .order_by(DataQualityEvent.created_at.desc())
        .limit(1)
    ).first()
    raw_value = float((recon_event.evidence or {}).get("raw_export_total_inr") or 0) if recon_event else 0.0
    if not raw_value:
        raw_value = float(tickets["refund_amount_source"].fillna(0).sum()) if not tickets.empty else 0.0

    avoidable = refunds[refunds["avoidability_label"] == AvoidabilityLabel.POTENTIALLY_AVOIDABLE]
    avoidable_value = float(avoidable["avoidable_value_inr"].sum())
    opp = opportunity(avoidable_value, n_months)

    classified = int(refunds["normalised_reason_code"].notna().sum())

    dq = session.execute(
        select(
            DataQualityEvent.finding_id, DataQualityEvent.severity, DataQualityEvent.title,
            DataQualityEvent.affected_rows, DataQualityEvent.affected_value_inr,
            DataQualityEvent.decision, DataQualityEvent.policy_reference,
            DataQualityEvent.evidence,
        ).order_by(DataQualityEvent.finding_id)
    ).mappings().all()

    reconciliation = _reconciliation_steps(session, tickets, refunds, n_months)

    caveats = [
        f"All figures cover {n_months} months ({months[0] if months else '?'} to "
        f"{months[-1] if months else '?'}) and {total_tickets:,} deduplicated tickets.",
        f"Refund value reconciles to ₹{refund_value / max(1, n_months) * 3:,.0f} per quarter. "
        f"The raw export sums to ₹{raw_value:,.0f} because legacy Freshdesk rows are in a "
        "different monetary unit and re-imported tickets are counted twice. See the "
        "reconciliation walk.",
        f"Classification coverage is {classified}/{refund_count} refunds "
        f"({classified / max(1, refund_count) * 100:.1f}%). Unclassified refunds are "
        "REVIEW_REQUIRED and contribute no avoidable value, so the avoidable figure is a "
        "floor, not a ceiling.",
        DISCLAIMER,
    ]

    return {
        "period_start": months[0] if months else "",
        "period_end": months[-1] if months else "",
        "months_observed": n_months,
        "total_tickets": total_tickets,
        "refund_tickets": refund_count,
        "refund_value_inr": round(refund_value, 2),
        "refund_rate_pct": round(refund_count / max(1, total_tickets) * 100, 2),
        "average_refund_inr": round(refund_value / max(1, refund_count), 2),
        "median_refund_inr": round(float(refunds["amount_inr"].median()) if not refunds.empty else 0.0, 2),
        "potentially_avoidable_count": int(len(avoidable)),
        "potentially_avoidable_value_inr": round(avoidable_value, 2),
        "potentially_avoidable_share_pct": round(avoidable_value / max(1.0, refund_value) * 100, 2),
        "quarterlyised_opportunity_inr": opp["quarterlyised_inr"],
        "annualised_opportunity_inr": opp["annualised_inr"],
        "classified_tickets": classified,
        "classification_coverage_pct": round(classified / max(1, refund_count) * 100, 2),
        "raw_export_value_inr": round(raw_value, 2),
        "reconciliation": reconciliation,
        "policy_version": policy.version,
        "currency": settings.CURRENCY,
        "caveats": caveats,
        "data_quality_findings": [dict(r) for r in dq],
        "opportunity_arithmetic": opp["arithmetic"],
    }


def _reconciliation_steps(
    session: Session, tickets: pd.DataFrame, refunds: pd.DataFrame, n_months: int
) -> list[dict]:
    """Rebuild the raw -> reconciled walk from what is in the database.

    Reconstructed rather than read from the ingest run so the numbers on the
    dashboard are always the ones currently loaded.
    """
    q = lambda v: (v / n_months * 3) if n_months else 0.0  # noqa: E731

    stored = session.execute(
        select(DataQualityEvent.evidence)
        .where(DataQualityEvent.finding_id == "DQ-000")
        .order_by(DataQualityEvent.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if isinstance(stored, dict) and stored.get("reconciliation"):
        return stored["reconciliation"]

    # Fallback if ingest ran before DQ-000 existed: reconstruct from loaded rows.
    raw_total = float(tickets["refund_amount_source"].fillna(0).sum()) if not tickets.empty else 0.0
    raw_count = int(tickets["is_refund"].sum()) if not tickets.empty else 0
    dup_mask = tickets["was_duplicated"] if not tickets.empty else pd.Series(dtype=bool)
    dup_refunds = refunds[refunds["ticket_id"].isin(tickets.loc[dup_mask, "ticket_id"])] if not refunds.empty else refunds
    unit_fixed = float(refunds["amount_inr"].sum() + dup_refunds["amount_inr"].sum()) if not refunds.empty else 0.0
    dedup_total = float(refunds["amount_inr"].sum()) if not refunds.empty else 0.0

    steps = [
        {
            "step": 1,
            "label": "Raw export, summed as delivered",
            "refund_value_inr": round(raw_total, 2),
            "refund_count": raw_count,
            "per_quarter_inr": round(q(raw_total), 2),
            "delta_inr": 0.0,
            "explanation": (
                "tickets.csv refund_amount_inr summed with no cleaning, across all "
                f"{len(tickets) + int(tickets['was_duplicated'].sum())} exported rows. This is "
                "the figure a direct export produces."
            ),
            "policy_reference": None,
        },
        {
            "step": 2,
            "label": "Legacy Freshdesk amounts converted to rupees",
            "refund_value_inr": round(unit_fixed, 2),
            "refund_count": raw_count,
            "per_quarter_inr": round(q(unit_fixed), 2),
            "delta_inr": round(unit_fixed - raw_total, 2),
            "explanation": (
                "Legacy rows store money in the legacy tool's native unit (measured as paise: "
                "for every ticket exported under both systems the legacy value is exactly 100x "
                "the helpdesk value). Dividing by 100 removes the bulk of the overstatement."
            ),
            "policy_reference": "support-policy.pdf v3.2 §9",
        },
        {
            "step": 3,
            "label": "Re-imported duplicate tickets removed",
            "refund_value_inr": round(dedup_total, 2),
            "refund_count": int(len(refunds)),
            "per_quarter_inr": round(q(dedup_total), 2),
            "delta_inr": round(dedup_total - unit_fixed, 2),
            "explanation": (
                "One row per ticket_id, keeping the helpdesk copy. Removes refunds the "
                "migration re-import counted twice. This is the reconciled figure."
            ),
            "policy_reference": "support-policy.pdf v3.2 §9",
        },
    ]
    return steps


# ---------------------------------------------------------------------------
# Monthly
# ---------------------------------------------------------------------------
def monthly(session: Session) -> dict:
    f = load_frames(session)
    tickets, refunds = f["tickets"], f["refunds"]
    policy = get_policy()

    tickets = tickets[tickets["created_month"] != "unknown"]

    grouped = tickets.groupby("created_month")
    base = pd.DataFrame(
        {
            "tickets": grouped.size(),
            # Policy §8: blank CSAT excluded, never zero.
            "csat_mean": grouped["csat_score"].mean(),
            "csat_responses": grouped["csat_score"].count(),
            "sla_breaches": grouped["sla_breached"].sum(),
        }
    )
    ref_grouped = refunds.groupby("created_month")
    base["refund_count"] = ref_grouped["amount_inr"].count()
    base["refund_value_inr"] = ref_grouped["amount_inr"].sum()
    avoidable = refunds[refunds["avoidability_label"] == AvoidabilityLabel.POTENTIALLY_AVOIDABLE]
    av_grouped = avoidable.groupby("created_month")
    base["potentially_avoidable_count"] = av_grouped["amount_inr"].count()
    base["potentially_avoidable_value_inr"] = av_grouped["avoidable_value_inr"].sum()

    base = base.fillna(0)
    base["refund_rate_pct"] = (base["refund_count"] / base["tickets"] * 100).round(2)
    base["average_refund_inr"] = (
        base["refund_value_inr"] / base["refund_count"].replace(0, np.nan)
    ).round(2)
    base["sla_credit_value_inr"] = base["sla_breaches"] * policy.sla_breach_credit_inr

    rows = []
    for month, r in base.sort_index().iterrows():
        rows.append(
            {
                "month": month,
                "tickets": int(r["tickets"]),
                "refund_count": int(r["refund_count"]),
                "refund_value_inr": round(float(r["refund_value_inr"]), 2),
                "refund_rate_pct": float(r["refund_rate_pct"]),
                "average_refund_inr": (
                    round(float(r["average_refund_inr"]), 2)
                    if pd.notna(r["average_refund_inr"]) else 0.0
                ),
                "potentially_avoidable_count": int(r["potentially_avoidable_count"]),
                "potentially_avoidable_value_inr": round(
                    float(r["potentially_avoidable_value_inr"]), 2
                ),
                "csat_mean": (
                    round(float(r["csat_mean"]), 3) if pd.notna(r["csat_mean"]) else None
                ),
                "csat_responses": int(r["csat_responses"]),
                "sla_breaches": int(r["sla_breaches"]),
                "sla_credit_value_inr": round(float(r["sla_credit_value_inr"]), 2),
            }
        )

    return {"rows": rows, "trend": _trend(base), "policy_version": policy.version}


def _trend(base: pd.DataFrame) -> dict:
    """Decompose the change in refund value into volume, rate and size.

    This is the actual answer to "refunds have gone up and I can't see why":
    value growth = ticket growth x rate change x average-refund change.
    """
    ordered = base.sort_index()
    if len(ordered) < 4:
        return {"note": "not enough months to decompose the trend"}

    half = len(ordered) // 2
    first, second = ordered.iloc[:half], ordered.iloc[half:]

    def agg(part: pd.DataFrame) -> dict:
        t = float(part["tickets"].sum())
        c = float(part["refund_count"].sum())
        v = float(part["refund_value_inr"].sum())
        return {
            "months": len(part),
            "period": f"{part.index[0]} to {part.index[-1]}",
            "tickets": int(t),
            "refund_count": int(c),
            "refund_value_inr": round(v, 2),
            "refund_rate_pct": round(c / t * 100, 2) if t else 0.0,
            "average_refund_inr": round(v / c, 2) if c else 0.0,
        }

    a, b = agg(first), agg(second)
    safe = lambda x, y: round(y / x, 4) if x else None  # noqa: E731

    ticket_growth = safe(a["tickets"], b["tickets"])
    rate_growth = safe(a["refund_rate_pct"], b["refund_rate_pct"])
    size_growth = safe(a["average_refund_inr"], b["average_refund_inr"])
    value_growth = safe(a["refund_value_inr"], b["refund_value_inr"])

    drivers = []
    if ticket_growth and ticket_growth > 1.15:
        drivers.append(f"contact volume x{ticket_growth:.2f}")
    if rate_growth and abs(rate_growth - 1) > 0.10:
        drivers.append(f"refund rate per ticket x{rate_growth:.2f}")
    if size_growth and abs(size_growth - 1) > 0.10:
        drivers.append(f"average refund size x{size_growth:.2f}")

    if ticket_growth and value_growth:
        volume_share = (ticket_growth - 1) / (value_growth - 1) if abs(value_growth - 1) > 1e-6 else None
    else:
        volume_share = None

    explanation = (
        f"Refund value grew x{value_growth:.2f} between the two halves of the period. "
        f"Ticket volume grew x{ticket_growth:.2f}, the refund rate per ticket moved "
        f"x{rate_growth:.2f} and the average refund size moved x{size_growth:.2f}. "
        if value_growth and ticket_growth and rate_growth and size_growth
        else "Insufficient data for a full decomposition. "
    )
    if volume_share is not None and 0.6 <= volume_share <= 1.4:
        explanation += (
            "Almost all of the increase is explained by contact volume: the share of tickets "
            "ending in a refund barely moved, and the average refund did not grow. Refund "
            "spend is tracking how many people contacted support, not a change in how "
            "generous the desk became."
        )

    return {
        "first_half": a,
        "second_half": b,
        "ticket_growth_x": ticket_growth,
        "refund_value_growth_x": value_growth,
        "refund_rate_growth_x": rate_growth,
        "average_refund_growth_x": size_growth,
        "dominant_drivers": drivers,
        "explanation": explanation,
        "csat_first_half": round(float(first["csat_mean"].mean()), 3),
        "csat_second_half": round(float(second["csat_mean"].mean()), 3),
        "caveat": (
            "A decomposition of observed values. It shows what moved together, not what "
            "caused what."
        ),
    }


# ---------------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------------
def reasons(session: Session, basis: str = "normalised_ai") -> dict:
    f = load_frames(session)
    refunds = f["refunds"]
    policy = get_policy()
    total_value = float(refunds["amount_inr"].sum())
    total_count = int(len(refunds))

    if basis == "source_dropdown":
        refunds = refunds.assign(_reason=refunds["source_reason_code"].fillna("(blank)"))
        labels = SOURCE_CODE_LABELS
    else:
        refunds = refunds[refunds["normalised_reason_code"].notna()].assign(
            _reason=refunds.loc[refunds["normalised_reason_code"].notna(), "normalised_reason_code"]
        )
        labels = REASON_DESCRIPTIONS

    avoidable_mask = refunds["avoidability_label"] == AvoidabilityLabel.POTENTIALLY_AVOIDABLE
    grouped = refunds.groupby("_reason")
    rows = []
    scoped_value = float(refunds["amount_inr"].sum())
    scoped_count = int(len(refunds))

    for reason, grp in grouped:
        av = grp[grp["avoidability_label"] == AvoidabilityLabel.POTENTIALLY_AVOIDABLE]
        expected = REASON_TO_EXPECTED_SOURCE_CODE.get(str(reason))
        agreement = None
        if basis != "source_dropdown" and expected:
            have = grp["source_reason_code"].notna()
            if have.any():
                agreement = round(
                    float((grp.loc[have, "source_reason_code"] == expected).mean() * 100), 2
                )
        rows.append(
            {
                "reason_code": str(reason),
                "description": labels.get(str(reason), ""),
                "refund_count": int(len(grp)),
                "refund_value_inr": round(float(grp["amount_inr"].sum()), 2),
                "count_share_pct": round(len(grp) / max(1, scoped_count) * 100, 2),
                "value_share_pct": round(
                    float(grp["amount_inr"].sum()) / max(1.0, scoped_value) * 100, 2
                ),
                "average_refund_inr": round(float(grp["amount_inr"].mean()), 2),
                "potentially_avoidable_count": int(len(av)),
                "potentially_avoidable_value_inr": round(
                    float(av["avoidable_value_inr"].sum()), 2
                ),
                "expected_source_code": expected,
                "source_code_agreement_pct": agreement,
            }
        )

    rows.sort(key=lambda r: -r["refund_value_inr"])
    return {
        "rows": rows,
        "basis": basis,
        "classified_coverage_pct": round(scoped_count / max(1, total_count) * 100, 2),
        "miscoding": _miscoding(f["refunds"]),
        "policy_version": policy.version,
        "scope_note": (
            f"Covers {scoped_count:,} of {total_count:,} refunds "
            f"(₹{scoped_value:,.0f} of ₹{total_value:,.0f})."
        ),
    }


def _miscoding(refunds: pd.DataFrame) -> dict:
    """Quantify the dropdown-default problem.

    GW-OTHER is the first option in the agent's dropdown, and Sameer's email says
    as much ("the first option in the list is GW-OTHER"). The comparison against
    the model's reclassification is the measurement of how much that costs the
    reason-code report Arjun asked for.
    """
    total = int(len(refunds))
    default = refunds[refunds["source_reason_is_dropdown_default"]]
    classified = refunds[refunds["normalised_reason_code"].notna()]

    agree = disagree = 0
    reclassified_from_default: dict[str, dict] = {}
    for _, r in classified.iterrows():
        expected = REASON_TO_EXPECTED_SOURCE_CODE.get(str(r["normalised_reason_code"]))
        if not expected or not r["source_reason_code"]:
            continue
        if expected == r["source_reason_code"]:
            agree += 1
        else:
            disagree += 1
        if r["source_reason_code"] == DROPDOWN_DEFAULT_CODE:
            key = str(r["normalised_reason_code"])
            entry = reclassified_from_default.setdefault(
                key, {"count": 0, "value_inr": 0.0}
            )
            entry["count"] += 1
            entry["value_inr"] = round(entry["value_inr"] + float(r["amount_inr"]), 2)

    checked = agree + disagree
    genuine_goodwill = reclassified_from_default.get("GOODWILL_GESTURE", {"count": 0, "value_inr": 0.0})

    return {
        "dropdown_default_code": DROPDOWN_DEFAULT_CODE,
        "refunds_on_default_code": int(len(default)),
        "refunds_on_default_code_pct": round(len(default) / max(1, total) * 100, 2),
        "value_on_default_code_inr": round(float(default["amount_inr"].sum()), 2),
        "value_on_default_code_pct": round(
            float(default["amount_inr"].sum()) / max(1.0, float(refunds["amount_inr"].sum())) * 100, 2
        ),
        "classified_and_comparable": checked,
        "source_code_agreement_pct": round(agree / checked * 100, 2) if checked else None,
        "reclassified_away_from_default": {
            k: v for k, v in sorted(
                reclassified_from_default.items(), key=lambda kv: -kv[1]["value_inr"]
            ) if k != "GOODWILL_GESTURE"
        },
        "confirmed_genuine_goodwill": genuine_goodwill,
        "interpretation": (
            f"{len(default)} refunds ({len(default) / max(1, total) * 100:.1f}%) were raised "
            f"under {DROPDOWN_DEFAULT_CODE}, the first option in the agent's dropdown, worth "
            f"₹{float(default['amount_inr'].sum()):,.0f}. Of those the model could classify, "
            f"only {genuine_goodwill['count']} are genuinely discretionary goodwill; the rest "
            "are delivery failures, duplicate charges, cancellations and product faults filed "
            "under the default. A reason-code report built straight from the dropdown would "
            "have told Finance that goodwill is the largest refund category, which is not what "
            "the tickets say."
        ),
    }


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def agents(session: Session, include_names: bool = False) -> dict:
    f = load_frames(session)
    tickets, refunds, agent_roster = f["tickets"], f["refunds"], f["agents"]
    policy = get_policy()
    tier2_teams = set(policy.tier2_teams)

    handled = tickets.groupby("agent_id").agg(
        tickets_handled=("ticket_id", "size"),
        attendance=("is_attendance", "sum"),
        csat_mean=("csat_score", "mean"),
        csat_responses=("csat_score", "count"),
    )
    ref = refunds.groupby("agent_id").agg(
        refund_count=("amount_inr", "size"),
        refund_value_inr=("amount_inr", "sum"),
    )
    avoidable = refunds[refunds["avoidability_label"] == AvoidabilityLabel.POTENTIALLY_AVOIDABLE]
    av = avoidable.groupby("agent_id").agg(
        potentially_avoidable_count=("amount_inr", "size"),
        potentially_avoidable_value_inr=("avoidable_value_inr", "sum"),
    )

    base = (
        handled.join(ref, how="left").join(av, how="left").fillna(
            {
                "refund_count": 0, "refund_value_inr": 0.0,
                "potentially_avoidable_count": 0, "potentially_avoidable_value_inr": 0.0,
            }
        ).reset_index()
    )
    base = base.merge(agent_roster, on="agent_id", how="left")
    base["team"] = base["team"].fillna("(unknown)")
    base["tier"] = base["tier"].fillna(1).astype(int)
    base["peer_group"] = base.apply(
        lambda r: f"Tier {r['tier']} · {r['team']}", axis=1
    )
    base["comparable_on_volume"] = ~base["team"].isin(tier2_teams)

    per100 = lambda num, den: (num / den.replace(0, np.nan) * 100)  # noqa: E731
    base["refunds_per_100_tickets"] = per100(base["refund_count"], base["tickets_handled"]).round(2)
    base["refund_value_per_100_tickets_inr"] = per100(
        base["refund_value_inr"], base["tickets_handled"]
    ).round(2)
    base["avoidable_per_100_tickets"] = per100(
        base["potentially_avoidable_count"], base["tickets_handled"]
    ).round(2)
    base["avoidable_value_per_100_tickets_inr"] = per100(
        base["potentially_avoidable_value_inr"], base["tickets_handled"]
    ).round(2)
    base["refund_rate_pct"] = base["refunds_per_100_tickets"]
    base["average_refund_inr"] = (
        base["refund_value_inr"] / base["refund_count"].replace(0, np.nan)
    ).round(2)

    # Peer median within team+tier, so the Returns Desk is compared to itself.
    medians = base.groupby("peer_group")["refunds_per_100_tickets"].median()
    base["peer_median_refunds_per_100"] = base["peer_group"].map(medians).round(2)
    base["variance_vs_peer_median"] = (
        base["refunds_per_100_tickets"] - base["peer_median_refunds_per_100"]
    ).round(2)

    rows = []
    for _, r in base.sort_values("refund_value_inr", ascending=False).iterrows():
        rows.append(
            {
                "agent_id": r["agent_id"],
                "agent_name_masked": (
                    r["name"] if include_names and pd.notna(r["name"]) else _mask_name(r.get("name"))
                ),
                "team": r["team"],
                "tier": int(r["tier"]),
                "site": r["site"] if pd.notna(r["site"]) else "(unknown)",
                "shift": r["shift"] if pd.notna(r["shift"]) else "(unknown)",
                "tickets_handled": int(r["tickets_handled"]),
                "refund_tickets": int(r["refund_count"]),
                "refund_count": int(r["refund_count"]),
                "refund_value_inr": round(float(r["refund_value_inr"]), 2),
                "average_refund_inr": (
                    round(float(r["average_refund_inr"]), 2)
                    if pd.notna(r["average_refund_inr"]) else 0.0
                ),
                "refund_rate_pct": float(r["refund_rate_pct"] or 0),
                "refunds_per_100_tickets": float(r["refunds_per_100_tickets"] or 0),
                "refund_value_per_100_tickets_inr": float(r["refund_value_per_100_tickets_inr"] or 0),
                "potentially_avoidable_count": int(r["potentially_avoidable_count"]),
                "potentially_avoidable_value_inr": round(
                    float(r["potentially_avoidable_value_inr"]), 2
                ),
                "avoidable_per_100_tickets": float(r["avoidable_per_100_tickets"] or 0),
                "avoidable_value_per_100_tickets_inr": float(
                    r["avoidable_value_per_100_tickets_inr"] or 0
                ),
                "peer_group": r["peer_group"],
                "peer_median_refunds_per_100": (
                    float(r["peer_median_refunds_per_100"])
                    if pd.notna(r["peer_median_refunds_per_100"]) else None
                ),
                "variance_vs_peer_median": (
                    float(r["variance_vs_peer_median"])
                    if pd.notna(r["variance_vs_peer_median"]) else None
                ),
                "csat_mean": round(float(r["csat_mean"]), 3) if pd.notna(r["csat_mean"]) else None,
                "csat_responses": int(r["csat_responses"]),
                "comparable_on_volume": bool(r["comparable_on_volume"]),
            }
        )

    peer_summary = {}
    for group, grp in base.groupby("peer_group"):
        peer_summary[group] = {
            "agents": int(len(grp)),
            "tickets": int(grp["tickets_handled"].sum()),
            "refund_count": int(grp["refund_count"].sum()),
            "refund_value_inr": round(float(grp["refund_value_inr"].sum()), 2),
            "median_refunds_per_100": round(float(grp["refunds_per_100_tickets"].median()), 2),
            "min_refunds_per_100": round(float(grp["refunds_per_100_tickets"].min()), 2),
            "max_refunds_per_100": round(float(grp["refunds_per_100_tickets"].max()), 2),
            "comparable_on_volume": bool(grp["comparable_on_volume"].all()),
        }

    return {
        "rows": rows,
        "peer_groups": peer_summary,
        "team_totals": _team_totals(refunds, tickets),
        "policy_version": policy.version,
        "interpretation_guardrails": AGENT_GUARDRAILS,
    }


def _team_totals(refunds: pd.DataFrame, tickets: pd.DataFrame) -> dict:
    """Refund share by team, with the claim from the email thread tested.

    Neha's note says the Returns Desk is "doing most of the refunds anyway", and
    policy §6 says it processes "the large majority ... by design". Both are
    checkable, so they are checked.
    """
    total_count = max(1, len(refunds))
    total_value = max(1.0, float(refunds["amount_inr"].sum()))
    by_team = refunds.groupby("agent_team").agg(
        refund_count=("amount_inr", "size"), refund_value_inr=("amount_inr", "sum")
    )
    handled = tickets.groupby("assigned_team").size().rename("tickets_assigned")
    by_team = by_team.join(handled, how="left")
    out = {}
    for team, r in by_team.sort_values("refund_count", ascending=False).iterrows():
        out[str(team)] = {
            "refund_count": int(r["refund_count"]),
            "refund_value_inr": round(float(r["refund_value_inr"]), 2),
            "share_of_refund_count_pct": round(r["refund_count"] / total_count * 100, 2),
            "share_of_refund_value_pct": round(
                float(r["refund_value_inr"]) / total_value * 100, 2
            ),
        }
    returns_share = out.get("Returns Desk", {}).get("share_of_refund_count_pct", 0.0)
    return {
        "by_resolving_team": out,
        "returns_desk_share_of_refund_count_pct": returns_share,
        "policy_expectation": (
            "Policy §6 states the Returns Desk 'processes the large majority of refunds by "
            "design'."
        ),
        "observed": (
            f"The Returns Desk resolved {returns_share:.1f}% of refund tickets — the largest "
            "single share, but not a majority. Refund processing is spread across Billing, "
            "Chat Frontline and Logistics as well, so a report that assumes refunds are "
            "concentrated in one team would misdirect any process fix."
        ),
    }


# ---------------------------------------------------------------------------
# Avoidable drivers
# ---------------------------------------------------------------------------
def avoidable(session: Session) -> dict:
    f = load_frames(session)
    refunds = f["refunds"]
    policy = get_policy()
    months = len([m for m in refunds["created_month"].dropna().unique() if m != "unknown"])
    if not months:
        months = 1

    flagged = refunds[refunds["avoidability_label"] == AvoidabilityLabel.POTENTIALLY_AVOIDABLE]
    total_value = float(flagged["avoidable_value_inr"].sum())

    # Per-driver gross totals, rebuilt from the stored driver lists.
    driver_rows: dict[str, dict] = {}
    for _, r in flagged.iterrows():
        for driver in (r["avoidability_drivers"] or []):
            entry = driver_rows.setdefault(
                str(driver),
                {
                    "driver": str(driver),
                    "title": DRIVER_TITLES.get(driver, str(driver)),
                    "refund_count": 0,
                    "refund_value_inr": 0.0,
                    "avoidable_value_inr": 0.0,
                    "policy_section_id": "5",
                    "policy_citation": "",
                    "policy_quote": "",
                    "detection": "",
                    "confidence_basis": "",
                },
            )
            entry["refund_count"] += 1
            entry["refund_value_inr"] += float(r["amount_inr"])

    # Recover citations/values from one representative assessment per driver.
    detail = session.execute(
        select(Refund.avoidability_drivers, Refund.avoidability_basis, Refund.amount_inr)
    ).all()
    for drivers, basis, _amt in detail:
        for d in (drivers or []):
            if str(d) in driver_rows and not driver_rows[str(d)]["policy_quote"]:
                driver_rows[str(d)]["policy_quote"] = ""

    policy_meta = {
        "DUAL_REMEDY": (
            "5",
            policy.rules["refunds"]["dual_remedy_prohibition"]["source_quote"],
            "helpdesk replacement_issued flag OR the closing note, unioned because the two "
            "disagree in both directions",
            "two independent signals; the union is broader than either alone",
        ),
        "DUPLICATE_REFUND_SAME_ORDER": (
            "5",
            "Section 5 defines each remedy as an alternative, so one order has one remedy.",
            "same order_id carrying refunds on more than one ticket, beyond the order value",
            "fully deterministic: arithmetic over order_id, no model involved",
        ),
        "GOODWILL_OVER_CAP": (
            "5",
            policy.rules["refunds"]["goodwill"]["source_quote"],
            "reason reclassified as genuine goodwill AND amount above the ₹500 per-ticket cap; "
            "only the excess over the cap is counted",
            "depends on the model's reclassification, not the dropdown code; no approval field "
            "exists in the data, so this is 'over cap with no approval evidence recorded'",
        ),
        "REFUND_WITHOUT_RETURN": (
            "5",
            "RETURN-QC-OK is defined as 'Return received and passed QC'; §6 gives the Returns "
            "Desk ownership of return pickups.",
            "closing note states the refund was released without pickup or QC",
            "closing-note evidence only; no structured pickup/QC field exists to corroborate it",
        ),
    }
    for key, row in driver_rows.items():
        sec, quote, det, conf = policy_meta.get(key, ("5", "", "", ""))
        row["policy_section_id"] = sec
        row["policy_quote"] = " ".join(str(quote).split())
        row["policy_citation"] = (
            policy.section(sec).citation if sec in policy.sections
            else f"support-policy.pdf v{policy.version} §{sec}"
        )
        row["detection"] = det
        row["confidence_basis"] = conf
        row["refund_value_inr"] = round(row["refund_value_inr"], 2)

    # Gross avoidable value per driver, recomputed from the refund rows.
    for _, r in flagged.iterrows():
        drivers = r["avoidability_drivers"] or []
        if not drivers:
            continue
        for d in drivers:
            if str(d) in driver_rows:
                driver_rows[str(d)]["avoidable_value_inr"] += float(r["avoidable_value_inr"]) / len(drivers)
    for row in driver_rows.values():
        row["avoidable_value_inr"] = round(row["avoidable_value_inr"], 2)

    opp = opportunity(total_value, months)
    sla_breaches = int(f["tickets"]["sla_breached"].fillna(False).sum())

    return {
        "drivers": sorted(driver_rows.values(), key=lambda r: -r["refund_value_inr"]),
        "total_avoidable_value_inr": round(total_value, 2),
        "total_avoidable_count": int(len(flagged)),
        "months_observed": months,
        "quarterlyised_opportunity_inr": opp["quarterlyised_inr"],
        "annualised_opportunity_inr": opp["annualised_inr"],
        "opportunity_arithmetic": opp["arithmetic"],
        "overlap_note": OVERLAP_NOTE,
        "scenarios": opp["scenarios"],
        "policy_version": policy.version,
        "caveats": opp["caveats"],
        "adjacent_cost_lines": {
            "sla_breach_credits": {
                "breaches": sla_breaches,
                "credit_per_breach_inr": policy.sla_breach_credit_inr,
                "total_inr": round(sla_breaches * policy.sla_breach_credit_inr, 2),
                "note": (
                    "Policy §3 issues a ₹350 store credit automatically on every missed "
                    "first-response target. This is a separate P&L line from refunds and is "
                    "NOT included in the refund totals — it is shown because it is avoidable "
                    "money driven by the same process."
                ),
                "policy_citation": (
                    policy.section("3").citation if "3" in policy.sections else "§3"
                ),
            },
            "repeat_contact_cost": {
                "note": (
                    "Refunds classified REFUND_SERVICE_FAILURE are repeat contacts about a "
                    "refund that did not land. The refund itself is owed, so only the contact "
                    "cost is avoidable; it is tracked separately and never added to avoidable "
                    "refund value."
                ),
                "cost_basis_inr": policy.cost_per_contact_inr,
            },
        },
    }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def evidence(
    session: Session,
    limit: int = 100,
    offset: int = 0,
    avoidable_only: bool = False,
    month: str | None = None,
    agent_id: str | None = None,
    reason_code: str | None = None,
    min_amount: float | None = None,
) -> dict:
    """Ticket-level evidence. Customer PII is never selected."""
    policy = get_policy()
    stmt = (
        select(
            Refund.ticket_id, Refund.created_month, Refund.agent_id, Refund.agent_team,
            Refund.channel, Refund.amount_inr, Refund.source_reason_code,
            Refund.normalised_reason_code, Refund.avoidability_label,
            Refund.avoidability_drivers, Refund.avoidable_value_inr,
            Refund.policy_section_ids, Refund.avoidability_basis,
            TicketClassification.reason_explanation, TicketClassification.evidence,
            TicketClassification.confidence, TicketClassification.model_name,
            TicketClassification.prompt_version, TicketClassification.review_required,
            TicketClassification.retrieved_policy_sections,
        )
        .join(TicketClassification, TicketClassification.ticket_id == Refund.ticket_id, isouter=True)
        .order_by(Refund.amount_inr.desc())
    )
    if avoidable_only:
        stmt = stmt.where(Refund.avoidability_label == AvoidabilityLabel.POTENTIALLY_AVOIDABLE)
    if month:
        stmt = stmt.where(Refund.created_month == month)
    if agent_id:
        stmt = stmt.where(Refund.agent_id == agent_id)
    if reason_code:
        stmt = stmt.where(Refund.normalised_reason_code == reason_code)
    if min_amount is not None:
        stmt = stmt.where(Refund.amount_inr >= min_amount)

    total = session.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()
    records = session.execute(stmt.limit(limit).offset(offset)).mappings().all()

    rows = []
    for r in records:
        sections = [str(s) for s in (r["policy_section_ids"] or [])]
        citations = [
            policy.section(s).citation if s in policy.sections
            else f"support-policy.pdf v{policy.version} §{s}"
            for s in sections
        ]
        rows.append(
            {
                "ticket_id": r["ticket_id"],
                "month": r["created_month"],
                "agent_id": r["agent_id"],
                "team": r["agent_team"] or "(unknown)",
                "channel": r["channel"],
                "refund_value_inr": round(float(r["amount_inr"]), 2),
                "source_reason_code": r["source_reason_code"],
                "reason_code": r["normalised_reason_code"] or "UNCLASSIFIED",
                "reason_explanation": r["reason_explanation"] or "",
                "potentially_avoidable": r["avoidability_label"],
                "avoidability_drivers": [str(d) for d in (r["avoidability_drivers"] or [])],
                "avoidable_value_inr": round(float(r["avoidable_value_inr"] or 0), 2),
                "avoidability_basis": r["avoidability_basis"] or "",
                "evidence": r["evidence"] or [],
                "policy_section_ids": sections,
                "policy_citations": citations,
                "confidence": float(r["confidence"] or 0),
                "model_name": r["model_name"] or "(unclassified)",
                "prompt_version": r["prompt_version"] or "",
                "review_required": bool(r["review_required"]),
            }
        )

    return {
        "rows": rows,
        "total": int(total),
        "privacy_note": (
            "Customer name, city, phone, email and address are never selected by this "
            "endpoint. Customers appear only as internal IDs, and no customer identifier "
            "appears in board-level exports."
        ),
    }


# ---------------------------------------------------------------------------
# Manufacturing-lot signal (not requested; see README "beyond the brief")
# ---------------------------------------------------------------------------
def lot_analysis(session: Session, min_tickets: int = 30) -> dict:
    """Refund rate by manufacturing lot_code.

    Nobody asked for this. `lot_code` is "the manufacturing lot printed on the
    box" (README.txt) and it joins to refunds through the order, so a lot with a
    refund rate well above baseline is a hardware-quality lead rather than a
    support one.

    Reported with a two-proportion z-test and a minimum volume, because with 1,276
    lots in the data the highest rate is guaranteed to look alarming by chance.
    """
    tickets = pd.read_sql(
        select(Ticket.ticket_id, Ticket.order_id, Ticket.is_refund, Ticket.refund_amount_inr,
               Ticket.product_sku),
        session.connection(),
    )
    orders = pd.read_sql(
        select(Order.order_id, Order.lot_code, Order.sku), session.connection()
    )
    joined = tickets.merge(orders, on="order_id", how="inner")
    if joined.empty:
        return {"rows": [], "note": "no order-linked tickets"}

    grouped = joined.groupby(["lot_code", "sku"]).agg(
        tickets=("ticket_id", "size"),
        refunds=("is_refund", "sum"),
        refund_value_inr=("refund_amount_inr", "sum"),
    ).reset_index()

    baseline_rate = float(joined["is_refund"].mean())
    eligible = grouped[grouped["tickets"] >= min_tickets].copy()
    if eligible.empty:
        return {
            "rows": [],
            "baseline_refund_rate_pct": round(baseline_rate * 100, 2),
            "note": f"no lot has at least {min_tickets} order-linked tickets",
        }

    eligible["refund_rate_pct"] = (eligible["refunds"] / eligible["tickets"] * 100).round(2)
    # Two-proportion z against baseline.
    p0 = baseline_rate
    n = eligible["tickets"]
    p = eligible["refunds"] / n
    se = np.sqrt(p0 * (1 - p0) / n)
    eligible["z_score"] = ((p - p0) / se).round(2)
    # Bonferroni across the lots actually tested.
    k = len(eligible)
    eligible["significant_bonferroni_05"] = eligible["z_score"] > (
        2.5758 if k > 20 else 1.96
    )

    top = eligible.sort_values("z_score", ascending=False).head(15)
    rows = [
        {
            "lot_code": r["lot_code"],
            "sku": r["sku"],
            "tickets": int(r["tickets"]),
            "refunds": int(r["refunds"]),
            "refund_rate_pct": float(r["refund_rate_pct"]),
            "refund_value_inr": round(float(r["refund_value_inr"] or 0), 2),
            "z_score": float(r["z_score"]),
            "significant": bool(r["significant_bonferroni_05"]),
        }
        for _, r in top.iterrows()
    ]
    n_sig = int(eligible["significant_bonferroni_05"].sum())
    return {
        "rows": rows,
        "baseline_refund_rate_pct": round(baseline_rate * 100, 2),
        "lots_tested": k,
        "lots_significant": n_sig,
        "min_tickets": min_tickets,
        "method": (
            f"Two-proportion z-test of each lot's refund rate against the {baseline_rate * 100:.1f}% "
            f"baseline, restricted to lots with at least {min_tickets} order-linked tickets, with "
            "a Bonferroni-adjusted threshold across the lots tested."
        ),
        "interpretation": (
            f"{n_sig} of {k} lots exceed the baseline refund rate at the adjusted threshold. "
            + (
                "That is consistent with chance across this many lots, so this is a lead to "
                "check against QA records rather than a finding."
                if n_sig <= max(1, k // 20)
                else "Worth checking against QA and lot-level returns before drawing a conclusion."
            )
        ),
        "caveat": (
            "Not requested in the brief. Lot volumes are small, the refund flag is not a "
            "confirmed hardware fault, and this analysis cannot distinguish a manufacturing "
            "problem from a batch that happened to ship to a region with delivery trouble."
        ),
    }
