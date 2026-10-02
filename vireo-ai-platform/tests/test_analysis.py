"""Avoidability drivers and monthly/agent aggregations."""

from __future__ import annotations

from pathlib import Path

from app.schemas.taxonomy import AvoidabilityDriver, AvoidabilityLabel, RefundReason
from app.services.analysis import agents, monthly, overview
from app.services.avoidability import RefundFacts, assess
from app.services.baseline import baseline_dual_remedy, classify_baseline
from app.services.classification import apply_deterministic_avoidability
from app.services.ingest import run_ingest

FIXTURES = Path(__file__).parent / "fixtures" / "mini"


def test_dual_remedy_from_flag_without_llm():
    facts = RefundFacts(
        ticket_id="TK-1",
        amount_inr=2499,
        replacement_issued_flag=True,
        source_reason_code="GW-OTHER",
    )
    out = assess(facts)
    assert out.label == AvoidabilityLabel.POTENTIALLY_AVOIDABLE
    assert AvoidabilityDriver.DUAL_REMEDY in [d.driver for d in out.drivers]
    assert out.avoidable_value_inr == 2499


def test_goodwill_cap_requires_classification():
    facts = RefundFacts(
        ticket_id="TK-1",
        amount_inr=2000,
        source_reason_code="GW-OTHER",
        classification_valid=False,
    )
    out = assess(facts)
    assert out.label == AvoidabilityLabel.REVIEW_REQUIRED
    assert out.avoidable_value_inr == 0


def test_duplicate_refund_excess():
    facts = RefundFacts(
        ticket_id="TK-2",
        amount_inr=1999,
        order_id="VR1",
        order_value_inr=1999,
        order_refund_ticket_count=2,
        order_refund_excess_inr=1999,
        classification_valid=True,
        ai_reason_code=RefundReason.DUPLICATE_OR_FAILED_PAYMENT,
    )
    out = assess(facts)
    assert out.label == AvoidabilityLabel.POTENTIALLY_AVOIDABLE
    assert out.avoidable_value_inr == 1999


def test_baseline_reads_cancel_and_negation():
    assert classify_baseline("stop the shipment", "cancelled before dispatch") == RefundReason.CANCELLED_BEFORE_DISPATCH
    assert baseline_dual_remedy("issued refund + replacement both, tl aware") is True
    assert baseline_dual_remedy("cx asked for replacement + refund, explained policy, refund only") is False


def test_monthly_and_agent_metrics(db):
    run_ingest(db, raw_dir=FIXTURES, truncate=True)
    apply_deterministic_avoidability(db)
    ov = overview(db)
    assert ov["total_tickets"] == 4
    assert ov["refund_tickets"] == 3
    assert ov["refund_value_inr"] == 900 + 1999 + 2499
    assert ov["potentially_avoidable_count"] >= 1  # dual-remedy flag on TK-MINI-001
    m = monthly(db)
    months = {r["month"] for r in m["rows"]}
    assert "2025-01" in months
    ag = agents(db)
    returns = [r for r in ag["rows"] if r["team"] == "Returns Desk"][0]
    chat = [r for r in ag["rows"] if r["team"] == "Chat Frontline"][0]
    assert returns["refunds_per_100_tickets"] > chat["refunds_per_100_tickets"]
    assert chat["comparable_on_volume"] is True
