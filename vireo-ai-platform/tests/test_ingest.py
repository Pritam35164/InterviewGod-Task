"""Data ingest: unit detection, dedup, joins, money."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.ingest import detect_legacy_money_unit, deduplicate_tickets, load_raw, run_ingest
from app.services.ingest_rules import FindingLog

FIXTURES = Path(__file__).parent / "fixtures" / "mini"


def test_legacy_unit_is_measured_not_guessed():
    tickets = pd.read_csv(FIXTURES / "tickets.csv", dtype=str, keep_default_na=False)
    log = FindingLog("t")
    det = detect_legacy_money_unit(tickets, log)
    assert det["conclusive"] is True
    assert det["divisor"] == 100.0
    assert det["paired_tickets"] >= 1
    assert log.get("DQ-001") is not None


def test_dedup_prefers_helpdesk_and_keeps_one_row():
    tickets = pd.read_csv(FIXTURES / "tickets.csv", dtype=str, keep_default_na=False)
    log = FindingLog("t")
    det = detect_legacy_money_unit(tickets, log)
    out = deduplicate_tickets(tickets, log, det["divisor"])
    assert out["ticket_id"].is_unique
    kept = out.loc[out["ticket_id"] == "TK-MINI-001"].iloc[0]
    assert kept["source_system"] == "helpdesk"
    assert bool(kept["was_duplicated"]) is True


def test_ingest_reconciles_mini_pack(db):
    result = run_ingest(db, raw_dir=FIXTURES, truncate=True)
    assert result.counts["tickets_loaded"] == 4
    assert result.counts["refunds_loaded"] == 3
    # TK-MINI-001 helpdesk 900 + legacy 90000 counted twice in the raw file.
    assert result.money["raw_export_total_inr"] == 900 + 90000 + 1999 + 249900
    assert result.money["after_dedup_inr"] == 900 + 1999 + 2499
    from app.models.refund import Refund
    from sqlalchemy import select

    refunds = db.execute(select(Refund)).scalars().all()
    by_id = {r.ticket_id: r.amount_inr for r in refunds}
    assert by_id["TK-MINI-001"] == 900
    assert by_id["TK-MINI-003"] == 1999
    assert by_id["TK-MINI-004"] == 2499


def test_load_raw_uses_real_column_names():
    raw = load_raw()
    assert "ticket_id" in raw["tickets"].columns
    assert "refund_amount_inr" in raw["tickets"].columns
    assert "lot_code" in raw["orders"].columns
    assert "care_plus" in raw["customers"].columns
