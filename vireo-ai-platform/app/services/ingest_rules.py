"""Data-quality rules and the decisions taken for each one.

Every landmine in this data pack is handled by a named rule here, so the answer
to "why is your total different from the export" is a row in
`data_quality_events` rather than a paragraph in a README.

The rules that change the money:

  DQ-001  Legacy monetary unit. Policy §9 says the legacy tool "stored monetary
          values in its own native unit" without naming it. The unit is measured,
          not assumed: 125 tickets exist under BOTH source systems, and for every
          one of them legacy = helpdesk x 100 exactly. The divisor is asserted
          before use, and if the evidence is not unanimous no conversion happens
          and the run is marked critical.

  DQ-002  Re-imported duplicates. Policy §9 warns a subset of legacy tickets
          "may appear in exports under both source systems". 638 ticket_ids do.
          The helpdesk row is kept because the current helpdesk stores rupees.

Together these two explain the whole gap between Arjun's "well over a crore a
quarter" and Sameer's "around Rs 11 lakh a quarter".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_INFO = "info"


@dataclass
class Finding:
    finding_id: str
    severity: str
    title: str
    affected_rows: int = 0
    affected_value_inr: float | None = None
    decision: str = ""
    policy_reference: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "title": self.title,
            "affected_rows": self.affected_rows,
            "affected_value_inr": self.affected_value_inr,
            "decision": self.decision,
            "policy_reference": self.policy_reference,
            "evidence": self.evidence,
        }


class FindingLog:
    """Collects findings during a run so they can be persisted and reported."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.findings: list[Finding] = []

    def add(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        return finding

    def get(self, finding_id: str) -> Finding | None:
        return next((f for f in self.findings if f.finding_id == finding_id), None)

    def as_dicts(self) -> list[dict]:
        order = {
            SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1,
            SEVERITY_MEDIUM: 2, SEVERITY_INFO: 3,
        }
        return [
            f.to_dict()
            for f in sorted(self.findings, key=lambda x: (order.get(x.severity, 9), x.finding_id))
        ]

    def has_critical(self) -> bool:
        return any(f.severity == SEVERITY_CRITICAL for f in self.findings)


# Policy §3 first-response targets, in minutes, keyed by channel.
SLA_TARGET_MINUTES = {"chat": 15, "voice": 120, "social": 240, "email": 480}

# Ticket-level data-quality flags attached to individual rows.
FLAG_NO_ORDER_ID = "order_id_missing_in_export"
FLAG_ORDER_AMBIGUOUS = "order_resolved_by_nearest_date"
FLAG_ORDER_UNRESOLVED = "order_could_not_be_resolved"
FLAG_TICKET_BEFORE_ORDER = "ticket_created_before_order_date"
FLAG_REFUND_ON_OPEN_TICKET = "refund_on_open_or_pending_ticket"
FLAG_LEGACY_UNIT_CONVERTED = "legacy_amount_converted_to_inr"
FLAG_REIMPORT_DUPLICATE = "reimported_under_both_source_systems"
FLAG_CSAT_BLANK = "csat_blank_excluded_from_averages"
FLAG_REFUND_EXCEEDS_ORDER = "refund_exceeds_order_value"
FLAG_NO_RESOLVED_AT = "resolved_at_blank"
