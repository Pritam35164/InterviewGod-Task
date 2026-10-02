"""Deterministic, policy-cited avoidability.

Division of labour, which is the whole point of the hybrid design:

  Nemotron reads the note and reports *facts about the text* — the true reason,
  whether both remedies were actually given, whether money went out without the
  unit coming back. It never sees the amount and never decides avoidability.

  This module applies the policy rules to those facts and to the deterministic
  fields, and computes every rupee. Each driver cites the clause it comes from;
  there are no unsourced drivers.

Language discipline (brief §23): POTENTIALLY_AVOIDABLE means the evidence
suggests the payment was preventable under the documented policy. It is not an
allegation of fraud or misconduct, and no output of this module names or
characterises an agent.

Overlap: one refund can trigger several drivers. The ticket's avoidable value is
the **maximum** across its drivers, never the sum, so a refund that is both a
dual remedy and a duplicate is counted once at the larger figure. Per-driver
totals are reported gross and are therefore expected to sum to more than the
deduplicated total; `overlap_note` states this on every response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.policy.loader import PolicyDocument, get_policy
from app.schemas.taxonomy import (
    AvoidabilityDriver,
    AvoidabilityLabel,
    RefundReason,
)

log = get_logger(__name__)


@dataclass
class RefundFacts:
    """Everything the rules need about one refund.

    Deterministic fields come from PostgreSQL; the `ai_*` fields come from the
    validated Nemotron classification and are None when the ticket is
    unclassified, which is what forces REVIEW_REQUIRED rather than a guess.
    """

    ticket_id: str
    amount_inr: float
    source_reason_code: str | None = None
    replacement_issued_flag: bool = False
    order_id: str | None = None
    order_value_inr: float | None = None
    order_refund_ticket_count: int = 1
    order_refund_excess_inr: float = 0.0
    agent_tier: int | None = None
    agent_team: str | None = None
    sla_breached: bool | None = None
    channel: str = "chat"
    data_quality_flags: list[str] = field(default_factory=list)

    # ---- from the validated LLM classification ---------------------------
    ai_reason_code: str | None = None
    ai_dual_remedy: bool | None = None
    ai_refund_without_return: bool | None = None
    ai_replacement_mentioned: bool | None = None
    ai_confidence: float | None = None
    ai_evidence: list[dict] = field(default_factory=list)
    classification_valid: bool = False

    def evidence_quote_for(self, *keywords: str) -> str | None:
        """Pick the stored evidence quote that best supports a driver."""
        for span in self.ai_evidence:
            quote = (span.get("quote") or "").lower()
            if any(k in quote for k in keywords):
                return span.get("quote")
        return self.ai_evidence[0].get("quote") if self.ai_evidence else None


@dataclass
class DriverHit:
    driver: AvoidabilityDriver
    avoidable_value_inr: float
    policy_section_id: str
    policy_citation: str
    policy_quote: str
    detection: str
    evidence_quote: str | None = None
    confidence_basis: str = ""

    def to_dict(self) -> dict:
        return {
            "driver": str(self.driver),
            "avoidable_value_inr": round(self.avoidable_value_inr, 2),
            "policy_section_id": self.policy_section_id,
            "policy_citation": self.policy_citation,
            "policy_quote": self.policy_quote,
            "detection": self.detection,
            "evidence_quote": self.evidence_quote,
            "confidence_basis": self.confidence_basis,
        }


@dataclass
class AvoidabilityAssessment:
    label: AvoidabilityLabel
    drivers: list[DriverHit]
    avoidable_value_inr: float
    basis: str
    policy_section_ids: list[str]
    avoidable_contact_cost_inr: float = 0.0
    review_reason: str | None = None

    def driver_names(self) -> list[str]:
        return [str(d.driver) for d in self.drivers]

    def to_dict(self) -> dict:
        return {
            "label": str(self.label),
            "drivers": [d.to_dict() for d in self.drivers],
            "avoidable_value_inr": round(self.avoidable_value_inr, 2),
            "avoidable_contact_cost_inr": round(self.avoidable_contact_cost_inr, 2),
            "basis": self.basis,
            "policy_section_ids": self.policy_section_ids,
            "review_reason": self.review_reason,
        }


OVERLAP_NOTE = (
    "A refund can trigger more than one driver. Per-driver values below are gross and "
    "overlap; the headline total counts each refund once at its largest driver value, "
    "so the driver rows sum to more than the total. Both figures are shown deliberately."
)

DISCLAIMER = (
    "'Potentially avoidable' means the available evidence suggests the payment may have "
    "been preventable under Vireo's documented policy and process. It is not a finding of "
    "fraud, misconduct or wrongdoing by any individual, and the Returns Desk processing a "
    "high share of refunds is by design (policy §6)."
)


def _quote(policy: PolicyDocument, path: list[str]) -> str:
    node: Any = policy.rules
    for key in path:
        node = node[key]
    return " ".join(str(node).split())


def _citation(policy: PolicyDocument, section_id: str) -> str:
    try:
        return policy.section(section_id).citation
    except Exception:
        return f"support-policy.pdf v{policy.version} §{section_id}"


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _dual_remedy(facts: RefundFacts, policy: PolicyDocument) -> DriverHit | None:
    """Policy §5: an order must never receive both a refund and a replacement.

    Two independent detectors, deliberately unioned:
      * `replacement_issued` — the structured flag the agent set.
      * Nemotron's reading of the closing note.

    They disagree often, in both directions: notes say "issued refund +
    replacement both, tl aware" with the flag on N, and the flag is set Y on
    tickets whose note says "replacement not applicable (out of stock)". Neither
    signal alone is trustworthy, so the union is taken and the source of the
    detection is recorded for each hit.
    """
    flag = bool(facts.replacement_issued_flag)
    ai = facts.ai_dual_remedy is True
    if not (flag or ai):
        return None

    if flag and ai:
        detection = "flagged by both the helpdesk replacement_issued field and the closing note"
        basis = "two independent signals agree"
    elif flag:
        detection = "helpdesk replacement_issued = Y"
        basis = "structured field only; the closing note does not confirm it"
    else:
        detection = "closing note states both a refund and a replacement were given"
        basis = "closing note only; the replacement_issued field was left N"

    return DriverHit(
        driver=AvoidabilityDriver.DUAL_REMEDY,
        avoidable_value_inr=facts.amount_inr,
        policy_section_id="5",
        policy_citation=_citation(policy, "5"),
        policy_quote=_quote(policy, ["refunds", "dual_remedy_prohibition", "source_quote"]),
        detection=detection,
        evidence_quote=facts.evidence_quote_for(
            "replacement", "rplc", "repl", "new unit", "fresh unit", "new one"
        ),
        confidence_basis=basis,
    )


def _duplicate_refund(facts: RefundFacts, policy: PolicyDocument) -> DriverHit | None:
    """The same order refunded on more than one ticket, beyond its value.

    Fully deterministic: computed in ingest from order_id, so it needs no model
    and cannot be wrong about the arithmetic.
    """
    if facts.order_refund_ticket_count < 2 or facts.order_refund_excess_inr <= 0:
        return None
    return DriverHit(
        driver=AvoidabilityDriver.DUPLICATE_REFUND_SAME_ORDER,
        avoidable_value_inr=facts.order_refund_excess_inr,
        policy_section_id="5",
        policy_citation=_citation(policy, "5"),
        policy_quote=(
            "Section 5 defines each remedy as an alternative ('customer chooses a full "
            "refund or a replacement'; 'reshipment or refund'), so one order has one remedy."
        ),
        detection=(
            f"order {facts.order_id} carries refunds on {facts.order_refund_ticket_count} "
            f"tickets totalling more than its value of ₹{facts.order_value_inr:,.2f}"
            if facts.order_value_inr
            else f"order {facts.order_id} carries refunds on "
            f"{facts.order_refund_ticket_count} separate tickets"
        ),
        confidence_basis="deterministic: computed from order_id and refund amounts, no model involved",
    )


def _goodwill_over_cap(facts: RefundFacts, policy: PolicyDocument) -> DriverHit | None:
    """Policy §5: goodwill credits are capped at ₹500 per ticket.

    Uses the *reclassified* reason, not the dropdown code. GW-OTHER is the first
    option in the dropdown and carries 42% of refunds, so applying the cap to the
    dropdown code would overstate this driver by counting every mis-coded delivery
    failure and duplicate charge as goodwill. An unclassified ticket therefore
    produces no hit here at all.
    """
    if not facts.classification_valid or facts.ai_reason_code != RefundReason.GOODWILL_GESTURE:
        return None
    cap = policy.goodwill_cap_inr
    if facts.amount_inr <= cap:
        return None
    return DriverHit(
        driver=AvoidabilityDriver.GOODWILL_OVER_CAP,
        avoidable_value_inr=facts.amount_inr - cap,
        policy_section_id="5",
        policy_citation=_citation(policy, "5"),
        policy_quote=_quote(policy, ["refunds", "goodwill", "source_quote"]),
        detection=(
            f"classified as a discretionary goodwill payment of ₹{facts.amount_inr:,.2f}, "
            f"which is ₹{facts.amount_inr - cap:,.2f} above the ₹{cap:,.0f} per-ticket cap. "
            "Only the excess over the cap is counted as avoidable, since ₹"
            f"{cap:,.0f} was authorised."
        ),
        evidence_quote=facts.evidence_quote_for("goodwill", "gesture", "courtesy", "one-time"),
        confidence_basis=(
            "depends on the model's reason classification, not on the agent's dropdown code; "
            "no field in the data records whether Team Lead approval was obtained, so this is "
            "'over cap with no approval evidence recorded', not 'unapproved'"
        ),
    )


def _refund_without_return(facts: RefundFacts, policy: PolicyDocument) -> DriverHit | None:
    """Money released without the unit coming back or passing QC.

    Policy §5 defines RETURN-QC-OK as "Return received and passed QC" and §6 gives
    the Returns Desk ownership of return pickups, so releasing a refund while the
    pickup is still outstanding is a process failure. No structured field records
    pickup or QC state, so this is detected from the closing note only, and that
    limitation is stated in `confidence_basis`.
    """
    if not facts.classification_valid:
        return None
    hit = facts.ai_refund_without_return is True or (
        facts.ai_reason_code == RefundReason.RETURN_PICKUP_FAILURE
    )
    if not hit:
        return None
    return DriverHit(
        driver=AvoidabilityDriver.REFUND_WITHOUT_RETURN,
        avoidable_value_inr=facts.amount_inr,
        policy_section_id="5",
        policy_citation=_citation(policy, "5"),
        policy_quote=(
            "Section 5 reason code RETURN-QC-OK is defined as 'Return received and passed QC'; "
            "section 6: 'Returns Desk owns return pickups and refund processing'."
        ),
        detection=(
            "closing note indicates the refund was released without the unit being collected "
            "or passing QC"
        ),
        evidence_quote=facts.evidence_quote_for(
            "without pickup", "pickup", "pkp", "qc", "not done", "missed", "pending"
        ),
        confidence_basis=(
            "closing-note evidence only: the data pack has no structured pickup or QC "
            "completion field, so this cannot be corroborated against a system of record"
        ),
    )


DRIVER_TITLES = {
    AvoidabilityDriver.DUAL_REMEDY: "Refund and replacement both given for the same order",
    AvoidabilityDriver.DUPLICATE_REFUND_SAME_ORDER: "Same order refunded on more than one ticket",
    AvoidabilityDriver.GOODWILL_OVER_CAP: "Goodwill payment above the ₹500 per-ticket cap",
    AvoidabilityDriver.REFUND_WITHOUT_RETURN: "Refund released without the unit being returned or QC'd",
    AvoidabilityDriver.REFUND_AFTER_SERVICE_FAILURE: "Repeat contact caused by an earlier refund not landing",
}


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
def assess(facts: RefundFacts, policy: PolicyDocument | None = None) -> AvoidabilityAssessment:
    policy = policy or get_policy()

    hits: list[DriverHit] = []
    for rule in (_dual_remedy, _duplicate_refund, _goodwill_over_cap, _refund_without_return):
        hit = rule(facts, policy)
        if hit is not None and hit.avoidable_value_inr > 0:
            hits.append(hit)

    # Repeat contact caused by a failed earlier refund: the refund itself is owed,
    # so only the avoidable *contact cost* is counted, never the refund value.
    contact_cost = 0.0
    if facts.classification_valid and facts.ai_reason_code == RefundReason.REFUND_SERVICE_FAILURE:
        contact_cost = policy.cost_per_contact_inr.get(
            facts.channel, policy.blended_cost_per_contact_inr
        )

    if hits:
        # Count each refund once, at its largest driver.
        value = max(h.avoidable_value_inr for h in hits)
        label = AvoidabilityLabel.POTENTIALLY_AVOIDABLE
        basis = (
            f"{len(hits)} policy driver(s) matched: "
            + "; ".join(f"{h.driver} (₹{h.avoidable_value_inr:,.2f}) per §{h.policy_section_id}" for h in hits)
            + f". Counted once at ₹{value:,.2f}, the largest driver, to avoid double counting. "
            + DISCLAIMER
        )
        return AvoidabilityAssessment(
            label=label,
            drivers=hits,
            avoidable_value_inr=value,
            avoidable_contact_cost_inr=contact_cost,
            basis=basis,
            policy_section_ids=sorted({h.policy_section_id for h in hits}),
        )

    if not facts.classification_valid:
        return AvoidabilityAssessment(
            label=AvoidabilityLabel.REVIEW_REQUIRED,
            drivers=[],
            avoidable_value_inr=0.0,
            avoidable_contact_cost_inr=contact_cost,
            basis=(
                "No valid AI classification for this ticket, and no deterministic driver "
                "matched. Avoidability is left unassessed rather than defaulted to compliant."
            ),
            policy_section_ids=[],
            review_reason="ticket not classified",
        )

    if facts.ai_reason_code in {RefundReason.AMBIGUOUS, RefundReason.UNKNOWN}:
        return AvoidabilityAssessment(
            label=AvoidabilityLabel.AMBIGUOUS,
            drivers=[],
            avoidable_value_inr=0.0,
            avoidable_contact_cost_inr=contact_cost,
            basis=(
                f"Reason classified as {facts.ai_reason_code}: the ticket text does not "
                "support a confident reading, so no avoidability judgement is made. "
                + DISCLAIMER
            ),
            policy_section_ids=[],
        )

    return AvoidabilityAssessment(
        label=AvoidabilityLabel.POLICY_COMPLIANT,
        drivers=[],
        avoidable_value_inr=0.0,
        avoidable_contact_cost_inr=contact_cost,
        basis=(
            f"Classified as {facts.ai_reason_code} with no policy driver matched: the "
            "documented policy supports this payment as it happened. " + DISCLAIMER
        ),
        policy_section_ids=["5"],
    )


def opportunity(
    avoidable_value_inr: float, months_observed: int
) -> dict[str, Any]:
    """Annualise the observed figure. No reduction target is assumed.

    The brief is explicit (§26): quarterlyise the *observed* value and label any
    hypothetical separately. Nothing here claims a saving.
    """
    if months_observed <= 0:
        return {
            "observed_value_inr": round(avoidable_value_inr, 2),
            "months_observed": 0,
            "quarterlyised_inr": 0.0,
            "annualised_inr": 0.0,
            "arithmetic": "not computable: zero months observed",
            "scenarios": [],
        }

    per_month = avoidable_value_inr / months_observed
    quarterly = per_month * 3
    annual = per_month * 12
    return {
        "observed_value_inr": round(avoidable_value_inr, 2),
        "months_observed": months_observed,
        "per_month_inr": round(per_month, 2),
        "quarterlyised_inr": round(quarterly, 2),
        "annualised_inr": round(annual, 2),
        "arithmetic": (
            f"₹{avoidable_value_inr:,.2f} observed over {months_observed} months "
            f"= ₹{per_month:,.2f}/month; x3 = ₹{quarterly:,.2f}/quarter; "
            f"x12 = ₹{annual:,.2f}/year"
        ),
        "scenarios": [
            {
                "label": f"SCENARIO (illustrative, not a forecast): eliminate {pct}% of the observed drivers",
                "reduction_pct": pct,
                "quarterly_inr": round(quarterly * pct / 100, 2),
                "annual_inr": round(annual * pct / 100, 2),
                "caveat": (
                    "Hypothetical. Derived by scaling the observed figure, not from any "
                    "measured intervention. Vireo has run no such intervention and this "
                    "platform has not tested one."
                ),
            }
            for pct in (25, 50)
        ],
        "caveats": [
            "Annualisation assumes the observed rate continues. Contact volume nearly "
            "doubled across the period, so a flat extrapolation is optimistic on count and "
            "conservative on nothing.",
            "Avoidable value is what the policy says should not have been paid. It is not a "
            "cash saving: recovering it needs process change, and some of it is money a "
            "customer would have received under a different remedy anyway.",
            DISCLAIMER,
        ],
    }
