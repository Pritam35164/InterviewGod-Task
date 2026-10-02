"""The policy engine: the only component allowed to authorise an action.

Authority order, enforced in `evaluate_outbound_action` and reported back to the
caller in `decision_hierarchy` so an overruled Laya call is visible in the API
response and not buried in a log line:

    1. Security constraints
    2. Hard business/policy rules (the prohibitions in support-policy.pdf §5)
    3. Policy engine operational controls
    4. Laya routing decision
    5. Nemotron reasoning
    6. Final action

Two properties this file is built around:

  * Laya can only ever *raise* the required scrutiny, never lower it. If Laya
    says `requires_human=False` and a policy rule says approval is required, the
    rule wins. There is no code path where a Laya or LLM output relaxes a check.
  * Fail closed. Missing policy, missing amount, missing reason, unknown reason,
    Laya degraded on a high-risk action — all resolve to
    REQUIRES_HUMAN_REVIEW or DENY, never to ALLOW.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.core.errors import PolicyUnavailable
from app.core.logging import get_logger
from app.policy.loader import PolicyDocument, get_policy
from app.schemas.api import OutboundActionRequest, PolicyCheck
from app.schemas.laya import LayaDecisionEnvelope
from app.schemas.taxonomy import (
    HIGH_RISK_ACTIONS,
    ActionType,
    PolicyOutcome,
    RiskLevel,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule identifiers. Each one cites the policy section it comes from, or is
# explicitly marked as an operational control rather than policy.
# ---------------------------------------------------------------------------
class Rule:
    SECURITY_INJECTION = "SEC-001-prompt-injection"
    SECURITY_GUARD = "SEC-002-laya-guard"
    POLICY_AVAILABLE = "SYS-001-policy-available"
    LAYA_AVAILABLE = "SYS-002-laya-available-for-high-risk"
    DUAL_REMEDY = "POL-5-001-dual-remedy-prohibited"
    GOODWILL_CAP = "POL-5-002-goodwill-cap-500"
    REASON_CODE_VALID = "POL-5-003-reason-code-in-taxonomy"
    WARRANTY_TIER2 = "POL-6-001-warranty-replacement-tier2"
    DUPLICATE_ORDER_REFUND = "POL-5-004-no-second-refund-same-order"
    AMOUNT_REQUIRED = "CTL-001-amount-required"
    AUTO_APPROVE_CEILING = "CTL-002-auto-approve-ceiling"
    CRITICAL_AMOUNT = "CTL-003-critical-amount-manager"
    REFUND_EXCEEDS_ORDER = "CTL-004-refund-exceeds-order-value"


_SEVERITY_ORDER = {PolicyOutcome.ALLOW: 0, PolicyOutcome.REQUIRES_HUMAN_REVIEW: 1, PolicyOutcome.DENY: 2}


@dataclass
class ActionContext:
    """Facts the engine needs that do not come from the request body.

    All of these are looked up deterministically from PostgreSQL by
    `app/services/decision_engine.py`. The engine never asks an LLM for them.
    """

    agent_tier: int | None = None
    agent_team: str | None = None
    order_value_inr: float | None = None
    existing_refund_count_for_order: int = 0
    existing_refund_value_for_order: float = 0.0
    replacement_already_issued_for_order: bool = False
    injection_flags: list[str] = field(default_factory=list)
    policy_available: bool = True


@dataclass
class PolicyEvaluation:
    final_decision: PolicyOutcome
    checks: list[PolicyCheck]
    risk_level: RiskLevel
    requires_human: bool
    approver_required: str | None
    reason: str
    policy_version: str
    policy_section_ids: list[str]
    decision_hierarchy: list[str]
    laya_overruled: bool = False
    laya_overrule_detail: str | None = None
    fail_closed: bool = False

    @property
    def executable(self) -> bool:
        return self.final_decision == PolicyOutcome.ALLOW

    def breached_rules(self) -> list[str]:
        return [c.rule_id for c in self.checks if c.breached]


def _cite(policy: PolicyDocument, section_id: str) -> str:
    try:
        return policy.section(section_id).citation
    except Exception:
        return f"support-policy.pdf v{policy.version} §{section_id}"


def _worst(checks: list[PolicyCheck]) -> PolicyOutcome:
    if not checks:
        return PolicyOutcome.ALLOW
    return max((c.outcome for c in checks), key=lambda o: _SEVERITY_ORDER[o])


def _risk_from_amount(amount: float | None, action: ActionType) -> RiskLevel:
    """Deterministic risk floor from the amount. Laya may raise this, not lower it."""
    if action not in HIGH_RISK_ACTIONS:
        return RiskLevel.LOW
    if amount is None:
        return RiskLevel.HIGH  # unknown money is not low risk
    if amount >= settings.REFUND_CRITICAL_RISK_INR:
        return RiskLevel.CRITICAL
    if amount > settings.REFUND_AUTO_APPROVE_MAX_INR:
        return RiskLevel.HIGH
    return RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Outbound authorisation
# ---------------------------------------------------------------------------
def evaluate_outbound_action(
    request: OutboundActionRequest,
    context: ActionContext,
    laya: LayaDecisionEnvelope | None = None,
) -> PolicyEvaluation:
    checks: list[PolicyCheck] = []
    hierarchy: list[str] = []
    sections: set[str] = set()
    fail_closed = False

    # ---- 1. security ------------------------------------------------------
    hierarchy.append("1. security constraints")
    if context.injection_flags:
        checks.append(
            PolicyCheck(
                rule_id=Rule.SECURITY_INJECTION,
                outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                detail=(
                    "Request text contains prompt-injection patterns "
                    f"({', '.join(context.injection_flags[:4])}). An action requested "
                    "through injected text is never executed automatically."
                ),
                breached=True,
            )
        )
    if laya is not None and laya.guard is not None:
        g = laya.guard
        if g.jailbreak or g.prompt_injection or g.harm_severity >= 2:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.SECURITY_GUARD,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    detail=(
                        f"Laya input guardrail flagged: jailbreak={g.jailbreak}, "
                        f"prompt_injection={g.prompt_injection}, harm_severity={g.harm_severity}"
                    ),
                    breached=True,
                )
            )

    # ---- 2. policy availability: fail closed ------------------------------
    hierarchy.append("2. hard policy rules")
    try:
        policy = get_policy()
        if not context.policy_available:
            raise PolicyUnavailable("policy marked unavailable by caller")
    except PolicyUnavailable as exc:
        # Brief §43: refund approval requires a policy check; no policy means
        # REQUIRES_HUMAN_REVIEW, never APPROVE.
        log.error("policy.unavailable_for_action", extra={"action": request.action, "error": str(exc)})
        return PolicyEvaluation(
            final_decision=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
            checks=[
                PolicyCheck(
                    rule_id=Rule.POLICY_AVAILABLE,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    detail=(
                        "Support policy could not be loaded, so no authorisation rule "
                        "could be evaluated. Failing closed."
                    ),
                    breached=True,
                )
            ],
            risk_level=RiskLevel.CRITICAL,
            requires_human=True,
            approver_required="Team Lead",
            reason="Policy unavailable: action withheld pending human review.",
            policy_version=settings.POLICY_VERSION,
            policy_section_ids=[],
            decision_hierarchy=hierarchy + ["FAIL CLOSED: policy unavailable"],
            fail_closed=True,
        )

    is_high_risk = request.action in HIGH_RISK_ACTIONS
    amount = request.amount_inr

    # ---- 2a. dual remedy: the one absolute prohibition in §5 --------------
    if request.action in {ActionType.REFUND, ActionType.REPLACEMENT}:
        sections.add("5")
        quote = policy.rules["refunds"]["dual_remedy_prohibition"]["source_quote"].strip()
        would_dual = (
            (request.action == ActionType.REFUND and (request.replacement_issued or context.replacement_already_issued_for_order))
            or (request.action == ActionType.REPLACEMENT and context.existing_refund_count_for_order > 0)
        )
        if would_dual and policy.dual_remedy_prohibited:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.DUAL_REMEDY,
                    outcome=PolicyOutcome.DENY,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail=(
                        f"Denied: this would give the same order both a refund and a "
                        f"replacement. Policy §5: \"{quote}\" This must be escalated to "
                        f"the Team Lead and Finance the same day."
                    ),
                    breached=True,
                )
            )
        else:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.DUAL_REMEDY,
                    outcome=PolicyOutcome.ALLOW,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail="No dual remedy detected for this order.",
                )
            )

    # ---- 2b. second refund on the same order -----------------------------
    if request.action == ActionType.REFUND and context.existing_refund_count_for_order > 0:
        sections.add("5")
        already = context.existing_refund_value_for_order
        order_value = context.order_value_inr
        would_exceed = (
            order_value is not None and amount is not None and (already + amount) > order_value + 0.01
        )
        checks.append(
            PolicyCheck(
                rule_id=Rule.DUPLICATE_ORDER_REFUND,
                outcome=PolicyOutcome.DENY if would_exceed else PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                policy_section_id="5",
                policy_citation=_cite(policy, "5"),
                detail=(
                    f"Order {request.order_id} already has {context.existing_refund_count_for_order} "
                    f"refund(s) totalling ₹{already:,.2f}"
                    + (
                        f"; adding ₹{amount:,.2f} would exceed the order value of ₹{order_value:,.2f}."
                        if would_exceed
                        else "; a second refund on one order needs human confirmation."
                    )
                ),
                breached=True,
            )
        )

    # ---- 2c. goodwill cap -------------------------------------------------
    reason = (request.reason_code or "").upper()
    is_goodwill = request.action == ActionType.GOODWILL_CREDIT or reason in {
        "GW-OTHER", "GOODWILL_GESTURE",
    }
    if is_goodwill:
        sections.add("5")
        cap = policy.goodwill_cap_inr
        quote = policy.rules["refunds"]["goodwill"]["source_quote"].strip()
        if amount is None:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.GOODWILL_CAP,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail="Goodwill payment with no amount supplied; cannot check the ₹500 cap.",
                    breached=True,
                )
            )
        elif amount > cap:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.GOODWILL_CAP,
                    outcome=PolicyOutcome.DENY,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail=(
                        f"Denied: ₹{amount:,.2f} exceeds the goodwill cap of ₹{cap:,.0f} "
                        f"per ticket. Policy §5: \"{quote}\" Excess over cap: "
                        f"₹{amount - cap:,.2f}."
                    ),
                    breached=True,
                )
            )
        else:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.GOODWILL_CAP,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail=(
                        f"₹{amount:,.2f} is within the ₹{cap:,.0f} goodwill cap, but policy §5 "
                        f"requires {policy.goodwill_approver} approval for every goodwill credit."
                    ),
                )
            )

    # ---- 2d. warranty replacement certification --------------------------
    if request.action == ActionType.REPLACEMENT and reason in {
        "WTY-BUYBACK", "PRODUCT_FAULT_IN_WARRANTY", "WARRANTY",
    }:
        sections.add("6")
        required_tier = int(policy.rules["teams"]["warranty_replacement_approval_requires_tier"])
        if context.agent_tier is None:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.WARRANTY_TIER2,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    policy_section_id="6",
                    policy_citation=_cite(policy, "6"),
                    detail="Requesting agent's tier is unknown; cannot verify Tier 2 certification.",
                    breached=True,
                )
            )
        elif context.agent_tier < required_tier:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.WARRANTY_TIER2,
                    outcome=PolicyOutcome.DENY,
                    policy_section_id="6",
                    policy_citation=_cite(policy, "6"),
                    detail=(
                        f"Denied: policy §6 states only Tier {required_tier} agents may approve "
                        f"warranty replacements; requesting agent is Tier {context.agent_tier}."
                    ),
                    breached=True,
                )
            )

    # ---- 2e. reason code must be known -----------------------------------
    if request.action == ActionType.REFUND:
        sections.add("5")
        valid = set(policy.reason_codes) | {
            "GOODWILL_GESTURE", "PRODUCT_FAULT_DOA", "PRODUCT_FAULT_IN_WARRANTY",
            "DAMAGED_IN_TRANSIT", "NOT_DELIVERED_OR_LOST", "WRONG_ITEM_SHIPPED",
            "RETURN_RECEIVED_QC_PASSED", "RETURN_PICKUP_FAILURE",
            "DUPLICATE_OR_FAILED_PAYMENT", "CANCELLED_BEFORE_DISPATCH",
            "CUSTOMER_CHANGED_MIND", "PRICE_OR_PROMO_ADJUSTMENT",
            "REFUND_SERVICE_FAILURE",
        }
        if not reason:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.REASON_CODE_VALID,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail="No reason code supplied. Policy §5 requires a reason code when a refund is raised.",
                    breached=True,
                )
            )
        elif reason not in valid:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.REASON_CODE_VALID,
                    outcome=PolicyOutcome.DENY,
                    policy_section_id="5",
                    policy_citation=_cite(policy, "5"),
                    detail=f"Reason code {reason!r} is not in the policy §5 taxonomy.",
                    breached=True,
                )
            )

    # ---- 3. operational controls -----------------------------------------
    hierarchy.append("3. policy engine operational controls")
    if is_high_risk:
        if amount is None and request.action in {ActionType.REFUND, ActionType.GOODWILL_CREDIT}:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.AMOUNT_REQUIRED,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    detail="A money-moving action was requested with no amount. Failing closed.",
                    breached=True,
                )
            )
            fail_closed = True
        elif amount is not None:
            if amount >= settings.REFUND_CRITICAL_RISK_INR:
                checks.append(
                    PolicyCheck(
                        rule_id=Rule.CRITICAL_AMOUNT,
                        outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                        detail=(
                            f"₹{amount:,.2f} is at or above the ₹{settings.REFUND_CRITICAL_RISK_INR:,.0f} "
                            "critical threshold and requires manager approval. This is an "
                            "operational control configured for this platform, not a policy rule: "
                            "support-policy.pdf defines no per-agent refund ceiling."
                        ),
                    )
                )
            elif amount > settings.REFUND_AUTO_APPROVE_MAX_INR:
                checks.append(
                    PolicyCheck(
                        rule_id=Rule.AUTO_APPROVE_CEILING,
                        outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                        detail=(
                            f"₹{amount:,.2f} exceeds the ₹{settings.REFUND_AUTO_APPROVE_MAX_INR:,.0f} "
                            "automatic-approval ceiling. Operational control, not a policy rule."
                        ),
                    )
                )

        if (
            request.action == ActionType.REFUND
            and amount is not None
            and context.order_value_inr is not None
            and amount > context.order_value_inr + 0.01
        ):
            checks.append(
                PolicyCheck(
                    rule_id=Rule.REFUND_EXCEEDS_ORDER,
                    outcome=PolicyOutcome.DENY,
                    detail=(
                        f"Denied: refund of ₹{amount:,.2f} exceeds the order value of "
                        f"₹{context.order_value_inr:,.2f}."
                    ),
                    breached=True,
                )
            )

    # ---- 4. Laya: can raise scrutiny, never lower it ---------------------
    hierarchy.append("4. Laya routing decision")
    risk = _risk_from_amount(amount, request.action)
    laya_overruled, overrule_detail = False, None

    if laya is not None:
        laya_risk = RiskLevel(laya.decision.risk_level)
        if laya_risk.rank > risk.rank:
            risk = laya_risk  # Laya escalating is honoured
        if laya.decision.requires_human:
            checks.append(
                PolicyCheck(
                    rule_id="LAYA-001-requires-human",
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    detail=f"Laya flagged this action as requiring human approval (risk={laya_risk}).",
                )
            )
        # Laya being unavailable on a high-risk action is itself a failure to close.
        if laya.degraded and is_high_risk and settings.LAYA_FAIL_CLOSED:
            checks.append(
                PolicyCheck(
                    rule_id=Rule.LAYA_AVAILABLE,
                    outcome=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                    detail=(
                        "Laya was unavailable and this is a high-risk action "
                        f"({laya.degraded_reason}). Failing closed to human review."
                    ),
                    breached=True,
                )
            )
            fail_closed = True

        # Record the overrule explicitly when Laya was more permissive.
        engine_outcome = _worst(checks)
        if not laya.decision.requires_human and engine_outcome != PolicyOutcome.ALLOW:
            laya_overruled = True
            blocking = [c.rule_id for c in checks if c.outcome != PolicyOutcome.ALLOW]
            overrule_detail = (
                f"Laya returned requires_human=False and risk={laya_risk}, but the policy "
                f"engine returned {engine_outcome} on {', '.join(blocking)}. "
                "POLICY ENGINE WINS."
            )
            hierarchy.append(f"   -> Laya OVERRULED: {overrule_detail}")
            log.warning(
                "policy.laya_overruled",
                extra={
                    "action": str(request.action),
                    "engine_outcome": str(engine_outcome),
                    "laya_risk": str(laya_risk),
                    "blocking_rules": blocking,
                },
            )

    hierarchy.append("5. Nemotron reasoning (advisory only; never authorises)")

    # ---- 6. final --------------------------------------------------------
    outcome = _worst(checks)

    # An approver being named cannot turn a DENY into an ALLOW.
    approver_required: str | None = None
    if outcome == PolicyOutcome.REQUIRES_HUMAN_REVIEW:
        if any(c.rule_id == Rule.CRITICAL_AMOUNT for c in checks):
            approver_required = "Manager"
        elif is_goodwill or any(c.rule_id == Rule.GOODWILL_CAP for c in checks):
            approver_required = policy.goodwill_approver
        else:
            approver_required = "Team Lead"

        if request.approved_by:
            outcome = PolicyOutcome.ALLOW
            checks.append(
                PolicyCheck(
                    rule_id="CTL-005-human-approval-recorded",
                    outcome=PolicyOutcome.ALLOW,
                    detail=(
                        f"Human approval recorded from {request.approved_by}; "
                        f"{approver_required} authorisation satisfied."
                    ),
                )
            )

    hierarchy.append(f"6. final decision: {outcome}")

    deny_reasons = [c.detail for c in checks if c.outcome == PolicyOutcome.DENY]
    review_reasons = [c.detail for c in checks if c.outcome == PolicyOutcome.REQUIRES_HUMAN_REVIEW]
    if outcome == PolicyOutcome.DENY:
        reason_text = " ".join(deny_reasons)[:1500]
    elif outcome == PolicyOutcome.REQUIRES_HUMAN_REVIEW:
        reason_text = " ".join(review_reasons)[:1500]
    else:
        reason_text = "All applicable policy checks passed."

    return PolicyEvaluation(
        final_decision=outcome,
        checks=checks,
        risk_level=risk,
        requires_human=outcome == PolicyOutcome.REQUIRES_HUMAN_REVIEW,
        approver_required=approver_required,
        reason=reason_text,
        policy_version=policy.version,
        policy_section_ids=sorted(sections, key=lambda s: int(s)),
        decision_hierarchy=hierarchy,
        laya_overruled=laya_overruled,
        laya_overrule_detail=overrule_detail,
        fail_closed=fail_closed,
    )


# ---------------------------------------------------------------------------
# Deterministic answers that need no model at all
# ---------------------------------------------------------------------------
def deterministic_answer(intent: str, question: str) -> tuple[str, str] | None:
    """Answer straight from the policy where a rule fully determines it.

    Returns (answer, policy_section_id) or None. This is route B in the inbound
    flow and is why a large share of policy questions never reach the LLM.
    """
    try:
        policy = get_policy()
    except PolicyUnavailable:
        return None

    q = (question or "").lower()

    if any(k in q for k in ("warranty period", "warranty last", "how long is the warranty", "warranty cover")):
        return (
            "Warranty length is set per product. Every Vireo audio product (earbuds, "
            "headphones, speakers and watches) carries a 12-month warranty; accessories "
            "such as chargers, cables and spare charging cases carry 6 months. An "
            "in-warranty fault is resolved by repair or replacement.",
            "5",
        )

    if any(k in q for k in ("first response", "how quickly", "response time", "how soon will you reply")):
        targets = policy.sla_targets_minutes
        return (
            "First-response targets are measured from when the ticket is created: "
            f"chat {targets['chat']} minutes, voice callback {targets['voice'] // 60} hours, "
            f"social {targets['social'] // 60} hours, email {targets['email'] // 60} hours. "
            f"If we miss the target, a ₹{policy.sla_breach_credit_inr:,.0f} store credit is "
            "issued automatically when the ticket is resolved.",
            "3",
        )

    if any(k in q for k in ("dead on arrival", "doa", "arrived dead", "faulty on arrival")):
        ent = policy.entitlement("DOA")
        if ent:
            return (
                f"If a product is dead on arrival within {ent['window_days']} days of delivery, "
                "you choose either a full refund or a replacement. Those are alternatives: "
                "an order cannot receive both.",
                "5",
            )

    if any(k in q for k in ("refund and replacement", "both refund and", "refund plus replacement")):
        return (
            "An order receives either a refund or a replacement, never both. Where both "
            "have been issued in error it is escalated to the Team Lead and Finance the "
            "same day.",
            "5",
        )

    if any(k in q for k in ("lost in transit", "damaged in transit", "never arrived", "lost package")):
        return (
            "If an order is lost or damaged in transit, we either reship it or refund it.",
            "5",
        )

    if any(k in q for k in ("support hours", "are you open", "24x7", "when can i call", "chat hours")):
        return (
            "Chat is available 24x7, with overnight cover from the Indore night shift. "
            "Email is accepted 24x7 and worked in queue order. Voice callbacks are taken "
            "between 08:00 and 22:00 IST. Social channels are worked by the chat team.",
            "2",
        )

    if any(k in q for k in ("csat", "survey", "rating request", "feedback survey")):
        return (
            "A one-question 1-5 satisfaction survey is sent when a ticket is resolved. "
            "Responding is optional.",
            "8",
        )

    return None
