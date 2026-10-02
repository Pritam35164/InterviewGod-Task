"""Deterministic routing used when Laya is unreachable.

This is the fail-closed path (brief §42/§43), and it is deliberately blunt:

  * Anything that looks like money or an action gets `requires_human=True` and
    risk >= high. If the fast decision layer is down we do not automate money.
  * `cacheable` is False for everything except an obviously generic policy
    question, because a wrong cache write outlives the outage.
  * Every decision produced here is marked `degraded=True`, so the audit trail
    distinguishes "Laya said this" from "Laya was down and rules said this".

It exists so a Laya outage degrades the system instead of stopping it, not as a
second-rate classifier to fall back on silently.
"""

from __future__ import annotations

import re

from app.schemas.laya import LayaDecisionEnvelope, LayaGuardDecision, LayaRoutingDecision
from app.schemas.taxonomy import Complexity, Direction, Intent, RiskLevel

_INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (Intent.REFUND_STATUS, re.compile(r"\b(where is my refund|refund not (credited|received)|refund (status|pending|delay)|still waiting for my refund|money for the return)\b", re.I)),
    (Intent.CANCELLATION, re.compile(r"\b(cancel|stop the shipment|don'?t ship|do not ship|change of mind|changed my mind)\b", re.I)),
    (Intent.REFUND, re.compile(r"\b(refund|money back|reverse the payment|return the money|charged twice|double charge|duplicate payment)\b", re.I)),
    (Intent.REPLACEMENT, re.compile(r"\b(replacement|replace|new unit|exchange)\b", re.I)),
    (Intent.WARRANTY, re.compile(r"\b(warranty|rma|service cent(re|er)|repair)\b", re.I)),
    (Intent.DELIVERY, re.compile(r"\b(not delivered|delivery|shipment|courier|awb|tracking|pickup|pkp)\b", re.I)),
    (Intent.BILLING, re.compile(r"\b(invoice|gst|payment|deducted|coupon|promo|discount|price)\b", re.I)),
    (Intent.ACCOUNT, re.compile(r"\b(log ?in|login|otp|password|account|address (change|update))\b", re.I)),
    (Intent.POLICY_QUESTION, re.compile(r"\b(what is (the|your)|how long|policy|how many days|am i eligible|do you (offer|allow))\b", re.I)),
    (Intent.PRODUCT_INFO, re.compile(r"\b(compatib|does it (work|support)|specification|pre-?sales|which model)\b", re.I)),
]

_MONEY_INTENTS = {Intent.REFUND, Intent.BILLING, Intent.CANCELLATION, Intent.REPLACEMENT}

# A generic policy question has no first-person account reference in it.
_PERSONAL_MARKER = re.compile(
    r"\b(my|mine|i (have|had|was|am|ordered|bought|paid)|order [a-z]{0,2}\d|vr\d{5,}|tk-\d+|c1\d{5})\b",
    re.I,
)


def classify_intent(message: str) -> Intent:
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message or ""):
            return intent
    return Intent.OTHER


def fallback_decision(
    message: str,
    reason: str,
    *,
    injection_flags: list[str] | None = None,
    has_customer_context: bool = False,
) -> LayaDecisionEnvelope:
    intent = classify_intent(message)
    personal = has_customer_context or bool(_PERSONAL_MARKER.search(message or ""))
    flagged = bool(injection_flags)

    money = intent in _MONEY_INTENTS
    generic_policy = intent in {Intent.POLICY_QUESTION, Intent.PRODUCT_INFO} and not personal

    if flagged:
        risk = RiskLevel.HIGH
    elif money:
        risk = RiskLevel.HIGH  # fail closed: no automated money while Laya is down
    elif personal:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.LOW

    decision = LayaRoutingDecision(
        direction=Direction.INBOUND,
        intent=intent,
        complexity=Complexity.HIGH if money else (Complexity.LOW if generic_policy else Complexity.MEDIUM),
        requires_llm=not generic_policy,
        requires_policy=money or intent in {Intent.WARRANTY, Intent.POLICY_QUESTION, Intent.REFUND_STATUS},
        requires_human=money or flagged,
        cacheable=generic_policy and not flagged,
        risk_level=risk,
    )
    return LayaDecisionEnvelope(
        decision=decision,
        confidence={},
        guard=LayaGuardDecision(prompt_injection=flagged, jailbreak=flagged),
        source="fallback",
        model_name="deterministic-fallback-rules",
        degraded=True,
        degraded_reason=reason,
        calibrated=False,
    )
