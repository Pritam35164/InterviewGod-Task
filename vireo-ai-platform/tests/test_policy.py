"""Policy engine: hard rules, fail-closed, Laya cannot override DENY."""

from __future__ import annotations

from app.policy.engine import ActionContext, evaluate_outbound_action
from app.policy.loader import get_policy
from app.schemas.api import OutboundActionRequest
from app.schemas.laya import LayaDecisionEnvelope, LayaRoutingDecision
from app.schemas.taxonomy import ActionType, PolicyOutcome, RiskLevel


def _laya(requires_human: bool = False, risk: str = "low") -> LayaDecisionEnvelope:
    return LayaDecisionEnvelope(
        decision=LayaRoutingDecision(
            direction="inbound",
            intent="refund",
            complexity="medium",
            requires_llm=True,
            requires_policy=True,
            requires_human=requires_human,
            cacheable=False,
            risk_level=risk,
        ),
        source="laya",
        calibrated=False,
    )


def test_policy_pdf_quotes_match():
    doc = get_policy()
    assert doc.version == "3.2"
    assert doc.goodwill_cap_inr == 500
    assert doc.dual_remedy_prohibited is True
    assert "GW-OTHER" in doc.reason_codes
    assert doc.verification["status"] in {"ok", "drift_detected"}
    assert doc.verification["checked"] > 0


def test_dual_remedy_is_denied_even_if_laya_says_safe():
    req = OutboundActionRequest(
        action=ActionType.REFUND,
        ticket_id="TK-1",
        amount_inr=1999,
        reason_code="DOA-REPL",
        order_id="VR1",
        replacement_issued=True,
    )
    result = evaluate_outbound_action(req, ActionContext(), _laya(requires_human=False, risk="low"))
    assert result.final_decision == PolicyOutcome.DENY
    assert result.laya_overruled is True
    assert any(c.rule_id.startswith("POL-5-001") for c in result.checks)


def test_goodwill_over_cap_denied():
    req = OutboundActionRequest(
        action=ActionType.GOODWILL_CREDIT,
        ticket_id="TK-1",
        amount_inr=2000,
        reason_code="GW-OTHER",
    )
    result = evaluate_outbound_action(req, ActionContext(), _laya())
    assert result.final_decision == PolicyOutcome.DENY


def test_missing_amount_fails_closed():
    req = OutboundActionRequest(action=ActionType.REFUND, ticket_id="TK-1", reason_code="CANCEL")
    result = evaluate_outbound_action(req, ActionContext(), _laya())
    assert result.final_decision == PolicyOutcome.REQUIRES_HUMAN_REVIEW
    assert result.fail_closed is True


def test_policy_unavailable_fails_closed(monkeypatch):
    from app.policy import engine as eng
    from app.core.errors import PolicyUnavailable

    def boom():
        raise PolicyUnavailable("gone")

    monkeypatch.setattr(eng, "get_policy", boom)
    req = OutboundActionRequest(
        action=ActionType.REFUND, ticket_id="TK-1", amount_inr=100, reason_code="CANCEL"
    )
    result = evaluate_outbound_action(req, ActionContext(policy_available=False), _laya())
    assert result.final_decision == PolicyOutcome.REQUIRES_HUMAN_REVIEW
    assert result.fail_closed is True


def test_approved_by_cannot_override_deny():
    req = OutboundActionRequest(
        action=ActionType.REFUND,
        ticket_id="TK-1",
        amount_inr=1999,
        reason_code="DOA-REPL",
        replacement_issued=True,
        approved_by="A9999",
    )
    result = evaluate_outbound_action(req, ActionContext(), _laya())
    assert result.final_decision == PolicyOutcome.DENY


def test_deterministic_warranty_answer():
    from app.policy.engine import deterministic_answer

    ans = deterministic_answer("policy_question", "How long is the warranty?")
    assert ans is not None
    assert "12-month" in ans[0] or "12" in ans[0]
