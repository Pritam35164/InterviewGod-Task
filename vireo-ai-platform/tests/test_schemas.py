"""Pydantic schemas, evidence grounding, taxonomy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm.client import extract_json
from app.schemas.llm import EvidenceSpan, RefundClassification
from app.schemas.taxonomy import RefundReason


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        RefundClassification(
            reason_code=RefundReason.CANCELLED_BEFORE_DISPATCH,
            reason_explanation="Customer cancelled before dispatch of the order.",
            policy_relevant=True,
            potentially_avoidable="POLICY_COMPLIANT",
            evidence=[EvidenceSpan(source="agent_notes", quote="cancelled before dispatch")],
            confidence=1.2,
        )


def test_evidence_required_except_unknown():
    with pytest.raises(ValidationError):
        RefundClassification(
            reason_code=RefundReason.CANCELLED_BEFORE_DISPATCH,
            reason_explanation="Customer cancelled before dispatch of the order.",
            policy_relevant=True,
            potentially_avoidable="POLICY_COMPLIANT",
            evidence=[],
            confidence=0.9,
        )
    ok = RefundClassification(
        reason_code=RefundReason.UNKNOWN,
        reason_explanation="Closing note is empty of substance.",
        policy_relevant=False,
        potentially_avoidable="AMBIGUOUS",
        evidence=[],
        confidence=0.2,
    )
    assert ok.reason_code == RefundReason.UNKNOWN


def test_evidence_must_be_verbatim():
    span = EvidenceSpan(source="agent_notes", quote="cancelled before dispatch")
    assert span.verify_against("hi", "cancelled before dispatch ~Sneha") is True
    assert span.verify_against("hi", "we processed a cancellation") is False


def test_extract_json_from_prose_and_fences():
    obj, err = extract_json('```json\n{"a": 1}\n```')
    assert obj == {"a": 1} and err is None
    obj, err = extract_json('sure, here: {"a": 2} thanks')
    assert obj == {"a": 2}
    obj, err = extract_json("no json here")
    assert obj is None and err


def test_reason_code_must_be_taxonomy():
    with pytest.raises(ValidationError):
        RefundClassification(
            reason_code="LATE_DELIVERY_REFUND",
            reason_explanation="Invented code that is not in the taxonomy at all.",
            policy_relevant=True,
            potentially_avoidable="POLICY_COMPLIANT",
            evidence=[EvidenceSpan(source="agent_notes", quote="late delivery")],
            confidence=0.5,
        )
