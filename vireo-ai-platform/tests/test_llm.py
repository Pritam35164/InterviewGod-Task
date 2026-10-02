"""LLM structured output: parse, retry, fail closed. Never hits OpenRouter."""

from __future__ import annotations

import json

from app.core.config import settings
from app.llm.client import OpenRouterClient, extract_json
from app.llm.prompts import PromptPair
from app.schemas.llm import RefundClassification


VALID = {
    "reason_code": "CANCELLED_BEFORE_DISPATCH",
    "reason_explanation": "The note says the order was cancelled before dispatch.",
    "policy_relevant": True,
    "policy_section_ids": ["5"],
    "potentially_avoidable": "POLICY_COMPLIANT",
    "avoidability_rationale": "Cancellation before dispatch is an entitlement under section 5.",
    "dual_remedy_indicated": False,
    "refund_without_return_indicated": False,
    "replacement_mentioned": False,
    "evidence": [
        {"source": "agent_notes", "quote": "cancelled before dispatch", "supports": "cancel"}
    ],
    "confidence": 0.91,
    "ambiguity_note": "",
}


class ScriptedClient(OpenRouterClient):
    def __init__(self, payloads: list[dict]):
        super().__init__(api_key="sk-test-not-real", model="stub-model")
        self._payloads = list(payloads)
        self.calls = 0

    def _post_completion(self, messages, max_tokens, temperature, json_mode):
        body = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return body, 12.0


def test_retry_on_invalid_json_then_success(monkeypatch):
    monkeypatch.setattr(settings, "MAX_LLM_RETRIES", 2)
    client = ScriptedClient(
        [
            {"choices": [{"message": {"content": "nope"}}], "usage": {"prompt_tokens": 8, "completion_tokens": 2}},
            {
                "choices": [{"message": {"content": json.dumps(VALID)}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 40},
            },
        ]
    )
    prompt = PromptPair(system="sys", user="cancelled before dispatch", version="v6")
    result = client.generate_structured(prompt, RefundClassification)
    assert result.ok
    assert result.retries == 1
    assert result.validation_status == "repaired"
    assert result.value.reason_code == "CANCELLED_BEFORE_DISPATCH"


def test_invalid_after_retries_does_not_invent_an_object(monkeypatch):
    monkeypatch.setattr(settings, "MAX_LLM_RETRIES", 1)
    client = ScriptedClient(
        [
            {"choices": [{"message": {"content": "{"}}], "usage": {}},
            {"choices": [{"message": {"content": "still bad"}}], "usage": {}},
        ]
    )
    prompt = PromptPair(system="sys", user="x", version="v6")
    result = client.generate_structured(prompt, RefundClassification)
    assert result.ok is False
    assert result.value is None
    assert result.validation_status == "invalid_after_retries"


def test_unavailable_without_key():
    client = OpenRouterClient(api_key="")
    assert client.available is False
    prompt = PromptPair(system="s", user="u", version="v6")
    result = client.generate_structured(prompt, RefundClassification)
    assert result.validation_status == "unavailable"
    assert result.value is None


def test_extract_json_strips_think_blocks():
    obj, err = extract_json('<think>plan</think>\n{"a": true}')
    assert obj == {"a": True} and err is None
