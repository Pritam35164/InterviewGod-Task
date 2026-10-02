"""
OpenRouter / NVIDIA Nemotron Client for Vireo Audio Refund Analysis.
Reads OPENROUTER_API_KEY and OPENROUTER_MODEL from environment variables.
Handles retries, JSON extraction, thinking tag removal, and safe fallbacks.
"""

import os
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import requests

_FENCE_REGEX = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_THINK_REGEX = re.compile(r"<think>.*?</think>", re.S | re.I)

@dataclass
class StructuredResult:
    value: Optional[Dict[str, Any]]
    raw_text: str = ""
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    retries: int = 0
    validation_status: str = "valid"  # valid | repaired | invalid_after_retries | unavailable
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_inr: float = 0.0

    @property
    def ok(self) -> bool:
        return self.value is not None

def extract_json_payload(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "Empty response from LLM"

    cleaned = _THINK_REGEX.sub("", text).strip()
    fenced = _FENCE_REGEX.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, f"Expected JSON object, got {type(obj).__name__}"
    except json.JSONDecodeError:
        pass

    # Brace extraction fallback
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end+1])
            if isinstance(obj, dict):
                return obj, None
        except Exception:
            pass

    return None, "Failed to parse valid JSON from LLM output"

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-4-340b-instruct")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_openrouter_api_key_here")

    def generate_structured(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> StructuredResult:
        if not self.is_available():
            return StructuredResult(
                value=None,
                errors=["OPENROUTER_API_KEY not configured."],
                validation_status="unavailable"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vireo-audio.local",
            "X-Title": "Vireo Audio Refund Analyzer"
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        retries = 0
        accumulated_errors = []

        for attempt in range(1, max_retries + 2):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
                if resp.status_code != 200:
                    err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    accumulated_errors.append(err_msg)
                    if resp.status_code in [429, 500, 502, 503, 504] and attempt <= max_retries:
                        time.sleep(1.0 * attempt)
                        retries += 1
                        continue
                    else:
                        break

                data = resp.json()
                choice = data.get("choices", [{}])[0]
                raw_text = choice.get("message", {}).get("content", "")

                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                # Cost estimate: ~$0.50/M prompt, $1.50/M completion => ~₹42/M prompt, ~₹126/M completion
                cost_inr = (p_tokens * 0.000042) + (c_tokens * 0.000126)

                parsed_json, parse_err = extract_json_payload(raw_text)
                if parsed_json is not None:
                    status = "valid" if attempt == 1 else "repaired"
                    return StructuredResult(
                        value=parsed_json,
                        raw_text=raw_text,
                        attempts=attempt,
                        retries=retries,
                        validation_status=status,
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        estimated_cost_inr=cost_inr
                    )

                accumulated_errors.append(parse_err or "Invalid JSON structure")
                if attempt <= max_retries:
                    # Append repair prompt message
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({"role": "user", "content": f"Your response was not valid JSON. Error: {parse_err}. Please output ONLY valid JSON format."})
                    retries += 1
                    time.sleep(0.5)

            except Exception as e:
                accumulated_errors.append(str(e))
                if attempt <= max_retries:
                    time.sleep(1.0 * attempt)
                    retries += 1

        return StructuredResult(
            value=None,
            errors=accumulated_errors,
            attempts=attempt,
            retries=retries,
            validation_status="invalid_after_retries"
        )
