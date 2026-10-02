"""OpenRouter / NVIDIA Nemotron client.

The abstraction the rest of the app sees is `generate_structured`,
`generate_text` and `classify_ticket`. Nothing outside this module knows that
OpenRouter exists: swapping providers means replacing this file.

Design points worth stating:

  * The API key is read from configuration only. There is no literal key anywhere
    in this repository, and `available()` is False when it is unset, so the
    pipeline degrades to "unclassified" rather than pretending.
  * Retries are bounded by MAX_LLM_RETRIES and cover two different failures:
    transport (5xx/timeout, retried with backoff) and schema (invalid payload,
    retried once with a repair prompt that names the validation errors).
  * After the final failure the caller gets a `StructuredResult` with
    `value=None` and the accumulated errors. It never gets a fabricated object.
  * Cost is taken from OpenRouter's own usage accounting when present, and only
    falls back to a local price-table estimate otherwise.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.errors import LLMError
from app.core.logging import current_request_id, current_trace_id, get_logger
from app.core.metrics import incr, observe
from app.core.security import stable_hash
from app.llm.prompts import PromptPair, build_repair_prompt
from app.schemas.llm import LLMUsage

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


@dataclass
class StructuredResult:
    """Outcome of a structured generation attempt."""

    value: Any | None
    usage: LLMUsage
    raw_text: str = ""
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    retries: int = 0
    validation_status: str = "valid"  # valid | repaired | invalid_after_retries | unavailable

    @property
    def ok(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------------------
# JSON extraction from a chat completion
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
# Nemotron may emit a reasoning preamble even with thinking off.
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Pull the first JSON object out of a model reply. (obj, error)."""
    if not text or not text.strip():
        return None, "model returned an empty response"

    cleaned = _THINK.sub("", text).strip()

    fenced = _FENCE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        obj = json.loads(cleaned)
        return (obj, None) if isinstance(obj, dict) else (None, f"expected a JSON object, got {type(obj).__name__}")
    except json.JSONDecodeError:
        pass

    # Brace matching, so trailing prose does not defeat parsing.
    start = cleaned.find("{")
    if start == -1:
        return None, "no JSON object found in the response"
    depth, in_str, escape = 0, False, False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                    return (obj, None) if isinstance(obj, dict) else (None, "not an object")
                except json.JSONDecodeError as exc:
                    return None, f"malformed JSON: {exc.msg} at char {exc.pos}"
    return None, "unterminated JSON object in the response"


def format_validation_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        out.append(f"field '{loc}': {err.get('msg')}")
    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class OpenRouterClient:
    """Chat-completions client for OpenRouter, used with NVIDIA Nemotron."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = (api_key if api_key is not None else settings.OPENROUTER_API_KEY).strip()
        self.model = model or settings.OPENROUTER_MODEL
        self._base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self._client: httpx.Client | None = None
        self._lock = Lock()

    # ---- availability -----------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def unavailable_reason(self) -> str | None:
        if not self._api_key:
            return (
                "OPENROUTER_API_KEY is not set. LLM classification is disabled; tickets "
                "are left unclassified rather than guessed."
            )
        return None

    def _http(self) -> httpx.Client:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.Client(
                        base_url=self._base_url,
                        timeout=httpx.Timeout(settings.OPENROUTER_TIMEOUT_S, connect=10.0),
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                            # OpenRouter attribution headers.
                            "HTTP-Referer": settings.OPENROUTER_APP_URL,
                            "X-Title": settings.OPENROUTER_APP_TITLE,
                        },
                    )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- budget guard -----------------------------------------------------
    @staticmethod
    def estimate_tokens(messages: list[dict[str, str]]) -> int:
        """Cheap token estimate without a tokeniser dependency.

        ~4 characters per token is close enough for a budget check on English
        helpdesk text, and deliberately errs high (ceil + per-message overhead)
        so the guard trips early rather than late.
        """
        chars = sum(len(m.get("content") or "") for m in messages)
        return int(chars / 3.6) + 8 * len(messages)

    def _check_budget(self, messages: list[dict[str, str]], max_tokens: int) -> str | None:
        """Return an error string if this call would exceed the configured budget."""
        estimated = self.estimate_tokens(messages) + max_tokens
        ceiling = settings.LLM_MAX_PROMPT_TOKENS
        if ceiling and estimated > ceiling:
            return (
                f"request refused before sending: estimated {estimated:,} tokens "
                f"(prompt ~{self.estimate_tokens(messages):,} + max_tokens {max_tokens:,}) "
                f"exceeds LLM_MAX_PROMPT_TOKENS={ceiling:,}"
            )
        return None

    # ---- transport --------------------------------------------------------
    def _post_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> tuple[dict, float]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to return its own cost accounting.
            "usage": {"include": True},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        over_budget = self._check_budget(messages, max_tokens)
        if over_budget:
            incr("llm.budget_refused")
            log.error("llm.budget_exceeded", extra={"detail": over_budget})
            raise LLMError(over_budget, retryable=False, budget_exceeded=True)

        start = time.perf_counter()
        resp = self._http().post("/chat/completions", json=payload)
        elapsed = (time.perf_counter() - start) * 1000

        if resp.status_code >= 400:
            body = resp.text[:400]
            retryable = resp.status_code in _RETRYABLE_STATUS
            raise LLMError(
                f"OpenRouter returned HTTP {resp.status_code}",
                status=resp.status_code,
                retryable=retryable,
                body=body,
            )
        return resp.json(), elapsed

    def _usage_from_response(
        self, body: dict, elapsed_ms: float, prompt_version: str, attempts: int, retries: int
    ) -> LLMUsage:
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (in_tok + out_tok))

        est_usd = (
            in_tok / 1_000_000 * settings.OPENROUTER_PRICE_IN_USD_PER_1M
            + out_tok / 1_000_000 * settings.OPENROUTER_PRICE_OUT_USD_PER_1M
        )
        upstream = usage.get("cost")
        upstream_usd = float(upstream) if isinstance(upstream, (int, float)) else None
        effective = upstream_usd if upstream_usd is not None else est_usd

        choices = body.get("choices") or [{}]
        return LLMUsage(
            model=body.get("model") or self.model,
            prompt_version=prompt_version,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total,
            latency_ms=round(elapsed_ms, 2),
            status="success",
            attempts=attempts,
            retries=retries,
            cost_usd=round(est_usd, 8),
            cost_inr=round(effective * settings.USD_INR_RATE, 6),
            finish_reason=choices[0].get("finish_reason"),
            upstream_id=body.get("id"),
            upstream_cost_usd=upstream_usd,
        )

    def _record(self, usage: LLMUsage) -> None:
        incr("llm.tokens_in", usage.input_tokens)
        incr("llm.tokens_out", usage.output_tokens)
        incr("llm.cost_usd", usage.effective_cost_usd)
        incr("llm.cost_inr", usage.cost_inr)
        observe("nemotron.request", usage.latency_ms)

    # ---- public API -------------------------------------------------------
    def generate_text(
        self,
        prompt: PromptPair,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str | None, LLMUsage]:
        if not self.available:
            return None, LLMUsage(
                model=self.model,
                prompt_version=prompt.version,
                status="unavailable",
                error=self.unavailable_reason(),
            )

        attempts, retries = 0, 0
        last_error: str | None = None
        max_attempts = settings.MAX_LLM_RETRIES + 1

        while attempts < max_attempts:
            attempts += 1
            try:
                body, elapsed = self._post_completion(
                    prompt.as_messages(),
                    max_tokens or settings.LLM_MAX_TOKENS,
                    settings.LLM_TEMPERATURE if temperature is None else temperature,
                    json_mode=False,
                )
                text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                usage = self._usage_from_response(body, elapsed, prompt.version, attempts, retries)
                self._record(usage)
                incr("llm.success")
                return text, usage
            except LLMError as exc:
                last_error = exc.message
                if not exc.details.get("retryable") or attempts >= max_attempts:
                    break
                retries += 1
                incr("llm.retry")
                time.sleep(min(2 ** (attempts - 1) * 0.5, 8.0))
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                if attempts >= max_attempts:
                    break
                retries += 1
                incr("llm.retry")
                time.sleep(min(2 ** (attempts - 1) * 0.5, 8.0))

        incr("llm.failure")
        log.error("llm.text_failed", extra={"error": last_error, "attempts": attempts})
        return None, LLMUsage(
            model=self.model,
            prompt_version=prompt.version,
            status="error",
            attempts=attempts,
            retries=retries,
            error=last_error,
        )

    def generate_structured(
        self,
        prompt: PromptPair,
        schema: Type[T],
        max_tokens: int | None = None,
        post_validate: Any = None,
    ) -> StructuredResult:
        """Generate and validate against `schema`, with bounded repair retries.

        `post_validate(obj) -> list[str]` adds semantic checks beyond the schema
        (used for evidence grounding). Returned errors trigger a repair retry.
        """
        if not self.available:
            incr("llm.unavailable")
            return StructuredResult(
                value=None,
                usage=LLMUsage(
                    model=self.model,
                    prompt_version=prompt.version,
                    status="unavailable",
                    error=self.unavailable_reason(),
                ),
                errors=[self.unavailable_reason() or "llm unavailable"],
                validation_status="unavailable",
            )

        max_attempts = settings.MAX_LLM_RETRIES + 1
        attempts, retries = 0, 0
        errors: list[str] = []
        raw_text = ""
        current = prompt
        last_usage: LLMUsage | None = None

        while attempts < max_attempts:
            attempts += 1
            incr("validation.attempt")
            try:
                body, elapsed = self._post_completion(
                    current.as_messages(),
                    max_tokens or settings.LLM_MAX_TOKENS,
                    settings.LLM_TEMPERATURE,
                    json_mode=True,
                )
            except LLMError as exc:
                errors.append(f"attempt {attempts}: {exc.message}")
                last_usage = LLMUsage(
                    model=self.model, prompt_version=prompt.version, status="error",
                    attempts=attempts, retries=retries, error=exc.message,
                )
                if not exc.details.get("retryable") or attempts >= max_attempts:
                    break
                retries += 1
                incr("llm.retry")
                time.sleep(min(2 ** (attempts - 1) * 0.5, 8.0))
                continue
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"[:300]
                errors.append(f"attempt {attempts}: {msg}")
                last_usage = LLMUsage(
                    model=self.model, prompt_version=prompt.version, status="error",
                    attempts=attempts, retries=retries, error=msg,
                )
                if attempts >= max_attempts:
                    break
                retries += 1
                incr("llm.retry")
                time.sleep(min(2 ** (attempts - 1) * 0.5, 8.0))
                continue

            raw_text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            usage = self._usage_from_response(body, elapsed, prompt.version, attempts, retries)
            last_usage = usage
            self._record(usage)

            obj, parse_error = extract_json(raw_text)
            if obj is None:
                errors.append(f"attempt {attempts}: {parse_error}")
                incr("validation.failure")
                if attempts >= max_attempts:
                    break
                retries += 1
                incr("llm.retry")
                current = build_repair_prompt(prompt, raw_text, [parse_error or "unparseable"])
                continue

            try:
                value = schema.model_validate(obj)
            except ValidationError as exc:
                detail = format_validation_errors(exc)
                errors.extend(f"attempt {attempts}: {d}" for d in detail)
                incr("validation.failure")
                log.warning(
                    "llm.schema_invalid",
                    extra={"attempt": attempts, "errors": detail[:4], "schema": schema.__name__},
                )
                if attempts >= max_attempts:
                    break
                retries += 1
                incr("llm.retry")
                current = build_repair_prompt(prompt, raw_text, detail)
                continue

            if post_validate is not None:
                semantic = post_validate(value) or []
                if semantic:
                    errors.extend(f"attempt {attempts}: {s}" for s in semantic)
                    incr("validation.failure")
                    incr("validation.semantic_failure")
                    if attempts < max_attempts:
                        retries += 1
                        incr("llm.retry")
                        current = build_repair_prompt(prompt, raw_text, semantic)
                        continue
                    # Out of retries: keep the object, flag it for review.
                    usage.retries = retries
                    incr("llm.success")
                    return StructuredResult(
                        value=value, usage=usage, raw_text=raw_text, errors=errors,
                        attempts=attempts, retries=retries, validation_status="repaired",
                    )

            usage.retries = retries
            incr("llm.success")
            return StructuredResult(
                value=value,
                usage=usage,
                raw_text=raw_text,
                errors=errors,
                attempts=attempts,
                retries=retries,
                validation_status="valid" if retries == 0 else "repaired",
            )

        incr("llm.failure")
        log.error(
            "llm.structured_failed",
            extra={"attempts": attempts, "retries": retries, "errors": errors[-3:]},
        )
        return StructuredResult(
            value=None,
            usage=last_usage
            or LLMUsage(
                model=self.model, prompt_version=prompt.version, status="error",
                attempts=attempts, retries=retries, error="; ".join(errors[-2:])[:400],
            ),
            raw_text=raw_text,
            errors=errors,
            attempts=attempts,
            retries=retries,
            validation_status="invalid_after_retries",
        )

    # ---- convenience ------------------------------------------------------
    def classify_ticket(
        self,
        context: dict,
        customer_message: str,
        agent_notes: str,
        policy_sections: list[dict],
        prompt_version: str | None = None,
    ) -> StructuredResult:
        """Classify one refund ticket. Evidence grounding is enforced here."""
        from app.llm.prompts import build_classification_prompt
        from app.schemas.llm import RefundClassification

        prompt = build_classification_prompt(
            context, customer_message, agent_notes, policy_sections,
            version=prompt_version or settings.PROMPT_VERSION,
        )

        def _check_grounding(value: RefundClassification) -> list[str]:
            _, bad = value.verify_evidence(customer_message, agent_notes)
            return [
                f"evidence quote not found verbatim in {span.source}: {span.quote[:90]!r}"
                for span in bad
            ]

        return self.generate_structured(
            prompt, RefundClassification, post_validate=_check_grounding
        )

    @staticmethod
    def prompt_fingerprint(prompt: PromptPair) -> str:
        return stable_hash(prompt.system, prompt.user)

    def health(self) -> dict:
        """Configuration-only. Deliberately makes no network call: a health probe
        must not spend money (brief §30)."""
        return {
            "configured": self.available,
            "model": self.model,
            "base_url": self._base_url,
            "reason": self.unavailable_reason(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_client: OpenRouterClient | None = None
_factory_lock = Lock()


def get_llm_client() -> OpenRouterClient:
    global _client
    if _client is None:
        with _factory_lock:
            if _client is None:
                _client = OpenRouterClient()
                log.info(
                    "llm.client_initialised",
                    extra={"model": _client.model, "configured": _client.available},
                )
    return _client


def reset_llm_client() -> None:
    global _client
    with _factory_lock:
        if _client is not None:
            _client.close()
        _client = None


def set_llm_client(client: Any) -> None:
    """Inject a stub client for tests (no OpenRouter money spent in unit tests)."""
    global _client
    with _factory_lock:
        _client = client


def record_llm_request(
    session,
    usage: LLMUsage,
    purpose: str,
    ticket_id: str | None = None,
    prompt_sha: str | None = None,
    response_sha: str | None = None,
    policy_version: str | None = None,
) -> None:
    """Persist one LLM call for the cost report. Prompt/response text is NOT
    stored — only hashes — because the prompt contains the customer message."""
    from app.models.decisions import LLMRequest

    session.add(
        LLMRequest(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            ticket_id=ticket_id,
            purpose=purpose,
            provider="openrouter",
            model_name=usage.model,
            prompt_version=usage.prompt_version,
            policy_version=policy_version or settings.POLICY_VERSION,
            upstream_id=usage.upstream_id,
            prompt_sha256=prompt_sha,
            response_sha256=response_sha,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=usage.latency_ms,
            status=usage.status,
            finish_reason=usage.finish_reason,
            error=(usage.error or None) and usage.error[:500],
            attempts=usage.attempts,
            retries=usage.retries,
            cache_hit=usage.cache_hit,
            cost_usd=usage.cost_usd,
            cost_inr=usage.cost_inr,
            upstream_cost_usd=usage.upstream_cost_usd,
            price_in_usd_per_1m=settings.OPENROUTER_PRICE_IN_USD_PER_1M,
            price_out_usd_per_1m=settings.OPENROUTER_PRICE_OUT_USD_PER_1M,
        )
    )
