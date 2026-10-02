"""Laya client: HTTP to the laya service, or embedded in-process.

Laya's job here is System-1 only — classification, routing, gating, risk, cache
eligibility. It is never asked for prose, and `decide()` returns a typed
`LayaRoutingDecision` because `laya.decide()` is driven by the Pydantic schema
itself rather than by a hand-written prompt.

Laya is also not the authority. `LayaDecisionEnvelope` is an input to the policy
engine, and `app/services/decision_engine.py` overrules it whenever policy and
Laya disagree.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Protocol, Sequence

import httpx

from app.cache import redis_client
from app.cache.keys import laya_decision_key
from app.cache.response_cache import get_json, set_json
from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import incr, observe
from app.core.security import normalise_text, scan_injection
from app.laya.fallback import fallback_decision
from app.schemas.laya import (
    LayaDecisionEnvelope,
    LayaGuardDecision,
    LayaRoutingDecision,
)

log = get_logger(__name__)


class LayaClient(Protocol):
    def decide(
        self, message: str, *, include_guard: bool = True, has_customer_context: bool = False
    ) -> LayaDecisionEnvelope: ...

    def decide_batch(self, messages: Sequence[str]) -> list[LayaDecisionEnvelope]: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def health(self) -> dict: ...

    @property
    def mode(self) -> str: ...


def _envelope_from_payload(payload: dict, latency_ms: float) -> LayaDecisionEnvelope:
    return LayaDecisionEnvelope(
        decision=LayaRoutingDecision.model_validate(payload["decision"]),
        confidence=payload.get("confidence") or {},
        probabilities=payload.get("probabilities") or {},
        guard=(LayaGuardDecision.model_validate(payload["guard"]) if payload.get("guard") else None),
        low_confidence_fields=payload.get("low_confidence_fields") or [],
        source=payload.get("source", "laya"),
        model_name=payload.get("model_name", settings.LAYA_MODEL),
        latency_ms=payload.get("latency_ms", latency_ms),
        degraded=payload.get("degraded", False),
        degraded_reason=payload.get("degraded_reason"),
        calibrated=payload.get("calibrated", False),
    )


class HttpLayaClient:
    """Talks to the standalone laya service (the production-oriented default)."""

    mode = "http"

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if settings.LAYA_API_KEY:
            headers["X-API-Key"] = settings.LAYA_API_KEY
        self._client = httpx.Client(
            base_url=self._base, timeout=settings.LAYA_TIMEOUT_S, headers=headers
        )

    def decide(
        self, message: str, *, include_guard: bool = True, has_customer_context: bool = False
    ) -> LayaDecisionEnvelope:
        start = time.perf_counter()
        try:
            resp = self._client.post(
                "/decide", json={"message": message, "include_guard": include_guard}
            )
            resp.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            observe("laya.decide", elapsed)
            incr("laya.success")
            return _envelope_from_payload(resp.json(), elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            observe("laya.decide", elapsed)
            incr("laya.failure")
            log.error("laya.http_failed", extra={"error": str(exc)[:300], "url": self._base})
            env = fallback_decision(
                message,
                reason=f"laya service unreachable: {type(exc).__name__}",
                injection_flags=scan_injection(message),
                has_customer_context=has_customer_context,
            )
            env.latency_ms = elapsed
            return env

    def decide_batch(self, messages: Sequence[str]) -> list[LayaDecisionEnvelope]:
        start = time.perf_counter()
        try:
            resp = self._client.post(
                "/decide/batch", json={"messages": list(messages), "include_guard": False}
            )
            resp.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            observe("laya.decide_batch", elapsed)
            incr("laya.success", len(messages))
            return [
                _envelope_from_payload(item, elapsed / max(1, len(messages)))
                for item in resp.json()["decisions"]
            ]
        except Exception as exc:  # noqa: BLE001
            incr("laya.failure", len(messages))
            log.error("laya.http_batch_failed", extra={"error": str(exc)[:300]})
            return [
                fallback_decision(m, reason=f"laya batch unreachable: {type(exc).__name__}")
                for m in messages
            ]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._client.post("/embed", json={"texts": list(texts)})
        resp.raise_for_status()
        return resp.json()["vectors"]

    def health(self) -> dict:
        try:
            start = time.perf_counter()
            resp = self._client.get("/health", timeout=5.0)
            resp.raise_for_status()
            payload = resp.json()
            payload["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            payload["mode"] = "http"
            return payload
        except Exception as exc:  # noqa: BLE001
            return {"status": "down", "mode": "http", "error": str(exc)[:200], "model_loaded": False}


class EmbeddedLayaClient:
    """Runs Laya in-process. Used for local development and unit tests.

    The model is loaded lazily on first use so importing the app does not pull a
    250 MB checkpoint, and so a missing checkpoint degrades to the fallback rules
    rather than preventing startup.
    """

    mode = "embedded"

    def __init__(self) -> None:
        self._agent: Any = None
        self._embed_fn: Any = None
        self._lock = Lock()
        self._load_error: str | None = None
        self._load_seconds: float | None = None
        self._warnings: list[str] = []

    def _ensure_agent(self):
        if self._agent is not None or self._load_error is not None:
            return self._agent
        with self._lock:
            if self._agent is not None or self._load_error is not None:
                return self._agent
            try:
                import warnings as _warnings

                import laya

                start = time.perf_counter()
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter("always")
                    self._agent = laya.load(settings.LAYA_MODEL)
                    self._warnings = [
                        str(w.message)[:300] for w in caught if "laya" in str(w.message).lower()
                    ]
                self._load_seconds = time.perf_counter() - start
                log.info(
                    "laya.loaded",
                    extra={
                        "model": settings.LAYA_MODEL,
                        "load_seconds": round(self._load_seconds, 2),
                        "warnings": len(self._warnings),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"[:300]
                log.error("laya.load_failed", extra={"error": self._load_error})
        return self._agent

    def decide(
        self, message: str, *, include_guard: bool = True, has_customer_context: bool = False
    ) -> LayaDecisionEnvelope:
        agent = self._ensure_agent()
        if agent is None:
            incr("laya.failure")
            return fallback_decision(
                message,
                reason=f"laya model unavailable: {self._load_error}",
                injection_flags=scan_injection(message),
                has_customer_context=has_customer_context,
            )
        try:
            import laya

            start = time.perf_counter()
            result = laya.decide(
                agent,
                {"message": message},
                LayaRoutingDecision,
                return_details=True,
                min_confidence=None,
            )
            guard = None
            if include_guard:
                graw = laya.decide(
                    agent,
                    {"prompt": message},
                    questions=laya.guard_questions(),
                    return_details=True,
                )
                guard = LayaGuardDecision.model_validate(
                    {
                        "jailbreak": bool(graw.values.get("jailbreak", False)),
                        "prompt_injection": bool(graw.values.get("prompt_injection", False)),
                        "sensitive_data": bool(graw.values.get("sensitive_data", False)),
                        "harm_severity": int(graw.values.get("harm_severity", 0) or 0),
                        "topic": str(graw.values.get("topic", "other")),
                    }
                )
            elapsed = (time.perf_counter() - start) * 1000
            observe("laya.decide", elapsed)
            incr("laya.success")

            conf = {k: v for k, v in (result.answer_confidence or {}).items()}
            low = [
                k for k, v in conf.items()
                if v is not None and v < settings.LAYA_MIN_CONFIDENCE
            ]
            return LayaDecisionEnvelope(
                decision=LayaRoutingDecision.model_validate(result.values),
                confidence=conf,
                probabilities=_trim_probabilities(result.probabilities or {}),
                guard=guard,
                low_confidence_fields=low,
                source="laya",
                model_name=settings.LAYA_MODEL,
                latency_ms=elapsed,
                calibrated=False,  # checkpoint ships uncalibrated temperatures
            )
        except Exception as exc:  # noqa: BLE001
            incr("laya.failure")
            log.error("laya.decide_failed", extra={"error": str(exc)[:300]})
            return fallback_decision(
                message,
                reason=f"laya decide failed: {type(exc).__name__}",
                injection_flags=scan_injection(message),
                has_customer_context=has_customer_context,
            )

    def decide_batch(self, messages: Sequence[str]) -> list[LayaDecisionEnvelope]:
        agent = self._ensure_agent()
        if agent is None:
            return [
                fallback_decision(m, reason=f"laya model unavailable: {self._load_error}")
                for m in messages
            ]
        try:
            import laya

            start = time.perf_counter()
            results = laya.decide_batch(
                agent,
                [{"message": m} for m in messages],
                LayaRoutingDecision,
                return_details=True,
            )
            elapsed = (time.perf_counter() - start) * 1000
            observe("laya.decide_batch", elapsed)
            incr("laya.success", len(messages))
            per_item = elapsed / max(1, len(messages))
            out = []
            for res in results:
                conf = dict(res.answer_confidence or {})
                out.append(
                    LayaDecisionEnvelope(
                        decision=LayaRoutingDecision.model_validate(res.values),
                        confidence=conf,
                        probabilities=_trim_probabilities(res.probabilities or {}),
                        low_confidence_fields=[
                            k for k, v in conf.items()
                            if v is not None and v < settings.LAYA_MIN_CONFIDENCE
                        ],
                        source="laya",
                        model_name=settings.LAYA_MODEL,
                        latency_ms=per_item,
                        calibrated=False,
                    )
                )
            return out
        except Exception as exc:  # noqa: BLE001
            incr("laya.failure", len(messages))
            log.error("laya.decide_batch_failed", extra={"error": str(exc)[:300]})
            return [
                fallback_decision(m, reason=f"laya batch failed: {type(exc).__name__}")
                for m in messages
            ]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        agent = self._ensure_agent()
        if agent is None:
            raise RuntimeError(f"laya model unavailable: {self._load_error}")
        if self._embed_fn is None:
            import laya

            self._embed_fn = laya.cached_embed_fn(laya.embed_fn_from_agent(agent))
        vectors = self._embed_fn(list(texts))
        return [list(map(float, v)) for v in vectors]

    def health(self) -> dict:
        loaded = self._agent is not None
        return {
            "status": "ok" if loaded or self._load_error is None else "down",
            "mode": "embedded",
            "model_loaded": loaded,
            "model_name": settings.LAYA_MODEL,
            "load_seconds": round(self._load_seconds, 2) if self._load_seconds else None,
            "error": self._load_error,
            "calibrated": False,
            "warnings": self._warnings,
        }


def _trim_probabilities(probs: dict) -> dict:
    """Keep the top 4 options per field: enough to audit, small enough to store."""
    out: dict = {}
    for field, dist in (probs or {}).items():
        if isinstance(dist, dict):
            items = sorted(
                ((k, float(v)) for k, v in dist.items() if isinstance(v, (int, float))),
                key=lambda kv: -kv[1],
            )[:4]
            out[field] = {k: round(v, 4) for k, v in items}
    return out


# ---------------------------------------------------------------------------
# Caching wrapper + factory
# ---------------------------------------------------------------------------
class CachedLayaClient:
    """Caches Laya decisions in Redis (brief §12.7).

    Safe because a Laya decision is a pure function of the message text and the
    Laya model version, both of which are in the key. Degraded (fallback)
    decisions are never cached — caching an outage would extend it.
    """

    def __init__(self, inner: LayaClient) -> None:
        self._inner = inner

    @property
    def mode(self) -> str:
        return self._inner.mode

    def decide(
        self, message: str, *, include_guard: bool = True, has_customer_context: bool = False
    ) -> LayaDecisionEnvelope:
        norm = normalise_text(message).strip().lower()
        key = laya_decision_key(f"{norm}|guard={include_guard}")

        if settings.CACHE_ENABLED and redis_client.available():
            cached = get_json(key)
            if cached:
                incr("laya.cache.hit")
                env = LayaDecisionEnvelope.model_validate(cached)
                env.latency_ms = 0.0
                return env

        env = self._inner.decide(
            message, include_guard=include_guard, has_customer_context=has_customer_context
        )
        if settings.CACHE_ENABLED and not env.degraded:
            set_json(key, env.model_dump(mode="json"), ttl=settings.CACHE_TTL_LAYA)
            incr("laya.cache.write")
        return env

    def decide_batch(self, messages: Sequence[str]) -> list[LayaDecisionEnvelope]:
        return self._inner.decide_batch(messages)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed(texts)

    def health(self) -> dict:
        return self._inner.health()


_client: Any = None
_client_lock = Lock()


def get_laya_client() -> Any:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                inner = (
                    HttpLayaClient(settings.LAYA_URL)
                    if settings.LAYA_URL.strip()
                    else EmbeddedLayaClient()
                )
                log.info("laya.client_initialised", extra={"mode": inner.mode, "url": settings.LAYA_URL or "embedded"})
                _client = CachedLayaClient(inner)
    return _client


def reset_laya_client() -> None:
    global _client
    with _client_lock:
        _client = None


def set_laya_client(client: Any) -> None:
    """Inject a stub client for tests."""
    global _client
    with _client_lock:
        _client = client
