"""In-process latency/counter registry, mirrored into Redis so the dashboard
can read numbers produced by the API and the worker together.

Deliberately not Prometheus: nothing in this take-home scrapes it, and a fake
metrics endpoint is worse than an honest counter the dashboard actually reads.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from threading import Lock
from typing import Iterator

from app.core.logging import get_logger

log = get_logger(__name__)

_LOCK = Lock()
_COUNTERS: dict[str, float] = {}
_TIMINGS: dict[str, list[float]] = {}

_REDIS_COUNTER_HASH = "vireo:metrics:counters"
_REDIS_TIMING_PREFIX = "vireo:metrics:timings:"
_MAX_SAMPLES = 2000


def _redis():
    try:
        from app.cache.redis_client import get_redis

        return get_redis()
    except Exception:
        return None


def incr(name: str, value: float = 1.0) -> None:
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0.0) + value
    r = _redis()
    if r is not None:
        try:
            r.hincrbyfloat(_REDIS_COUNTER_HASH, name, value)
        except Exception:
            pass


def observe(name: str, millis: float) -> None:
    with _LOCK:
        samples = _TIMINGS.setdefault(name, [])
        samples.append(millis)
        if len(samples) > _MAX_SAMPLES:
            del samples[: len(samples) - _MAX_SAMPLES]
    r = _redis()
    if r is not None:
        try:
            key = _REDIS_TIMING_PREFIX + name
            pipe = r.pipeline()
            pipe.rpush(key, f"{millis:.3f}")
            pipe.ltrim(key, -_MAX_SAMPLES, -1)
            pipe.expire(key, 7 * 24 * 3600)
            pipe.execute()
        except Exception:
            pass


@contextmanager
def timed(name: str, **log_fields) -> Iterator[dict]:
    """Time a block, record it, and hand back a dict the caller can annotate."""
    box: dict = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        box["latency_ms"] = round(elapsed, 2)
        observe(name, elapsed)
        if log_fields:
            log.debug("stage.complete", extra={"stage": name, "latency_ms": box["latency_ms"], **log_fields})


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def snapshot(include_redis: bool = True) -> dict:
    """Merged view of local + Redis metrics."""
    with _LOCK:
        counters = dict(_COUNTERS)
        timings = {k: list(v) for k, v in _TIMINGS.items()}

    if include_redis:
        r = _redis()
        if r is not None:
            try:
                for k, v in (r.hgetall(_REDIS_COUNTER_HASH) or {}).items():
                    counters[k] = float(v)
                for key in r.scan_iter(match=_REDIS_TIMING_PREFIX + "*", count=200):
                    name = key.split(_REDIS_TIMING_PREFIX, 1)[-1]
                    vals = [float(x) for x in r.lrange(key, 0, -1)]
                    if vals:
                        timings[name] = vals
            except Exception:
                pass

    latency = {
        name: {
            "count": len(vals),
            "p50_ms": _percentile(vals, 50),
            "p95_ms": _percentile(vals, 95),
            "p99_ms": _percentile(vals, 99),
            "max_ms": round(max(vals), 2),
            "mean_ms": round(sum(vals) / len(vals), 2),
        }
        for name, vals in sorted(timings.items())
        if vals
    }

    hits = counters.get("cache.exact.hit", 0) + counters.get("cache.semantic.hit", 0)
    misses = counters.get("cache.miss", 0)
    llm_ok = counters.get("llm.success", 0)
    llm_fail = counters.get("llm.failure", 0)
    llm_calls = llm_ok + llm_fail
    validations = counters.get("validation.attempt", 0)

    return {
        "counters": {k: round(v, 4) for k, v in sorted(counters.items())},
        "latency": latency,
        "derived": {
            "cache_hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else None,
            "llm_failure_rate": round(llm_fail / llm_calls, 4) if llm_calls else None,
            "validation_failure_rate": (
                round(counters.get("validation.failure", 0) / validations, 4) if validations else None
            ),
            "retry_rate": round(counters.get("llm.retry", 0) / llm_calls, 4) if llm_calls else None,
            "human_escalation_rate": (
                round(
                    counters.get("decision.requires_human", 0)
                    / counters.get("decision.total", 0),
                    4,
                )
                if counters.get("decision.total", 0)
                else None
            ),
            "estimated_cost_inr": round(counters.get("llm.cost_inr", 0.0), 4),
            "estimated_cost_usd": round(counters.get("llm.cost_usd", 0.0), 6),
            "total_input_tokens": int(counters.get("llm.tokens_in", 0)),
            "total_output_tokens": int(counters.get("llm.tokens_out", 0)),
        },
    }


def reset() -> None:
    with _LOCK:
        _COUNTERS.clear()
        _TIMINGS.clear()
    r = _redis()
    if r is not None:
        try:
            r.delete(_REDIS_COUNTER_HASH)
            for key in r.scan_iter(match=_REDIS_TIMING_PREFIX + "*", count=200):
                r.delete(key)
        except Exception:
            pass
