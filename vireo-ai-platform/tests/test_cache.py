"""Redis cache, invalidation, rate limit, idempotency."""

from __future__ import annotations

import pytest

from app.cache import idempotency, rate_limit
from app.cache.keys import CacheScope, Safety
from app.cache.response_cache import ResponseCache, normalise_query
from app.core.errors import IdempotencyConflict
from app.db.session import session_scope


def test_exact_cache_hit_and_policy_namespace():
    cache = ResponseCache(embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    a = CacheScope(intent="policy_question", safety=Safety.PUBLIC, policy_version="3.2")
    b = CacheScope(intent="policy_question", safety=Safety.PUBLIC, policy_version="3.3")
    assert a.namespace() != b.namespace()
    assert cache.set(a, "How long is the warranty?", {"answer": "12 months"})
    hit = cache.get(a, "how long is the warranty")
    assert hit.hit and hit.kind == "exact"
    miss = cache.get(b, "how long is the warranty")
    assert miss.hit is False


def test_personalised_intent_never_cached():
    cache = ResponseCache()
    scope = CacheScope(intent="refund_status", safety=Safety.PERSONALISED)
    assert scope.cacheable is False
    assert cache.set(scope, "what was my refund", {"answer": "nope"}) is False
    assert cache.get(scope, "what was my refund").hit is False


def test_semantic_cache_only_inside_public_namespace():
    def embed(texts):
        # Very similar vectors for warranty phrasings.
        return [[0.9, 0.1, 0.0] if "warranty" in t.lower() else [0.0, 0.0, 1.0] for t in texts]

    cache = ResponseCache(embed_fn=embed)
    scope = CacheScope(intent="policy_question", safety=Safety.PUBLIC)
    cache.set(scope, "What is the warranty period?", {"answer": "12 months"})
    hit = cache.get(scope, "How long does the warranty last?")
    assert hit.hit is True
    assert hit.kind == "semantic"

    other = CacheScope(intent="refund_status", safety=Safety.PERSONALISED)
    assert cache.get(other, "What is the warranty period?").hit is False


def test_cache_invalidation_on_policy_bump():
    cache = ResponseCache()
    scope = CacheScope(intent="policy_question", safety=Safety.PUBLIC, policy_version="3.2")
    cache.set(scope, "hours", {"answer": "24x7"})
    deleted = cache.invalidate_namespace(scope)
    assert deleted >= 1
    assert cache.get(scope, "hours").hit is False


def test_rate_limit_rejects_after_limit(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_PUBLIC", "3/60")
    identity = "ip:test-rl"
    rate_limit.reset("public", identity)
    for _ in range(3):
        assert rate_limit.check("public", identity).allowed
    blocked = rate_limit.check("public", identity)
    assert blocked.allowed is False
    assert blocked.retry_after_seconds >= 1


def test_idempotency_replay_and_conflict():
    with session_scope() as session:
        first = idempotency.claim(session, "refund", "key-1", {"amount": 100})
        assert first.is_replay is False
        idempotency.complete(session, "key-1", {"ok": True})
        session.commit()

    with session_scope() as session:
        replay = idempotency.claim(session, "refund", "key-1", {"amount": 100})
        assert replay.is_replay is True
        assert replay.response == {"ok": True}

    with session_scope() as session:
        with pytest.raises(IdempotencyConflict):
            idempotency.claim(session, "refund", "key-1", {"amount": 999})


def test_normalise_query():
    assert normalise_query("  How long???  ") == "how long"
