"""Shared test fixtures. Env is set before any application import."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.update(
    {
        "APP_ENV": "test",
        "LOG_JSON": "false",
        "LOG_LEVEL": "WARNING",
        "DATABASE_URL": "sqlite+pysqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/15",
        "INTERNAL_API_KEYS": "test-key",
        "DASHBOARD_API_KEY": "test-key",
        "LAYA_URL": "",
        "LAYA_API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "CACHE_ENABLED": "true",
        "RATE_LIMIT_ENABLED": "true",
        "SEMANTIC_CACHE_ENABLED": "true",
        "POLICY_PDF": "data/raw/support-policy.pdf",
        "POLICY_FILE": "app/policy/policy_v3_2.yaml",
    }
)

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.cache.redis_client import reset_client, set_client  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_engine, reset_engine, session_scope  # noqa: E402
from app.laya.client import set_laya_client  # noqa: E402
from app.laya.fallback import fallback_decision  # noqa: E402
from app.llm.client import set_llm_client  # noqa: E402


class StubLaya:
    mode = "stub"

    def decide(self, message, *, include_guard=True, has_customer_context=False):
        return fallback_decision(
            message,
            reason="test stub",
            has_customer_context=has_customer_context,
        )

    def decide_batch(self, messages):
        return [self.decide(m) for m in messages]

    def embed(self, texts):
        return [[float((hash(t) % 100) / 100.0), 0.2, 0.3, 0.4] for t in texts]

    def health(self):
        return {"status": "ok", "model_loaded": True, "mode": "stub", "model_name": "stub"}


class StubLLM:
    available = False
    model = "stub-nemotron"

    def unavailable_reason(self):
        return "tests do not call OpenRouter"

    def health(self):
        return {"configured": False, "model": self.model, "base_url": "", "reason": self.unavailable_reason()}

    def generate_structured(self, *a, **k):
        raise AssertionError("unit tests must not call the LLM")

    def generate_text(self, *a, **k):
        raise AssertionError("unit tests must not call the LLM")

    def classify_ticket(self, *a, **k):
        raise AssertionError("unit tests must not call the LLM")


@pytest.fixture(autouse=True)
def _infra():
    reset_engine()
    get_settings.cache_clear()
    engine = get_engine()
    import app.models  # noqa: F401

    Base.metadata.create_all(engine)
    fake = fakeredis.FakeRedis(decode_responses=True)
    set_client(fake)
    set_laya_client(StubLaya())
    set_llm_client(StubLLM())
    yield
    reset_engine()
    reset_client()


@pytest.fixture
def db():
    with session_scope() as session:
        yield session


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key"}
