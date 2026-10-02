"""Centralised configuration. Everything comes from the environment.

No credential has a usable default. `OPENROUTER_API_KEY` empty means the LLM layer
reports itself unavailable rather than silently inventing classifications.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application -----------------------------------------------------
    APP_ENV: Literal["local", "dev", "test", "prod"] = "local"
    APP_NAME: str = "vireo-refund-intelligence"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    TENANT: str = "vireo-audio"
    LOCALE: str = "en-IN"
    CURRENCY: str = "INR"

    # ---- infrastructure --------------------------------------------------
    DATABASE_URL: str = "postgresql+psycopg://vireo:vireo@localhost:5432/vireo"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---- Laya System-1 decision layer ------------------------------------
    # http://laya:8100 in compose. Empty string => embedded in-process Laya.
    LAYA_URL: str = ""
    LAYA_API_KEY: str = ""
    LAYA_MODEL: str = "convaiinnovations/laya"
    LAYA_TIMEOUT_S: float = 30.0
    LAYA_MIN_CONFIDENCE: float = 0.35
    # Fail closed: when Laya is down, outbound/high-risk work goes to humans.
    LAYA_FAIL_CLOSED: bool = True

    # ---- OpenRouter / NVIDIA Nemotron ------------------------------------
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TIMEOUT_S: float = 120.0
    OPENROUTER_APP_URL: str = "https://github.com/vireo-audio/refund-intelligence"
    OPENROUTER_APP_TITLE: str = "Vireo Refund Intelligence"
    MAX_LLM_RETRIES: int = 2
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 700
    LLM_CONCURRENCY: int = 4

    # Hard per-request budget. Nemotron Ultra offers a very large context window,
    # but a refund ticket needs ~1.5k tokens: the policy sections plus a 130-char
    # customer message and an 85-char note. Anything approaching this ceiling means
    # something is wrong (runaway retrieval, a pathological ticket), so the client
    # refuses the call instead of quietly spending on a huge prompt.
    LLM_MAX_PROMPT_TOKENS: int = 10_000
    # Optional spend guard for a batch run. 0 disables it.
    LLM_MAX_TOTAL_TOKENS_PER_RUN: int = 0

    # Prices are USD per 1M tokens and MUST match the account's actual rate card.
    # Verified against the OpenRouter model page at the time of the run; override
    # via env when the rate card changes. Used only for cost reporting.
    OPENROUTER_PRICE_IN_USD_PER_1M: float = 0.60
    OPENROUTER_PRICE_OUT_USD_PER_1M: float = 1.80
    USD_INR_RATE: float = 88.0

    # ---- cache -----------------------------------------------------------
    CACHE_TTL: int = 3600
    CACHE_TTL_POLICY_ANSWER: int = 86400
    CACHE_TTL_LAYA: int = 900
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.93
    SEMANTIC_CACHE_MAX_CANDIDATES: int = 200
    CACHE_ENABLED: bool = True

    # ---- rate limiting (requests per window, seconds) --------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PUBLIC: str = "30/60"
    RATE_LIMIT_AUTHENTICATED: str = "120/60"
    RATE_LIMIT_INTERNAL: str = "600/60"
    RATE_LIMIT_LLM: str = "60/60"

    # ---- authorisation ---------------------------------------------------
    # Internal endpoints require one of these keys. Comma separated.
    INTERNAL_API_KEYS: str = ""
    DASHBOARD_API_KEY: str = ""

    # ---- policy ----------------------------------------------------------
    POLICY_VERSION: str = "3.2"
    POLICY_FILE: str = "app/policy/policy_v3_2.yaml"
    POLICY_PDF: str = "data/raw/support-policy.pdf"

    # Operational control, NOT a policy rule: the policy document defines no
    # per-agent refund ceiling. Refunds above this need a human in the loop.
    REFUND_AUTO_APPROVE_MAX_INR: float = 500.0
    REFUND_CRITICAL_RISK_INR: float = 5000.0

    # ---- prompt / model versioning --------------------------------------
    PROMPT_VERSION: str = "v6"
    CLASSIFIER_VERSION: str = "refund-classifier-1.0.0"

    # ---- data ------------------------------------------------------------
    DATA_RAW_DIR: str = "data/raw"
    OUTPUT_DIR: str = "outputs"

    # ---- worker ----------------------------------------------------------
    JOB_QUEUE_KEY: str = "vireo:jobs:classification"
    JOB_STATE_TTL: int = 86400
    WORKER_POLL_INTERVAL_S: float = 2.0
    WORKER_BATCH_SIZE: int = 25

    # ---- business planning constants (from the brief, not from the data) --
    PLANNING_TICKETS_PER_WEEK: int = 650

    @field_validator("LOG_LEVEL")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    # ---- derived ---------------------------------------------------------
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def policy_path(self) -> Path:
        return REPO_ROOT / self.POLICY_FILE

    @property
    def policy_pdf_path(self) -> Path:
        return REPO_ROOT / self.POLICY_PDF

    @property
    def raw_dir(self) -> Path:
        return REPO_ROOT / self.DATA_RAW_DIR

    @property
    def output_dir(self) -> Path:
        p = REPO_ROOT / self.OUTPUT_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def llm_available(self) -> bool:
        return bool(self.OPENROUTER_API_KEY.strip())

    @property
    def internal_keys(self) -> set[str]:
        return {k.strip() for k in self.INTERNAL_API_KEYS.split(",") if k.strip()}

    @property
    def planning_tickets_per_month(self) -> float:
        """650 tickets/week -> 650 * 52 / 12 tickets/month."""
        return self.PLANNING_TICKETS_PER_WEEK * 52 / 12

    def rate_limit(self, bucket: str) -> tuple[int, int]:
        raw = {
            "public": self.RATE_LIMIT_PUBLIC,
            "authenticated": self.RATE_LIMIT_AUTHENTICATED,
            "internal": self.RATE_LIMIT_INTERNAL,
            "llm": self.RATE_LIMIT_LLM,
        }[bucket]
        limit, _, window = raw.partition("/")
        return int(limit), int(window)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
