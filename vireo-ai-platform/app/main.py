"""FastAPI application entrypoint."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import admin, analytics, decisions, health, jobs, tickets
from app.core.config import settings
from app.core.errors import RateLimitExceeded, VireoError
from app.core.logging import (
    configure_logging,
    get_logger,
    new_request_id,
    request_id_var,
    trace_id_var,
)
from app.core.metrics import incr, observe
from app.policy.loader import get_policy

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Load and verify the policy at startup: a transcription drift or a missing
    # PDF should be visible immediately, not on the first refund decision.
    try:
        policy = get_policy()
        log.info(
            "startup.policy_ready",
            extra={
                "policy_version": policy.version,
                "sections": len(policy.sections),
                "verification": policy.verification.get("status"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.error("startup.policy_failed", extra={"error": str(exc)[:300]})

    log.info(
        "startup.complete",
        extra={
            "env": settings.APP_ENV,
            "version": __version__,
            "laya_mode": "http" if settings.LAYA_URL else "embedded",
            "llm_configured": settings.llm_available,
            "cache_enabled": settings.CACHE_ENABLED,
            "rate_limit_enabled": settings.RATE_LIMIT_ENABLED,
        },
    )
    yield
    log.info("shutdown.complete")


app = FastAPI(
    title="Vireo Audio — Refund Intelligence and Support Decision Platform",
    version=__version__,
    description=(
        "Turns a messy helpdesk export into an auditable refund analysis.\n\n"
        "**Architecture.** Laya makes the fast System-1 routing decision; Redis serves cache, "
        "rate limiting and idempotency; a deterministic policy engine owns every authorisation; "
        "NVIDIA Nemotron (through OpenRouter) reads the natural language; Pydantic validates "
        "every model output; Python computes every number; PostgreSQL holds the audit trail.\n\n"
        "**What the AI is not allowed to do.** No model decides a refund amount, computes a "
        "total, or authorises a payment. A DENY from the policy engine cannot be overridden by "
        "Laya or by the LLM, and a missing dependency fails closed to human review.\n\n"
        "Most endpoints require an `X-API-Key` header."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness and observability."},
        {"name": "decisions", "description": "Inbound answering and outbound authorisation."},
        {"name": "tickets", "description": "Classification and per-ticket decision lineage."},
        {"name": "analytics", "description": "Refund analysis. Every number computed in Python."},
        {"name": "jobs", "description": "Background bulk classification."},
        {"name": "admin", "description": "Ingest, policy, cache, audit, evaluation."},
    ],
)


# ---------------------------------------------------------------------------
# Middleware: correlation + request logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    incoming = request.headers.get("x-request-id")
    request_id = incoming or new_request_id()
    trace_id = request.headers.get("x-trace-id") or request_id
    rid_token = request_id_var.set(request_id)
    tid_token = trace_id_var.set(trace_id)

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        observe("http.request", elapsed)
        incr("http.error")
        log.error(
            "http.unhandled_exception",
            extra={
                "method": request.method,
                "path": request.url.path,
                "latency_ms": round(elapsed, 2),
            },
            exc_info=True,
        )
        request_id_var.reset(rid_token)
        trace_id_var.reset(tid_token)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-Id": request_id},
        )

    elapsed = (time.perf_counter() - start) * 1000
    observe("http.request", elapsed)
    observe(f"http.{request.url.path}", elapsed)
    incr("http.request")
    incr(f"http.status.{response.status_code // 100}xx")

    response.headers["X-Request-Id"] = request_id
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.2f}"

    # /health is polled by Docker every 15s; logging it buries everything else.
    if request.url.path not in {"/health", "/ready"}:
        log.info(
            "http.request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed, 2),
            },
        )

    request_id_var.reset(rid_token)
    trace_id_var.reset(tid_token)
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    payload = exc.to_payload()
    payload["error"]["request_id"] = request_id_var.get()
    return JSONResponse(
        status_code=429,
        content=payload,
        headers={"Retry-After": str(exc.retry_after), "X-Request-Id": request_id_var.get()},
    )


@app.exception_handler(VireoError)
async def vireo_error_handler(request: Request, exc: VireoError):
    payload = exc.to_payload()
    payload["error"]["request_id"] = request_id_var.get()
    if exc.status_code >= 500:
        log.error(
            "app.error",
            extra={"code": exc.code, "path": request.url.path, "detail": exc.details},
        )
    else:
        log.warning(
            "app.client_error",
            extra={"code": exc.code, "path": request.url.path, "status": exc.status_code},
        )
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    incr("http.validation_error")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "request_validation_failed",
                "message": "The request body or parameters did not validate.",
                "details": {"errors": exc.errors()[:10]},
                "request_id": request_id_var.get(),
            }
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(decisions.router)
app.include_router(tickets.router)
app.include_router(analytics.router)
app.include_router(jobs.router)
app.include_router(admin.router)


@app.get("/", tags=["health"], summary="Service banner")
def root() -> dict:
    return {
        "service": settings.APP_NAME,
        "version": __version__,
        "env": settings.APP_ENV,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
        "architecture": [
            "Laya — fast System-1 routing, gating and risk. Never writes prose.",
            "Redis — exact cache, scoped semantic cache, rate limiting, idempotency, job state.",
            "Policy engine — the only component that authorises an action. Overrules Laya.",
            "NVIDIA Nemotron via OpenRouter — natural-language reading only. Never money.",
            "Pydantic — validates every model output, with evidence grounding enforced.",
            "Python — every number, every total, every ratio.",
            "PostgreSQL — persistent records and the audit trail.",
        ],
    }
