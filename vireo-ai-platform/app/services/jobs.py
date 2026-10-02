"""Redis-backed job queue for bulk classification.

Classifying 2,340 tickets is thousands of LLM calls; an HTTP request must not wait
for that (brief §31). So the API enqueues a job and returns immediately, and the
worker drains the queue.

A Redis list plus a status hash, not Kafka and not Celery. The requirement is
"one producer, one consumer, restartable, observable progress", and a broker would
be infrastructure nobody here needs to operate.

Redis is not authoritative: every classification lands in PostgreSQL as it
completes, so losing the job state loses the progress bar, not the work.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache import redis_client
from app.cache.keys import job_key, job_queue_key
from app.core.config import settings
from app.core.errors import DependencyUnavailable
from app.core.logging import get_logger
from app.models.decisions import TicketClassification
from app.models.refund import Refund
from app.models.source import Ticket
from app.schemas.api import ClassificationJobRequest
from app.schemas.taxonomy import JobStatus

log = get_logger(__name__)

WORKER_HEARTBEAT_KEY = "vireo:worker:heartbeat"
WORKER_HEARTBEAT_TTL = 90


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_ticket_ids(session: Session, request: ClassificationJobRequest) -> list[str]:
    """Resolve a job scope to a concrete, ordered list of ticket ids.

    Ordered by refund value descending so that a capped or interrupted run has
    covered the money that matters most.
    """
    stmt = select(Ticket.ticket_id).order_by(Ticket.refund_amount_inr.desc().nullslast())

    if request.scope in {"refunds", "unclassified", "sample"}:
        stmt = stmt.where(Ticket.is_refund.is_(True))

    if request.scope == "unclassified":
        stmt = stmt.where(
            Ticket.ticket_id.not_in(select(TicketClassification.ticket_id))
        )

    if request.scope == "evaluation_set":
        from app.services.evaluation import load_eval_set

        ids = [r["ticket_id"] for r in load_eval_set()]
        if not ids:
            return []
        stmt = select(Ticket.ticket_id).where(Ticket.ticket_id.in_(ids))

    if request.month_from:
        stmt = stmt.where(Ticket.created_month >= request.month_from)
    if request.month_to:
        stmt = stmt.where(Ticket.created_month <= request.month_to)

    ids = list(session.execute(stmt).scalars().all())

    if request.scope == "sample" and request.limit:
        # Deterministic spread across months rather than the top-N by value, so a
        # sample is representative instead of being all the big refunds.
        months: dict[str, list[str]] = {}
        rows = session.execute(
            select(Ticket.ticket_id, Ticket.created_month)
            .where(Ticket.ticket_id.in_(ids))
            .order_by(Ticket.refund_amount_inr.desc().nullslast())
        ).all()
        for tid, month in rows:
            months.setdefault(month, []).append(tid)
        picked: list[str] = []
        idx = 0
        while len(picked) < request.limit and any(
            idx < len(v) for v in months.values()
        ):
            for month in sorted(months):
                bucket = months[month]
                if idx < len(bucket) and len(picked) < request.limit:
                    picked.append(bucket[idx])
            idx += 1
        return picked

    if request.limit:
        ids = ids[: request.limit]
    return ids


def enqueue(session: Session, request: ClassificationJobRequest) -> dict:
    if not redis_client.available():
        raise DependencyUnavailable(
            "Redis is unavailable, so background jobs cannot be queued. Classify individual "
            "tickets through POST /v1/tickets/classify, or restore Redis."
        )

    ticket_ids = select_ticket_ids(session, request)
    job_id = uuid.uuid4().hex
    state = {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "scope": request.scope,
        "force": request.force,
        "total": len(ticket_ids),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "review_required": 0,
        "cache_hits": 0,
        "cost_inr": 0.0,
        "cost_usd": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "created_at": _now(),
        "updated_at": _now(),
        "finished_at": None,
        "error": None,
        "llm_configured": settings.llm_available,
    }

    r = redis_client.get_redis()
    pipe = r.pipeline()
    pipe.set(job_key(job_id), json.dumps(state), ex=settings.JOB_STATE_TTL)
    pipe.set(
        job_key(job_id) + ":tickets",
        json.dumps(ticket_ids),
        ex=settings.JOB_STATE_TTL,
    )
    pipe.lpush(job_queue_key(), job_id)
    pipe.execute()

    log.info(
        "job.enqueued",
        extra={"job_id": job_id, "scope": request.scope, "total": len(ticket_ids)},
    )
    return state


def get_state(job_id: str) -> dict | None:
    if not redis_client.available():
        return None
    raw = redis_client.get_redis().get(job_key(job_id))
    return json.loads(raw) if raw else None


def update_state(job_id: str, **changes: Any) -> dict | None:
    if not redis_client.available():
        return None
    r = redis_client.get_redis()
    raw = r.get(job_key(job_id))
    if not raw:
        return None
    state = json.loads(raw)
    state.update(changes)
    state["updated_at"] = _now()
    r.set(job_key(job_id), json.dumps(state), ex=settings.JOB_STATE_TTL)
    return state


def get_tickets(job_id: str) -> list[str]:
    if not redis_client.available():
        return []
    raw = redis_client.get_redis().get(job_key(job_id) + ":tickets")
    return json.loads(raw) if raw else []


def claim_next(timeout: int = 5) -> str | None:
    """Block for up to `timeout` seconds waiting for a job id."""
    if not redis_client.available():
        return None
    try:
        popped = redis_client.get_redis().brpop(job_queue_key(), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("job.claim_failed", extra={"error": str(exc)[:200]})
        return None
    return popped[1] if popped else None


def cancel(job_id: str) -> dict | None:
    return update_state(job_id, status=JobStatus.CANCELLED, finished_at=_now())


def list_jobs(limit: int = 25) -> list[dict]:
    if not redis_client.available():
        return []
    r = redis_client.get_redis()
    jobs = []
    for key in r.scan_iter(match="vireo:job:*", count=200):
        if key.endswith(":tickets"):
            continue
        raw = r.get(key)
        if raw:
            try:
                jobs.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs[:limit]


def heartbeat() -> None:
    if redis_client.available():
        try:
            redis_client.get_redis().set(
                WORKER_HEARTBEAT_KEY, _now(), ex=WORKER_HEARTBEAT_TTL
            )
        except Exception:
            pass


def worker_alive() -> dict:
    if not redis_client.available():
        return {"alive": False, "reason": "redis unavailable"}
    try:
        beat = redis_client.get_redis().get(WORKER_HEARTBEAT_KEY)
    except Exception as exc:  # noqa: BLE001
        return {"alive": False, "reason": str(exc)[:120]}
    return {
        "alive": bool(beat),
        "last_heartbeat": beat,
        "ttl_seconds": WORKER_HEARTBEAT_TTL,
    }
