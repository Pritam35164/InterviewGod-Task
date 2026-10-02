"""LLM cost accounting.

Two figures are reported and never conflated:

  * Development cost — what this project actually spent, summed from the
    `llm_requests` rows produced by real calls. If the run used free credits the
    cash cost is zero, and the token-priced equivalent is reported alongside it so
    "we spent nothing" is not mistaken for "it costs nothing".
  * Production projection — the same measured per-ticket token cost applied to
    Vireo's stated 650 tickets/week, with the arithmetic printed.

The projection is based on measured tokens, not a guess. Where no calls have been
made yet, the report says so instead of producing an estimate.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.metrics import snapshot
from app.models.decisions import LLMRequest
from app.models.refund import Refund
from app.models.source import Ticket


def cost_report(session: Session) -> dict:
    agg = session.execute(
        select(
            func.count(LLMRequest.id),
            func.coalesce(func.sum(LLMRequest.input_tokens), 0),
            func.coalesce(func.sum(LLMRequest.output_tokens), 0),
            func.coalesce(func.sum(LLMRequest.cost_usd), 0.0),
            func.coalesce(func.sum(LLMRequest.cost_inr), 0.0),
            func.coalesce(func.sum(LLMRequest.upstream_cost_usd), 0.0),
            func.coalesce(func.avg(LLMRequest.latency_ms), 0.0),
            func.coalesce(func.sum(LLMRequest.retries), 0),
        )
    ).one()
    (
        requests_total, tokens_in, tokens_out,
        cost_usd_est, cost_inr_est, upstream_usd, avg_latency, total_retries,
    ) = agg

    succeeded = session.execute(
        select(func.count(LLMRequest.id)).where(LLMRequest.status == "success")
    ).scalar_one()
    failed = requests_total - succeeded
    cache_hits = session.execute(
        select(func.count(LLMRequest.id)).where(LLMRequest.cache_hit.is_(True))
    ).scalar_one()

    classified = session.execute(
        select(func.count(Refund.id)).where(Refund.normalised_reason_code.isnot(None))
    ).scalar_one()
    refund_tickets = session.execute(select(func.count(Refund.id))).scalar_one()
    total_tickets = session.execute(select(func.count(Ticket.id))).scalar_one()

    models = session.execute(
        select(LLMRequest.model_name, func.count(LLMRequest.id))
        .group_by(LLMRequest.model_name)
    ).all()
    prompts = session.execute(
        select(LLMRequest.prompt_version, func.count(LLMRequest.id))
        .group_by(LLMRequest.prompt_version)
    ).all()

    if requests_total == 0:
        return {
            "status": "no_llm_calls_recorded",
            "model": settings.OPENROUTER_MODEL,
            "prompt_version": settings.PROMPT_VERSION,
            "requests_total": 0,
            "message": (
                "No LLM requests have been made against this database, so there is nothing to "
                "cost. Cost is reported from measured token usage only; no estimate is "
                "produced from an unmeasured run."
            ),
            "llm_configured": settings.llm_available,
            "projection": {
                "tickets_per_week": settings.PLANNING_TICKETS_PER_WEEK,
                "tickets_per_month": round(settings.planning_tickets_per_month, 1),
                "note": "Cannot be projected without measured per-ticket token usage.",
            },
        }

    avg_in = tokens_in / requests_total
    avg_out = tokens_out / requests_total

    # Actual cash cost: OpenRouter's own number when it reported one.
    actual_usd = upstream_usd if upstream_usd > 0 else cost_usd_est
    actual_inr = actual_usd * settings.USD_INR_RATE

    per_request_usd = actual_usd / requests_total
    per_request_inr = per_request_usd * settings.USD_INR_RATE
    # One classified ticket can cost more than one request when a retry happened.
    per_classified_inr = (actual_inr / classified) if classified else per_request_inr

    tickets_per_month = settings.planning_tickets_per_month
    refund_share = (refund_tickets / total_tickets) if total_tickets else 0.0

    monthly_all = per_classified_inr * tickets_per_month
    monthly_refunds_only = per_classified_inr * tickets_per_month * refund_share
    weekly_refunds_only = per_classified_inr * settings.PLANNING_TICKETS_PER_WEEK * refund_share

    arithmetic = [
        f"Measured: {requests_total:,} LLM requests, {tokens_in:,} input tokens, "
        f"{tokens_out:,} output tokens.",
        f"Average per request: {avg_in:,.0f} input + {avg_out:,.0f} output tokens.",
        f"Rate card: ${settings.OPENROUTER_PRICE_IN_USD_PER_1M:.2f}/1M input, "
        f"${settings.OPENROUTER_PRICE_OUT_USD_PER_1M:.2f}/1M output.",
        f"Token-priced cost = {tokens_in:,}/1e6 x ${settings.OPENROUTER_PRICE_IN_USD_PER_1M:.2f} "
        f"+ {tokens_out:,}/1e6 x ${settings.OPENROUTER_PRICE_OUT_USD_PER_1M:.2f} "
        f"= ${cost_usd_est:.4f}.",
        (
            f"OpenRouter's own reported cost = ${upstream_usd:.4f} (used in preference to the "
            "local estimate)."
            if upstream_usd > 0
            else "OpenRouter did not return a per-call cost; the token-priced estimate is used."
        ),
        f"At ₹{settings.USD_INR_RATE:.2f}/USD that is ₹{actual_inr:,.2f} for the whole run.",
        f"{classified:,} refund tickets were classified, so ₹{actual_inr:,.2f} / {classified:,} "
        f"= ₹{per_classified_inr:.4f} per classified ticket.",
        f"Vireo volume: {settings.PLANNING_TICKETS_PER_WEEK} tickets/week "
        f"x 52 / 12 = {tickets_per_month:,.0f} tickets/month.",
        f"Only refund tickets need classification. Observed refund share = {refund_tickets:,}/"
        f"{total_tickets:,} = {refund_share * 100:.1f}%.",
        f"Projected monthly cost, refund tickets only = ₹{per_classified_inr:.4f} x "
        f"{tickets_per_month:,.0f} x {refund_share:.3f} = ₹{monthly_refunds_only:,.2f}.",
        f"Projected monthly cost if EVERY ticket were classified = ₹{per_classified_inr:.4f} x "
        f"{tickets_per_month:,.0f} = ₹{monthly_all:,.2f}.",
    ]

    snap = snapshot()
    latency = {
        k: v for k, v in snap.get("latency", {}).items()
        if k in {"nemotron.request", "laya.decide", "redis.get", "db.load_frames",
                 "pipeline.nemotron", "pipeline.laya", "pipeline.policy_retrieval"}
    }

    return {
        "status": "measured",
        "model": models[0][0] if models else settings.OPENROUTER_MODEL,
        "models_used": {m: int(c) for m, c in models},
        "prompt_versions_used": {p: int(c) for p, c in prompts},
        "prompt_version": settings.PROMPT_VERSION,
        "requests_total": int(requests_total),
        "requests_succeeded": int(succeeded),
        "requests_failed": int(failed),
        "total_retries": int(total_retries),
        "cache_hits": int(cache_hits),
        "cache_hit_rate": round(cache_hits / requests_total, 4) if requests_total else None,
        "input_tokens": int(tokens_in),
        "output_tokens": int(tokens_out),
        "avg_input_tokens": round(avg_in, 1),
        "avg_output_tokens": round(avg_out, 1),
        "price_in_usd_per_1m": settings.OPENROUTER_PRICE_IN_USD_PER_1M,
        "price_out_usd_per_1m": settings.OPENROUTER_PRICE_OUT_USD_PER_1M,
        "usd_inr_rate": settings.USD_INR_RATE,
        "development_cost_usd": round(actual_usd, 6),
        "development_cost_inr": round(actual_inr, 4),
        "development_cost_note": (
            "This is the cost of the development/analysis run recorded in this database. If the "
            "account was on free credits the cash outlay was ₹0, and the figure above is the "
            "token-priced equivalent of the same work at the configured rate card — reported so "
            "that a zero invoice is not read as a zero running cost."
        ),
        "token_priced_equivalent_usd": round(cost_usd_est, 6),
        "token_priced_equivalent_inr": round(cost_inr_est, 4),
        "cost_per_request_usd": round(per_request_usd, 8),
        "cost_per_request_inr": round(per_request_inr, 6),
        "cost_per_ticket_usd": round(per_classified_inr / settings.USD_INR_RATE, 8),
        "cost_per_ticket_inr": round(per_classified_inr, 6),
        "cost_per_classified_ticket_inr": round(per_classified_inr, 6),
        "classified_tickets": int(classified),
        "refund_tickets": int(refund_tickets),
        "total_tickets": int(total_tickets),
        "arithmetic": arithmetic,
        "projected_tickets_per_week": settings.PLANNING_TICKETS_PER_WEEK,
        "projected_tickets_per_month": round(tickets_per_month, 1),
        "projected_refund_share": round(refund_share, 4),
        "projected_monthly_cost_inr_all_tickets": round(monthly_all, 2),
        "projected_monthly_cost_inr_refunds_only": round(monthly_refunds_only, 2),
        "projected_weekly_cost_inr": round(weekly_refunds_only, 2),
        "avg_latency_ms": round(float(avg_latency), 2),
        "latency": latency,
        "caveats": [
            "Projection assumes the same prompt version, the same policy context size and the "
            "same average ticket length. A longer policy or a chattier model changes it.",
            "Retries are included in the measured cost, so the per-ticket figure already "
            "carries the observed schema-failure overhead.",
            "The rate card is configuration (OPENROUTER_PRICE_*). Verify it against the "
            "OpenRouter model page before quoting these numbers to Finance.",
            "The cache is not modelled in the projection. In production, repeated policy "
            "questions hit the Redis cache and cost nothing, so the projection is an upper bound.",
        ],
    }
