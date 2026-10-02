"""Inbound and outbound decision flows.

Inbound:
    normalise -> Laya -> hard security checks -> Redis (if cacheable)
    -> deterministic answer OR policy answer OR Nemotron -> validate -> audit

Outbound:
    proposed action -> Laya -> policy engine -> risk -> human approval if needed
    -> executor -> audit

The ordering is not cosmetic. The security check runs before the cache so a
flagged message cannot be served a cached answer, and the policy engine runs after
Laya so Laya's verdict can be overruled rather than obeyed.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache import idempotency
from app.cache.keys import CacheScope, Safety, classify_safety
from app.cache.response_cache import ResponseCache
from app.core.config import settings
from app.core.logging import current_request_id, current_trace_id, get_logger
from app.core.metrics import incr, timed
from app.core.security import normalise_text, scan_injection
from app.laya.client import get_laya_client
from app.llm.client import get_llm_client, record_llm_request
from app.models.decisions import AuditEvent, PolicyDecision, RoutingDecision
from app.models.refund import Refund
from app.models.source import Agent, Order
from app.policy.engine import ActionContext, evaluate_outbound_action, deterministic_answer
from app.policy.loader import get_policy, get_retrieval_index
from app.schemas.api import (
    DecisionTrace,
    InboundRequest,
    InboundResponse,
    OutboundActionRequest,
    OutboundActionResponse,
)
from app.schemas.llm import PolicyAnswer
from app.schemas.taxonomy import DecisionRoute, PolicyOutcome, RiskLevel

log = get_logger(__name__)

HUMAN_REVIEW_MESSAGE = (
    "This request needs a person. It has been routed to a support agent for review "
    "rather than answered automatically."
)
BLOCKED_MESSAGE = (
    "This request could not be processed automatically. It has been recorded and "
    "routed to a human reviewer."
)


def _embed_fn():
    """Embeddings for the semantic cache come from the Laya service."""
    def embed(texts):
        return get_laya_client().embed(texts)
    return embed


# ---------------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------------
def handle_inbound(session: Session, request: InboundRequest) -> InboundResponse:
    started = time.perf_counter()
    trace: list[DecisionTrace] = []
    policy = get_policy()

    # ---- 1. normalise -----------------------------------------------------
    with timed("inbound.normalise") as t:
        message = normalise_text(request.message).strip()
        injection_flags = scan_injection(message)
    trace.append(
        DecisionTrace(
            stage="normalise",
            outcome="normalised",
            latency_ms=t["latency_ms"],
            detail={
                "original_chars": len(request.message),
                "normalised_chars": len(message),
                "injection_patterns": injection_flags,
            },
        )
    )

    # ---- 2. Laya ----------------------------------------------------------
    laya = get_laya_client()
    has_customer_context = bool(request.customer_id or request.order_id or request.ticket_id)
    with timed("inbound.laya") as t:
        laya_env = laya.decide(
            message, include_guard=True, has_customer_context=has_customer_context
        )
    trace.append(
        DecisionTrace(
            stage="laya",
            outcome=f"intent={laya_env.decision.intent} risk={laya_env.decision.risk_level}",
            latency_ms=t["latency_ms"],
            detail={
                **laya_env.decision.model_dump(),
                "source": laya_env.source,
                "degraded": laya_env.degraded,
                "degraded_reason": laya_env.degraded_reason,
                "min_confidence": laya_env.min_confidence(),
                "low_confidence_fields": laya_env.low_confidence_fields,
                "calibrated": laya_env.calibrated,
            },
        )
    )
    incr("decision.total")

    intent = str(laya_env.decision.intent)
    guard_flagged = bool(
        laya_env.guard and (laya_env.guard.jailbreak or laya_env.guard.prompt_injection)
    )

    # ---- 3. hard security checks (before any cache read) -------------------
    if injection_flags or guard_flagged:
        incr("decision.blocked_security")
        incr("decision.requires_human")
        trace.append(
            DecisionTrace(
                stage="security",
                outcome="blocked",
                detail={
                    "regex_patterns": injection_flags,
                    "laya_guard": laya_env.guard.model_dump() if laya_env.guard else None,
                    "action": "no cached answer served, no LLM call made, routed to human",
                },
            )
        )
        response = InboundResponse(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            route=DecisionRoute.BLOCKED,
            answer=BLOCKED_MESSAGE,
            requires_human=True,
            human_review_reason=(
                "Message contains content aimed at the AI system rather than a genuine "
                "request: " + ", ".join(injection_flags or ["flagged by Laya guardrail"])
            ),
            laya=laya_env,
            policy_version=policy.version,
            injection_flags=injection_flags,
            trace=trace,
        )
        _record_routing(session, laya_env, DecisionRoute.BLOCKED, request.ticket_id, injection_flags)
        _audit_inbound(session, request, response, "inbound_blocked_security")
        response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response

    # ---- 4. cache ---------------------------------------------------------
    safety = classify_safety(
        intent,
        has_customer_context=has_customer_context,
        injection_flags=injection_flags,
        guard_flagged=guard_flagged,
    )
    scope = CacheScope(intent=intent, safety=safety, locale=request.locale)
    cache = ResponseCache(embed_fn=_embed_fn() if settings.SEMANTIC_CACHE_ENABLED else None)

    cacheable = bool(laya_env.decision.cacheable) and scope.cacheable
    if cacheable:
        with timed("inbound.cache") as t:
            hit = cache.get(scope, message)
        trace.append(
            DecisionTrace(
                stage="cache",
                outcome=hit.kind if hit.hit else "miss",
                latency_ms=t["latency_ms"],
                detail={**hit.to_dict(), "scope": scope.metadata()},
            )
        )
        if hit.hit and hit.value:
            response = InboundResponse(
                request_id=current_request_id(),
                trace_id=current_trace_id(),
                route=(
                    DecisionRoute.CACHE_SEMANTIC if hit.kind == "semantic"
                    else DecisionRoute.CACHE_EXACT
                ),
                answer=hit.value.get("answer"),
                laya=laya_env,
                policy_sections=hit.value.get("policy_sections", []),
                policy_version=policy.version,
                cache=hit.to_dict(),
                trace=trace,
            )
            _record_routing(session, laya_env, response.route, request.ticket_id, injection_flags)
            _audit_inbound(session, request, response, "inbound_cache_hit")
            response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return response
    else:
        trace.append(
            DecisionTrace(
                stage="cache",
                outcome="skipped",
                detail={
                    "reason": (
                        "Laya marked the request not cacheable"
                        if not laya_env.decision.cacheable
                        else f"safety class '{safety}' is never cached"
                    ),
                    "safety": safety,
                    "scope": scope.metadata(),
                },
            )
        )

    # ---- 5. route B: deterministic ---------------------------------------
    with timed("inbound.deterministic") as t:
        det = deterministic_answer(intent, message)
    if det is not None and safety == Safety.PUBLIC:
        answer, section_id = det
        sections = [policy.section(section_id).to_dict()] if section_id in policy.sections else []
        trace.append(
            DecisionTrace(
                stage="deterministic",
                outcome="answered",
                latency_ms=t["latency_ms"],
                detail={
                    "policy_section_id": section_id,
                    "note": "answered from policy rules in code; no LLM call made",
                },
            )
        )
        response = InboundResponse(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            route=DecisionRoute.DETERMINISTIC,
            answer=answer,
            laya=laya_env,
            policy_sections=sections,
            policy_version=policy.version,
            trace=trace,
        )
        if cacheable:
            cache.set(
                scope, message,
                {"answer": answer, "policy_sections": sections, "route": "deterministic"},
                ttl=settings.CACHE_TTL_POLICY_ANSWER,
            )
        _record_routing(session, laya_env, DecisionRoute.DETERMINISTIC, request.ticket_id, injection_flags)
        _audit_inbound(session, request, response, "inbound_deterministic")
        response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response
    trace.append(
        DecisionTrace(
            stage="deterministic",
            outcome="no_rule_matched",
            latency_ms=t["latency_ms"],
            detail={"safety": safety},
        )
    )

    # ---- 6. human review gate --------------------------------------------
    needs_human = bool(laya_env.decision.requires_human)
    risk = RiskLevel(laya_env.decision.risk_level)
    if laya_env.degraded and risk.rank >= RiskLevel.HIGH.rank and settings.LAYA_FAIL_CLOSED:
        needs_human = True
    if needs_human:
        incr("decision.requires_human")
        reason = (
            f"Laya flagged requires_human=True at risk={risk}"
            + (f" (degraded: {laya_env.degraded_reason})" if laya_env.degraded else "")
        )
        trace.append(
            DecisionTrace(
                stage="human_gate", outcome="escalated",
                detail={"risk_level": str(risk), "degraded": laya_env.degraded, "reason": reason},
            )
        )
        response = InboundResponse(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            route=DecisionRoute.HUMAN,
            answer=HUMAN_REVIEW_MESSAGE,
            requires_human=True,
            human_review_reason=reason,
            laya=laya_env,
            policy_version=policy.version,
            trace=trace,
        )
        _record_routing(session, laya_env, DecisionRoute.HUMAN, request.ticket_id, injection_flags)
        _audit_inbound(session, request, response, "inbound_human_review")
        response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response

    # ---- 7. policy retrieval ---------------------------------------------
    with timed("inbound.policy_retrieval") as t:
        retriever = get_retrieval_index()
        retrieved = [r.to_dict() for r in retriever.retrieve(message, intent=intent, top_k=3)]
    trace.append(
        DecisionTrace(
            stage="policy_retrieval",
            outcome=f"{len(retrieved)} section(s)",
            latency_ms=t["latency_ms"],
            detail={
                "sections": [
                    {
                        "id": s["policy_section_id"],
                        "title": s["title"],
                        "score": s["relevance"]["score"],
                        "why": s["relevance"]["selected_because"],
                    }
                    for s in retrieved
                ],
                "method": "BM25 over 10 policy sections plus intent priors; no vector database",
            },
        )
    )

    # ---- 8. Nemotron ------------------------------------------------------
    llm = get_llm_client()
    if not request.allow_llm or not llm.available:
        reason = (
            "allow_llm=false on the request"
            if not request.allow_llm
            else (llm.unavailable_reason() or "LLM unavailable")
        )
        trace.append(DecisionTrace(stage="nemotron", outcome="skipped", detail={"reason": reason}))
        incr("decision.requires_human")
        response = InboundResponse(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            route=DecisionRoute.POLICY if retrieved else DecisionRoute.HUMAN,
            answer=(
                "The relevant policy sections are attached. An agent will confirm how they "
                "apply to this request."
            ),
            requires_human=True,
            human_review_reason=reason,
            laya=laya_env,
            policy_sections=retrieved,
            policy_version=policy.version,
            trace=trace,
        )
        _record_routing(session, laya_env, response.route, request.ticket_id, injection_flags)
        _audit_inbound(session, request, response, "inbound_llm_unavailable")
        response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response

    from app.llm.prompts import build_policy_answer_prompt

    with timed("inbound.nemotron") as t:
        prompt = build_policy_answer_prompt(message, retrieved)
        result = llm.generate_structured(prompt, PolicyAnswer)

    if result.usage is not None:
        record_llm_request(
            session, result.usage, purpose="policy_answer",
            ticket_id=request.ticket_id, policy_version=policy.version,
        )

    if not result.ok:
        trace.append(
            DecisionTrace(
                stage="nemotron", outcome="failed", latency_ms=t["latency_ms"],
                detail={
                    "attempts": result.attempts, "retries": result.retries,
                    "errors": result.errors[-3:],
                    "validation_status": result.validation_status,
                },
            )
        )
        incr("decision.requires_human")
        response = InboundResponse(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            route=DecisionRoute.HUMAN,
            answer=HUMAN_REVIEW_MESSAGE,
            requires_human=True,
            human_review_reason=(
                f"Model output failed schema validation after {result.retries} retries; "
                "no answer was produced rather than an unvalidated one."
            ),
            laya=laya_env,
            policy_sections=retrieved,
            policy_version=policy.version,
            llm={
                "status": "invalid_after_retries",
                "attempts": result.attempts,
                "retries": result.retries,
            },
            trace=trace,
        )
        _record_routing(session, laya_env, DecisionRoute.HUMAN, request.ticket_id, injection_flags)
        _audit_inbound(session, request, response, "inbound_llm_validation_failed")
        response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response

    answer_obj: PolicyAnswer = result.value
    trace.append(
        DecisionTrace(
            stage="nemotron", outcome="answered", latency_ms=t["latency_ms"],
            detail={
                "model": result.usage.model,
                "prompt_version": result.usage.prompt_version,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_inr": round(result.usage.cost_inr, 6),
                "retries": result.retries,
                "validation_status": result.validation_status,
            },
        )
    )

    # ---- 9. grounding check: refuse to present a guess as policy ----------
    if not answer_obj.answered_from_policy:
        trace.append(
            DecisionTrace(
                stage="policy_validation", outcome="not_grounded",
                detail={
                    "reason": (
                        "the model reported that the retrieved policy sections do not contain "
                        "the answer, so it is escalated rather than presented as policy"
                    )
                },
            )
        )
        incr("decision.requires_human")
        response = InboundResponse(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            route=DecisionRoute.HUMAN,
            answer=HUMAN_REVIEW_MESSAGE,
            requires_human=True,
            human_review_reason="Answer is not supported by the written policy.",
            laya=laya_env,
            policy_sections=retrieved,
            policy_version=policy.version,
            llm={"answered_from_policy": False, "confidence": answer_obj.confidence},
            trace=trace,
        )
        _record_routing(session, laya_env, DecisionRoute.HUMAN, request.ticket_id, injection_flags)
        _audit_inbound(session, request, response, "inbound_answer_not_grounded")
        response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response

    trace.append(
        DecisionTrace(
            stage="policy_validation", outcome="grounded",
            detail={
                "cited_sections": answer_obj.policy_section_ids,
                "confidence": answer_obj.confidence,
            },
        )
    )

    response = InboundResponse(
        request_id=current_request_id(),
        trace_id=current_trace_id(),
        route=DecisionRoute.NEMOTRON,
        answer=answer_obj.answer,
        laya=laya_env,
        policy_sections=retrieved,
        policy_version=policy.version,
        llm={
            "model": result.usage.model,
            "prompt_version": result.usage.prompt_version,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "latency_ms": result.usage.latency_ms,
            "cost_inr": round(result.usage.cost_inr, 6),
            "retries": result.retries,
            "validation_status": result.validation_status,
            "confidence": answer_obj.confidence,
        },
        trace=trace,
    )
    if cacheable:
        cache.set(
            scope, message,
            {
                "answer": answer_obj.answer,
                "policy_sections": retrieved,
                "route": "nemotron",
                "confidence": answer_obj.confidence,
            },
            ttl=settings.CACHE_TTL,
        )
    _record_routing(session, laya_env, DecisionRoute.NEMOTRON, request.ticket_id, injection_flags)
    _audit_inbound(session, request, response, "inbound_nemotron_answer")
    response.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return response


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------
def handle_outbound(
    session: Session,
    request: OutboundActionRequest,
    idempotency_key: str | None = None,
) -> OutboundActionResponse:
    trace: list[DecisionTrace] = []
    policy = get_policy()
    payload = request.model_dump(mode="json")

    # ---- 0. idempotency: before anything can execute ----------------------
    replay = None
    if idempotency_key:
        with timed("outbound.idempotency") as t:
            replay = idempotency.claim(
                session, scope=f"outbound:{request.action}", key=idempotency_key, payload=payload
            )
        trace.append(
            DecisionTrace(
                stage="idempotency",
                outcome="replay" if replay.is_replay else "claimed",
                latency_ms=t["latency_ms"],
                detail={
                    "key": idempotency_key,
                    "store": replay.store,
                    "status": replay.status,
                    "note": (
                        "PostgreSQL is authoritative for idempotency so a Redis flush cannot "
                        "allow a refund to execute twice."
                    ),
                },
            )
        )
        if replay.is_replay and replay.response:
            stored = OutboundActionResponse.model_validate(replay.response)
            stored.idempotent_replay = True
            stored.trace = trace + list(stored.trace)
            incr("outbound.idempotent_replay")
            return stored
        if replay.is_replay and not replay.response:
            incr("outbound.idempotent_in_flight")
            return OutboundActionResponse(
                request_id=current_request_id(),
                idempotency_key=idempotency_key,
                idempotent_replay=True,
                final_decision=PolicyOutcome.REQUIRES_HUMAN_REVIEW,
                executed=False,
                risk_level=RiskLevel.MEDIUM,
                requires_human=True,
                policy_version=policy.version,
                reason=(
                    "An identical request is already in flight under this idempotency key. "
                    "The action was not executed a second time."
                ),
                trace=trace,
            )

    # ---- 1. Laya ---------------------------------------------------------
    laya = get_laya_client()
    action_text = (
        f"Support agent proposes to {request.action.replace('_', ' ')}"
        + (f" of Rs {request.amount_inr:,.0f}" if request.amount_inr else "")
        + (f" for reason {request.reason_code}" if request.reason_code else "")
        + (f". Justification: {request.justification}" if request.justification else "")
    )
    injection_flags = scan_injection(f"{request.justification} {request.reason_code or ''}")
    with timed("outbound.laya") as t:
        laya_env = laya.decide(action_text, include_guard=True, has_customer_context=True)
    trace.append(
        DecisionTrace(
            stage="laya",
            outcome=f"risk={laya_env.decision.risk_level} requires_human={laya_env.decision.requires_human}",
            latency_ms=t["latency_ms"],
            detail={
                **laya_env.decision.model_dump(),
                "source": laya_env.source,
                "degraded": laya_env.degraded,
                "confidence": laya_env.confidence,
            },
        )
    )
    incr("decision.total")

    # ---- 2. deterministic context lookups --------------------------------
    with timed("outbound.context") as t:
        context = _build_action_context(session, request, injection_flags)
    trace.append(
        DecisionTrace(
            stage="context_lookup",
            outcome="loaded",
            latency_ms=t["latency_ms"],
            detail={
                "agent_tier": context.agent_tier,
                "agent_team": context.agent_team,
                "order_value_inr": context.order_value_inr,
                "existing_refunds_for_order": context.existing_refund_count_for_order,
                "existing_refund_value_for_order": context.existing_refund_value_for_order,
                "replacement_already_issued_for_order": context.replacement_already_issued_for_order,
                "source": "PostgreSQL only; no model consulted for these facts",
            },
        )
    )

    # ---- 3. policy engine ------------------------------------------------
    with timed("outbound.policy_engine") as t:
        evaluation = evaluate_outbound_action(request, context, laya_env)
    trace.append(
        DecisionTrace(
            stage="policy_engine",
            outcome=str(evaluation.final_decision),
            latency_ms=t["latency_ms"],
            detail={
                "checks": [c.model_dump() for c in evaluation.checks],
                "breached_rules": evaluation.breached_rules(),
                "laya_overruled": evaluation.laya_overruled,
                "laya_overrule_detail": evaluation.laya_overrule_detail,
                "fail_closed": evaluation.fail_closed,
                "hierarchy": evaluation.decision_hierarchy,
            },
        )
    )

    # ---- 4. execute or withhold ------------------------------------------
    executed = False
    if evaluation.final_decision == PolicyOutcome.ALLOW:
        executed = _execute_action(session, request)
        trace.append(
            DecisionTrace(
                stage="action_executor",
                outcome="executed" if executed else "execution_failed",
                detail={
                    "action": str(request.action),
                    "amount_inr": request.amount_inr,
                    "note": (
                        "This platform records the authorised action in the audit trail. It "
                        "does not call a payment gateway: no such integration exists here, and "
                        "pretending otherwise would be a fake integration."
                    ),
                },
            )
        )
        incr("outbound.executed")
    else:
        trace.append(
            DecisionTrace(
                stage="action_executor",
                outcome="withheld",
                detail={
                    "decision": str(evaluation.final_decision),
                    "approver_required": evaluation.approver_required,
                },
            )
        )
        incr("outbound.withheld")
        if evaluation.requires_human:
            incr("decision.requires_human")

    # ---- 5. persist ------------------------------------------------------
    decision_row = PolicyDecision(
        request_id=current_request_id(),
        trace_id=current_trace_id(),
        ticket_id=request.ticket_id,
        idempotency_key=idempotency_key,
        action=str(request.action),
        amount_inr=request.amount_inr,
        reason_code=request.reason_code,
        requested_by=request.requested_by or None,
        approved_by=request.approved_by,
        final_decision=str(evaluation.final_decision),
        executed=executed,
        risk_level=str(evaluation.risk_level),
        requires_human=evaluation.requires_human,
        approver_required=evaluation.approver_required,
        checks=[c.model_dump() for c in evaluation.checks],
        policy_version=evaluation.policy_version,
        policy_section_ids=evaluation.policy_section_ids,
        decision_hierarchy=evaluation.decision_hierarchy,
        laya_decision=laya_env.to_audit(),
        laya_overruled=evaluation.laya_overruled,
        reason=evaluation.reason,
        fail_closed=evaluation.fail_closed,
    )
    session.add(decision_row)
    session.flush()

    audit = AuditEvent(
        request_id=current_request_id(),
        trace_id=current_trace_id(),
        event_type=f"outbound_{request.action}",
        entity_type="ticket",
        entity_id=request.ticket_id,
        actor=request.requested_by or "unknown",
        actor_type="agent",
        outcome=str(evaluation.final_decision),
        policy_version=evaluation.policy_version,
        payload={
            "action": str(request.action),
            "amount_inr": request.amount_inr,
            "reason_code": request.reason_code,
            "order_id": request.order_id,
            "executed": executed,
            "risk_level": str(evaluation.risk_level),
            "breached_rules": evaluation.breached_rules(),
            "laya_overruled": evaluation.laya_overruled,
            "fail_closed": evaluation.fail_closed,
            "approver_required": evaluation.approver_required,
            "approved_by": request.approved_by,
            "idempotency_key": idempotency_key,
            "policy_decision_id": decision_row.id,
        },
    )
    session.add(audit)
    session.flush()

    _record_routing(
        session, laya_env,
        DecisionRoute.HUMAN if evaluation.requires_human else DecisionRoute.DETERMINISTIC,
        request.ticket_id, injection_flags,
        overruled=evaluation.laya_overruled,
        overrule_detail=evaluation.laya_overrule_detail,
    )

    response = OutboundActionResponse(
        request_id=current_request_id(),
        idempotency_key=idempotency_key,
        idempotent_replay=False,
        final_decision=evaluation.final_decision,
        executed=executed,
        risk_level=evaluation.risk_level,
        requires_human=evaluation.requires_human,
        approver_required=evaluation.approver_required,
        laya=laya_env,
        policy_checks=evaluation.checks,
        policy_version=evaluation.policy_version,
        decision_hierarchy=evaluation.decision_hierarchy,
        reason=evaluation.reason,
        audit_event_id=audit.id,
        trace=trace,
    )

    if idempotency_key:
        idempotency.complete(
            session, idempotency_key,
            response.model_dump(mode="json"),
            policy_decision_id=decision_row.id,
        )

    log.info(
        "outbound.decided",
        extra={
            "action": str(request.action),
            "ticket_id": request.ticket_id,
            "decision": str(evaluation.final_decision),
            "executed": executed,
            "risk": str(evaluation.risk_level),
            "laya_overruled": evaluation.laya_overruled,
            "fail_closed": evaluation.fail_closed,
        },
    )
    return response


def _build_action_context(
    session: Session, request: OutboundActionRequest, injection_flags: list[str]
) -> ActionContext:
    agent_tier = agent_team = None
    if request.requested_by:
        agent = session.execute(
            select(Agent).where(Agent.agent_id == request.requested_by)
        ).scalars().first()
        if agent is not None:
            agent_tier, agent_team = agent.tier, agent.team

    order_value = None
    existing_count = 0
    existing_value = 0.0
    replacement_issued = False

    if request.order_id:
        order = session.execute(
            select(Order).where(Order.order_id == request.order_id)
        ).scalar_one_or_none()
        if order is not None:
            order_value = float(order.order_value_inr)

        prior = session.execute(
            select(Refund).where(Refund.order_id == request.order_id)
        ).scalars().all()
        prior = [p for p in prior if p.ticket_id != request.ticket_id]
        existing_count = len(prior)
        existing_value = float(sum(p.amount_inr for p in prior))
        replacement_issued = any(p.replacement_issued_flag for p in prior)

    try:
        get_policy()
        policy_available = True
    except Exception:
        policy_available = False

    return ActionContext(
        agent_tier=agent_tier,
        agent_team=agent_team,
        order_value_inr=order_value,
        existing_refund_count_for_order=existing_count,
        existing_refund_value_for_order=existing_value,
        replacement_already_issued_for_order=replacement_issued,
        injection_flags=injection_flags,
        policy_available=policy_available,
    )


def _execute_action(session: Session, request: OutboundActionRequest) -> bool:
    """Record an authorised action.

    There is no payment gateway behind this. The platform's job is the decision
    and the audit record; wiring a fake external call would be exactly the "fake
    integration" the brief rules out.
    """
    session.add(
        AuditEvent(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            event_type="action_executed",
            entity_type="ticket",
            entity_id=request.ticket_id,
            actor=request.approved_by or request.requested_by or "system",
            actor_type="human" if request.approved_by else "agent",
            outcome="executed",
            payload={
                "action": str(request.action),
                "amount_inr": request.amount_inr,
                "reason_code": request.reason_code,
                "executor": "audit_record_only (no external payment integration exists)",
            },
        )
    )
    return True


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------
def _record_routing(
    session: Session,
    laya_env,
    route: DecisionRoute,
    ticket_id: str | None,
    injection_flags: list[str],
    overruled: bool = False,
    overrule_detail: str | None = None,
) -> None:
    d = laya_env.decision
    session.add(
        RoutingDecision(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            ticket_id=ticket_id,
            direction=str(d.direction),
            intent=str(d.intent),
            complexity=str(d.complexity),
            requires_llm=bool(d.requires_llm),
            requires_policy=bool(d.requires_policy),
            requires_human=bool(d.requires_human),
            cacheable=bool(d.cacheable),
            risk_level=str(d.risk_level),
            confidence={k: v for k, v in laya_env.confidence.items() if v is not None},
            probabilities=laya_env.probabilities,
            guard=laya_env.guard.model_dump() if laya_env.guard else {},
            low_confidence_fields=laya_env.low_confidence_fields,
            laya_source=laya_env.source,
            laya_model=laya_env.model_name,
            laya_calibrated=laya_env.calibrated,
            degraded=laya_env.degraded,
            degraded_reason=laya_env.degraded_reason,
            latency_ms=laya_env.latency_ms,
            chosen_route=str(route),
            overruled_by_policy=overruled,
            overruled_detail=overrule_detail,
            injection_flags=injection_flags,
        )
    )
    session.flush()


def _audit_inbound(
    session: Session, request: InboundRequest, response: InboundResponse, event_type: str
) -> None:
    """Audit an inbound decision.

    The customer's message is NOT stored: only its length and a note of which
    route served it. The message already lives in the helpdesk, and copying it
    into the audit log would spread PII for no added traceability.
    """
    session.add(
        AuditEvent(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            event_type=event_type,
            entity_type="ticket" if request.ticket_id else "inbound_request",
            entity_id=request.ticket_id or current_request_id(),
            actor="customer" if not request.ticket_id else "agent",
            actor_type="external",
            outcome=str(response.route),
            policy_version=response.policy_version,
            payload={
                "route": str(response.route),
                "requires_human": response.requires_human,
                "human_review_reason": response.human_review_reason,
                "message_chars": len(request.message),
                "intent": str(response.laya.decision.intent) if response.laya else None,
                "risk_level": str(response.laya.decision.risk_level) if response.laya else None,
                "laya_degraded": response.laya.degraded if response.laya else None,
                "cache": response.cache,
                "injection_flags": response.injection_flags,
                "policy_sections": [s.get("policy_section_id") for s in response.policy_sections],
                "llm": response.llm,
            },
        )
    )
    session.flush()
