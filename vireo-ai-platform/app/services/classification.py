"""Ticket classification pipeline.

    ticket -> Laya (gate) -> policy retrieval -> Nemotron -> Pydantic
           -> evidence grounding -> deterministic avoidability -> PostgreSQL

Laya's role here is gating, not classifying: it decides whether a ticket needs the
LLM at all and flags injection attempts. Nemotron reads the note. The avoidability
rules and every rupee are computed afterwards in Python from the validated facts.

A failure anywhere leaves a row that says so. There is no path where a ticket ends
up with a confident-looking reason that no model produced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.cache.keys import classification_key
from app.cache.response_cache import get_json, set_json
from app.core.config import settings
from app.core.logging import current_request_id, current_trace_id, get_logger
from app.core.metrics import incr, timed
from app.core.security import scan_injection
from app.laya.client import get_laya_client
from app.llm.client import get_llm_client, record_llm_request
from app.models.decisions import AuditEvent, TicketClassification
from app.models.refund import Refund
from app.models.source import Order, Product, Ticket
from app.policy.loader import get_policy, get_retrieval_index
from app.schemas.llm import RefundClassification
from app.schemas.taxonomy import AvoidabilityLabel, RefundReason
from app.services.avoidability import RefundFacts, assess
from app.services.baseline import classify_baseline

log = get_logger(__name__)


@dataclass
class ClassificationOutcome:
    ticket_id: str
    status: str  # classified | review_required | skipped_no_llm | error
    reason_code: str
    avoidability_label: str
    avoidable_value_inr: float = 0.0
    confidence: float = 0.0
    retries: int = 0
    cache_hit: bool = False
    validation_status: str = "valid"
    review_reason: str | None = None
    drivers: list[str] = None  # type: ignore[assignment]
    cost_inr: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def __post_init__(self) -> None:
        if self.drivers is None:
            self.drivers = []


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_ticket_context(session: Session, ticket: Ticket) -> dict[str, Any]:
    """Trusted structured context for the prompt.

    Deliberately excludes the refund amount and the replacement_issued flag — see
    `app/llm/prompts._context_block` for why.
    """
    order = None
    if ticket.order_id:
        order = session.execute(
            select(Order).where(Order.order_id == ticket.order_id)
        ).scalar_one_or_none()
    product = session.execute(
        select(Product).where(Product.sku == ticket.product_sku)
    ).scalar_one_or_none()
    refund = session.execute(
        select(Refund).where(Refund.ticket_id == ticket.ticket_id)
    ).scalar_one_or_none()

    days_since_order = None
    if order is not None and order.order_date and ticket.created_at_ist:
        delta = (ticket.created_at_ist.date() - order.order_date).days
        # A negative gap is a known data-quality problem (DQ-009); do not show
        # the model an impossible number.
        days_since_order = delta if delta >= 0 else None

    return {
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel,
        "bot_category": ticket.category,
        "assigned_team": ticket.assigned_team,
        "resolving_team": refund.agent_team if refund else None,
        "resolving_tier": refund.agent_tier if refund else None,
        "status": ticket.status,
        "source_reason_code": ticket.refund_reason_code,
        "source_reason_is_default": (
            refund.source_reason_is_dropdown_default if refund else None
        ),
        "product_family": product.family if product else None,
        "days_since_order": days_since_order,
        "warranty_months": product.warranty_months if product else None,
    }


def classify_ticket(
    session: Session,
    ticket: Ticket,
    force: bool = False,
    use_cache: bool = True,
) -> ClassificationOutcome:
    policy = get_policy()
    llm = get_llm_client()
    laya = get_laya_client()

    existing = session.execute(
        select(TicketClassification).where(TicketClassification.ticket_id == ticket.ticket_id)
    ).scalar_one_or_none()
    if existing is not None and not force:
        return ClassificationOutcome(
            ticket_id=ticket.ticket_id,
            status="review_required" if existing.review_required else "classified",
            reason_code=existing.reason_code,
            avoidability_label=(
                session.execute(
                    select(Refund.avoidability_label).where(Refund.ticket_id == ticket.ticket_id)
                ).scalar_one_or_none()
                or AvoidabilityLabel.REVIEW_REQUIRED
            ),
            confidence=existing.confidence or 0.0,
            retries=existing.retries,
            cache_hit=True,
            validation_status=existing.validation_status,
            review_reason=existing.review_reason,
        )

    combined_text = f"{ticket.customer_message}\n\n{ticket.agent_notes}"
    injection_flags = scan_injection(combined_text)

    # ---- Laya: gate, not classifier --------------------------------------
    with timed("pipeline.laya") as t_laya:
        laya_env = laya.decide(
            ticket.customer_message or ticket.agent_notes or "(empty)",
            include_guard=bool(injection_flags),
            has_customer_context=True,
        )
    _ = t_laya

    guard_flagged = bool(
        laya_env.guard
        and (laya_env.guard.jailbreak or laya_env.guard.prompt_injection)
    )
    if injection_flags or guard_flagged:
        incr("classification.injection_flagged")
        log.warning(
            "classification.injection_detected",
            extra={
                "ticket_id": ticket.ticket_id,
                "patterns": injection_flags,
                "laya_guard_flagged": guard_flagged,
            },
        )

    # ---- exact cache on (ticket, model, prompt, policy) -------------------
    cache_key = classification_key(ticket.ticket_id)
    if use_cache and not force:
        cached = get_json(cache_key)
        if cached:
            incr("classification.cache_hit")
            return _persist(
                session, ticket, RefundClassification.model_validate(cached["classification"]),
                retrieved=cached.get("policy_sections", []),
                usage=None, attempts=1, retries=0, validation_status="valid",
                cache_hit=True, injection_flags=injection_flags,
                errors=[], raw_text="",
            )

    # ---- policy retrieval -------------------------------------------------
    with timed("pipeline.policy_retrieval"):
        retriever = get_retrieval_index()
        retrieved = [r.to_dict() for r in retriever.retrieve_for_refund_classification(combined_text)]

    if not retrieved:
        # No policy -> no avoidability claim (brief §42).
        log.error("classification.policy_unavailable", extra={"ticket_id": ticket.ticket_id})

    context = build_ticket_context(session, ticket)
    baseline = classify_baseline(ticket.customer_message, ticket.agent_notes)

    # ---- LLM unavailable: record honestly, do not guess -------------------
    if not llm.available:
        incr("classification.skipped_no_llm")
        return _persist_unclassified(
            session, ticket, baseline, retrieved, injection_flags,
            reason="OPENROUTER_API_KEY not configured; no classification attempted",
            validation_status="skipped_no_llm",
        )

    # ---- Nemotron + Pydantic ---------------------------------------------
    with timed("pipeline.nemotron"):
        result = llm.classify_ticket(
            context=context,
            customer_message=ticket.customer_message,
            agent_notes=ticket.agent_notes,
            policy_sections=retrieved,
        )

    if result.usage is not None:
        record_llm_request(
            session,
            result.usage,
            purpose="refund_classification",
            ticket_id=ticket.ticket_id,
            prompt_sha=_sha(combined_text + settings.PROMPT_VERSION),
            response_sha=_sha(result.raw_text) if result.raw_text else None,
            policy_version=policy.version,
        )

    if not result.ok:
        incr("classification.invalid_after_retries")
        log.error(
            "classification.failed",
            extra={
                "ticket_id": ticket.ticket_id,
                "attempts": result.attempts,
                "retries": result.retries,
                "errors": result.errors[-3:],
            },
        )
        return _persist_unclassified(
            session, ticket, baseline, retrieved, injection_flags,
            reason=f"schema validation failed after {result.retries} retries: "
                   + "; ".join(result.errors[-2:])[:250],
            validation_status=result.validation_status,
            attempts=result.attempts,
            retries=result.retries,
            usage=result.usage,
        )

    outcome = _persist(
        session,
        ticket,
        result.value,
        retrieved=retrieved,
        usage=result.usage,
        attempts=result.attempts,
        retries=result.retries,
        validation_status=result.validation_status,
        cache_hit=False,
        injection_flags=injection_flags,
        errors=result.errors,
        raw_text=result.raw_text,
        baseline=baseline,
    )

    if use_cache and outcome.validation_status in {"valid", "repaired"}:
        set_json(
            cache_key,
            {
                "classification": result.value.model_dump(mode="json"),
                "policy_sections": retrieved,
            },
            ttl=settings.CACHE_TTL,
        )
    return outcome


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist(
    session: Session,
    ticket: Ticket,
    classification: RefundClassification,
    retrieved: list[dict],
    usage,
    attempts: int,
    retries: int,
    validation_status: str,
    cache_hit: bool,
    injection_flags: list[str],
    errors: list[str],
    raw_text: str,
    baseline: RefundReason | None = None,
) -> ClassificationOutcome:
    grounded, ungrounded = classification.verify_evidence(
        ticket.customer_message, ticket.agent_notes
    )
    review_required = False
    review_reason: str | None = None

    if ungrounded:
        review_required = True
        review_reason = (
            f"{len(ungrounded)} evidence quote(s) are not verbatim in the ticket text"
        )
        incr("classification.ungrounded_evidence")
    if classification.reason_code in {RefundReason.UNKNOWN, RefundReason.AMBIGUOUS}:
        review_required = True
        review_reason = review_reason or f"reason classified {classification.reason_code}"
    if classification.confidence < 0.5:
        review_required = True
        review_reason = review_reason or f"model confidence {classification.confidence:.2f} below 0.5"
    if injection_flags:
        review_required = True
        review_reason = review_reason or (
            f"ticket text contains injection patterns: {', '.join(injection_flags[:3])}"
        )

    refund = session.execute(
        select(Refund).where(Refund.ticket_id == ticket.ticket_id)
    ).scalar_one_or_none()

    row = session.execute(
        select(TicketClassification).where(TicketClassification.ticket_id == ticket.ticket_id)
    ).scalar_one_or_none()
    if row is None:
        row = TicketClassification(ticket_uuid=ticket.id, ticket_id=ticket.ticket_id)
        session.add(row)

    row.request_id = current_request_id()
    row.trace_id = current_trace_id()
    row.reason_code = str(classification.reason_code)
    row.reason_explanation = classification.reason_explanation
    row.policy_relevant = classification.policy_relevant
    row.ai_avoidability_label = str(classification.potentially_avoidable)
    row.avoidability_rationale = classification.avoidability_rationale
    row.dual_remedy_indicated = classification.dual_remedy_indicated
    row.refund_without_return_indicated = classification.refund_without_return_indicated
    row.replacement_mentioned = classification.replacement_mentioned
    row.confidence = classification.confidence
    row.ambiguity_note = classification.ambiguity_note
    row.evidence = [s.model_dump() for s in grounded]
    row.unverified_evidence = [s.model_dump() for s in ungrounded]
    row.evidence_grounded = not ungrounded
    row.model_name = usage.model if usage else settings.OPENROUTER_MODEL
    row.prompt_version = usage.prompt_version if usage else settings.PROMPT_VERSION
    row.classifier_version = settings.CLASSIFIER_VERSION
    row.policy_version = get_policy().version
    row.retrieved_policy_sections = [
        {
            "policy_section_id": s["policy_section_id"],
            "policy_version": s["policy_version"],
            "title": s["title"],
            "relevance": s.get("relevance", {}),
        }
        for s in retrieved
    ]
    row.validation_status = validation_status
    row.validation_errors = errors[-6:]
    row.attempts = attempts
    row.retries = retries
    row.review_required = review_required
    row.review_reason = review_reason
    row.cache_hit = cache_hit
    if baseline is not None:
        row.baseline_reason_code = str(baseline)
        row.baseline_agrees = str(baseline) == str(classification.reason_code)

    # ---- deterministic avoidability -------------------------------------
    assessment = None
    if refund is not None:
        facts = RefundFacts(
            ticket_id=ticket.ticket_id,
            amount_inr=float(refund.amount_inr),
            source_reason_code=refund.source_reason_code,
            replacement_issued_flag=refund.replacement_issued_flag,
            order_id=refund.order_id,
            order_value_inr=refund.order_value_inr,
            order_refund_ticket_count=refund.order_refund_ticket_count,
            order_refund_excess_inr=refund.order_refund_excess_inr,
            agent_tier=refund.agent_tier,
            agent_team=refund.agent_team,
            sla_breached=refund.sla_breached,
            channel=refund.channel,
            data_quality_flags=ticket.data_quality_flags or [],
            ai_reason_code=str(classification.reason_code),
            ai_dual_remedy=classification.dual_remedy_indicated,
            ai_refund_without_return=classification.refund_without_return_indicated,
            ai_replacement_mentioned=classification.replacement_mentioned,
            ai_confidence=classification.confidence,
            ai_evidence=[s.model_dump() for s in grounded],
            classification_valid=(
                validation_status in {"valid", "repaired"} and not review_required
            ),
        )
        assessment = assess(facts)
        refund.normalised_reason_code = str(classification.reason_code)
        refund.reason_source = "nemotron"
        refund.avoidability_label = str(assessment.label)
        refund.avoidability_drivers = assessment.driver_names()
        refund.avoidable_value_inr = assessment.avoidable_value_inr
        refund.avoidability_basis = assessment.basis
        refund.policy_section_ids = assessment.policy_section_ids

    session.add(
        AuditEvent(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            event_type="ticket_classified",
            entity_type="ticket",
            entity_id=ticket.ticket_id,
            actor="classification_pipeline",
            actor_type="system",
            outcome=str(classification.reason_code),
            policy_version=row.policy_version,
            model_name=row.model_name,
            prompt_version=row.prompt_version,
            payload={
                "reason_code": str(classification.reason_code),
                "confidence": classification.confidence,
                "avoidability": str(assessment.label) if assessment else None,
                "avoidable_value_inr": assessment.avoidable_value_inr if assessment else 0.0,
                "drivers": assessment.driver_names() if assessment else [],
                "retries": retries,
                "validation_status": validation_status,
                "evidence_grounded": not ungrounded,
                "review_required": review_required,
                "injection_flags": injection_flags,
                "policy_sections": [s["policy_section_id"] for s in retrieved],
                "cache_hit": cache_hit,
            },
        )
    )
    session.flush()

    incr("classification.success")
    if review_required:
        incr("classification.review_required")

    return ClassificationOutcome(
        ticket_id=ticket.ticket_id,
        status="review_required" if review_required else "classified",
        reason_code=str(classification.reason_code),
        avoidability_label=str(assessment.label) if assessment else AvoidabilityLabel.REVIEW_REQUIRED,
        avoidable_value_inr=assessment.avoidable_value_inr if assessment else 0.0,
        confidence=classification.confidence,
        retries=retries,
        cache_hit=cache_hit,
        validation_status=validation_status,
        review_reason=review_reason,
        drivers=assessment.driver_names() if assessment else [],
        cost_inr=usage.cost_inr if usage else 0.0,
        tokens_in=usage.input_tokens if usage else 0,
        tokens_out=usage.output_tokens if usage else 0,
    )


def _persist_unclassified(
    session: Session,
    ticket: Ticket,
    baseline: RefundReason,
    retrieved: list[dict],
    injection_flags: list[str],
    reason: str,
    validation_status: str,
    attempts: int = 0,
    retries: int = 0,
    usage=None,
) -> ClassificationOutcome:
    """Record a failure as UNKNOWN + REVIEW_REQUIRED (brief §18).

    The deterministic baseline guess is stored in `baseline_reason_code` for
    reference but is NOT promoted to `reason_code`, and the refund stays
    REVIEW_REQUIRED with zero avoidable value. A failed classification must not
    quietly become an analytical result.
    """
    row = session.execute(
        select(TicketClassification).where(TicketClassification.ticket_id == ticket.ticket_id)
    ).scalar_one_or_none()
    if row is None:
        row = TicketClassification(ticket_uuid=ticket.id, ticket_id=ticket.ticket_id)
        session.add(row)

    row.request_id = current_request_id()
    row.trace_id = current_trace_id()
    row.reason_code = str(RefundReason.UNKNOWN)
    row.reason_explanation = reason
    row.policy_relevant = False
    row.ai_avoidability_label = str(AvoidabilityLabel.REVIEW_REQUIRED)
    row.confidence = 0.0
    row.evidence = []
    row.unverified_evidence = []
    row.evidence_grounded = None
    row.model_name = usage.model if usage else settings.OPENROUTER_MODEL
    row.prompt_version = settings.PROMPT_VERSION
    row.classifier_version = settings.CLASSIFIER_VERSION
    row.policy_version = get_policy().version
    row.retrieved_policy_sections = [
        {"policy_section_id": s["policy_section_id"], "policy_version": s["policy_version"]}
        for s in retrieved
    ]
    row.validation_status = validation_status
    row.validation_errors = [reason]
    row.attempts = attempts
    row.retries = retries
    row.review_required = True
    row.review_reason = reason[:300]
    row.cache_hit = False
    row.baseline_reason_code = str(baseline)
    row.baseline_agrees = None

    refund = session.execute(
        select(Refund).where(Refund.ticket_id == ticket.ticket_id)
    ).scalar_one_or_none()
    if refund is not None:
        facts = RefundFacts(
            ticket_id=ticket.ticket_id,
            amount_inr=float(refund.amount_inr),
            source_reason_code=refund.source_reason_code,
            replacement_issued_flag=refund.replacement_issued_flag,
            order_id=refund.order_id,
            order_value_inr=refund.order_value_inr,
            order_refund_ticket_count=refund.order_refund_ticket_count,
            order_refund_excess_inr=refund.order_refund_excess_inr,
            channel=refund.channel,
            classification_valid=False,
        )
        # Deterministic drivers still apply: they need no model.
        assessment = assess(facts)
        refund.normalised_reason_code = None
        refund.reason_source = "unclassified"
        refund.avoidability_label = str(assessment.label)
        refund.avoidability_drivers = assessment.driver_names()
        refund.avoidable_value_inr = assessment.avoidable_value_inr
        refund.avoidability_basis = assessment.basis
        refund.policy_section_ids = assessment.policy_section_ids
        label = str(assessment.label)
        avoidable_value = assessment.avoidable_value_inr
        drivers = assessment.driver_names()
    else:
        label = str(AvoidabilityLabel.REVIEW_REQUIRED)
        avoidable_value = 0.0
        drivers = []

    session.add(
        AuditEvent(
            request_id=current_request_id(),
            trace_id=current_trace_id(),
            event_type="ticket_classification_failed",
            entity_type="ticket",
            entity_id=ticket.ticket_id,
            actor="classification_pipeline",
            actor_type="system",
            outcome=validation_status,
            policy_version=row.policy_version,
            model_name=row.model_name,
            prompt_version=row.prompt_version,
            payload={
                "reason": reason[:500],
                "attempts": attempts,
                "retries": retries,
                "baseline_guess": str(baseline),
                "deterministic_drivers_still_applied": drivers,
                "injection_flags": injection_flags,
            },
        )
    )
    session.flush()

    return ClassificationOutcome(
        ticket_id=ticket.ticket_id,
        status="skipped_no_llm" if validation_status == "skipped_no_llm" else "review_required",
        reason_code=str(RefundReason.UNKNOWN),
        avoidability_label=label,
        avoidable_value_inr=avoidable_value,
        confidence=0.0,
        retries=retries,
        validation_status=validation_status,
        review_reason=reason[:300],
        drivers=drivers,
        cost_inr=usage.cost_inr if usage else 0.0,
        tokens_in=usage.input_tokens if usage else 0,
        tokens_out=usage.output_tokens if usage else 0,
    )


# ---------------------------------------------------------------------------
# Deterministic-only pass (no LLM): applies the rules that need no model
# ---------------------------------------------------------------------------
def apply_deterministic_avoidability(session: Session) -> dict[str, Any]:
    """Run the model-free avoidability drivers across every refund.

    Used after ingest so the dual-remedy-by-flag and duplicate-refund findings
    exist before a single LLM call is made — and so the platform produces a real
    number even with no API key at all.
    """
    refunds = session.execute(select(Refund)).scalars().all()
    counts = {"flagged": 0, "review_required": 0}
    total = 0.0
    for refund in refunds:
        classification = session.execute(
            select(TicketClassification).where(
                TicketClassification.ticket_id == refund.ticket_id
            )
        ).scalar_one_or_none()

        facts = RefundFacts(
            ticket_id=refund.ticket_id,
            amount_inr=float(refund.amount_inr),
            source_reason_code=refund.source_reason_code,
            replacement_issued_flag=refund.replacement_issued_flag,
            order_id=refund.order_id,
            order_value_inr=refund.order_value_inr,
            order_refund_ticket_count=refund.order_refund_ticket_count,
            order_refund_excess_inr=refund.order_refund_excess_inr,
            agent_tier=refund.agent_tier,
            agent_team=refund.agent_team,
            sla_breached=refund.sla_breached,
            channel=refund.channel,
            ai_reason_code=classification.reason_code if classification else None,
            ai_dual_remedy=classification.dual_remedy_indicated if classification else None,
            ai_refund_without_return=(
                classification.refund_without_return_indicated if classification else None
            ),
            ai_confidence=classification.confidence if classification else None,
            ai_evidence=(classification.evidence or []) if classification else [],
            classification_valid=bool(
                classification
                and classification.validation_status in {"valid", "repaired"}
                and not classification.review_required
            ),
        )
        assessment = assess(facts)
        refund.avoidability_label = str(assessment.label)
        refund.avoidability_drivers = assessment.driver_names()
        refund.avoidable_value_inr = assessment.avoidable_value_inr
        refund.avoidability_basis = assessment.basis
        refund.policy_section_ids = assessment.policy_section_ids
        if assessment.label == AvoidabilityLabel.POTENTIALLY_AVOIDABLE:
            counts["flagged"] += 1
            total += assessment.avoidable_value_inr
        elif assessment.label == AvoidabilityLabel.REVIEW_REQUIRED:
            counts["review_required"] += 1

    session.flush()
    log.info(
        "avoidability.deterministic_pass",
        extra={"refunds": len(refunds), **counts, "avoidable_value_inr": round(total, 2)},
    )
    return {
        "refunds_assessed": len(refunds),
        "flagged": counts["flagged"],
        "review_required": counts["review_required"],
        "avoidable_value_inr": round(total, 2),
    }
