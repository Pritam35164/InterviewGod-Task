"""
AI Classifier for Vireo Audio Support Tickets.
Interprets customer opening messages and agent closing notes using OpenRouter Nemotron.
Outputs structured JSON with evidence, reason codes, confidence, and policy relevance.
"""

from typing import Dict, Any, Optional
from src.llm_client import OpenRouterClient, StructuredResult

SYSTEM_PROMPT = """You are an expert AI customer support analyst for Vireo Audio.
Your job is to read a support ticket's customer message and agent closing note, interpret the underlying refund reason, extract verbatim evidence, and output valid JSON.

Reason Codes (controlled taxonomy):
- PRODUCT_FAULT_DOA: Defective product, dead on arrival, hardware failure within warranty.
- PRODUCT_DAMAGED_TRANSIT: Product damaged or broken during shipping/transit.
- WRONG_ITEM_DELIVERED: Customer received a different item/sku than ordered.
- DELIVERY_ISSUE_LOST: Order undelivered, lost package, extreme delay.
- CUSTOMER_CANCELLATION: Order cancelled by customer prior to dispatch or buyer remorse.
- BILLING_DUPLICATE_PAYMENT: Double charge, failed payment debited, coupon not applied.
- GOODWILL_POLICY_EXCEPTION: Goodwill refund given without defective return, policy exception, CSAT concession.
- WARRANTY_BUYBACK: Out of stock warranty claim buyback or store credit refund.
- OTHER_UNKNOWN: Unclear, missing notes, or unclassifiable.

Return ONLY a valid JSON object with these keys:
{
    "reason_code": "<ONE OF THE REASON CODES ABOVE>",
    "reason_explanation": "<Brief 1-2 sentence explanation of why this code was selected>",
    "evidence": "<Direct quote or key phrase from customer message or agent note>",
    "policy_relevant": <true or false - true if refund or policy issue>,
    "potentially_avoidable": <true or false - true if refund appears preventable under policy rules (e.g. dual remedy, unverified goodwill over cap, customer error)>,
    "dual_remedy_issued": <true or false - true if both replacement product and cash refund were issued>,
    "confidence": <float between 0.0 and 1.0>
}
"""

def classify_ticket(
    ticket_id: str,
    customer_message: str,
    agent_notes: str,
    category: str = "",
    llm_client: Optional[OpenRouterClient] = None
) -> Dict[str, Any]:
    if llm_client is None:
        llm_client = OpenRouterClient()

    if not llm_client.is_available():
        return {
            "ticket_id": ticket_id,
            "reason_code": "UNKNOWN",
            "reason_explanation": "LLM API key not configured.",
            "evidence": "",
            "policy_relevant": False,
            "potentially_avoidable": False,
            "dual_remedy_issued": False,
            "confidence": 0.0,
            "status": "unavailable"
        }

    user_prompt = f"""Ticket ID: {ticket_id}
Category Tag: {category}
Customer Opening Message: {customer_message}
Agent Closing Notes: {agent_notes}
"""

    res: StructuredResult = llm_client.generate_structured(SYSTEM_PROMPT, user_prompt, max_retries=2)

    if res.ok and isinstance(res.value, dict):
        val = res.value
        val["ticket_id"] = ticket_id
        val["status"] = res.validation_status
        val["retries"] = res.retries
        return val

    return {
        "ticket_id": ticket_id,
        "reason_code": "UNKNOWN",
        "reason_explanation": f"LLM parsing failed after retries: {'; '.join(res.errors)}",
        "evidence": "",
        "policy_relevant": False,
        "potentially_avoidable": False,
        "dual_remedy_issued": False,
        "confidence": 0.0,
        "status": "unresolved"
    }
