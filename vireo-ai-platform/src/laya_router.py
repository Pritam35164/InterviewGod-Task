"""
Lightweight Laya Decision & Intent Router for Vireo Audio.
Demonstrates gating, routing, and intent determination prior to calling heavy LLM pipeline.
"""

from typing import Dict, Any

class LayaRouter:
    """
    Lightweight decision router for support tickets.
    Determines ticket complexity, routing intent, policy relevance, and LLM gating.
    """
    def __init__(self):
        pass

    def route_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        refund_amount = float(ticket.get("clean_refund_inr", 0) or ticket.get("refund_amount_inr", 0) or 0)
        replacement = str(ticket.get("replacement_issued", "")).upper() == "Y"
        reason_code = str(ticket.get("refund_reason_code", ""))
        cust_msg = str(ticket.get("customer_message", "")).lower()
        agent_note = str(ticket.get("agent_notes", "")).lower()

        is_refund = refund_amount > 0

        # Gating logic
        needs_llm = False
        requires_human_review = False
        policy_context_required = False

        if is_refund:
            policy_context_required = True
            # Low risk deterministic cases vs high complexity text
            if replacement and refund_amount > 0:
                intent = "DUAL_REMEDY_PROHIBITED_CHECK"
                complexity = "HIGH"
                requires_human_review = True
            elif reason_code == "GW-OTHER" and refund_amount > 500:
                intent = "GOODWILL_CAP_EXCEEDED_CHECK"
                complexity = "MEDIUM"
                needs_llm = True
            elif "delay" in cust_msg or "cancel" in cust_msg:
                intent = "SHIPPING_OR_CANCELLATION"
                complexity = "LOW"
                needs_llm = True
            else:
                intent = "GENERAL_REFUND_CLASSIFICATION"
                complexity = "MEDIUM"
                needs_llm = True
        else:
            intent = "NON_REFUND_GENERAL_QUERY"
            complexity = "LOW"
            needs_llm = False

        return {
            "ticket_id": ticket.get("ticket_id"),
            "intent": intent,
            "complexity": complexity,
            "needs_llm": needs_llm,
            "policy_context_required": policy_context_required,
            "requires_human_review": requires_human_review,
            "routing_destination": "FINANCE_RECONCILIATION" if is_refund else "SUPPORT_QUEUE"
        }
