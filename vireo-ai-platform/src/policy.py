"""
Support Policy v3.2 Definitions and Deterministic Avoidability Engine.
Source of truth: support-policy.pdf
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Official Reason Codes from Policy Section 5
OFFICIAL_POLICY_TAXONOMY = {
    "GW-OTHER": "Goodwill / Other",
    "DOA-REPL": "Dead on arrival, refund chosen",
    "LOST-TRANSIT": "Lost or undelivered",
    "DUP-PAYMENT": "Duplicate or failed payment",
    "CANCEL": "Cancellation before dispatch",
    "PRICE-ADJ": "Price or coupon adjustment",
    "RETURN-QC-OK": "Return received and passed QC",
    "WTY-BUYBACK": "Warranty buy-back"
}

# Standard Cost Constants from Policy Section 4
COST_STANDARDS = {
    "chat_contact_inr": 210.0,
    "email_contact_inr": 260.0,
    "voice_contact_inr": 520.0,
    "social_contact_inr": 240.0,
    "blended_contact_inr": 290.0,
    "transfer_cost_inr": 305.0,
    "agent_hourly_rate_inr": 165.0,
    "sla_breach_credit_inr": 350.0,
    "goodwill_cap_inr": 500.0,
    "shipping_and_handling_inr": 340.0
}

@dataclass
class AvoidabilityAssessment:
    is_avoidable: bool
    avoidable_amount_inr: float
    drivers: List[str]
    policy_clauses: List[str]
    explanation: str

def assess_ticket_avoidability(row: Dict[str, Any], ai_classification: Optional[Dict[str, Any]] = None) -> AvoidabilityAssessment:
    """
    Evaluates policy avoidability deterministically.
    Avoidable amount is computed taking the maximum across applicable drivers (never double counted).
    """
    is_refund = row.get("is_refund", False) or (row.get("clean_refund_inr", 0) > 0)
    refund_val = float(row.get("clean_refund_inr", 0))

    if not is_refund or refund_val <= 0:
        return AvoidabilityAssessment(
            is_avoidable=False,
            avoidable_amount_inr=0.0,
            drivers=[],
            policy_clauses=[],
            explanation="Not a refund ticket."
        )

    drivers = []
    clauses = []
    driver_amounts = []

    # Driver 1: Prohibited Dual Remedy (Policy Section 5)
    # "In no case is a customer to receive both a refund and a replacement for the same order"
    is_repl = str(row.get("replacement_issued", "")).upper() == "Y"
    if ai_classification and ai_classification.get("dual_remedy_issued") is True:
        is_repl = True

    if is_repl:
        drivers.append("DUAL_REMEDY_PROHIBITED")
        clauses.append("§5")
        driver_amounts.append(refund_val)

    # Driver 2: Goodwill Credit Exceeding Rs 500 Cap (Policy Section 5)
    # "Goodwill credits are capped at Rs 500 per ticket and require Team Lead approval."
    reason_code = row.get("refund_reason_code", "")
    if ai_classification:
        norm_reason = ai_classification.get("reason_code", "")
        if norm_reason in ["GOODWILL_POLICY_EXCEPTION", "GW-OTHER"]:
            reason_code = "GW-OTHER"

    if reason_code == "GW-OTHER" and refund_val > 500.0:
        excess = refund_val - 500.0
        drivers.append("GOODWILL_CAP_EXCEEDED")
        clauses.append("§5")
        driver_amounts.append(excess)

    # Driver 3: Duplicate Refund across Tickets for same Order
    # Handled at aggregate order level if order_id has multiple refunds > order value

    # Driver 4: AI identified policy exception / avoidable case
    if ai_classification and ai_classification.get("potentially_avoidable") is True:
        if "AI_POLICY_EXCEPTIONAL" not in drivers and not drivers:
            drivers.append("AI_POLICY_EXCEPTIONAL")
            clauses.append("§5/§6")
            driver_amounts.append(refund_val)

    if driver_amounts:
        # Avoid double-counting: avoidable amount is max across drivers
        max_avoidable = min(max(driver_amounts), refund_val)
        return AvoidabilityAssessment(
            is_avoidable=True,
            avoidable_amount_inr=max_avoidable,
            drivers=drivers,
            policy_clauses=clauses,
            explanation=f"Potentially avoidable under {', '.join(clauses)}: {', '.join(drivers)}"
        )
    else:
        return AvoidabilityAssessment(
            is_avoidable=False,
            avoidable_amount_inr=0.0,
            drivers=[],
            policy_clauses=[],
            explanation="Compliant with support policy."
        )

def compute_dataset_avoidability(df: pd.DataFrame, classifications: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    df_out = df.copy()
    avoidable_flags = []
    avoidable_amounts = []
    driver_lists = []

    for idx, row in df_out.iterrows():
        t_id = row.get("ticket_id")
        ai_info = classifications.get(t_id) if classifications else None
        res = assess_ticket_avoidability(row.to_dict(), ai_info)
        avoidable_flags.append(res.is_avoidable)
        avoidable_amounts.append(res.avoidable_amount_inr)
        driver_lists.append(", ".join(res.drivers))

    df_out["is_potentially_avoidable"] = avoidable_flags
    df_out["avoidable_amount_inr"] = avoidable_amounts
    df_out["avoidability_drivers"] = driver_lists
    return df_out
