"""Deterministic keyword baseline for refund reason classification.

This exists to answer a question the brief implies but does not ask directly: is
the language model worth its cost here, or would regex have done?

It is a genuine attempt, not a strawman — it covers every canonical issue phrase
observed in the notes, handles the common abbreviations ("rfnd", "pkp", "rplc",
"dlvry", "cx"), and is ordered so that specific patterns win over general ones.
What it cannot do is read negation and pragmatics, which is exactly where the
money is: the notes contain 412 distinct spellings of ~45 issues, and
distinguishing "issued refund + replacement both" from "replacement not
applicable (out of stock)" needs more than keywords.

`evaluation.py` scores this baseline on the same held-out set as Nemotron, so the
lift is measured rather than asserted.
"""

from __future__ import annotations

import re

from app.schemas.taxonomy import RefundReason

# Ordered: the first pattern that matches wins, so specific beats general.
_RULES: list[tuple[RefundReason, re.Pattern[str]]] = [
    # Duplicate / failed payment
    (RefundReason.DUPLICATE_OR_FAILED_PAYMENT, re.compile(
        r"(charg?ed?\s*twice|do?ub?le?\s*charge|duplicate\s*(pa[ym]ment|txn|refunded)"
        r"|amou?nt\s*dedu?cted\s*with?ou?t|payment\s*debited|failed\s*or?d(er)?\s*after\s*payment"
        r"|conf(irmed)?\s*utrs|pg\s*for\s*duplicate)", re.I)),
    # Cancellation before dispatch
    (RefundReason.CANCELLED_BEFORE_DISPATCH, re.compile(
        r"(cancell?ed\s*before\s*dispatch|cancell?ation\s*requ?e?st|or?d(er)?\s*cancell?ation"
        r"|cancel\s*or?d(er)?|cancel\s*button)", re.I)),
    # Price / promo
    (RefundReason.PRICE_OR_PROMO_ADJUSTMENT, re.compile(
        r"(cou?pon\s*not\s*appl|di[sc]cou?nt\s*not\s*appl|pro?mo\s*c[od]de?\s*inv?al"
        r"|price\s*(drop|adjust))", re.I)),
    # Return pickup failure -> money out without the unit back
    (RefundReason.RETURN_PICKUP_FAILURE, re.compile(
        r"(with?ou?t\s*pick?up|pi?c?k?up\s*(not\s*done|missed|msied)|pkp\s*(not\s*done|missed|msied)"
        r"|rev(erse)?\s*(pick?up|pkp)\s*pending)", re.I)),
    # Return received and QC passed
    (RefundReason.RETURN_RECEIVED_QC_PASSED, re.compile(
        r"(rev(erse)?\s*(pick?up|pkp)\s*qc|qc\s*(ok|passed|clear)|return\s*re?c?[ei]?[ve]*d"
        r"|passed\s*qc)", re.I)),
    # Refund service failure (about an earlier refund)
    (RefundReason.REFUND_SERVICE_FAILURE, re.compile(
        r"(r?f?n?d|ref?u?n?n?d)\s*(not\s*cred?it|pending|delay|de?elay|reprocess)"
        r"|refund\s*status\s*on\s*pg|arn\s*shared|still\s*waiting\s*for\s*my\s*refund", re.I)),
    # Transit damage
    (RefundReason.DAMAGED_IN_TRANSIT, re.compile(
        r"(tra?n?sit\s*d[ao]?ma?ge|d[ao]ma?ged?\s*in\s*tra?n?sit|unit\s*r?e?c?[ev]*d\s*d[ao]ma?ged"
        r"|damage\s*confirmed|photos\s*r?c?[ev]*d)", re.I)),
    # Wrong item
    (RefundReason.WRONG_ITEM_SHIPPED, re.compile(
        r"(in?correct\s*pro?du?ct\s*shipp?ed|wrong\s*i?te?m\s*deliver|wrong\s*vari[ae]nt"
        r"|wrong\s*colour)", re.I)),
    # Not delivered / lost
    (RefundReason.NOT_DELIVERED_OR_LOST, re.compile(
        r"(or?d(er)?\s*not\s*deliver|shipm?e?nt\s*not\s*r?e?c?[ev]*d|d?l?[iv]*ry\s*delay"
        r"|lost\s*in\s*tra?n?sit|rto\s*conf|not\s*deliver)", re.I)),
    # Warranty
    (RefundReason.PRODUCT_FAULT_IN_WARRANTY, re.compile(
        r"(w[ta]?r?ra?nty\s*(claim|buy)|rma\s*status|service\s*cent|repair\s*st?a?tu?s"
        r"|buy-?back)", re.I)),
    # Product fault
    (RefundReason.PRODUCT_FAULT_DOA, re.compile(
        r"(not\s*char?ging|no\s*l[ei]d|case\s*dead|unit\s*dead|no\s*power|not\s*discover"
        r"|pair(ing)?\s*fail|not\s*taking\s*charge|silent|no\s*audio|di?st?ort|cra?ckl"
        r"|battery\s*dra?in|disconnect|dropout|touch\s*not|unresponsive|strap\s*(broken|damage|pin)"
        r"|mic\s*(issue|not)|dead\s*on\s*arrival|doa)", re.I)),
    # Change of mind
    (RefundReason.CUSTOMER_CHANGED_MIND, re.compile(
        r"(change\s*of\s*mind|changed?\s*my\s*mind|with?ou?t\s*asking|no\s*longer\s*need"
        r"|don'?t\s*want)", re.I)),
    # Goodwill, last: only when nothing more specific matched
    (RefundReason.GOODWILL_GESTURE, re.compile(
        r"(goodwill|good\s*will|one-?time\s*gesture|courtesy|as\s*a\s*gesture)", re.I)),
]

# Notes that carry no information at all.
_EMPTYISH = re.compile(r"^\s*(cx\s*ok|ok|done|closed|resolved|rslvd|n/?a|-+|\.*)\s*\.?\s*$", re.I)


def classify_baseline(customer_message: str, agent_notes: str) -> RefundReason:
    """Best deterministic guess at the refund reason.

    Agent notes are searched first: they describe what the agent actually did,
    whereas the customer message describes the complaint, which is not always the
    reason money moved.
    """
    notes = agent_notes or ""
    message = customer_message or ""

    if _EMPTYISH.match(notes.strip()) and not message.strip():
        return RefundReason.UNKNOWN

    for reason, pattern in _RULES:
        if pattern.search(notes):
            return reason
    for reason, pattern in _RULES:
        if pattern.search(message):
            return reason

    if _EMPTYISH.match(notes.strip()):
        return RefundReason.UNKNOWN
    return RefundReason.UNKNOWN


_DUAL_POSITIVE = re.compile(
    r"(refund\s*\+\s*replacement|replacement\s*\+\s*refund|both\s*refund\s*(and|&)\s*repl"
    r"|refund\s*(and|&)\s*replacement\s*(both|given)|rfnd\s*\+\s*rplc"
    r"|replacement\s*dispatched|new\s*unit\s*(also\s*)?(going|sent|shipped)"
    r"|fresh\s*unit\s*shipped|dispatched\s*a\s*new\s*one|new\s*set)", re.I
)
_DUAL_NEGATIVE = re.compile(
    r"(replacement\s*not\s*applicable|rplc\s*(request\s*)?rejected|replacement\s*(request\s*)?rejected"
    r"|explained\s*policy,?\s*(refund|rfnd)\s*only|replacement\s*declined|opted\s*for\s*(a\s*)?refund"
    r"|out\s*of\s*stock|did\s*not\s*want\s*replacement|offered\s*earlier)", re.I
)


def baseline_dual_remedy(agent_notes: str) -> bool:
    """Keyword-only dual-remedy detection, with a negation blocklist.

    Kept separate so the evaluation can quantify what the negation handling costs
    a keyword approach — which is the concrete argument for the LLM layer.
    """
    notes = agent_notes or ""
    if _DUAL_NEGATIVE.search(notes):
        return False
    return bool(_DUAL_POSITIVE.search(notes))
