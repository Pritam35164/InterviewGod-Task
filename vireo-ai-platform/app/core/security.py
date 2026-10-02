"""Input hygiene and prompt-injection containment.

The LLM boundary treats ticket text as *data*. Two independent controls:

1. `wrap_untrusted` fences customer/agent text in explicit delimiters and strips
   the delimiter sequence from the payload so the text cannot close its own fence.
2. `scan_injection` is a cheap deterministic pre-filter. Laya's `guard_questions`
   provides the semantic check; this catches the literal patterns and, more
   importantly, records *why* a ticket was flagged for the audit trail.

Neither is claimed to be complete. They lower the blast radius; the real control
is that the LLM's output is a validated classification and can never authorise a
payment — the policy engine does that from deterministic fields.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata

# Delimiters are unlikely to occur in helpdesk exports and are stripped from the
# payload before fencing, so untrusted text cannot terminate its own block.
FENCE_OPEN = "<<<UNTRUSTED_{label}_BEGIN>>>"
FENCE_CLOSE = "<<<UNTRUSTED_{label}_END>>>"
_FENCE_LIKE = re.compile(r"<<<\s*/?\s*UNTRUSTED[^>]*>>>", re.I)

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("instruction_override", re.compile(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|all|your|the)\b[^.\n]{0,25}\b(instruction|prompt|rule|polic|direction|system)", re.I)),
    ("role_hijack", re.compile(r"\b(you are now|act as|pretend to be|from now on you|new persona|roleplay as)\b", re.I)),
    ("system_prompt_probe", re.compile(r"\b(system prompt|developer message|initial instructions|reveal your (prompt|instructions)|what are your instructions)\b", re.I)),
    ("authorisation_injection", re.compile(r"\b(approve|authorise|authorize|issue|release|sanction)\b[^.\n]{0,30}\b(my|this|the)\b[^.\n]{0,20}\b(refund|payment|credit|replacement)\b[^.\n]{0,40}\b(immediately|now|without|no need|bypass|skip)", re.I)),
    ("policy_rewrite", re.compile(r"\b(the )?(new|updated|revised) polic(y|ies)\b[^.\n]{0,30}\b(states|says|allows|permits)\b", re.I)),
    ("schema_tamper", re.compile(r"\b(output|respond|reply|return)\b[^.\n]{0,25}\b(json|schema|format|field)\b[^.\n]{0,30}\b(instead|only|as follows|exactly)", re.I)),
    ("delimiter_break", _FENCE_LIKE),
    ("fake_system_turn", re.compile(r"(^|\n)\s*(system|assistant|developer)\s*[:>]\s*", re.I)),
    ("encoded_payload", re.compile(r"\b(base64|rot13|hex ?decode|decode this)\b", re.I)),
]

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))


def normalise_text(text: str) -> str:
    """NFKC-normalise and drop zero-width characters used to split keywords."""
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    # Collapse runs of control characters but keep newlines/tabs.
    return "".join(ch for ch in cleaned if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def scan_injection(text: str) -> list[str]:
    """Return the names of the injection patterns present in `text`."""
    if not text:
        return []
    probe = normalise_text(text)
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(probe)]


def sanitise_untrusted(text: str, max_chars: int = 4000) -> str:
    """Make untrusted text safe to embed in a prompt as data."""
    cleaned = normalise_text(text or "")
    cleaned = _FENCE_LIKE.sub("[removed-delimiter]", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n[truncated]"
    return cleaned


def wrap_untrusted(text: str, label: str, max_chars: int = 4000) -> str:
    """Fence untrusted text so the model can see exactly where data starts/ends."""
    tag = re.sub(r"[^A-Z_]", "", label.upper().replace(" ", "_")) or "DATA"
    body = sanitise_untrusted(text, max_chars=max_chars)
    return f"{FENCE_OPEN.format(label=tag)}\n{body}\n{FENCE_CLOSE.format(label=tag)}"


def stable_hash(*parts: str) -> str:
    """Deterministic short digest used for cache keys and dedup fingerprints."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def constant_time_match(candidate: str, allowed: set[str]) -> bool:
    """Compare an API key against the allow-list without leaking timing."""
    if not candidate:
        return False
    return any(hmac.compare_digest(candidate, ok) for ok in allowed)


def pseudonymise(value: str, salt: str = "vireo") -> str:
    """One-way ID for anything that must appear in an export but not as PII."""
    return "px_" + hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:12]
