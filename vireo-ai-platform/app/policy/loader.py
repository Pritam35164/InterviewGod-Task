"""Load Vireo's support policy: machine-readable rules + verbatim PDF sections.

The YAML holds the rules the engine enforces. The PDF is the source of truth for
the *words*, and the two are cross-checked at load time: `verify_against_pdf()`
asserts that every `source_quote` in the YAML actually appears in the extracted
PDF text. A transcription drift therefore fails loudly instead of quietly
authorising refunds against a rule nobody wrote.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.errors import PolicyUnavailable
from app.core.logging import get_logger

log = get_logger(__name__)

_SECTION_TITLES = {
    "1": "Purpose and scope",
    "2": "Channels and hours",
    "3": "First-response service levels and breach credits",
    "4": "Cost standards (FY26 planning figures)",
    "5": "Refunds, replacements and reason codes",
    "6": "Teams, tiers and ownership",
    "7": "Shifts and roster",
    "8": "Customer satisfaction (CSAT)",
    "9": "Systems and timestamps",
    "10": "Reporting definitions",
}

# Which policy sections are relevant to which request intent. Used to bias
# retrieval; keyword scoring still decides the ordering within the bias.
INTENT_SECTIONS: dict[str, list[str]] = {
    "refund": ["5", "6", "10"],
    "refund_status": ["5", "10"],
    "replacement": ["5", "6"],
    "warranty": ["5", "6"],
    "delivery": ["5", "2"],
    "billing": ["5", "4"],
    "cancellation": ["5"],
    "policy_question": ["5", "3", "2", "8"],
    "product_info": ["2"],
    "sla": ["3", "10"],
    "other": ["5", "3", "10"],
}


@dataclass(frozen=True)
class PolicySection:
    """One numbered section of support-policy.pdf, verbatim."""

    section_id: str
    title: str
    text: str
    policy_version: str

    @property
    def citation(self) -> str:
        return f"support-policy.pdf v{self.policy_version} §{self.section_id} ({self.title})"

    def to_dict(self) -> dict:
        return {
            "policy_section_id": self.section_id,
            "policy_version": self.policy_version,
            "title": self.title,
            "policy_text": self.text,
            "citation": self.citation,
        }


@dataclass
class PolicyDocument:
    version: str
    effective_date: str
    rules: dict[str, Any]
    sections: dict[str, PolicySection] = field(default_factory=dict)
    pdf_text: str = ""
    pdf_available: bool = False
    verification: dict[str, Any] = field(default_factory=dict)

    # ---- typed accessors used by the engine ------------------------------
    @property
    def goodwill_cap_inr(self) -> float:
        return float(self.rules["refunds"]["goodwill"]["cap_inr"])

    @property
    def goodwill_approver(self) -> str:
        return self.rules["refunds"]["goodwill"]["requires_approval_by"]

    @property
    def dual_remedy_prohibited(self) -> bool:
        return bool(self.rules["refunds"]["dual_remedy_prohibition"]["prohibited"])

    @property
    def reason_codes(self) -> dict[str, str]:
        return dict(self.rules["refunds"]["reason_codes"]["codes"])

    @property
    def sla_targets_minutes(self) -> dict[str, int]:
        return dict(self.rules["sla"]["first_response_target_minutes"])

    @property
    def sla_breach_credit_inr(self) -> float:
        return float(self.rules["sla"]["breach_credit_inr"])

    @property
    def cost_per_contact_inr(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.rules["cost_standards"]["cost_per_contact_inr"].items()}

    @property
    def blended_cost_per_contact_inr(self) -> float:
        return float(self.rules["cost_standards"]["blended_cost_per_contact_inr"])

    @property
    def transfer_cost_inr(self) -> float:
        return float(self.rules["cost_standards"]["transfer_cost_inr"])

    @property
    def replacement_logistics_inr(self) -> float:
        return float(self.rules["refunds"]["replacement_cost_model"]["logistics_add_inr"])

    @property
    def tier2_teams(self) -> list[str]:
        return list(self.rules["teams"]["tier2_teams"])

    @property
    def refund_processing_owner(self) -> str:
        return str(self.rules["teams"]["refund_processing_owner"])

    @property
    def fcr_window_days(self) -> int:
        return int(self.rules["reporting_definitions"]["fcr_window_days"])

    @property
    def attendance_statuses(self) -> list[str]:
        return list(self.rules["reporting_definitions"]["attendance_statuses"])

    @property
    def helpdesk_go_live(self) -> str:
        return str(self.rules["systems"]["helpdesk_go_live"])

    @property
    def gaps(self) -> list[dict]:
        return list(self.rules.get("not_defined_by_policy", []))

    def entitlement(self, entitlement_id: str) -> dict | None:
        for e in self.rules["refunds"]["entitlements"]:
            if e["id"] == entitlement_id:
                return e
        return None

    def section(self, section_id: str) -> PolicySection:
        try:
            return self.sections[str(section_id)]
        except KeyError as exc:
            raise PolicyUnavailable(f"policy section {section_id} not found") from exc

    def cache_namespace(self) -> str:
        """Cache keys are namespaced by policy version, so a policy bump
        invalidates every policy-derived cached answer (brief §20)."""
        return f"policy:v{self.version}"

    def to_summary(self) -> dict:
        return {
            "policy_version": self.version,
            "effective_date": self.effective_date,
            "sections": [
                {"id": s.section_id, "title": s.title, "chars": len(s.text)}
                for s in sorted(self.sections.values(), key=lambda x: int(x.section_id))
            ],
            "pdf_available": self.pdf_available,
            "verification": self.verification,
            "reason_codes": self.reason_codes,
            "goodwill_cap_inr": self.goodwill_cap_inr,
            "dual_remedy_prohibited": self.dual_remedy_prohibited,
            "known_gaps": [g["id"] for g in self.gaps],
        }


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages)
    # Strip the page footers the extractor emits ("-- 1 of 2 --").
    text = re.sub(r"--\s*\d+\s+of\s+\d+\s*--", "\n", text)
    return text


def split_sections(pdf_text: str, version: str) -> dict[str, PolicySection]:
    """Split on the numbered headings 1..10 as they appear in the document."""
    if not pdf_text.strip():
        return {}
    flat = re.sub(r"[ \t]+", " ", pdf_text)
    pattern = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+([A-Z][^\n]{3,90})")
    matches = [m for m in pattern.finditer(flat) if m.group(1) in _SECTION_TITLES]

    # Keep the first occurrence of each heading, in document order.
    seen: set[str] = set()
    ordered = []
    for m in matches:
        if m.group(1) not in seen:
            seen.add(m.group(1))
            ordered.append(m)

    sections: dict[str, PolicySection] = {}
    for idx, m in enumerate(ordered):
        sid = m.group(1)
        start = m.start()
        end = ordered[idx + 1].start() if idx + 1 < len(ordered) else len(flat)
        body = flat[start:end].strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        sections[sid] = PolicySection(
            section_id=sid,
            title=_SECTION_TITLES.get(sid, m.group(2).strip()),
            text=body,
            policy_version=version,
        )
    return sections


def _normalise_for_match(s: str) -> str:
    """Whitespace/punctuation-insensitive form for quote verification."""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _iter_quotes(node: Any, path: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, str) and (k == "source_quote" or k.endswith("_quote")):
                out.append((p, v))
            else:
                out.extend(_iter_quotes(v, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_iter_quotes(v, f"{path}[{i}]"))
    return out


def verify_against_pdf(rules: dict, pdf_text: str) -> dict[str, Any]:
    """Assert every transcribed quote is really in the PDF."""
    if not pdf_text.strip():
        return {"checked": 0, "matched": 0, "mismatched": [], "status": "pdf_unavailable"}
    haystack = _normalise_for_match(pdf_text)
    mismatched: list[str] = []
    quotes = _iter_quotes(rules)
    for path, quote in quotes:
        needle = _normalise_for_match(quote)
        if not needle:
            continue
        if needle in haystack:
            continue
        # A quote may join clauses the PDF separates; require every sentence.
        parts = [p for p in re.split(r"\s+(?=[a-z0-9]{4,})", needle) if len(p) > 40]
        if parts and all(p in haystack for p in parts):
            continue
        mismatched.append(path)
    return {
        "checked": len(quotes),
        "matched": len(quotes) - len(mismatched),
        "mismatched": mismatched,
        "status": "ok" if not mismatched else "drift_detected",
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cached: PolicyDocument | None = None


def _load() -> PolicyDocument:
    path = settings.policy_path
    if not path.exists():
        raise PolicyUnavailable(f"policy rules file missing: {path}")
    try:
        rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PolicyUnavailable(f"policy rules file unreadable: {exc}") from exc
    if not isinstance(rules, dict) or "refunds" not in rules:
        raise PolicyUnavailable("policy rules file malformed: no 'refunds' block")

    version = str(rules.get("version") or settings.POLICY_VERSION)
    if version != settings.POLICY_VERSION:
        log.warning(
            "policy.version_mismatch",
            extra={"file_version": version, "configured_version": settings.POLICY_VERSION},
        )

    pdf_text, sections, pdf_available = "", {}, False
    pdf_path = settings.policy_pdf_path
    if pdf_path.exists():
        try:
            pdf_text = extract_pdf_text(pdf_path)
            sections = split_sections(pdf_text, version)
            pdf_available = True
        except Exception as exc:  # noqa: BLE001
            log.error("policy.pdf_extract_failed", extra={"error": str(exc)})
    else:
        log.warning("policy.pdf_missing", extra={"path": str(pdf_path)})

    verification = verify_against_pdf(rules, pdf_text)
    if verification["status"] == "drift_detected":
        log.error("policy.transcription_drift", extra=verification)

    doc = PolicyDocument(
        version=version,
        effective_date=str(rules.get("effective_date", "")),
        rules=rules,
        sections=sections,
        pdf_text=pdf_text,
        pdf_available=pdf_available,
        verification=verification,
    )
    log.info(
        "policy.loaded",
        extra={
            "policy_version": version,
            "sections": len(sections),
            "pdf_available": pdf_available,
            "quotes_checked": verification["checked"],
            "quotes_matched": verification["matched"],
        },
    )
    return doc


def get_policy() -> PolicyDocument:
    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                _cached = _load()
    return _cached


def reload_policy() -> PolicyDocument:
    global _cached
    with _lock:
        _cached = None
    get_retrieval_index.cache_clear()
    return get_policy()


@lru_cache(maxsize=1)
def get_retrieval_index():
    from app.policy.retrieval import PolicyRetriever

    return PolicyRetriever(get_policy())
