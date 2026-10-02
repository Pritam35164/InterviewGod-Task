"""Policy retrieval over a 2-page, 10-section document.

Why there is no vector database here
------------------------------------
support-policy.pdf is 6.7 KB of text in 10 numbered sections. The entire document
fits in a single Nemotron prompt several times over. Embedding it would add a
service, a migration and an index to maintain in exchange for choosing between
ten short candidates — and BM25 over ten sections is both cheaper and auditable,
because the score decomposes into the terms that matched.

So retrieval is: intent-based section priors (from the policy structure itself)
combined with BM25 term scoring, returning sections with their citation and the
matched terms as relevance metadata. `§5` is always retained for any refund or
money-moving request, because that is where every refund authorisation rule lives.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.policy.loader import INTENT_SECTIONS, PolicyDocument, PolicySection

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "by",
    "it", "be", "as", "at", "that", "this", "with", "from", "not", "no", "any", "all",
    "i", "my", "me", "you", "your", "we", "they", "was", "were", "has", "have", "had",
    "do", "does", "did", "can", "will", "would", "should", "please", "hi", "hello",
    "sir", "madam", "dear", "team", "thanks", "regards", "order", "what", "when",
    "how", "why", "which", "who",
}

# Intents whose answer can move money: §5 is force-included.
_MONEY_INTENTS = {
    "refund", "refund_status", "replacement", "warranty", "billing",
    "cancellation", "other",
}


def tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 2]


@dataclass
class RetrievedSection:
    section: PolicySection
    score: float
    matched_terms: list[str]
    reason: str

    def to_dict(self) -> dict:
        d = self.section.to_dict()
        d["relevance"] = {
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms[:12],
            "selected_because": self.reason,
        }
        return d


class PolicyRetriever:
    """BM25 over the policy sections, with intent priors."""

    K1 = 1.4
    B = 0.75

    def __init__(self, policy: PolicyDocument) -> None:
        self.policy = policy
        self._ids: list[str] = sorted(policy.sections, key=lambda s: int(s))
        self._tokens: dict[str, list[str]] = {
            sid: tokenise(policy.sections[sid].text) for sid in self._ids
        }
        self._tf: dict[str, dict[str, int]] = {}
        for sid, toks in self._tokens.items():
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self._tf[sid] = counts
        lengths = [len(t) for t in self._tokens.values()] or [1]
        self._avg_len = sum(lengths) / len(lengths)
        n_docs = max(1, len(self._ids))
        df: dict[str, int] = {}
        for counts in self._tf.values():
            for term in counts:
                df[term] = df.get(term, 0) + 1
        self._idf = {
            term: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    @property
    def available(self) -> bool:
        return bool(self._ids)

    def _bm25(self, sid: str, query_terms: list[str]) -> tuple[float, list[str]]:
        counts = self._tf.get(sid, {})
        dl = max(1, len(self._tokens.get(sid, [])))
        score, matched = 0.0, []
        for term in set(query_terms):
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = self._idf.get(term, 0.0)
            denom = tf + self.K1 * (1 - self.B + self.B * dl / self._avg_len)
            score += idf * (tf * (self.K1 + 1)) / denom
            matched.append(term)
        matched.sort(key=lambda t: -self._idf.get(t, 0.0))
        return score, matched

    def retrieve(
        self,
        query: str,
        intent: str | None = None,
        top_k: int = 3,
        extra_terms: list[str] | None = None,
    ) -> list[RetrievedSection]:
        if not self.available:
            return []

        terms = tokenise(query) + list(extra_terms or [])
        prior_ids = INTENT_SECTIONS.get((intent or "other").lower(), INTENT_SECTIONS["other"])

        scored: list[RetrievedSection] = []
        for sid in self._ids:
            bm, matched = self._bm25(sid, terms)
            prior = 0.0
            reason_bits: list[str] = []
            if sid in prior_ids:
                # Earlier in the prior list = stronger prior.
                prior = 1.6 - 0.4 * prior_ids.index(sid)
                reason_bits.append(f"intent '{intent or 'other'}' maps to §{sid}")
            if matched:
                reason_bits.append(f"matched {len(matched)} policy term(s)")
            total = bm + prior
            if total <= 0:
                continue
            scored.append(
                RetrievedSection(
                    section=self.policy.sections[sid],
                    score=total,
                    matched_terms=matched,
                    reason="; ".join(reason_bits) or "keyword match",
                )
            )

        scored.sort(key=lambda r: -r.score)
        selected = scored[:top_k]

        # §5 carries every refund authorisation rule. Never answer a money
        # question without it, regardless of what the keywords scored.
        if (intent or "other").lower() in _MONEY_INTENTS and "5" in self.policy.sections:
            if not any(r.section.section_id == "5" for r in selected):
                forced = RetrievedSection(
                    section=self.policy.section("5"),
                    score=0.0,
                    matched_terms=[],
                    reason="force-included: §5 holds all refund authorisation rules",
                )
                selected = [forced] + selected[: max(0, top_k - 1)]
        return selected

    def retrieve_for_refund_classification(self, ticket_text: str) -> list[RetrievedSection]:
        """Context for the Nemotron refund classifier: §5 always, plus §6/§10
        because avoidability reasoning depends on team ownership and on the
        reporting definitions."""
        out: list[RetrievedSection] = []
        for sid, why in (
            ("5", "refund rules, dual-remedy prohibition, goodwill cap, reason codes"),
            ("6", "team ownership: Returns Desk processes refunds by design; Tier 2 certification"),
            ("10", "reporting definitions: repeat contact, handle time, attendance"),
        ):
            if sid in self.policy.sections:
                _, matched = self._bm25(sid, tokenise(ticket_text))
                out.append(
                    RetrievedSection(
                        section=self.policy.section(sid),
                        score=1.0,
                        matched_terms=matched,
                        reason=f"always supplied for refund classification: {why}",
                    )
                )
        return out
