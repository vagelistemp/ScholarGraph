"""Guardrails for generated answers: citation checking, a cheap groundedness measure, and abstention.

A RAG answer is only worth trusting if it can be traced to the retrieved sources. Three inexpensive checks catch the most common failures:

1. **Citation validity**: every ``[P3]`` marker in the answer must refer to a source that was actually shown to the model.
2. **Groundedness**: for each sentence, the share of its content words that occur in the sources it cites (or in any source when it cites
   none). A sentence with low overlap is probably invented. This is a lexical proxy, not a proof; the LLM judge in ``evalkit`` is the
   stronger (and costlier) check.
3. **Abstention**: when retrieval finds nothing relevant, the system should say so instead of improvising.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from .bm25 import tokenize

CITE = re.compile(r"\[P(\d+)\]")
REFUSAL = "I could not find enough evidence in the indexed papers to answer this question."


def extract_citations(answer: str) -> List[int]:
    return [int(m) for m in CITE.findall(answer)]


def invalid_citations(answer: str, n_sources: int) -> List[int]:
    return sorted({c for c in extract_citations(answer) if not 1 <= c <= n_sources})


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def groundedness(answer: str, sources: Sequence[str]) -> float:
    """Average over sentences of the fraction of content words found in the cited sources (all sources if the sentence has no marker)."""
    scores = []
    for s in _sentences(answer):
        cited = [c for c in extract_citations(s) if 1 <= c <= len(sources)]
        pool = set()
        for c in (cited or range(1, len(sources) + 1)):
            pool |= set(tokenize(sources[c - 1]))
        words = tokenize(CITE.sub("", s))
        if words:
            scores.append(sum(w in pool for w in words) / len(words))
    return sum(scores) / len(scores) if scores else 0.0


@dataclass
class GuardrailReport:
    invalid_citations: List[int]
    groundedness: float
    has_citation: bool
    abstained: bool

    @property
    def ok(self) -> bool:
        return not self.invalid_citations and (self.abstained or (self.has_citation and self.groundedness >= 0.5))


def check_answer(answer: str, sources: Sequence[str]) -> GuardrailReport:
    abstained = answer.strip() == REFUSAL
    return GuardrailReport(invalid_citations(answer, len(sources)), groundedness(answer, sources), bool(extract_citations(answer)), abstained)
