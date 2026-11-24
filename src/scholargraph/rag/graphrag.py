"""GraphRAG: retrieve papers, add their graph context, and let an LLM answer with citations.

Pipeline for a question:

1. hybrid retrieval (BM25 + dense) returns the best papers, optionally boosted by citation links between them (``retrieval.py``);
2. the **context** lists each paper as a numbered source ``[P1]``, ``[P2]``... with title, year, authors and abstract, followed by a block
   of graph facts (``[P1] cites [P3]``, concepts, citation counts) that no text search could provide;
3. the LLM is instructed to answer only from the sources and to cite them with the ``[Pn]`` markers;
4. guardrails verify the citations and the groundedness of the answer; with weak retrieval the system abstains.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..graph.store import GraphStore
from ..llm import LLM, LLMResult
from .guardrails import REFUSAL, GuardrailReport, check_answer
from .retrieval import Hit, HybridRetriever

SYSTEM = (
    "You are a research assistant answering questions about scientific papers. Use ONLY the numbered sources and graph facts provided. "
    "Cite the sources you use with their markers, for example [P2]. If the sources do not contain the answer, say exactly: "
    f"{REFUSAL} Be concise: at most four sentences."
)


@dataclass
class Source:
    marker: int
    paper_id: str
    title: str
    year: int
    authors: List[str]
    abstract: str
    score: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[Source]
    report: GuardrailReport
    prompt_chars: int = 0
    seconds: float = 0.0
    llm_tokens: Dict[str, int] = field(default_factory=dict)


def build_context(sources: List[Source], store: Optional[GraphStore], max_abstract_chars: int = 700) -> str:
    lines = ["SOURCES"]
    for s in sources:
        authors = ", ".join(s.authors[:4]) + (" et al." if len(s.authors) > 4 else "")
        lines.append(f"[P{s.marker}] {s.title} ({s.year}) by {authors}\n    {s.abstract[:max_abstract_chars]}")
    if store is not None:
        facts = []
        by_id = {s.paper_id: s.marker for s in sources}
        for s in sources:
            for ref in store.references(s.paper_id):
                if ref in by_id:
                    facts.append(f"[P{s.marker}] cites [P{by_id[ref]}]")
            n = store.paper(s.paper_id) or {}
            facts.append(f"[P{s.marker}] is cited by {n.get('citations_in_graph', 0)} papers in the graph")
            concepts = store.concepts_of(s.paper_id)
            if concepts:
                facts.append(f"[P{s.marker}] mentions: {', '.join(concepts[:6])}")
        lines += ["", "GRAPH FACTS"] + facts
    return "\n".join(lines)


class GraphRAG:
    def __init__(self, retriever: HybridRetriever, store: GraphStore, llm: LLM, k: int = 5, graph_weight: float = 0.3, min_hits: int = 1):
        self.retriever, self.store, self.llm, self.k, self.graph_weight, self.min_hits = retriever, store, llm, k, graph_weight, min_hits

    def retrieve(self, question: str, year_from: Optional[int] = None) -> List[Source]:
        hits: List[Hit] = self.retriever.search(question, k=self.k, year_from=year_from, graph_weight=self.graph_weight)
        out = []
        for i, h in enumerate(hits, 1):
            p = self.store.paper(h.paper_id) or {}
            out.append(Source(i, h.paper_id, p.get("title", ""), int(p.get("year", 0) or 0),
                              [a["name"] or "" for a in self.store.authors_of(h.paper_id)], p.get("abstract", ""), h.score))
        return out

    def answer(self, question: str, year_from: Optional[int] = None) -> RAGAnswer:
        t0 = time.time()
        sources = self.retrieve(question, year_from)
        if len(sources) < self.min_hits:
            return RAGAnswer(question, REFUSAL, [], check_answer(REFUSAL, []), 0, time.time() - t0)
        context = build_context(sources, self.store)
        r: LLMResult = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{context}\n\nQUESTION: {question}"}],
                                     temperature=0.0, max_tokens=300)
        texts = [f"{s.title}. {s.abstract}" for s in sources]
        report = check_answer(r.text, texts)
        return RAGAnswer(question, r.text.strip(), sources, report, len(context), time.time() - t0,
                         {"prompt": r.prompt_tokens, "completion": r.completion_tokens})
