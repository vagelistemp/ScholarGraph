"""Query-side improvements: rewriting, expansion, hypothetical documents, and multi-query retrieval.

A user's question is often a poor search query ("how do people do X?"). Three standard fixes:

* **lexicon expansion** (free, deterministic): add the canonical names and aliases of known concepts ("gnn" also searches for
  "graph neural network");
* **LLM rewriting**: ask a model for up to three alternative keyword queries (structured output, validated);
* **HyDE** (hypothetical document embeddings): ask a model to write the abstract an ideal answer paper would have, and search with *that*;
  an abstract looks like the documents far more than a question does.

``multi_query_search`` runs every variant and fuses the ranked lists with reciprocal rank fusion, so a document found by several variants wins.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..graph.concepts import RuleConceptExtractor, load_lexicon, normalise_phrase
from ..llm import LLM
from ..structured import generate_structured
from .retrieval import Hit, HybridRetriever, rrf

REWRITE_SCHEMA = {"type": "object", "required": ["queries"], "properties": {"queries": {"type": "array", "maxItems": 3, "items": {"type": "string", "minLength": 3}}}}
HYDE_SCHEMA = {"type": "object", "required": ["abstract"], "properties": {"abstract": {"type": "string", "minLength": 20}}}

_EXTRACTOR: Optional[RuleConceptExtractor] = None


def _extractor() -> RuleConceptExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = RuleConceptExtractor()
    return _EXTRACTOR


def expand_with_lexicon(query: str) -> str:
    """Append the canonical concept name and all aliases of every concept recognised in the query."""
    ex = _extractor()
    found = [c for c, _ in ex.extract(query)]
    extra: List[str] = []
    for canon in found:
        extra.append(canon)
        extra += [a for a, (c, _) in ex.alias.items() if c == canon and a != canon][:3]
    seen = set(normalise_phrase(query).split())
    add = [w for w in dict.fromkeys(" ".join(extra).split()) if w not in seen]
    return query + (" " + " ".join(add) if add else "")


def rewrite_queries(llm: LLM, question: str) -> List[str]:
    res = generate_structured(llm, [{"role": "system", "content": "Rewrite the question as up to 3 short keyword search queries for a database of scientific "
                                     'abstracts. Reply with JSON: {"queries": ["...", "..."]}'}, {"role": "user", "content": question}], REWRITE_SCHEMA, retries=1, max_tokens=150)
    return res.value["queries"] if res.ok else []


def hyde(llm: LLM, question: str) -> Optional[str]:
    res = generate_structured(llm, [{"role": "system", "content": "Write a plausible 3-sentence scientific abstract that would answer the question. "
                                     'Reply with JSON: {"abstract": "..."}'}, {"role": "user", "content": question}], HYDE_SCHEMA, retries=1, max_tokens=250)
    return res.value["abstract"] if res.ok else None


def multi_query_search(retriever: HybridRetriever, queries: Sequence[str], k: int = 5, pool: int = 30, **kw) -> List[Hit]:
    """Search with every query variant and fuse the ranked lists with reciprocal rank fusion."""
    lists = []
    store = {}
    for q in dict.fromkeys(queries):                                    # unique, in order
        hits = retriever.search(q, k=pool, **kw)
        lists.append([retriever.idx[h.paper_id] for h in hits])
        for h in hits:
            store.setdefault(h.paper_id, h)
    fused = rrf(lists)
    ranked = sorted(fused, key=lambda i: (-fused[i], i))[:k]
    return [Hit(retriever.ids[i], fused[i], {"multi_query": fused[i], "n_variants": float(sum(i in l for l in lists))}) for i in ranked]
