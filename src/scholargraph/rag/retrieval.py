"""Hybrid retrieval (BM25 + dense embeddings, fused with reciprocal rank fusion) and graph-aware expansion.

* **BM25** finds papers that share exact words with the question (names, acronyms, rare terms).
* **Dense** retrieval finds papers with similar *meaning* (the fine-tuned encoder of Phase 4 is plugged in here).
* **Reciprocal rank fusion** combines the two ranked lists without having to calibrate their incompatible score scales:
  ``rrf(d) = sum over lists of 1 / (k + rank_list(d))``.
* **Graph expansion** (the *Graph* in GraphRAG): papers that are cited by, or cite, several of the best hits are likely relevant even when
  their own text matched weakly. Their fused score gets a bonus proportional to how many top hits they are linked to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..graph.store import GraphStore
from .bm25 import BM25


@dataclass
class Hit:
    paper_id: str
    score: float
    sources: Dict[str, float]        # which component contributed what: {"bm25": rrf part, "dense": rrf part, "graph": bonus}


def rrf(rank_lists: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for lst in rank_lists:
        for r, d in enumerate(lst, 1):
            out[d] = out.get(d, 0.0) + 1.0 / (k + r)
    return out


class HybridRetriever:
    def __init__(self, paper_ids: List[str], texts: Sequence[str], embeddings: Optional[np.ndarray] = None, encoder=None,
                 years: Optional[np.ndarray] = None, store: Optional[GraphStore] = None):
        self.ids, self.idx = list(paper_ids), {p: i for i, p in enumerate(paper_ids)}
        self.texts = list(texts)
        self.bm25 = BM25().fit(texts)
        self.emb, self.encoder, self.years, self.store = embeddings, encoder, years, store

    def dense_scores(self, query: str) -> Optional[np.ndarray]:
        if self.emb is None or self.encoder is None:
            return None
        q = self.encoder.encode([query])[0]
        return self.emb @ q

    def search(self, query: str, k: int = 5, pool: int = 50, year_from: Optional[int] = None, use_dense: bool = True,
               use_bm25: bool = True, graph_weight: float = 0.0, reranker=None, rerank_pool: int = 30, rerank_blend: float = 0.0) -> List[Hit]:
        allowed = None if (year_from is None or self.years is None) else (self.years >= year_from)
        lists, labels = [], []
        if use_bm25:
            s = self.bm25.scores(query)
            if allowed is not None:
                s = np.where(allowed, s, 0.0)
            order = np.argsort(-s, kind="stable")[:pool]
            lists.append([int(i) for i in order if s[i] > 0])
            labels.append("bm25")
        d = self.dense_scores(query) if use_dense else None
        if d is not None:
            if allowed is not None:
                d = np.where(allowed, d, -np.inf)
            order = np.argsort(-d, kind="stable")[:pool]
            lists.append([int(i) for i in order if np.isfinite(d[i])])
            labels.append("dense")
        parts = {lab: rrf([lst]) for lab, lst in zip(labels, lists)}
        fused: Dict[int, float] = {}
        for p in parts.values():
            for doc, v in p.items():
                fused[doc] = fused.get(doc, 0.0) + v
        graph_bonus: Dict[int, float] = {}
        if graph_weight > 0 and self.store is not None and fused:
            top = sorted(fused, key=fused.get, reverse=True)[:10]
            for h in top:
                pid = self.ids[h]
                for nb in set(self.store.references(pid)) | set(self.store.cited_by(pid)):
                    j = self.idx.get(nb)
                    if j is not None and (allowed is None or allowed[j]):
                        graph_bonus[j] = graph_bonus.get(j, 0.0) + graph_weight * fused[h]
            for j, b in graph_bonus.items():
                fused[j] = fused.get(j, 0.0) + b
        ranked = sorted(fused, key=lambda i: (-fused[i], i))[:max(k, rerank_pool) if reranker is not None else k]
        if reranker is not None and ranked:                      # second stage: re-score the best candidates with a stronger model
            from .rerank import rerank_ids
            order = rerank_ids(reranker, query, [self.ids[i] for i in ranked], [self.texts[i] for i in ranked], k, [fused[i] for i in ranked], rerank_blend)
            rs = reranker.score(query, [self.texts[ranked[j]] for j in order])
            return [Hit(self.ids[ranked[j]], float(rs[n]), {"rerank": float(rs[n]), "first_stage": fused[ranked[j]], "graph": graph_bonus.get(ranked[j], 0.0)})
                    for n, j in enumerate(order)]
        return [Hit(self.ids[i], fused[i], {**{lab: parts[lab].get(i, 0.0) for lab in labels}, "graph": graph_bonus.get(i, 0.0)}) for i in ranked]
