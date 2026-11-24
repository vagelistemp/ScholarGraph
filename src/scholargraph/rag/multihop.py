"""Multi-hop retrieval over the citation graph, with an explanation of how each paper was reached.

Some relevant papers share no words with the question and are not close to it in embedding space, yet are *linked* to papers that are
(the method paper a good application paper builds on). Multi-hop retrieval starts from the best text matches and walks the graph:

    hop 0: the top ``beam`` papers by hybrid text retrieval (the seeds)
    hop h: collect the unvisited neighbours (papers cited by, or citing, the current frontier), keep the ``beam`` neighbours most similar to
           the question, remember from which paper and through which relation each was reached

The result lists every kept paper with its **path** from a seed, so an answer can say *why* a paper is relevant ("seed W5 --cites--> W9").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..graph.store import GraphStore
from .retrieval import HybridRetriever


@dataclass
class Evidence:
    paper_id: str
    score: float
    hop: int
    path: List[Tuple[str, str, str]] = field(default_factory=list)      # (from paper, relation, to paper) from a seed to this paper

    def explain(self) -> str:
        if not self.path:
            return f"{self.paper_id}: direct text match"
        return f"{self.paper_id}: " + " ".join(f"{a} --{rel}--> {b}" for a, rel, b in self.path)


class MultiHopRetriever:
    def __init__(self, retriever: HybridRetriever, store: GraphStore, hops: int = 2, beam: int = 5, hop_decay: float = 0.9):
        self.retriever, self.store, self.hops, self.beam, self.hop_decay = retriever, store, hops, beam, hop_decay

    def search(self, question: str, k: int = 8) -> List[Evidence]:
        r = self.retriever
        seeds = r.search(question, k=self.beam, use_bm25=True)
        evid: Dict[str, Evidence] = {h.paper_id: Evidence(h.paper_id, 1.0, 0) for h in seeds}
        frontier = [h.paper_id for h in seeds]
        qvec = r.encoder.encode([question])[0] if (r.encoder is not None and r.emb is not None) else None
        for hop in range(1, self.hops + 1):
            cand: Dict[str, Tuple[str, str]] = {}                         # candidate -> (parent, relation)
            for p in frontier:
                for q in self.store.references(p):
                    if q not in evid:
                        cand.setdefault(q, (p, "cites"))
                for q in self.store.cited_by(p):
                    if q not in evid:
                        cand.setdefault(q, (p, "cited_by"))
            if not cand:
                break
            ids = list(cand)
            if qvec is not None:
                sims = r.emb[[r.idx[i] for i in ids]] @ qvec
            else:
                sims = np.array([r.bm25.scores(question)[r.idx[i]] for i in ids])
            order = np.argsort(-sims, kind="stable")[: self.beam]
            frontier = []
            for j in order:
                child = ids[j]
                parent, rel = cand[child]
                a, b = (parent, child) if rel == "cites" else (child, parent)
                label = "cites"
                evid[child] = Evidence(child, float(sims[j]) * (self.hop_decay ** hop), hop, evid[parent].path + [(a, label, b)])
                frontier.append(child)
        ranked = sorted(evid.values(), key=lambda e: (-e.score, e.paper_id))
        return ranked[:k]
