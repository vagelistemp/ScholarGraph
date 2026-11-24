"""Communities and global questions: the part of GraphRAG that plain RAG cannot do.

Ordinary retrieval answers *local* questions ("what does paper X do?"). A *global* question ("what are the main themes in this field, and
how do they relate?") has no single relevant paper; the answer is spread over thousands. The GraphRAG recipe:

1. **detect communities** in the citation graph (groups of papers that cite each other much more than they cite outsiders);
2. write a **report** for every community (its size, years, most cited papers, frequent concepts, topics);
3. answer a global question by a **map-reduce over reports**: each relevant community reports what it knows (map), then the partial
   answers are combined with citations to the community ids (reduce).

Community detection here is **label propagation**: every node starts with its own label and repeatedly adopts the most common label of
its neighbours; densely connected groups converge to one label. It is simple, near-linear in the number of edges and needs no parameters.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ..graph.store import MemoryGraphStore
from ..llm import LLM
from ..structured import generate_structured
from .bm25 import BM25
from .guardrails import REFUSAL

NONE_MARK = "nothing relevant"


def label_propagation(adj: sp.spmatrix, iterations: int = 15, seed: int = 0) -> np.ndarray:
    """Return a community label for every node of an (undirected, symmetrised) graph."""
    A = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    labels = np.arange(n)
    for _ in range(iterations):
        changed = 0
        for i in rng.permutation(n):
            nb = A.indices[A.indptr[i]:A.indptr[i + 1]]
            if len(nb) == 0:
                continue
            cnt = np.bincount(labels[nb])
            best = np.flatnonzero(cnt == cnt.max())
            new = labels[i] if labels[i] in best else int(best[rng.integers(len(best))])
            if new != labels[i]:
                labels[i] = new
                changed += 1
        if changed == 0:
            break
    _, dense = np.unique(labels, return_inverse=True)
    return dense


def louvain_labels(adj: sp.spmatrix, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Louvain modularity optimisation (via networkx): finds finer, more balanced communities than label propagation, which tends to
    collapse a densely connected graph into a single giant community (observed on the synthetic network in the tests)."""
    import networkx as nx
    A = ((adj + adj.T) > 0).astype(np.int8).tocoo()
    g = nx.Graph()
    g.add_nodes_from(range(A.shape[0]))
    g.add_edges_from(zip(A.row.tolist(), A.col.tolist()))
    labels = np.zeros(A.shape[0], dtype=np.int64)
    for cid, nodes in enumerate(nx.community.louvain_communities(g, resolution=resolution, seed=seed)):
        labels[list(nodes)] = cid
    return labels


@dataclass
class Community:
    cid: int
    members: List[int]
    report: str = ""
    top_concepts: List[str] = field(default_factory=list)
    top_papers: List[str] = field(default_factory=list)


def build_communities(store: MemoryGraphStore, min_size: int = 5, seed: int = 0, top_papers: int = 5, top_concepts: int = 8,
                      method: str = "louvain", resolution: float = 1.0) -> List[Community]:
    labels = louvain_labels(store.A, resolution, seed) if method == "louvain" else label_propagation(store.A, seed=seed)
    members: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        members.setdefault(int(l), []).append(i)
    out: List[Community] = []
    for cid, idx in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if len(idx) < min_size:
            continue
        ids = [store.ids[i] for i in idx]
        by_cites = sorted(idx, key=lambda i: (-store.indeg[i], store.ids[i]))[:top_papers]
        cnt: Counter = Counter()
        topics: Counter = Counter()
        for i in idx:
            cnt.update(store.concepts_of(store.ids[i]))
            t = store._topic_of.get(store.ids[i])
            if t:
                topics[t] += 1
        years = [int(store._year[p]) for p in ids]
        top_c = [c for c, _ in cnt.most_common(top_concepts)]
        titles = "; ".join(f"{store.paper(store.ids[i])['title']} ({int(store.indeg[i])} citations)" for i in by_cites)
        report = (f"Community {len(out)}: {len(idx)} papers from {min(years)} to {max(years)}. Main topics: "
                  f"{', '.join(t for t, _ in topics.most_common(3)) or 'unknown'}. Frequent concepts: {', '.join(top_c) or 'none'}. Most cited papers: {titles}.")
        out.append(Community(len(out), idx, report, top_c, [store.ids[i] for i in by_cites]))
    return out


def llm_report(llm: LLM, community: Community, facts: str) -> str:
    """Optionally let a model write a fluent report from the extractive facts (map step of the indexing phase)."""
    r = llm.chat([{"role": "system", "content": "Write a 3-sentence summary of this research community from the facts. Use only the facts."},
                  {"role": "user", "content": facts}], temperature=0.0, max_tokens=200)
    return r.text.strip() or facts


class CommunityIndex:
    def __init__(self, communities: Sequence[Community]):
        self.communities = list(communities)
        self.bm25 = BM25().fit([c.report for c in self.communities])

    def search(self, question: str, k: int = 3) -> List[Community]:
        return [self.communities[i] for i, _ in self.bm25.search(question, k)]


MAP_SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}


class GlobalAnswerer:
    """Map-reduce over community reports with citations like ``[C3]``."""

    def __init__(self, index: CommunityIndex, llm: LLM, k: int = 3):
        self.index, self.llm, self.k = index, llm, k

    def answer(self, question: str) -> Dict:
        comms = self.index.search(question, self.k)
        partials = []
        for c in comms:                                                         # map: one focused call per community
            res = generate_structured(self.llm, [
                {"role": "system", "content": f'Using ONLY the report, give a 1-2 sentence answer to the question, or answer exactly "{NONE_MARK}". Reply with JSON: {{"answer": "..."}}'},
                {"role": "user", "content": f"REPORT [C{c.cid}]: {c.report}\n\nQUESTION: {question}"}], MAP_SCHEMA, retries=1, max_tokens=150)
            if res.ok and NONE_MARK not in res.value["answer"].lower():
                partials.append((c.cid, res.value["answer"]))
        if not partials:
            return {"answer": REFUSAL, "communities": [c.cid for c in comms], "partials": [], "abstained": True}
        body = "\n".join(f"[C{cid}] {a}" for cid, a in partials)                 # reduce: combine, keeping the citations
        r = self.llm.chat([{"role": "system", "content": "Combine the partial answers into a short final answer. Cite the communities you use as [C#]. Use only what the partial answers say."},
                           {"role": "user", "content": f"QUESTION: {question}\n\nPARTIAL ANSWERS:\n{body}"}], temperature=0.0, max_tokens=250)
        cited = sorted({int(x) for x in re.findall(r"\[C(\d+)\]", r.text)})
        valid = {cid for cid, _ in partials}
        return {"answer": r.text.strip(), "communities": [c.cid for c in comms], "partials": partials, "cited": cited,
                "invalid_citations": [c for c in cited if c not in valid], "abstained": False}
