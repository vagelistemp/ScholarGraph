"""The knowledge-graph query interface and its in-memory implementation.

``GraphStore`` is the *only* way the agent (Phase 5) and the evaluation harness touch the graph, so they work identically on top of

* ``MemoryGraphStore``  built from the gold tables with sparse matrices (no server, used by tests and for local runs), and
* ``Neo4jGraphStore``   (``neo4j_store.py``) running the same questions as Cypher queries on a Neo4j server.

A parity test in ``tests/test_graph.py`` checks that the memory store answers every method correctly on a graph whose answers are
known from the way it was generated; the Neo4j store has the same method signatures and is checked against the memory store when a
server is available (set ``NEO4J_URI``).

Graph model (see ``schema.py`` for the Cypher side)::

    (Author)-[:WROTE {position}]->(Paper)-[:CITES]->(Paper)
    (Paper)-[:PUBLISHED_IN]->(Venue)    (Paper)-[:ABOUT {score}]->(Topic)    (Paper)-[:MENTIONS {source}]->(Concept)
    (Author)-[:AFFILIATED_WITH]->(Institution)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


class GraphStore(ABC):
    @abstractmethod
    def stats(self) -> Dict[str, int]: ...

    @abstractmethod
    def paper(self, paper_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def references(self, paper_id: str) -> List[str]:
        """Papers that ``paper_id`` cites."""

    @abstractmethod
    def cited_by(self, paper_id: str) -> List[str]:
        """Papers that cite ``paper_id``."""

    @abstractmethod
    def authors_of(self, paper_id: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def papers_of_author(self, author_id: str) -> List[str]: ...

    @abstractmethod
    def find_author(self, name: str) -> List[Dict[str, Any]]:
        """Authors whose name contains ``name`` (case-insensitive)."""

    @abstractmethod
    def coauthors(self, author_id: str) -> List[Tuple[str, int]]:
        """(author_id, number of shared papers), most frequent first."""

    @abstractmethod
    def concepts_of(self, paper_id: str) -> List[str]: ...

    @abstractmethod
    def papers_with_concept(self, concept: str, year_from: Optional[int] = None) -> List[str]: ...

    @abstractmethod
    def co_cited(self, a: str, b: str) -> List[str]:
        """Papers that cite both ``a`` and ``b``."""

    @abstractmethod
    def shortest_path(self, a: str, b: str, max_hops: int = 4) -> Optional[List[str]]:
        """Shortest path between two papers over citation links in either direction (None if farther than ``max_hops``)."""

    @abstractmethod
    def top_cited(self, k: int = 10, year: Optional[int] = None, topic: Optional[str] = None) -> List[str]:
        """Most cited papers by their number of citations *inside the graph*."""

    @abstractmethod
    def find_papers_by_title(self, text: str, k: int = 5) -> List[str]: ...


class MemoryGraphStore(GraphStore):
    def __init__(self, tables: Mapping[str, pd.DataFrame], concepts: Optional[pd.DataFrame] = None):
        p = tables["papers"].reset_index(drop=True)
        self.papers = p
        self.idx = {pid: i for i, pid in enumerate(p["paper_id"])}
        self.ids = p["paper_id"].tolist()
        n = len(p)
        c = tables["citations"]
        r = c["citing_id"].map(self.idx).to_numpy()
        col = c["cited_id"].map(self.idx).to_numpy()
        self.A = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, col)), shape=(n, n))      # citing -> cited
        self.AT = self.A.T.tocsr()
        self.indeg = np.asarray(self.AT.sum(axis=1)).ravel()
        ash = tables["authorships"].sort_values(["paper_id", "position"])
        self._authors_of = {pid: g for pid, g in ash.groupby("paper_id")}
        self._papers_of = {aid: g["paper_id"].tolist() for aid, g in ash.groupby("author_id")}
        self.authors = tables["authors"].set_index("author_id")
        self._author_names_lower = self.authors["name"].fillna("").str.lower()
        self._topic_of = dict(zip(p["paper_id"], p["primary_topic_name"]))
        self._title_tokens = [set(str(t).lower().split()) for t in p["title"]]
        self.concepts = concepts if concepts is not None else pd.DataFrame(columns=["paper_id", "concept", "kind", "source"])
        self._concepts_of = {pid: g["concept"].tolist() for pid, g in self.concepts.groupby("paper_id")}
        self._by_concept = {cn: g["paper_id"].tolist() for cn, g in self.concepts.groupby("concept")}
        self._year = dict(zip(p["paper_id"], p["year"]))

    # -------------------------------------------------------------------------------------------- lookups
    def stats(self) -> Dict[str, int]:
        return {"papers": len(self.papers), "authors": len(self.authors), "citations": int(self.A.nnz),
                "authorships": int(sum(len(g) for g in self._authors_of.values())), "concept_links": len(self.concepts)}

    def paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        i = self.idx.get(paper_id)
        if i is None:
            return None
        row = self.papers.iloc[i].to_dict()
        row["citations_in_graph"] = int(self.indeg[i])
        return row

    def references(self, paper_id: str) -> List[str]:
        i = self.idx.get(paper_id)
        return [] if i is None else [self.ids[j] for j in self.A.indices[self.A.indptr[i]:self.A.indptr[i + 1]]]

    def cited_by(self, paper_id: str) -> List[str]:
        i = self.idx.get(paper_id)
        return [] if i is None else [self.ids[j] for j in self.AT.indices[self.AT.indptr[i]:self.AT.indptr[i + 1]]]

    def authors_of(self, paper_id: str) -> List[Dict[str, Any]]:
        g = self._authors_of.get(paper_id)
        if g is None:
            return []
        return [{"author_id": a, "name": self.authors["name"].get(a), "position": int(pos)} for a, pos in zip(g["author_id"], g["position"])]

    def papers_of_author(self, author_id: str) -> List[str]:
        return list(self._papers_of.get(author_id, []))

    def find_author(self, name: str) -> List[Dict[str, Any]]:
        hit = self.authors[self._author_names_lower.str.contains(name.lower(), regex=False)]
        return [{"author_id": a, "name": r["name"], "n_papers": int(r["n_papers"])} for a, r in hit.sort_values("n_papers", ascending=False).iterrows()]

    def coauthors(self, author_id: str) -> List[Tuple[str, int]]:
        cnt: Counter = Counter()
        for pid in self._papers_of.get(author_id, []):
            for a in self._authors_of[pid]["author_id"]:
                if a != author_id:
                    cnt[a] += 1
        return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))

    def concepts_of(self, paper_id: str) -> List[str]:
        return list(self._concepts_of.get(paper_id, []))

    def papers_with_concept(self, concept: str, year_from: Optional[int] = None) -> List[str]:
        ids = self._by_concept.get(concept.lower(), [])
        return [p for p in ids if year_from is None or self._year[p] >= year_from]

    def co_cited(self, a: str, b: str) -> List[str]:
        return sorted(set(self.cited_by(a)) & set(self.cited_by(b)))

    def shortest_path(self, a: str, b: str, max_hops: int = 4) -> Optional[List[str]]:
        if a not in self.idx or b not in self.idx:
            return None
        if a == b:
            return [a]
        prev = {self.idx[a]: None}
        frontier = deque([(self.idx[a], 0)])
        target = self.idx[b]
        while frontier:
            u, d = frontier.popleft()
            if d >= max_hops:
                continue
            nbrs = np.concatenate([self.A.indices[self.A.indptr[u]:self.A.indptr[u + 1]], self.AT.indices[self.AT.indptr[u]:self.AT.indptr[u + 1]]])
            for v in np.unique(nbrs):
                v = int(v)
                if v in prev:
                    continue
                prev[v] = u
                if v == target:
                    path, cur = [], v
                    while cur is not None:
                        path.append(self.ids[cur])
                        cur = prev[cur]
                    return path[::-1]
                frontier.append((v, d + 1))
        return None

    def top_cited(self, k: int = 10, year: Optional[int] = None, topic: Optional[str] = None) -> List[str]:
        mask = np.ones(len(self.papers), dtype=bool)
        if year is not None:
            mask &= (self.papers["year"] == year).to_numpy()
        if topic is not None:
            mask &= (self.papers["primary_topic_name"] == topic).to_numpy()
        cand = np.flatnonzero(mask)
        order = cand[np.lexsort((np.array(self.ids, dtype=object)[cand], -self.indeg[cand]))]
        return [self.ids[i] for i in order[:k]]

    def find_papers_by_title(self, text: str, k: int = 5) -> List[str]:
        q = set(text.lower().split())
        if not q:
            return []
        scores = np.array([len(q & t) / len(q) for t in self._title_tokens])
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in order if scores[i] > 0]
