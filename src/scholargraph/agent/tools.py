"""The tools an agent can call: typed, documented functions over the knowledge graph and the retriever.

Each tool returns a small JSON-serialisable result (lists are capped), because every character of an observation is paid for in model
tokens and long observations make small models lose the thread.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..graph.store import GraphStore
from ..rag.retrieval import HybridRetriever

MAX_ITEMS = 12


@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, str]
    fn: Callable[..., Any]

    def spec(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.params.items())
        return f"- {self.name}({args}): {self.description}"


def _brief(store: GraphStore, ids: List[str]) -> List[Dict[str, Any]]:
    out = []
    for pid in ids[:MAX_ITEMS]:
        p = store.paper(pid) or {}
        out.append({"paper_id": pid, "title": p.get("title"), "year": p.get("year")})
    return out


def make_tools(store: GraphStore, retriever: Optional[HybridRetriever] = None) -> Dict[str, Tool]:
    def search_papers(query: str, k: int = 5, year_from: Optional[int] = None):
        ids: List[str] = []
        exact = store.find_papers_by_title(query, 1)                                      # a pasted title must find that paper, always
        if exact and " ".join((store.paper(exact[0]) or {}).get("title", "").lower().split()) == " ".join(query.lower().split()):
            ids.append(exact[0])
        if retriever is None:
            ids += [p for p in store.find_papers_by_title(query, int(k)) if p not in ids]
        else:
            ids += [h.paper_id for h in retriever.search(query, k=int(k), year_from=year_from, graph_weight=0.2) if h.paper_id not in ids]
        return {"results": _brief(store, ids[:max(int(k), 1)])}

    def get_paper(paper_id: str):
        p = store.paper(paper_id)
        if not p:
            return {"error": f"unknown paper_id {paper_id}"}
        return {"paper_id": paper_id, "title": p["title"], "year": p["year"], "abstract": (p.get("abstract") or "")[:600],
                "citations_in_graph": p.get("citations_in_graph"), "concepts": store.concepts_of(paper_id)[:8]}

    def get_authors(paper_id: str):
        return {"authors": [{"author_id": a["author_id"], "name": a["name"]} for a in store.authors_of(paper_id)]}

    def get_references(paper_id: str):
        ids = store.references(paper_id)
        return {"count": len(ids), "results": _brief(store, ids)}

    def get_citations(paper_id: str):
        ids = store.cited_by(paper_id)
        return {"count": len(ids), "results": _brief(store, ids)}

    def find_author(name: str):
        return {"authors": store.find_author(name)[:MAX_ITEMS]}

    def author_papers(author_id: str):
        ids = store.papers_of_author(author_id)
        return {"count": len(ids), "results": _brief(store, ids)}

    def coauthors(author_id: str, k: int = 5):
        rows = store.coauthors(author_id)[: int(k)]
        return {"coauthors": [{"author_id": aid, "name": store.author_name(aid), "shared_papers": n} for aid, n in rows]}

    def papers_with_concept(concept: str, year_from: Optional[int] = None):
        ids = store.papers_with_concept(concept, year_from)
        return {"count": len(ids), "results": _brief(store, ids)}

    def top_cited(k: int = 5, year: Optional[int] = None, topic: Optional[str] = None):
        return {"results": _brief(store, store.top_cited(int(k), year, topic))}

    def co_cited(paper_a: str, paper_b: str):
        ids = store.co_cited(paper_a, paper_b)
        return {"count": len(ids), "results": _brief(store, ids)}

    def shortest_path(paper_a: str, paper_b: str, max_hops: int = 4):
        path = store.shortest_path(paper_a, paper_b, int(max_hops))
        return {"path": None if path is None else _brief(store, path)}

    tools = [
        Tool("search_papers", "find papers relevant to a text query (title words or a topic)", {"query": "str", "k": "int=5", "year_from": "int|null"}, search_papers),
        Tool("get_paper", "title, year, abstract, citation count and concepts of one paper", {"paper_id": "str"}, get_paper),
        Tool("get_authors", "authors of a paper", {"paper_id": "str"}, get_authors),
        Tool("get_references", "papers cited BY the given paper", {"paper_id": "str"}, get_references),
        Tool("get_citations", "papers that cite the given paper (count and list)", {"paper_id": "str"}, get_citations),
        Tool("find_author", "authors whose name contains the text", {"name": "str"}, find_author),
        Tool("author_papers", "papers written by an author", {"author_id": "str"}, author_papers),
        Tool("coauthors", "the author's most frequent co-authors", {"author_id": "str", "k": "int=5"}, coauthors),
        Tool("papers_with_concept", "papers mentioning a method / task / dataset concept, optionally from a year on", {"concept": "str", "year_from": "int|null"}, papers_with_concept),
        Tool("top_cited", "most cited papers in the graph, optionally for one year or topic", {"k": "int=5", "year": "int|null", "topic": "str|null"}, top_cited),
        Tool("co_cited", "papers that cite both of two papers", {"paper_a": "str", "paper_b": "str"}, co_cited),
        Tool("shortest_path", "shortest citation path between two papers", {"paper_a": "str", "paper_b": "str", "max_hops": "int=4"}, shortest_path),
    ]
    return {t.name: t for t in tools}
