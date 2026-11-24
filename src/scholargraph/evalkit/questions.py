"""Evaluation questions generated *from the graph itself*, so that the correct answers are known without any hand labelling.

Seven question types, each answerable with the agent's tools and checkable against the graph:

===============  ====================================================================  =============================
type             template                                                              expected answer
===============  ====================================================================  =============================
author_of        Who wrote the paper titled "T"?                                       the set of author names
cites_count      How many papers in the graph cite the paper titled "T"?               a number
references_of    Which papers does the paper titled "T" cite?                          the set of titles
co_cited         Which papers cite both the paper titled "A" and the paper titled "B"? the set of titles
concept_year     Which papers published in Y or later mention the concept "C"?        the set of titles
top_cited_year   What is the most cited paper published in Y?                          one title
coauthor         Who is the most frequent co-author of A?                              one author name
===============  ====================================================================  =============================

Only unambiguous questions are generated (unique titles and names, answer sets of limited size, no ties for "most").
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..graph.store import MemoryGraphStore

EXPECTED_TOOLS = {
    "author_of": {"get_authors"}, "cites_count": {"get_citations"}, "references_of": {"get_references"}, "co_cited": {"co_cited"},
    "concept_year": {"papers_with_concept"}, "top_cited_year": {"top_cited"}, "coauthor": {"coauthors"},
}
MAX_ANSWER_SET = 10


@dataclass
class Question:
    qid: str
    qtype: str
    text: str
    answer: List[str]                       # normalised answer items (a number is a one-item list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def expected_tools(self):
        return EXPECTED_TOOLS[self.qtype]


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def generate_questions(store: MemoryGraphStore, per_type: int = 10, seed: int = 0) -> List[Question]:
    rng = np.random.default_rng(seed)
    papers = store.papers
    title_counts = Counter(papers["title"])
    unique = papers[papers["title"].map(title_counts) == 1]
    title = lambda pid: store.paper(pid)["title"]  # noqa: E731
    out: List[Question] = []

    def add(qtype, text, answer, **meta):
        out.append(Question(f"{qtype}-{sum(q.qtype == qtype for q in out):03d}", qtype, text, [norm(a) for a in answer], meta))

    def sample_ids(n):
        return unique["paper_id"].to_numpy()[rng.permutation(len(unique))][: n * 20]

    k = 0
    for pid in sample_ids(per_type):
        names = [a["name"] for a in store.authors_of(pid) if a["name"]]
        if names and len(set(names)) == len(names):
            add("author_of", f'Who wrote the paper titled "{title(pid)}"?', names, paper_id=pid)
            k += 1
        if k >= per_type:
            break
    k = 0
    for pid in sample_ids(per_type):
        add("cites_count", f'How many papers in the graph cite the paper titled "{title(pid)}"?', [str(len(store.cited_by(pid)))], paper_id=pid)
        k += 1
        if k >= per_type:
            break
    k = 0
    for pid in sample_ids(per_type):
        refs = store.references(pid)
        if 1 <= len(refs) <= MAX_ANSWER_SET and all(title_counts[title(r)] == 1 for r in refs):
            add("references_of", f'Which papers does the paper titled "{title(pid)}" cite?', [title(r) for r in refs], paper_id=pid)
            k += 1
        if k >= per_type:
            break
    k = 0
    top = store.top_cited(60)
    for _ in range(2000):
        a, b = rng.choice(top, 2, replace=False)
        common = store.co_cited(a, b)
        if 1 <= len(common) <= MAX_ANSWER_SET and all(title_counts[title(c)] == 1 for c in common) and title_counts[title(a)] == 1 and title_counts[title(b)] == 1:
            add("co_cited", f'Which papers cite both the paper titled "{title(a)}" and the paper titled "{title(b)}"?', [title(c) for c in common], a=a, b=b)
            k += 1
        if k >= per_type:
            break
    k = 0
    concepts = sorted(store._by_concept) if hasattr(store, "_by_concept") else []
    for _ in range(500 if concepts else 0):
        c, y = concepts[int(rng.integers(len(concepts)))], int(rng.integers(int(papers["year"].min()), int(papers["year"].max()) + 1))
        ids = store.papers_with_concept(c, y)
        if 1 <= len(ids) <= MAX_ANSWER_SET and all(title_counts[title(i)] == 1 for i in ids):
            add("concept_year", f'Which papers published in {y} or later mention the concept "{c}"?', [title(i) for i in ids], concept=c, year=y)
            k += 1
        if k >= per_type:
            break
    k = 0
    for y in rng.permutation(np.unique(papers["year"])):
        ids = store.top_cited(2, int(y))
        if len(ids) == 2 and len(store.cited_by(ids[0])) > len(store.cited_by(ids[1])) and title_counts[title(ids[0])] == 1:
            add("top_cited_year", f"What is the most cited paper published in {int(y)}?", [title(ids[0])], year=int(y))
            k += 1
        if k >= per_type:
            break
    k = 0
    name_counts = Counter(store.authors["name"])
    for aid in store.authors.sort_values("n_papers", ascending=False).index[:200]:
        co = store.coauthors(aid)
        nm = store.author_name(aid)
        if len(co) >= 2 and co[0][1] > co[1][1] and name_counts[nm] == 1 and name_counts[store.author_name(co[0][0])] == 1:
            add("coauthor", f"Who is the most frequent co-author of {nm}?", [store.author_name(co[0][0])], author_id=aid)
            k += 1
        if k >= per_type:
            break
    return out
