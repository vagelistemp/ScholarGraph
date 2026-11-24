"""Direct handlers: answer exact, structured questions from the graph with no language model at all.

If a question is "how many papers cite *this paper*", a graph lookup is faster (microseconds), free, deterministic and cannot
hallucinate. Each handler recognises a family of phrasings with regular expressions, resolves the entities (a title must match **exactly
one** paper, a name exactly one author) and answers from the ``GraphStore``. Anything ambiguous returns ``None`` and falls through to a
more capable (and more expensive) route. The handlers are deliberately conservative: a wrong direct answer is worse than a slow one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ..graph.store import GraphStore


@dataclass
class HandlerResult:
    handler: str
    answer: str
    items: List[str]
    evidence: Dict


def _title_key(s: str) -> str:
    return " ".join(s.lower().split())


class DirectHandlers:
    def __init__(self, store: GraphStore):
        self.store = store
        self._rules: List[Tuple[str, re.Pattern, Callable[[re.Match], Optional[HandlerResult]]]] = []
        q = r'["“”](.+?)["“”]'
        add = lambda name, pat, fn: self._rules.append((name, re.compile(pat, re.I), fn))  # noqa: E731
        add("author_of", rf"^(?:who wrote|who are the authors of|who authored|authors of)(?: the paper(?: titled)?)? {q}\??$", self._author_of)
        add("cites_count", rf"^(?:how many (?:papers|works)(?: in the graph)? cite(?: the paper(?: titled)?)?|how many citations does(?: the paper(?: titled)?)?) {q}(?: have)?\??$", self._cites_count)
        add("references_of", rf"^(?:which papers does the paper titled|what does the paper titled|list the references of(?: the paper(?: titled)?)?|which papers are cited by(?: the paper(?: titled)?)?) {q}(?: cite)?\??$", self._references_of)
        add("co_cited", rf"^which papers cite both(?: the paper titled)? {q} and(?: the paper titled)? {q}\??$", self._co_cited)
        add("concept_year", r'^which papers published in (\d{4}) or later mention the concept ["“](.+?)["”]\??$', self._concept_year)
        add("top_cited_year", r"^what is the most cited paper(?: published)? in (\d{4})\??$", self._top_cited_year)
        add("coauthor", r"^who is the most frequent co-?author of (.+?)\??$", self._coauthor)

    # ------------------------------------------------------------------------------------------ entity resolution
    def _paper_by_title(self, title: str) -> Optional[str]:
        key = _title_key(title)
        ids = [p for p in self.store.find_papers_by_title(title, 3) if _title_key((self.store.paper(p) or {}).get("title", "")) == key]
        return ids[0] if len(ids) == 1 else None

    def _title(self, pid: str) -> str:
        return (self.store.paper(pid) or {}).get("title", pid)

    def _author_by_name(self, name: str) -> Optional[str]:
        hits = [a for a in self.store.find_author(name) if _title_key(a["name"] or "") == _title_key(name)]
        return hits[0]["author_id"] if len(hits) == 1 else None

    # ------------------------------------------------------------------------------------------ handlers
    def _author_of(self, m):
        pid = self._paper_by_title(m.group(1))
        if pid is None:
            return None
        names = [a["name"] for a in self.store.authors_of(pid) if a["name"]]
        return HandlerResult("author_of", ", ".join(names), names, {"paper_id": pid})

    def _cites_count(self, m):
        pid = self._paper_by_title(m.group(1))
        if pid is None:
            return None
        n = len(self.store.cited_by(pid))
        return HandlerResult("cites_count", f"{n} papers in the graph cite it.", [str(n)], {"paper_id": pid})

    def _references_of(self, m):
        pid = self._paper_by_title(m.group(1))
        if pid is None:
            return None
        titles = [self._title(r) for r in self.store.references(pid)]
        return HandlerResult("references_of", "; ".join(titles), titles, {"paper_id": pid})

    def _co_cited(self, m):
        a, b = self._paper_by_title(m.group(1)), self._paper_by_title(m.group(2))
        if a is None or b is None:
            return None
        titles = [self._title(p) for p in self.store.co_cited(a, b)]
        return HandlerResult("co_cited", "; ".join(titles), titles, {"a": a, "b": b})

    def _concept_year(self, m):
        ids = self.store.papers_with_concept(m.group(2), int(m.group(1)))
        titles = [self._title(p) for p in ids]
        return HandlerResult("concept_year", "; ".join(titles), titles, {"concept": m.group(2), "year": int(m.group(1))})

    def _top_cited_year(self, m):
        ids = self.store.top_cited(2, int(m.group(1)))
        if not ids:
            return None
        if len(ids) > 1 and len(self.store.cited_by(ids[0])) == len(self.store.cited_by(ids[1])):
            return None                                              # a tie: the question has no single right answer
        t = self._title(ids[0])
        return HandlerResult("top_cited_year", t, [t], {"year": int(m.group(1))})

    def _coauthor(self, m):
        aid = self._author_by_name(m.group(1).strip().strip('"'))
        if aid is None:
            return None
        co = self.store.coauthors(aid)
        if not co or (len(co) > 1 and co[0][1] == co[1][1]):
            return None
        name = self.store.author_name(co[0][0]) or co[0][0]
        return HandlerResult("coauthor", name, [name], {"author_id": aid})

    def handle(self, question: str) -> Optional[HandlerResult]:
        q = " ".join(question.strip().split())
        for _, pat, fn in self._rules:
            m = pat.match(q)
            if m:
                return fn(m)
        return None
