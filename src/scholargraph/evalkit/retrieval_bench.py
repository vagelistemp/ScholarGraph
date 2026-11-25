"""Compare retrieval configurations on the same queries: the evidence for (or against) every retrieval upgrade.

Protocol (the same as in Phase 4): the queries are papers published after the validation year (we search with the paper's own title and
abstract), the relevant documents are the papers it cites, and only papers that existed at the query's date are candidates. Any
function ``search(query_text, query_index) -> list of paper indices`` can be evaluated, so BM25, dense, hybrid, graph-boosted, reranked
and multi-query configurations are all measured with identical code.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..gnn.data import CitationData


class BenchmarkLeak(ValueError):
    """The graph store used by the retriever contains test-period citations, i.e. the answers."""


def leak_free_tables(tables: Dict, data: CitationData) -> Dict:
    """A copy of the gold tables whose citations are only those made up to the validation year (the graph that exists before the test period)."""
    out = dict(tables)
    c = tables["citations"]
    out["citations"] = c[c["citing_year"] <= data.val_year].reset_index(drop=True)
    return out


def check_store_is_leak_free(store, data: CitationData, sample: int = 500) -> None:
    """Raise ``BenchmarkLeak`` if the store still knows the references of test-period papers.

    A query paper is itself in the index, so its own text ranks it first; a graph bonus then boosts *its neighbours*, which are exactly the
    answers. A new paper never comes with its references. (This was found when "hybrid + graph bonus" scored an implausible MRR of 0.76.)"""
    test = data.split()["test"]
    for u, v in test[:sample]:
        if data.ids[int(v)] in set(store.references(data.ids[int(u)])):
            raise BenchmarkLeak("the store contains test-period citations; build it from leak_free_tables(...)")


def evaluate_search_fn(search: Callable[[str, int], Sequence[int]], data: CitationData, texts: Sequence[str], ks: Sequence[int] = (5, 10, 20),
                       max_queries: Optional[int] = 200, seed: int = 0, store=None) -> Dict[str, float]:
    """Pass ``store`` (the graph store the search function uses, if any) so the leak guard can check it."""
    if store is not None:
        check_store_is_leak_free(store, data)
    rel: Dict[int, set] = {}
    for u, v in data.split()["test"]:
        rel.setdefault(int(u), set()).add(int(v))
    queries = np.array(sorted(rel))
    if max_queries and len(queries) > max_queries:
        queries = np.random.default_rng(seed).choice(queries, max_queries, replace=False)
    hits = {k: 0.0 for k in ks}
    mrr = 0.0
    kmax = max(ks)
    for q in queries:
        ranked = [int(i) for i in search(texts[int(q)], int(q))
                  if int(i) != int(q) and data.year[int(i)] <= data.year[int(q)]][:kmax]
        relevant = rel[int(q)]
        flags = [i in relevant for i in ranked]
        first = next((r for r, f in enumerate(flags, 1) if f), None)
        mrr += 1.0 / first if first else 0.0
        for k in ks:
            hits[k] += sum(flags[:k]) / len(relevant)
    n = len(queries)
    return {"n_queries": n, "mrr": mrr / n, **{f"recall@{k}": hits[k] / n for k in ks}}
