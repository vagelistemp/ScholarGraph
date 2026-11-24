"""Offline evaluation of a retriever on citation data: *given a new paper, can we find the papers it cites?*

Queries are papers published after ``val_year`` (the test period, never used for training). For each query the candidate pool is every
paper published no later than the query (excluding the query), and the relevant documents are the papers it actually cites. We report
Recall@k, MRR and nDCG@k averaged over queries. The same function evaluates any embedding matrix, so the base encoder, the fine-tuned
encoder and a TF-IDF baseline are compared on exactly the same queries.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ..gnn.data import CitationData


def evaluate_retrieval(emb, data: CitationData, ks: Sequence[int] = (1, 5, 10, 20, 50), max_queries: Optional[int] = 2000, seed: int = 0,
                       chunk: int = 256) -> Dict[str, float]:
    """``emb``: (n, d) dense L2-normalised matrix, or a scipy sparse matrix with L2-normalised rows (TF-IDF)."""
    test_edges = data.split()["test"]
    rel: Dict[int, set] = {}
    for u, v in test_edges:
        rel.setdefault(int(u), set()).add(int(v))
    queries = np.array(sorted(rel))
    if max_queries and len(queries) > max_queries:
        queries = np.random.default_rng(seed).choice(queries, max_queries, replace=False)
    kmax = max(ks)
    hits = {k: 0.0 for k in ks}
    mrr = 0.0
    ndcg = {k: 0.0 for k in ks}
    discount = 1.0 / np.log2(np.arange(2, kmax + 2))
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk]
        S = emb[q] @ emb.T
        S = S.toarray() if sp.issparse(S) else np.asarray(S)
        S[data.year[None, :] > data.year[q][:, None]] = -np.inf                       # only papers that existed at the query's date
        S[np.arange(len(q)), q] = -np.inf
        top = np.argpartition(-S, kth=min(kmax, S.shape[1] - 1), axis=1)[:, :kmax]
        for r, qi in enumerate(q):
            order = top[r][np.argsort(-S[r, top[r]])]
            relevant = rel[int(qi)]
            is_rel = np.array([int(j) in relevant for j in order])
            first = np.flatnonzero(is_rel)
            mrr += 1.0 / (first[0] + 1) if len(first) else 0.0
            for k in ks:
                hits[k] += is_rel[:k].sum() / len(relevant)                             # recall@k: share of the cited papers found in the top k
                dcg = (is_rel[:k] * discount[:k]).sum()
                ideal = discount[:min(len(relevant), k)].sum()
                ndcg[k] += dcg / ideal
    nq = len(queries)
    return {"n_queries": nq, "mrr": mrr / nq, **{f"recall@{k}": hits[k] / nq for k in ks}, **{f"ndcg@{k}": ndcg[k] / nq for k in ks}}
