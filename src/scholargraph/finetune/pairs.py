"""Training pairs for contrastive fine-tuning, built from citations.

A citation is free supervision: if paper A cites paper B, B is relevant to A. We train an embedding model so that **the text of A is
close to the text of B** (and far from unrelated papers), which turns a general-purpose sentence encoder into a *citation-aware*
retriever, the engine of the recommender and of the RAG system in Phase 5.

Leakage rules (tested in ``tests/test_finetune.py``):

* only citations made by papers published up to ``train_end`` are used, so the validation and test years are never seen;
* consequently no text of a paper published after ``train_end`` appears in any training pair, neither as anchor nor as positive
  (a positive is older than its anchor);
* hard negatives are older than their anchor, are not cited by it, and come from the training period only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..gnn.data import CitationData


@dataclass
class PairSet:
    anchor: np.ndarray            # (p,) paper index of the citing paper
    positive: np.ndarray          # (p,) paper index of the cited paper
    hard_negative: np.ndarray     # (p,) paper index of a hard negative, or -1

    def __len__(self) -> int:
        return len(self.anchor)


def build_pairs(data: CitationData, max_pairs: Optional[int] = None, seed: int = 0) -> PairSet:
    edges = data.split()["train"]
    rng = np.random.default_rng(seed)
    if max_pairs is not None and len(edges) > max_pairs:
        edges = edges[rng.choice(len(edges), max_pairs, replace=False)]
    return PairSet(edges[:, 0].copy(), edges[:, 1].copy(), np.full(len(edges), -1, dtype=np.int64))


def mine_hard_negatives(data: CitationData, pairs: PairSet, emb: np.ndarray, pool: int = 30, seed: int = 0, chunk: int = 512) -> PairSet:
    """For every anchor pick, among its ``pool`` nearest papers (by ``emb``, e.g. a TF-IDF or base-model embedding) that are older, were
    published in the training period and are not cited by it, one at random. These are *plausible but wrong* papers, the informative
    negatives: random papers are too easy to tell apart."""
    rng = np.random.default_rng(seed)
    n = data.n
    cited = {}
    tr = data.split()["train"]
    for u, v in tr:
        cited.setdefault(int(u), set()).add(int(v))
    eligible = data.year <= data.train_end
    hard = np.full(len(pairs), -1, dtype=np.int64)
    for i in range(0, len(pairs), chunk):
        a = pairs.anchor[i:i + chunk]
        sims = emb[a] @ emb.T
        sims[:, ~eligible] = -np.inf
        sims[data.year[None, :].repeat(len(a), 0) > data.year[a][:, None]] = -np.inf          # only older papers
        sims[np.arange(len(a)), a] = -np.inf
        top = np.argpartition(-sims, kth=min(pool, n - 1) - 1, axis=1)[:, :pool]
        for r, anchor in enumerate(a):
            ok = [int(j) for j in top[r] if np.isfinite(sims[r, j]) and int(j) not in cited.get(int(anchor), ())]
            if ok:
                hard[i + r] = rng.choice(ok)
    return PairSet(pairs.anchor, pairs.positive, hard)
