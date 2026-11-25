"""Citation-prediction data with a leakage-free *temporal* split.

Task: given a **new paper** (published in the validation or test years), rank the older papers it is likely to cite.

Splits are by the **citing paper's year**, so the future never leaks into the past:

    train  citations made by papers published up to ``train_end``       (the graph the model learns from)
    val    citations made by papers published in ``val_year``           (model selection, early stopping)
    test   citations made by papers published after ``val_year``        (reported once)

Message passing graph: train edges when validating, train + val edges when testing. The query paper itself has no outgoing edges in
that graph (its references are exactly what we try to predict), so the model must rely on its text, authors and the structure around
older papers: it is a **cold-start** problem.

Mimicking cold start while training (the lesson of the earlier NLP citation project: *features of a training example must be computed
on a graph that does not contain its own answer*): every epoch a random share of the training papers is "hidden": all their
outgoing edges become supervision targets and are removed from the message graph, so the hidden papers look exactly like the new
papers of the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import TRAIN_END, VAL_YEAR


@dataclass
class CitationData:
    ids: List[str]
    year: np.ndarray                    # (n,) publication year
    topic: np.ndarray                   # (n,) integer topic id
    venue: np.ndarray                   # (n,) integer venue id (-1 = unknown)
    edges: np.ndarray                   # (m, 2) citing, cited (paper indices)
    paper_author: sp.csr_matrix         # (n, n_authors) 0/1
    text: np.ndarray                    # (n, d) L2-normalised text embeddings
    train_end: int = TRAIN_END
    val_year: int = VAL_YEAR

    @property
    def n(self) -> int:
        return len(self.ids)

    @cached_property
    def topic_index(self):
        """(paper indices sorted by topic, start offset of each topic, size of each topic): O(1) sampling from a topic."""
        order = np.argsort(self.topic, kind="stable")
        counts = np.bincount(self.topic, minlength=int(self.topic.max()) + 1)
        start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return order, start, counts

    def edge_year(self) -> np.ndarray:
        return self.year[self.edges[:, 0]]

    def split(self) -> Dict[str, np.ndarray]:
        y = self.edge_year()
        return {"train": self.edges[y <= self.train_end], "val": self.edges[y == self.val_year], "test": self.edges[y > self.val_year]}

    def all_pairs_codes(self) -> np.ndarray:
        """Sorted codes u * n + v of every known citation (used to reject negatives that are real citations)."""
        return np.sort(self.edges[:, 0].astype(np.int64) * self.n + self.edges[:, 1])


def load_citation_data(gold: Dict[str, pd.DataFrame], text: np.ndarray, train_end: int = TRAIN_END, val_year: int = VAL_YEAR) -> CitationData:
    papers = gold["papers"].reset_index(drop=True)
    idx = {p: i for i, p in enumerate(papers["paper_id"])}
    c = gold["citations"]
    edges = np.stack([c["citing_id"].map(idx).to_numpy(), c["cited_id"].map(idx).to_numpy()], axis=1).astype(np.int64)
    ash = gold["authorships"]
    aidx = {a: i for i, a in enumerate(sorted(ash["author_id"].unique()))}
    P = sp.csr_matrix((np.ones(len(ash), dtype=np.float32), (ash["paper_id"].map(idx).to_numpy(), ash["author_id"].map(aidx).to_numpy())),
                      shape=(len(papers), max(len(aidx), 1)))
    topic = pd.factorize(papers["primary_topic_id"].fillna("?"))[0].astype(np.int64)
    venue = pd.factorize(papers["venue_id"].fillna("?"))[0].astype(np.int64)
    venue[papers["venue_id"].isna().to_numpy()] = -1
    assert len(text) == len(papers), "text embeddings must be aligned with gold papers (sorted by paper_id)"
    return CitationData(papers["paper_id"].tolist(), papers["year"].to_numpy(np.int64), topic, venue, edges, P, text.astype(np.float32), train_end, val_year)


# ---------------------------------------------------------------------------------------------- negatives
def _in_codes(codes: np.ndarray, sorted_codes: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_codes, codes)
    return (pos < len(sorted_codes)) & (sorted_codes[np.minimum(pos, len(sorted_codes) - 1)] == codes)


def sample_negatives(data: CitationData, src: np.ndarray, k: int, rng: np.random.Generator, forbidden: np.ndarray,
                     hard_frac: float = 0.0) -> np.ndarray:
    """``k`` negative cited papers for each source paper: published no later than the source, not the source itself and never a real
    citation of it. ``hard_frac`` of them are drawn from the source's own topic (harder than uniform random papers)."""
    n = data.n
    order, start, counts = data.topic_index                  # papers sorted by topic, built once per dataset
    out = np.full((len(src), k), -1, dtype=np.int64)
    for attempt in range(60):
        need = np.argwhere(out < 0)
        if len(need) == 0:
            break
        rows, cols = need[:, 0], need[:, 1]
        cand = rng.integers(0, n, size=len(need))
        if hard_frac > 0:                                    # vectorised draw of a random paper from the source's own topic
            hard = rng.random(len(need)) < hard_frac
            t = data.topic[src[rows]]
            pick = start[t] + (rng.random(len(need)) * counts[t]).astype(np.int64)
            cand = np.where(hard, order[np.minimum(pick, n - 1)], cand)
        u = src[rows]
        ok = (cand != u) & (data.year[cand] <= data.year[u]) & ~_in_codes(u.astype(np.int64) * n + cand, forbidden)
        out[rows[ok], cols[ok]] = cand[ok]
    if (out < 0).any():                                    # rare: a source with (almost) no admissible candidates; drop those rows' gaps
        out[out < 0] = rng.integers(0, n, size=int((out < 0).sum()))
    return out


# ---------------------------------------------------------------------------------------------- hiding papers during training
def hide_sources(train_edges: np.ndarray, frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Hide a random ``frac`` of the citing papers: (message edges, supervision edges).

    A hidden paper must look like a brand-new paper, which has **neither references nor citations** yet. So all its outgoing edges become
    the supervision targets, and its *incoming* edges (later papers citing it) are removed from the message graph too. Hiding only the
    outgoing edges (an earlier version of this function) left hidden papers with citers, information a new paper never has; the model
    learned to rely on it and validation performance collapsed (found on the first real-data run)."""
    srcs = np.unique(train_edges[:, 0])
    hidden = rng.choice(srcs, size=max(1, int(round(frac * len(srcs)))), replace=False)
    out_mask = np.isin(train_edges[:, 0], hidden)
    in_mask = np.isin(train_edges[:, 1], hidden)
    return train_edges[~(out_mask | in_mask)], train_edges[out_mask]


# ---------------------------------------------------------------------------------------------- evaluation sets
@dataclass
class EvalSet:
    pos: np.ndarray            # (p, 2) citing, cited
    neg: np.ndarray            # (p, K) negative cited papers for every positive pair


def make_eval_set(data: CitationData, edges: np.ndarray, k_neg: int = 99, max_pairs: Optional[int] = 20000, seed: int = 0) -> EvalSet:
    rng = np.random.default_rng(seed)
    if max_pairs is not None and len(edges) > max_pairs:
        edges = edges[rng.choice(len(edges), max_pairs, replace=False)]
    neg = sample_negatives(data, edges[:, 0], k_neg, rng, data.all_pairs_codes())
    return EvalSet(edges, neg)
