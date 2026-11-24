"""The feature-based baseline: hand-crafted pair features and a gradient-boosted classifier.

This is the classical recipe from the earlier NLP citation project (features of a pair -> boosted trees), adapted to the cold-start,
time-split task of this phase, and it is the fair opponent of the graph neural network: **same training pairs, same hidden-paper protocol,
same evaluation sets**. If a neural model cannot beat this, it has not earned its complexity.

Features of a pair (u, v), all computed on the message graph (which never contains u's own references):

=================  ======================================================================
text_cos           cosine similarity of the two text embeddings
log_indeg_v        log(1 + citations of v in the message graph): popularity / preferential attachment
year_gap           year(u) - year(v)
same_topic         1 if both have the same primary topic
same_venue         1 if both appeared in the same venue
common_authors     number of authors shared by u and v (self-citation)
author_prior       how often u's authors cited v in their earlier papers
author_prior_norm  the same, divided by the number of authors of u
=================  ======================================================================
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import CitationData, EvalSet, hide_sources, sample_negatives
from .metrics import all_metrics

FEATURES = ["text_cos", "log_indeg_v", "year_gap", "same_topic", "same_venue", "common_authors", "author_prior", "author_prior_norm"]


class PairFeaturizer:
    """Precomputes everything that depends on the message graph, then featurises any set of pairs."""

    def __init__(self, data: CitationData, message_edges: np.ndarray):
        self.d = data
        n = data.n
        refs = sp.csr_matrix((np.ones(len(message_edges), dtype=np.float32), (message_edges[:, 0], message_edges[:, 1])), shape=(n, n))
        self.indeg = np.asarray(refs.sum(axis=0)).ravel()
        self.C = (data.paper_author.T @ refs).tocsr()                  # author x paper: how often the author's papers cite the paper
        self.n_auth = np.asarray(data.paper_author.sum(axis=1)).ravel()

    def features(self, u: np.ndarray, v: np.ndarray) -> pd.DataFrame:
        d = self.d
        P = d.paper_author
        common = np.asarray(P[u].multiply(P[v]).sum(axis=1)).ravel()
        sub = P[u].tocoo()
        prior = np.bincount(sub.row, weights=np.asarray(self.C[sub.col, v[sub.row]]).ravel(), minlength=len(u))
        return pd.DataFrame({
            "text_cos": (d.text[u] * d.text[v]).sum(1),
            "log_indeg_v": np.log1p(self.indeg[v]),
            "year_gap": (d.year[u] - d.year[v]).astype(np.float32),
            "same_topic": (d.topic[u] == d.topic[v]).astype(np.float32),
            "same_venue": ((d.venue[u] == d.venue[v]) & (d.venue[u] >= 0)).astype(np.float32),
            "common_authors": common,
            "author_prior": prior,
            "author_prior_norm": prior / np.maximum(self.n_auth[u], 1),
        })[FEATURES]


def _classifier(seed: int):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)


def _score_eval(feat: PairFeaturizer, ev: EvalSet, scorer) -> Dict[str, float]:
    k = ev.neg.shape[1]
    pos = scorer(feat.features(ev.pos[:, 0], ev.pos[:, 1]))
    neg = scorer(feat.features(np.repeat(ev.pos[:, 0], k), ev.neg.reshape(-1))).reshape(len(ev.pos), k)
    return all_metrics(pos, neg)


def run_baseline(data: CitationData, val_set: EvalSet, test_set: EvalSet, seed: int = 0, hide_frac: float = 0.2, neg_per_pos: int = 1,
                 hard_frac: float = 0.5, log=print) -> Dict[str, Dict[str, float]]:
    """Train the boosted-tree baseline and the single-feature heuristics; return validation and test metrics for each."""
    rng = np.random.default_rng(seed)
    split = data.split()
    forbidden = data.all_pairs_codes()
    msg, sup = hide_sources(split["train"], hide_frac, rng)
    feat = PairFeaturizer(data, msg)
    neg_v = sample_negatives(data, sup[:, 0], neg_per_pos, rng, forbidden, hard_frac)
    u = np.concatenate([sup[:, 0], np.repeat(sup[:, 0], neg_per_pos)])
    v = np.concatenate([sup[:, 1], neg_v.reshape(-1)])
    y = np.concatenate([np.ones(len(sup)), np.zeros(len(sup) * neg_per_pos)])
    X = feat.features(u, v)
    clf = _classifier(seed).fit(X, y)
    log(f"baseline trained on {len(X):,} pairs with {len(FEATURES)} features")

    val_feat = PairFeaturizer(data, split["train"])                                                # validation: message graph = train
    test_feat = PairFeaturizer(data, np.vstack([split["train"], split["val"]]))                   # test: train + val
    boosted = lambda df: clf.predict_proba(df)[:, 1]  # noqa: E731
    out: Dict[str, Dict[str, float]] = {}
    for name, scorer in {"boosted_trees": boosted, "popularity": lambda df: df["log_indeg_v"].to_numpy(),
                         "text_cosine": lambda df: df["text_cos"].to_numpy(), "author_prior": lambda df: df["author_prior"].to_numpy()}.items():
        out[name] = {"val": _score_eval(val_feat, val_set, scorer), "test": _score_eval(test_feat, test_set, scorer)}
    out["boosted_trees"]["importance"] = dict(zip(FEATURES, getattr(clf, "feature_importances_", np.zeros(len(FEATURES))).astype(float).tolist()))
    return out
