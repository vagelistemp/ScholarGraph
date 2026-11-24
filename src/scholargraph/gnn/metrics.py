"""Ranking and classification metrics for citation prediction.

Each evaluation positive (u, v) is compared with K negatives (papers u could have cited at publication time but did not). From the
scores we report

* **Hits@k** the share of positives ranked in the top k among the K + 1 candidates,
* **MRR**    the mean reciprocal rank,
* **AUC / AP** over positives against one random negative each (a balanced set).

Ties are resolved by the average rank, so a model that gives every candidate the same score cannot look good.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranks(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Rank of each positive among its negatives (1 = best); ties count as half a position."""
    greater = (neg > pos[:, None]).sum(1)
    equal = (neg == pos[:, None]).sum(1)
    return 1.0 + greater + 0.5 * equal


def ranking_metrics(pos: np.ndarray, neg: np.ndarray, ks=(1, 5, 10, 50)) -> Dict[str, float]:
    r = ranks(pos, neg)
    out = {f"hits@{k}": float((r <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / r).mean())
    out["mean_rank"] = float(r.mean())
    return out


def balanced_metrics(pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(pos))])
    s = np.concatenate([pos, neg[:, 0]])
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def all_metrics(pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
    return {**balanced_metrics(pos, neg), **ranking_metrics(pos, neg)}
