"""An answer verifier: a small trained classifier that estimates whether an agent's answer can be trusted.

An answer is more trustworthy when the concrete items it names (titles, author names, numbers) *appear in what the tools actually
returned* and *exist in the graph*, when the run had no errors, and when it finished normally. Those are cheap, model-free features, and a
logistic regression turns them into a probability. The verifier decides two things in the query engine: whether an answer may be stored in
the cache, and whether the system should try again with a stronger route.

Training data is built from runs of the agent: the correct runs are positives; **corrupted** copies (items replaced by other real
entities, by invented ones, partly replaced, or dropped) are negatives. This teaches the classifier the signature of a fabricated
answer. Honest limit: corruptions are synthetic; how well they resemble real model mistakes is unknown until real traces exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agent.agent import AgentResult
from ..graph.store import GraphStore

FEATURES = ["n_items", "has_items", "in_observations", "exist_in_graph", "n_errors", "n_steps", "finished", "answer_chars"]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def entity_exists(store: GraphStore, item: str) -> bool:
    if re.fullmatch(r"\d+", item.strip()):
        return True                                                # a number is checked by the observation feature, not here
    key = _norm(item)
    if any(_norm((store.paper(p) or {}).get("title", "")) == key for p in store.find_papers_by_title(item, 3)):
        return True
    return any(_norm(a["name"] or "") == key for a in store.find_author(item))


def observations_text(res: AgentResult) -> str:
    return _norm(" ".join(m["content"] for m in res.messages if m["role"] == "user" and m["content"].startswith("OBSERVATION")))


def answer_features(res: AgentResult, store: GraphStore) -> List[float]:
    items = res.items
    obs = observations_text(res)
    n = len(items)
    in_obs = sum(_norm(i) in obs for i in items) / n if n else 0.0
    exists = sum(entity_exists(store, i) for i in items) / n if n else 0.0
    return [float(n), float(n > 0), in_obs, exists, float(sum(1 for s in res.steps if s.error)), float(len(res.steps)), float(res.finished), float(len(res.answer))]


def corrupt(res: AgentResult, pool: Sequence[str], rng: np.random.Generator) -> AgentResult:
    """A wrong-answer variant of a correct run (same tool trace, different items)."""
    mode = int(rng.integers(4))
    n = max(len(res.items), 1)
    if mode == 0:
        items = [str(pool[int(rng.integers(len(pool)))]) for _ in range(n)]                   # real entities, but the wrong ones
    elif mode == 1:
        items = [f"Invented Result {int(rng.integers(10 ** 6))}" for _ in range(n)]            # fabricated
    elif mode == 2:
        items = list(res.items[: max(1, len(res.items) // 2)]) + [f"Invented Result {int(rng.integers(10 ** 6))}"]
    else:
        items = []
    return AgentResult(res.question, ", ".join(items), items, res.steps, res.finished, res.seconds, res.usage, res.messages)


@dataclass
class VerifierRow:
    features: List[float]
    label: int
    qid: str


def build_rows(results: Sequence[Tuple[str, AgentResult, bool]], store: GraphStore, pool: Sequence[str], seed: int = 0) -> List[VerifierRow]:
    """``results``: (question id, agent result, was it exactly correct). Correct runs give positives and corrupted copies give negatives."""
    rng = np.random.default_rng(seed)
    rows: List[VerifierRow] = []
    for qid, res, correct in results:
        rows.append(VerifierRow(answer_features(res, store), int(correct), qid))
        if correct:
            rows.append(VerifierRow(answer_features(corrupt(res, pool, rng), store), 0, qid))
    return rows


class AnswerVerifier:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._mu = self._sd = None

    def fit(self, rows: Sequence[VerifierRow]) -> "AnswerVerifier":
        from sklearn.linear_model import LogisticRegression
        X = np.array([r.features for r in rows], dtype=float)
        y = np.array([r.label for r in rows])
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-6
        self.model = LogisticRegression(C=5.0, max_iter=2000).fit((X - self._mu) / self._sd, y)
        return self

    def score_features(self, feats: Sequence[Sequence[float]]) -> np.ndarray:
        return self.model.predict_proba((np.array(feats, dtype=float) - self._mu) / self._sd)[:, 1]

    def score(self, res: AgentResult, store: GraphStore) -> float:
        return float(self.score_features([answer_features(res, store)])[0])

    def verify(self, res: AgentResult, store: GraphStore) -> bool:
        return self.score(res, store) >= self.threshold
