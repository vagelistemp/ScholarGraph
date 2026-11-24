"""Second-stage ranking: re-score the best candidates with a more expensive, more accurate model.

First-stage retrieval (BM25 + dense vectors) is fast because it compares the query with *precomputed* document representations. A
**cross-encoder** reads the query and one document *together* and can model their interaction, which is more accurate but needs one model
call per candidate: affordable only for the top 20 to 50. A **late-interaction** model (ColBERT-style) sits in between: it keeps one
vector per token and scores by "for each query token, its best-matching document token" (MaxSim).

All rerankers share one interface: ``score(query, docs) -> array`` (higher = more relevant).
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

import numpy as np

from .bm25 import tokenize


class Reranker:
    name = "reranker"

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class LexicalReranker(Reranker):
    """Dependency-free reranker: query-term coverage, weighted extra for words that appear in the *title* (the first sentence)."""
    name = "lexical"

    def __init__(self, title_weight: float = 0.6):
        self.title_weight = title_weight

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(docs))
        for i, d in enumerate(docs):
            title = set(tokenize(re.split(r"(?<=[.?!])\s", d, maxsplit=1)[0]))
            body = set(tokenize(d))
            if q:
                out[i] = (1 - self.title_weight) * len(q & body) / len(q) + self.title_weight * len(q & title) / len(q)
        return out


class EncoderReranker(Reranker):
    """Cosine similarity with a bi-encoder (cheap; useful when a stronger model than the first-stage one is available)."""
    name = "bi-encoder"

    def __init__(self, encoder):
        self.encoder = encoder

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        q = self.encoder.encode([query])[0]
        return np.asarray(self.encoder.encode(list(docs))) @ q


class CrossEncoderReranker(Reranker):
    """A Hugging Face sequence-classification cross-encoder (default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``).

    Pass ``score_fn`` to inject a scoring function (used by the tests, so nothing is downloaded)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None, max_length: int = 384,
                 batch_size: int = 16, score_fn: Optional[Callable[[str, Sequence[str]], Sequence[float]]] = None):
        self.name = f"cross-encoder:{model_name}"
        self._fn = score_fn
        self.max_length, self.batch_size = max_length, batch_size
        if score_fn is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from ..common import pick_device
            self.device = device or pick_device()
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
            self._torch = torch

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        if self._fn is not None:
            return np.asarray(self._fn(query, docs), dtype=float)
        out: List[float] = []
        with self._torch.no_grad():
            for i in range(0, len(docs), self.batch_size):
                b = self.tok([query] * len(docs[i:i + self.batch_size]), list(docs[i:i + self.batch_size]), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**b).logits
                out += (logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]).float().cpu().tolist()
        return np.asarray(out)


def maxsim(q_tokens: np.ndarray, d_tokens: np.ndarray) -> float:
    """ColBERT's late-interaction score: for each query token take its best cosine match among the document tokens, then sum.

    Rows must be L2-normalised token vectors (query tokens x dim, document tokens x dim)."""
    if len(q_tokens) == 0 or len(d_tokens) == 0:
        return 0.0
    return float((q_tokens @ d_tokens.T).max(axis=1).sum())


class LateInteractionReranker(Reranker):
    """MaxSim reranker. ``token_encode(text) -> (n_tokens, dim)`` L2-normalised token vectors (e.g. from a Transformer's last hidden states)."""
    name = "late-interaction"

    def __init__(self, token_encode: Callable[[str], np.ndarray]):
        self.token_encode = token_encode

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        q = self.token_encode(query)
        return np.array([maxsim(q, self.token_encode(d)) for d in docs])


def rerank_ids(reranker: Reranker, query: str, ids: Sequence[str], texts: Sequence[str], top_k: int, original_scores: Optional[Sequence[float]] = None,
               blend: float = 0.0) -> List[int]:
    """Order candidate positions by reranker score (stable on ties). ``blend`` in [0, 1] mixes in the first-stage rank (0 = rerank only)."""
    s = np.asarray(reranker.score(query, texts), dtype=float)
    if blend > 0 and original_scores is not None:
        r1 = np.argsort(np.argsort(-np.asarray(original_scores))) + 1
        r2 = np.argsort(np.argsort(-s)) + 1
        s = -(blend * r1 + (1 - blend) * r2)
    order = np.argsort(-s, kind="stable")
    return [int(i) for i in order[:top_k]]
