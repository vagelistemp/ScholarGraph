"""BM25 ranking written from scratch with sparse matrices (the classic lexical retrieval function).

score(q, d) = sum over query terms t of  idf(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

Every document-term weight is precomputed once into a sparse matrix, so a query is just a sum of a few columns.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were which with we our can using use used based".split())


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: Dict[str, int] = {}
        self.W: sp.csc_matrix = sp.csc_matrix((0, 0))
        self.n_docs = 0

    def fit(self, texts: Sequence[str]) -> "BM25":
        rows, cols, vals = [], [], []
        vocab: Dict[str, int] = {}
        lengths = np.zeros(len(texts))
        for d, text in enumerate(texts):
            toks = tokenize(text)
            lengths[d] = len(toks)
            counts: Dict[int, int] = {}
            for t in toks:
                i = vocab.setdefault(t, len(vocab))
                counts[i] = counts.get(i, 0) + 1
            for i, c in counts.items():
                rows.append(d)
                cols.append(i)
                vals.append(c)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (len(texts) - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = max(lengths.mean(), 1e-9)
        tf = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * lengths[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm)
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.vocab, self.n_docs = vocab, len(texts)
        return self

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
