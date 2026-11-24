"""Text encoders: turn a paper's text into a vector.

Three interchangeable encoders share ``encode(texts) -> numpy array (n, dim)`` with L2-normalised rows (so a dot product is a
cosine similarity):

* ``TfidfSvdEncoder``    classical: TF-IDF followed by truncated SVD. No downloads, no GPU; the default in tests and a strong baseline.
* ``HashingBagEncoder``  a tiny *trainable* neural encoder (hashed word embeddings, averaged). Used by the tests and the ``demo`` so
                         that contrastive fine-tuning can run in seconds without downloading a model.
* ``HFEncoder``          a Transformer from the Hugging Face hub (default ``sentence-transformers/all-MiniLM-L6-v2``) with mean pooling.
                         This is the model Phase 4 fine-tunes. Needs ``torch`` and ``transformers``.

The two torch encoders are ``torch.nn.Module`` objects whose ``forward(texts)`` returns differentiable embeddings, which is what the
training loop of Phase 4 needs.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .common import pick_device


def l2_normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class TfidfSvdEncoder:
    name = "tfidf-svd"

    def __init__(self, dim: int = 128, max_features: int = 50_000):
        self.dim, self.max_features = dim, max_features
        self._tfidf = None
        self._svd = None

    def fit(self, texts: Sequence[str]) -> "TfidfSvdEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(max_features=self.max_features, sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english")
        X = self._tfidf.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[1] - 1, X.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def encode(self, texts: Sequence[str], batch_size: int = 0) -> np.ndarray:
        if self._tfidf is None:
            self.fit(texts)
        return l2_normalise(self._svd.transform(self._tfidf.transform(texts)).astype(np.float32))


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash(token: str, buckets: int) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % buckets


def _torch():
    import torch
    return torch


def make_torch_base():
    torch = _torch()

    class TorchTextEncoder(torch.nn.Module):
        """Base class: ``forward(texts)`` -> (batch, dim) tensor; ``encode`` -> L2-normalised numpy array, in batches, without gradients."""
        dim: int
        name: str

        def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
            was = self.training
            self.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    z = torch.nn.functional.normalize(self.forward(list(texts[i:i + batch_size])), dim=-1)
                    out.append(z.float().cpu().numpy())
            self.train(was)
            return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    return TorchTextEncoder


def HashingBagEncoder(dim: int = 64, buckets: int = 1 << 15, seed: int = 0):
    """Trainable bag-of-hashed-words encoder (tiny; for tests, demos and CPU-only experiments)."""
    torch = _torch()
    Base = make_torch_base()

    class _HashingBag(Base):
        name = "hashing-bag"

        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.dim, self.buckets = dim, buckets
            self.emb = torch.nn.EmbeddingBag(buckets, dim, mode="mean")
            self.proj = torch.nn.Linear(dim, dim)

        def forward(self, texts):
            ids, offsets = [], []
            for t in texts:
                offsets.append(len(ids))
                toks = _tokens(t) or ["<empty>"]
                ids.extend(_hash(w, self.buckets) for w in toks)
            dev = self.emb.weight.device
            z = self.emb(torch.tensor(ids, dtype=torch.long, device=dev), torch.tensor(offsets, dtype=torch.long, device=dev))
            return self.proj(z)

    return _HashingBag()


def HFEncoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 256, device: Optional[str] = None):
    """A Hugging Face Transformer with mean pooling over the non-padding tokens."""
    torch = _torch()
    from transformers import AutoModel, AutoTokenizer
    Base = make_torch_base()

    class _HF(Base):
        def __init__(self):
            super().__init__()
            self.name = model_name
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.dim = int(self.model.config.hidden_size)
            self.max_length = max_length
            self.to(device or pick_device())

        def forward(self, texts):
            dev = next(self.model.parameters()).device
            batch = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(dev)
            hidden = self.model(**batch).last_hidden_state                       # (batch, tokens, dim)
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)          # mean over real tokens only

    return _HF()


def embed_corpus(encoder, texts: Sequence[str], cache_path: Optional[Path] = None, batch_size: int = 64,
                 log: Callable[[str], None] = print) -> np.ndarray:
    """Encode all texts once and cache the matrix as ``.npy``. The cache is only valid for the same number of texts."""
    if cache_path and Path(cache_path).exists():
        arr = np.load(cache_path)
        if len(arr) == len(texts):
            return arr
    chunks = []
    for i in range(0, len(texts), 4096):
        chunks.append(encoder.encode(texts[i:i + 4096], batch_size=batch_size))
        log(f"  encoded {min(i + 4096, len(texts)):,} / {len(texts):,}")
    arr = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0, getattr(encoder, "dim", 0)), dtype=np.float32)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr
