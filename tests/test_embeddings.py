import tempfile
import unittest
from pathlib import Path

import numpy as np

from scholargraph.embeddings import HashingBagEncoder, TfidfSvdEncoder, embed_corpus, fingerprint


class CountingEncoder:
    """Wraps an encoder and counts how many texts it has been asked to encode."""
    dim = 8

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=0):
        self.calls += len(texts)
        rng = np.random.default_rng(len(texts))
        return rng.normal(size=(len(texts), self.dim)).astype(np.float32)


class EmbeddingCacheTest(unittest.TestCase):
    TEXTS = [f"paper number {i} about topic {i % 3}" for i in range(20)]

    def test_cache_is_reused_for_the_same_corpus(self):
        enc = CountingEncoder()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.npy"
            a = embed_corpus(enc, self.TEXTS, path, log=lambda *x: None, tag="t")
            first = enc.calls
            b = embed_corpus(enc, self.TEXTS, path, log=lambda *x: None, tag="t")
            self.assertEqual(enc.calls, first)                                  # second call read the cache
            np.testing.assert_array_equal(a, b)

    def test_cache_is_rejected_when_the_row_order_changes(self):
        """A reordered corpus has the same length but a different meaning for every row: the old check would have accepted it."""
        enc = CountingEncoder()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.npy"
            embed_corpus(enc, self.TEXTS, path, log=lambda *x: None)
            before = enc.calls
            embed_corpus(enc, list(reversed(self.TEXTS)), path, log=lambda *x: None)
            self.assertGreater(enc.calls, before)

    def test_cache_is_rejected_for_another_encoder_tag_or_missing_fingerprint(self):
        enc = CountingEncoder()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "e.npy"
            embed_corpus(enc, self.TEXTS, path, log=lambda *x: None, tag="a")
            before = enc.calls
            embed_corpus(enc, self.TEXTS, path, log=lambda *x: None, tag="b")
            self.assertGreater(enc.calls, before)
            before = enc.calls
            path.with_suffix(".fingerprint").unlink()
            embed_corpus(enc, self.TEXTS, path, log=lambda *x: None, tag="b")
            self.assertGreater(enc.calls, before)

    def test_fingerprint_depends_on_text_order_and_tag(self):
        self.assertNotEqual(fingerprint(["a", "b"]), fingerprint(["b", "a"]))
        self.assertNotEqual(fingerprint(["a", "b"], "x"), fingerprint(["a", "b"], "y"))
        self.assertEqual(fingerprint(["a", "b"], "x"), fingerprint(["a", "b"], "x"))

    def test_encoders_return_unit_vectors(self):
        texts = [f"graph neural network {i} learning" for i in range(30)]
        for enc in (TfidfSvdEncoder(8).fit(texts), HashingBagEncoder(16)):
            v = enc.encode(texts)
            np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
