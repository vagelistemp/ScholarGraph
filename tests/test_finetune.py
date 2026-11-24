import json
import unittest

import numpy as np
import torch

from scholargraph.demo import make_demo_gold, make_demo_text
from scholargraph.embeddings import HashingBagEncoder, TfidfSvdEncoder, embed_corpus
from scholargraph.finetune.contrastive import ContrastiveConfig, duplicate_mask, info_nce, train_contrastive
from scholargraph.finetune.lora_extractor import (ByteTokenizer, LoraConfigLite, attach_lora, build_sft_examples, evaluate_extractor,
                                                  format_target, make_batch, train_lora)
from scholargraph.finetune.pairs import PairSet, build_pairs, mine_hard_negatives
from scholargraph.finetune.retrieval_eval import evaluate_retrieval
from scholargraph.gnn.data import CitationData, load_citation_data


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = make_demo_gold(700, 21)
        cls.text = make_demo_text(cls.gold, 32)
        cls.data = load_citation_data(cls.gold, cls.text, 2022, 2023)
        p = cls.gold["papers"]
        cls.texts = (p["title"] + ". " + p["abstract"]).tolist()


class PairTest(Fixture):
    def test_no_future_information_in_training_pairs(self):
        d = self.data
        pairs = build_pairs(d, seed=0)
        self.assertGreater(len(pairs), 100)
        self.assertTrue((d.year[pairs.anchor] <= d.train_end).all())
        self.assertTrue((d.year[pairs.positive] <= d.year[pairs.anchor]).all())
        used = set(pairs.anchor.tolist()) | set(pairs.positive.tolist())
        late = set(np.flatnonzero(d.year > d.train_end).tolist())
        self.assertFalse(used & late)                                   # no text of a validation/test-period paper is ever trained on
        test = d.split()["test"]
        train_codes = set((pairs.anchor * d.n + pairs.positive).tolist())
        self.assertFalse(train_codes & set((test[:, 0] * d.n + test[:, 1]).tolist()))

    def test_hard_negatives_are_valid(self):
        d = self.data
        pairs = build_pairs(d, max_pairs=300, seed=1)
        emb = TfidfSvdEncoder(32).fit(self.texts).encode(self.texts)
        hard = mine_hard_negatives(d, pairs, emb, pool=15, seed=0)
        cited = {}
        for u, v in d.split()["train"]:
            cited.setdefault(int(u), set()).add(int(v))
        got = hard.hard_negative
        self.assertGreater((got >= 0).mean(), 0.9)
        for a, n in zip(hard.anchor[got >= 0], got[got >= 0]):
            self.assertLessEqual(d.year[n], d.year[a])
            self.assertLessEqual(d.year[n], d.train_end)
            self.assertNotIn(int(n), cited.get(int(a), set()))
            self.assertNotEqual(int(n), int(a))


class LossTest(unittest.TestCase):
    def test_duplicate_mask_marks_only_off_diagonal_repeats(self):
        cand = torch.tensor([5, 7, 5, 9])
        m = duplicate_mask(torch.arange(3), cand)
        self.assertTrue(bool(m[0, 2]) and bool(m[2, 0]))
        self.assertFalse(bool(m.diagonal().any()))
        self.assertEqual(int(m.sum()), 2)

    def test_loss_is_low_for_aligned_and_finite_with_duplicates(self):
        torch.manual_seed(0)
        a = torch.randn(8, 16)
        good = info_nce(a, a + 0.01 * torch.randn(8, 16), torch.arange(8), 0.05)
        bad = info_nce(a, torch.randn(8, 16), torch.arange(8), 0.05)
        self.assertLess(float(good), float(bad))
        dup = info_nce(a, a.clone(), torch.tensor([1, 1, 2, 3, 4, 5, 6, 7]), 0.05)
        self.assertTrue(torch.isfinite(dup))
        hard = info_nce(a, a.clone(), torch.arange(10), 0.05, n=torch.randn(2, 16))
        self.assertTrue(torch.isfinite(hard))


class RetrievalMetricTest(unittest.TestCase):
    def test_metrics_on_a_hand_made_ranking(self):
        # papers 0-3 are old; paper 4 (2022) cites 1 and 2. Similarity to the query: 1 > 0 > 2 > 3.
        import scipy.sparse as sp
        emb = np.array([[0.5, 0, 0], [0.9, 0, 0], [0.3, 0, 0], [0.1, 0, 0], [1.0, 0, 0]], dtype=np.float32)
        d = CitationData(ids=list("abcde"), year=np.array([2019, 2019, 2020, 2020, 2022]), topic=np.zeros(5, dtype=np.int64),
                         venue=-np.ones(5, dtype=np.int64), edges=np.array([[4, 1], [4, 2]]), paper_author=sp.csr_matrix((5, 1), dtype=np.float32),
                         text=emb, train_end=2020, val_year=2021)
        m = evaluate_retrieval(emb, d, ks=(1, 3), max_queries=None)
        self.assertEqual(m["n_queries"], 1)
        self.assertAlmostEqual(m["mrr"], 1.0)
        self.assertAlmostEqual(m["recall@1"], 0.5)
        self.assertAlmostEqual(m["recall@3"], 1.0)
        ideal = 1 + 1 / np.log2(3)
        self.assertAlmostEqual(m["ndcg@3"], (1 + 0.5) / ideal, places=5)


class TrainingTest(Fixture):
    def test_contrastive_training_improves_retrieval(self):
        d = self.data
        enc = HashingBagEncoder(dim=64, seed=0)
        before = evaluate_retrieval(embed_corpus(enc, self.texts, log=lambda *a: None), d, ks=(10,), max_queries=150)
        pairs = build_pairs(d, seed=0)
        hist = train_contrastive(enc, self.texts, pairs, ContrastiveConfig(epochs=3, batch_size=32, lr=5e-2, temperature=0.1, device="cpu",
                                                                           log_every=10 ** 9), log=lambda *a: None)
        after = evaluate_retrieval(embed_corpus(enc, self.texts, log=lambda *a: None), d, ks=(10,), max_queries=150)
        self.assertLess(np.mean([h["loss"] for h in hist[-5:]]), np.mean([h["loss"] for h in hist[:5]]))
        self.assertGreater(after["recall@10"], before["recall@10"])
        self.assertGreater(after["mrr"], before["mrr"])

    def test_sparse_tfidf_baseline_runs_through_the_same_evaluation(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(sublinear_tf=True).fit_transform(self.texts)
        m = evaluate_retrieval(X, self.data, ks=(10,), max_queries=100)
        self.assertGreater(m["recall@10"], 0.0)


class LoraTest(Fixture):
    def _tiny(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=259, n_positions=2048, n_embd=32, n_layer=2, n_head=2, pad_token_id=0, eos_token_id=1, bos_token_id=1)
        return GPT2LMHeadModel(cfg), ByteTokenizer()

    def test_prompt_tokens_are_masked_from_the_loss(self):
        tok = ByteTokenizer()
        ids, att, lab = make_batch(tok, [("abc", "xy")], 64, "cpu")
        self.assertEqual(ids.shape[1], 3 + 2 + 1)                                  # prompt + answer + eos
        self.assertEqual(lab[0, :3].tolist(), [-100] * 3)
        self.assertEqual(lab[0, 3:].tolist(), ids[0, 3:].tolist())
        ids, att, lab = make_batch(tok, [("a" * 100, "xy")], 20, "cpu")             # long prompts are cut from the left, never the answer
        self.assertEqual(ids.shape[1], 20)
        self.assertEqual(lab[0, -3:].tolist(), ids[0, -3:].tolist())

    def test_only_lora_weights_train_and_loss_falls(self):
        import copy
        model, tok = self._tiny()
        base_before = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})
        peft_model = attach_lora(model, LoraConfigLite(r=8, alpha=16, target_modules=("c_attn", "c_fc", "c_proj")), fan_in_fan_out=True)
        n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in peft_model.parameters())
        self.assertLess(n_train / n_all, 0.5)
        examples = [("### Abstract:\nthe cat sat\n\n### Concepts (JSON):\n", format_target(["transformer"], ["translation"], []))] * 16
        hist = train_lora(peft_model, tok, examples, LoraConfigLite(epochs=25, batch_size=4, lr=3e-3, device="cpu", max_len=256, log_every=10 ** 9), log=lambda *a: None)
        self.assertLess(hist[-1]["loss"], hist[0]["loss"] - 0.3)      # a frozen random toy model can only fit slowly; this checks the wiring
        after = peft_model.state_dict()
        import re
        for k, v in base_before.items():
            key = "base_model.model." + re.sub(r"(c_attn|c_fc|c_proj)\.(weight|bias)", r"\1.base_layer.\2", k)
            self.assertTrue(torch.equal(v, after[key]), f"frozen weight changed: {k}")
        self.assertTrue(any(("lora_B" in k) and bool((v != 0).any()) for k, v in after.items()))

    def test_sft_examples_and_evaluation_structure(self):
        papers = self.gold["papers"].head(5)
        import pandas as pd
        conc = pd.DataFrame({"paper_id": [papers["paper_id"].iloc[0]] * 2, "concept": ["transformer", "link prediction"], "kind": ["method", "task"], "source": ["t"] * 2})
        ex = build_sft_examples(papers, conc)
        self.assertEqual(len(ex), 5)
        self.assertEqual(json.loads(ex[0][1]), {"methods": ["transformer"], "tasks": ["link prediction"], "datasets": []})
        self.assertEqual(json.loads(ex[1][1]), {"methods": [], "tasks": [], "datasets": []})
        model, tok = self._tiny()
        res = evaluate_extractor(model.eval(), tok, ex[:2], max_new_tokens=8)
        self.assertEqual(set(res), {"json_valid", "precision", "recall", "f1"})


if __name__ == "__main__":
    unittest.main()
