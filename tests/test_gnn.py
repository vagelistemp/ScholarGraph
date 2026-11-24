import unittest

import numpy as np
import torch

from scholargraph.demo import make_demo_gold, make_demo_text
from scholargraph.gnn.baseline import PairFeaturizer, run_baseline
from scholargraph.gnn.data import hide_sources, load_citation_data, make_eval_set, sample_negatives
from scholargraph.gnn.metrics import all_metrics, ranking_metrics, ranks
from scholargraph.gnn.model import Adjacency, CitationGNN, node_features
from scholargraph.gnn.train import TrainConfig, train_gnn


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = make_demo_gold(700, 11)
        cls.text = make_demo_text(cls.gold, 32)
        cls.data = load_citation_data(cls.gold, cls.text, train_end=2022, val_year=2023)
        cls.split = cls.data.split()


class SplitTest(Fixture):
    def test_split_is_temporal_and_disjoint(self):
        d, s = self.data, self.split
        self.assertTrue((d.year[s["train"][:, 0]] <= 2022).all())
        self.assertTrue((d.year[s["val"][:, 0]] == 2023).all())
        self.assertTrue((d.year[s["test"][:, 0]] > 2023).all())
        self.assertEqual(sum(len(v) for v in s.values()), len(d.edges))
        codes = {k: set((v[:, 0] * d.n + v[:, 1]).tolist()) for k, v in s.items()}
        self.assertFalse(codes["train"] & codes["val"] or codes["train"] & codes["test"] or codes["val"] & codes["test"])

    def test_hidden_papers_have_no_references_in_message_graph(self):
        """The property the whole protocol rests on (compare the spy-builder test of the earlier project)."""
        rng = np.random.default_rng(0)
        msg, sup = hide_sources(self.split["train"], 0.2, rng)
        self.assertEqual(len(msg) + len(sup), len(self.split["train"]))
        mc = set((msg[:, 0] * self.data.n + msg[:, 1]).tolist())
        sc = set((sup[:, 0] * self.data.n + sup[:, 1]).tolist())
        self.assertFalse(mc & sc)                                           # no supervision edge inside the message graph
        self.assertFalse(set(sup[:, 0].tolist()) & set(msg[:, 0].tolist()))   # hidden papers keep NO outgoing edge
        adj = Adjacency(msg, self.data.n)
        self.assertTrue((adj.out_deg[np.unique(sup[:, 0])] == 0).all())

    def test_test_edges_never_enter_training_or_validation_graphs(self):
        s = self.split
        test_codes = set((s["test"][:, 0] * self.data.n + s["test"][:, 1]).tolist())
        for name, edges in {"train": s["train"], "train+val": np.vstack([s["train"], s["val"]])}.items():
            self.assertFalse(test_codes & set((edges[:, 0] * self.data.n + edges[:, 1]).tolist()), name)


class NegativeTest(Fixture):
    def test_negatives_are_valid(self):
        d = self.data
        rng = np.random.default_rng(1)
        src = self.split["val"][:, 0]
        neg = sample_negatives(d, src, 5, rng, d.all_pairs_codes(), hard_frac=0.0)
        self.assertEqual(neg.shape, (len(src), 5))
        codes = set((d.edges[:, 0] * d.n + d.edges[:, 1]).tolist())
        for u, row in zip(src, neg):
            for v in row:
                self.assertNotEqual(u, v)
                self.assertLessEqual(d.year[v], d.year[u])                    # a paper cannot cite one published later
                self.assertNotIn(int(u) * d.n + int(v), codes)                # and a negative is never a real citation

    def test_hard_negatives_share_the_topic_more_often(self):
        d = self.data
        src = np.repeat(self.split["train"][:, 0], 1)
        easy = sample_negatives(d, src, 1, np.random.default_rng(2), d.all_pairs_codes(), 0.0)[:, 0]
        hard = sample_negatives(d, src, 1, np.random.default_rng(2), d.all_pairs_codes(), 1.0)[:, 0]
        self.assertGreater((d.topic[src] == d.topic[hard]).mean(), (d.topic[src] == d.topic[easy]).mean() + 0.3)

    def test_eval_set_shapes(self):
        ev = make_eval_set(self.data, self.split["test"], 20, 50, seed=0)
        self.assertEqual(ev.neg.shape, (min(50, len(self.split["test"])), 20))


class MetricTest(unittest.TestCase):
    def test_perfect_and_tied_scores(self):
        pos, neg = np.array([5.0, 6.0]), np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
        self.assertEqual(ranking_metrics(pos, neg)["hits@1"], 1.0)
        self.assertEqual(ranking_metrics(pos, neg)["mrr"], 1.0)
        tied = ranks(np.zeros(4), np.zeros((4, 9)))
        self.assertTrue((tied == 5.5).all())                                    # average rank among 10 equal candidates
        self.assertLess(ranking_metrics(np.zeros(4), np.zeros((4, 9)))["hits@1"], 0.5)
        self.assertEqual(all_metrics(pos, neg)["auc"], 1.0)


class ModelTest(Fixture):
    def test_adjacency_is_mean_aggregation(self):
        e = np.array([[0, 1], [0, 2], [3, 2]])
        adj = Adjacency(e, 4)
        h = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        self.assertTrue(torch.allclose(adj.mean_refs(h)[0], (h[1] + h[2]) / 2))
        self.assertTrue(torch.allclose(adj.mean_cits(h)[2], (h[0] + h[3]) / 2))
        self.assertTrue(torch.allclose(adj.mean_refs(h)[1], torch.zeros(2)))     # no references -> zero vector, not NaN

    def test_forward_shapes_and_gradients(self):
        adj = Adjacency(self.split["train"], self.data.n)
        x = node_features(self.data, adj, "cpu")
        m = CitationGNN(x.shape[1], self.data.text.shape[1], 16, 2, 0.0)
        z = m.encode(x, adj)
        self.assertEqual(z.shape, (self.data.n, 16))
        u, v = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
        m.score(z, x, u, v).sum().backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_training_beats_chance_on_the_test_years(self):
        cfg = TrainConfig(epochs=12, hidden=32, k_neg=49, max_eval_pairs=400, batch_pairs=100_000, device="cpu", patience=4, seed=0)
        res = train_gnn(self.data, cfg, log=lambda *a: None)
        self.assertGreater(res.test["auc"], 0.7)
        self.assertGreater(res.test["hits@10"], 0.35)                           # chance with 50 candidates is 0.20
        self.assertGreater(res.history[0]["loss"], res.history[-1]["loss"] - 0.2)


class BaselineTest(Fixture):
    def test_pair_features_match_hand_computation(self):
        d = self.data
        feat = PairFeaturizer(d, self.split["train"])
        u, v = self.split["train"][0]
        f = feat.features(np.array([u]), np.array([v])).iloc[0]
        self.assertAlmostEqual(f["text_cos"], float(d.text[u] @ d.text[v]), places=5)
        self.assertEqual(f["year_gap"], d.year[u] - d.year[v])
        shared = int((d.paper_author[u].multiply(d.paper_author[v])).sum())
        self.assertEqual(f["common_authors"], shared)
        self.assertEqual(f["same_topic"], float(d.topic[u] == d.topic[v]))

    def test_baseline_runs_and_beats_a_random_ranking(self):
        va = make_eval_set(self.data, self.split["val"], 49, 300, seed=1)
        te = make_eval_set(self.data, self.split["test"], 49, 300, seed=2)
        out = run_baseline(self.data, va, te, log=lambda *a: None)
        self.assertGreater(out["boosted_trees"]["test"]["hits@10"], 0.35)
        self.assertEqual(set(out), {"boosted_trees", "popularity", "text_cosine", "author_prior"})


if __name__ == "__main__":
    unittest.main()
