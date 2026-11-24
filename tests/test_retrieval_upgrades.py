import json
import unittest

import numpy as np
import scipy.sparse as sp

from scholargraph.api.service import build_service
from scholargraph.demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold, make_demo_text
from scholargraph.evalkit.retrieval_bench import evaluate_search_fn
from scholargraph.gnn.data import load_citation_data
from scholargraph.graph.store import MemoryGraphStore
from scholargraph.llm import FakeLLM
from scholargraph.rag.communities import (CommunityIndex, GlobalAnswerer, build_communities, label_propagation)
from scholargraph.rag.contextual import build_contextual_texts
from scholargraph.rag.guardrails import REFUSAL
from scholargraph.rag.multihop import MultiHopRetriever
from scholargraph.rag.query import expand_with_lexicon, hyde, multi_query_search, rewrite_queries
from scholargraph.rag.rerank import (CrossEncoderReranker, EncoderReranker, LateInteractionReranker, LexicalReranker, maxsim, rerank_ids)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = make_demo_gold(450, 31)
        cls.concepts = make_demo_concepts(cls.gold)
        cls.svc = build_service(cls.gold, cls.concepts, ScriptedPolicyLLM())
        cls.store, cls.retriever = cls.svc.store, cls.svc.retriever
        cls.papers = cls.gold["papers"]


class RerankerTest(Fixture):
    def test_lexical_reranker_prefers_title_matches(self):
        rr = LexicalReranker()
        s = rr.score("graph neural network", ["Graph neural network for links. Something else entirely here.", "Other title. We use a graph neural network in the body."])
        self.assertGreater(s[0], s[1])

    def test_rerank_ids_orders_by_score_and_blend_keeps_first_stage_influence(self):
        class Fixed:
            def score(self, q, docs):
                return np.array([0.1, 0.9, 0.5])
        self.assertEqual(rerank_ids(Fixed(), "q", ["a", "b", "c"], ["x", "y", "z"], 3), [1, 2, 0])
        self.assertEqual(rerank_ids(Fixed(), "q", ["a", "b", "c"], ["x", "y", "z"], 3, original_scores=[3, 2, 1], blend=1.0), [0, 1, 2])
        self.assertEqual(rerank_ids(Fixed(), "q", ["a", "b", "c"], ["x", "y", "z"], 2), [1, 2])

    def test_cross_encoder_wrapper_with_an_injected_scorer(self):
        ce = CrossEncoderReranker(score_fn=lambda q, docs: [len(d) for d in docs])
        np.testing.assert_array_equal(ce.score("q", ["a", "abc", "ab"]), [1, 3, 2])

    def test_retriever_reranks_only_its_candidate_pool(self):
        q = "transformer question answering"
        base = self.retriever.search(q, k=5)
        marker = {self.retriever.texts[self.retriever.idx[h.paper_id]]: i for i, h in enumerate(base)}      # score = original position: reverses the order
        rr = CrossEncoderReranker(score_fn=lambda query, docs: [marker.get(d, -1) for d in docs])
        out = self.retriever.search(q, k=5, reranker=rr, rerank_pool=5)
        self.assertEqual([h.paper_id for h in out], [h.paper_id for h in base[::-1]])
        self.assertIn("rerank", out[0].sources)
        wide = self.retriever.search(q, k=5, reranker=LexicalReranker(), rerank_pool=40)
        self.assertEqual(len(wide), 5)
        self.assertEqual(len({h.paper_id for h in wide}), 5)

    def test_maxsim_by_hand_and_late_interaction_reranker(self):
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0, 0.0], [0.6, 0.8]])
        self.assertAlmostEqual(maxsim(q, d), 1.0 + 0.8)
        self.assertEqual(maxsim(q, np.zeros((0, 2))), 0.0)
        vocab = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.7, 0.7]}
        enc = lambda t: np.array([vocab[w] / np.linalg.norm(vocab[w]) for w in t.split()])  # noqa: E731
        s = LateInteractionReranker(enc).score("a b", ["a b", "a a", "c c"])
        self.assertEqual(int(np.argmax(s)), 0)


class QueryTest(Fixture):
    def test_lexicon_expansion_adds_aliases_without_duplicating_words(self):
        q = expand_with_lexicon("gnn for link prediction")
        self.assertIn("graph neural network", q)
        self.assertEqual(expand_with_lexicon("completely unknown words"), "completely unknown words")

    def test_llm_rewrites_and_hyde_are_validated(self):
        self.assertEqual(rewrite_queries(FakeLLM([json.dumps({"queries": ["graph networks citation", "link prediction gnn"]})]), "how do people predict citations"),
                         ["graph networks citation", "link prediction gnn"])
        self.assertEqual(rewrite_queries(FakeLLM(["nonsense"]), "q"), [])
        self.assertIsNone(hyde(FakeLLM([json.dumps({"abstract": "short"})]), "q"))
        self.assertIn("transformer", hyde(FakeLLM([json.dumps({"abstract": "We study a transformer model for translation tasks in depth."})]), "q"))

    def test_multi_query_search_fuses_variants(self):
        qs = ["transformer machine translation", "attention mechanism translation", "transformer translation"]
        hits = multi_query_search(self.retriever, qs, k=5)
        self.assertEqual(len(hits), 5)
        self.assertEqual(len({h.paper_id for h in hits}), 5)
        self.assertGreaterEqual(hits[0].sources["n_variants"], 2)
        self.assertTrue(all(hits[i].score >= hits[i + 1].score for i in range(4)))


class ContextualTest(Fixture):
    def test_context_is_prefixed_and_includes_concepts(self):
        texts = build_contextual_texts(self.papers.head(3), self.concepts)
        self.assertEqual(len(texts), 3)
        self.assertIn("[year ", texts[0])
        self.assertIn("topic ", texts[0])
        self.assertIn("concepts ", texts[0])
        self.assertTrue(texts[0].startswith(self.papers.iloc[0]["title"]))


class BenchTest(Fixture):
    def test_every_configuration_is_evaluated_with_the_same_protocol(self):
        text = make_demo_text(self.gold, 32)
        data = load_citation_data(self.gold, text, 2022, 2023)
        texts = (self.papers["title"] + ". " + self.papers["abstract"]).tolist()
        r = self.retriever
        configs = {"bm25": lambda q, i: [r.idx[h.paper_id] for h in r.search(q, k=20, use_dense=False)],
                   "dense": lambda q, i: [r.idx[h.paper_id] for h in r.search(q, k=20, use_bm25=False)],
                   "hybrid": lambda q, i: [r.idx[h.paper_id] for h in r.search(q, k=20)],
                   "hybrid+graph": lambda q, i: [r.idx[h.paper_id] for h in r.search(q, k=20, graph_weight=0.5)],
                   "hybrid+rerank": lambda q, i: [r.idx[h.paper_id] for h in r.search(q, k=20, reranker=LexicalReranker(), rerank_pool=40)]}
        res = {name: evaluate_search_fn(fn, data, texts, max_queries=60) for name, fn in configs.items()}
        for name, m in res.items():
            self.assertTrue(0.0 <= m["mrr"] <= 1.0 and 0.0 <= m["recall@10"] <= 1.0, (name, m))
            self.assertEqual(m["n_queries"], 60)
        self.assertGreaterEqual(res["hybrid"]["recall@20"], 0.0)

    def test_the_metric_is_exact_on_a_hand_made_search_function(self):
        text = make_demo_text(self.gold, 32)
        data = load_citation_data(self.gold, text, 2022, 2023)
        texts = ["x"] * data.n
        test = data.split()["test"]
        u = int(test[0][0])
        cited = [int(v) for uu, v in test if int(uu) == u]
        oracle = lambda q, i: cited[:3]  # noqa: E731
        m = evaluate_search_fn(oracle, data, texts, ks=(3,), max_queries=None)
        self.assertGreater(m["mrr"], 0.0)
        never = evaluate_search_fn(lambda q, i: [], data, texts, ks=(3,), max_queries=None)
        self.assertEqual((never["mrr"], never["recall@3"]), (0.0, 0.0))


class CommunityTest(Fixture):
    def test_label_propagation_recovers_planted_blocks(self):
        rng = np.random.default_rng(0)
        n, blocks = 120, 3
        truth = np.repeat(np.arange(blocks), n // blocks)
        P = np.where(truth[:, None] == truth[None, :], 0.35, 0.01)
        A = sp.csr_matrix(np.triu(rng.random((n, n)) < P, 1).astype(np.int8))
        labels = label_propagation(A, seed=1)
        purity = sum(np.bincount(truth[labels == l]).max() for l in np.unique(labels)) / n
        self.assertGreater(purity, 0.9)
        self.assertLessEqual(len(np.unique(labels)), 6)

    def test_reports_and_global_answers(self):
        comms = build_communities(self.store, min_size=8, seed=0)
        self.assertGreaterEqual(len(comms), 2)
        self.assertGreater(len(comms[0].members), len(comms[-1].members) - 1)
        self.assertIn("Community 0", comms[0].report)
        self.assertTrue(comms[0].top_papers)
        idx = CommunityIndex(comms)
        replies = [json.dumps({"answer": "This community studies transformers for translation."})] * 3 + ["Summary: transformers dominate [C0]."]
        llm = FakeLLM(replies)
        out = GlobalAnswerer(idx, llm, k=3).answer("what are the main themes about transformer translation")
        self.assertFalse(out["abstained"])
        self.assertEqual(len(llm.calls), len(out["partials"]) + 1)          # one map call per community, then one reduce call
        self.assertEqual(out["invalid_citations"], [])
        self.assertIn("[C0]", out["answer"])

    def test_global_answer_abstains_when_no_community_knows(self):
        comms = build_communities(self.store, min_size=8, seed=0)
        out = GlobalAnswerer(CommunityIndex(comms), FakeLLM([json.dumps({"answer": "nothing relevant"})]), k=3).answer("transformer translation")
        self.assertEqual(out["answer"], REFUSAL)
        self.assertTrue(out["abstained"])

    def test_communities_follow_the_planted_topics(self):
        comms = build_communities(self.store, min_size=8, seed=0)
        topic = self.store.papers["primary_topic_name"].to_numpy()
        purities = [np.unique(topic[c.members], return_counts=True)[1].max() / len(c.members) for c in comms]
        self.assertGreater(float(np.average(purities, weights=[len(c.members) for c in comms])), 0.5)


class MultiHopTest(Fixture):
    def test_a_paper_two_hops_away_is_found_with_a_valid_explanation(self):
        s, r = self.store, self.retriever
        # choose a target paper with a citation path of length 2 from some seed, and craft a question that matches only the seed's title
        for seed_pid in s.ids[100:200]:
            for mid in s.references(seed_pid):
                for target in s.references(mid):
                    if target not in (seed_pid, mid) and target not in s.references(seed_pid):
                        title = s.paper(seed_pid)["title"]
                        mh1 = MultiHopRetriever(r, s, hops=1, beam=3)
                        mh2 = MultiHopRetriever(r, s, hops=2, beam=200)
                        ev2 = {e.paper_id: e for e in mh2.search(title, k=500)}
                        if target in ev2 and ev2[target].hop == 2:
                            path = ev2[target].path
                            self.assertEqual(len(path), 2)
                            for a, rel, b in path:                              # every explained edge really exists in the graph
                                self.assertIn(b, s.references(a))
                            self.assertEqual(path[0][0] if path[0][1] == "cites" else None, path[0][0])
                            self.assertNotIn(target, {e.paper_id for e in mh1.search(title, k=500) if e.hop == 2})
                            self.assertIn("--cites-->", ev2[target].explain())
                            return
        self.fail("no two-hop example found in the synthetic graph")

    def test_seeds_have_hop_zero_and_no_path(self):
        out = MultiHopRetriever(self.retriever, self.store, hops=0, beam=3).search("transformer translation", k=3)
        self.assertTrue(all(e.hop == 0 and e.path == [] for e in out))
        self.assertIn("direct text match", out[0].explain())


if __name__ == "__main__":
    unittest.main()
