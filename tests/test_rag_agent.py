import json
import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

from scholargraph.agent.agent import Agent
from scholargraph.agent.tools import Tool, make_tools
from scholargraph.api.app import create_app
from scholargraph.api.service import build_service
from scholargraph.demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold
from scholargraph.evalkit.harness import judge_faithfulness, run_agent_eval, set_f1
from scholargraph.evalkit.questions import EXPECTED_TOOLS, generate_questions, norm
from scholargraph.graph.store import MemoryGraphStore
from scholargraph.llm import FakeLLM, MeteredLLM
from scholargraph.rag.bm25 import BM25, tokenize
from scholargraph.rag.graphrag import GraphRAG
from scholargraph.rag.guardrails import REFUSAL, check_answer, extract_citations, groundedness, invalid_citations
from scholargraph.rag.retrieval import HybridRetriever, rrf


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = make_demo_gold(450, 31)
        cls.concepts = make_demo_concepts(cls.gold)
        cls.service = build_service(cls.gold, cls.concepts, ScriptedPolicyLLM())
        cls.store = cls.service.store
        cls.papers = cls.gold["papers"]


class Bm25Test(unittest.TestCase):
    def test_matches_the_formula_on_a_tiny_corpus(self):
        docs = ["graph neural network graph", "language model", "graph theory"]
        bm = BM25(k1=1.5, b=0.75).fit(docs)
        toks = [tokenize(d) for d in docs]
        n, avg = len(docs), np.mean([len(t) for t in toks])
        df = sum("graph" in t for t in toks)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        expect = [idf * t.count("graph") * 2.5 / (t.count("graph") + 1.5 * (0.25 + 0.75 * len(t) / avg)) if "graph" in t else 0 for t in toks]
        np.testing.assert_allclose(bm.scores("graph"), expect, rtol=1e-5)

    def test_rare_terms_outweigh_common_ones_and_unknown_terms_score_zero(self):
        docs = ["common common rare", "common common common", "common other"]
        bm = BM25().fit(docs)
        self.assertEqual(bm.search("common rare", 1)[0][0], 0)
        self.assertEqual(bm.search("zzz", 3), [])
        self.assertEqual(float(bm.scores("").sum()), 0.0)


class RetrievalTest(Fixture):
    def test_rrf_example(self):
        out = rrf([[1, 2, 3], [3, 1]], k=60)
        self.assertAlmostEqual(out[1], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(out[3], 1 / 63 + 1 / 61)
        self.assertGreater(out[1], out[2])

    def test_a_paper_is_found_by_its_own_title(self):
        r = self.service.retriever
        found = 0
        ids = self.papers["paper_id"][:30]
        for pid in ids:
            title = self.store.paper(pid)["title"]
            found += pid in [h.paper_id for h in r.search(title, k=3)]
        self.assertGreaterEqual(found / len(ids), 0.9)                       # retrieval is fuzzy: top-3 recall, not top-1 certainty

    def test_year_filter_and_components(self):
        r = self.service.retriever
        hits = r.search("transformer question answering", k=20, year_from=2022)
        self.assertTrue(all(self.store.paper(h.paper_id)["year"] >= 2022 for h in hits))
        dense_only = r.search("transformer question answering", k=5, use_bm25=False)
        self.assertTrue(all(h.sources.get("bm25", 0.0) == 0.0 for h in dense_only))

    def test_graph_bonus_pulls_in_linked_papers(self):
        r = self.service.retriever
        plain = r.search("transformer for link prediction", k=15, graph_weight=0.0)
        boosted = r.search("transformer for link prediction", k=15, graph_weight=1.0)
        self.assertTrue(any(h.sources["graph"] > 0 for h in boosted))
        self.assertTrue(all(h.sources["graph"] == 0 for h in plain))


class GuardrailTest(unittest.TestCase):
    SOURCES = ["Graph neural networks learn node representations by message passing between neighbours."]

    def test_citation_parsing_and_validation(self):
        self.assertEqual(extract_citations("A [P1] and B [P3][P1]."), [1, 3, 1])
        self.assertEqual(invalid_citations("A [P1] and B [P3].", 2), [3])

    def test_groundedness_separates_supported_from_invented(self):
        good = groundedness("Graph neural networks learn node representations by message passing [P1].", self.SOURCES)
        bad = groundedness("Quantum annealers factorise primes using lattice surgery [P1].", self.SOURCES)
        self.assertGreater(good, 0.8)
        self.assertLess(bad, 0.2)

    def test_report_flags_missing_citation_and_accepts_refusal(self):
        self.assertFalse(check_answer("Graph neural networks learn node representations.", self.SOURCES).ok)     # no citation
        self.assertTrue(check_answer("Graph neural networks learn node representations [P1].", self.SOURCES).ok)
        self.assertTrue(check_answer(REFUSAL, self.SOURCES).ok)
        self.assertFalse(check_answer("Graph neural networks learn node representations [P4].", self.SOURCES).ok)


class GraphRagTest(Fixture):
    def test_answer_has_sources_and_prompt_contains_graph_facts(self):
        llm = FakeLLM(["Papers use transformers for translation [P1]."])
        rag = GraphRAG(self.service.retriever, self.store, llm, k=4)
        a = rag.answer("transformer for machine translation")
        self.assertEqual(len(a.sources), 4)
        prompt = llm.calls[0][-1]["content"]
        self.assertIn("[P1]", prompt)
        self.assertIn("GRAPH FACTS", prompt)
        self.assertIn("QUESTION: transformer for machine translation", prompt)
        self.assertEqual(llm.calls[0][0]["role"], "system")
        self.assertEqual(a.report.invalid_citations, [])

    def test_abstains_without_evidence_and_never_calls_the_model(self):
        r = HybridRetriever(self.papers["paper_id"].tolist(), (self.papers["title"] + ". " + self.papers["abstract"]).tolist(), None, None)
        llm = FakeLLM(["should not be used"])
        a = GraphRAG(r, self.store, llm).answer("zzzzqqq xxyyzz")
        self.assertEqual(a.answer, REFUSAL)
        self.assertEqual(len(llm.calls), 0)


class AgentRobustnessTest(Fixture):
    def tools(self):
        return make_tools(self.store, self.service.retriever)

    def test_recovers_from_garbage_unknown_tool_and_bad_args(self):
        replies = ["not json at all", json.dumps({"tool": "nope", "args": {}}), json.dumps({"tool": "get_paper", "args": {"wrong": 1}}),
                   json.dumps({"final": "done", "items": ["x"]})]
        res = Agent(FakeLLM(replies), self.tools()).run("anything")
        self.assertTrue(res.finished)
        self.assertEqual([s.error for s in res.steps], ["invalid_json", "unknown_tool", "bad_args"])
        self.assertEqual(res.items, ["x"])

    def test_step_budget_is_enforced(self):
        loop = json.dumps({"tool": "top_cited", "args": {"k": 1}})
        res = Agent(FakeLLM([loop]), self.tools(), max_steps=3).run("never ends")
        self.assertFalse(res.finished)
        self.assertEqual(len(res.steps), 3)

    def test_tool_exceptions_do_not_crash_the_agent(self):
        boom = Tool("boom", "always fails", {}, lambda: 1 / 0)
        res = Agent(FakeLLM([json.dumps({"tool": "boom", "args": {}}), json.dumps({"final": "ok", "items": []})]), {"boom": boom}).run("q")
        self.assertTrue(res.finished)
        self.assertEqual(res.steps[0].error, "tool_error")

    def test_system_prompt_lists_every_tool(self):
        ag = Agent(FakeLLM(["{}"]), self.tools())
        for name in self.tools():
            self.assertIn(name, ag.system)


class EvaluationTest(Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.questions = generate_questions(cls.store, per_type=6, seed=0)

    def test_every_question_type_is_generated(self):
        self.assertEqual({q.qtype for q in self.questions}, set(EXPECTED_TOOLS))

    def test_answers_are_correct_against_the_raw_tables(self):
        ash, papers, cit = self.gold["authorships"], self.papers.set_index("paper_id"), self.gold["citations"]
        for q in self.questions:
            if q.qtype == "author_of":
                names = set(self.store.authors.loc[ash[ash["paper_id"] == q.meta["paper_id"]]["author_id"], "name"].map(norm))
                self.assertEqual(set(q.answer), names)
            elif q.qtype == "cites_count":
                self.assertEqual(q.answer, [str(int((cit["cited_id"] == q.meta["paper_id"]).sum()))])
            elif q.qtype == "references_of":
                refs = cit[cit["citing_id"] == q.meta["paper_id"]]["cited_id"]
                self.assertEqual(set(q.answer), {norm(papers.loc[r, "title"]) for r in refs})
            elif q.qtype == "top_cited_year":
                y = q.meta["year"]
                ids = papers[papers["year"] == y].index
                counts = cit[cit["cited_id"].isin(ids)].groupby("cited_id").size()
                self.assertEqual(q.answer, [norm(papers.loc[counts.idxmax(), "title"])])

    def test_scoring_functions(self):
        self.assertEqual(set_f1(["A", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(set_f1(["a"], ["a", "b"]), 2 / 3)
        self.assertEqual(set_f1([], ["a"]), 0.0)
        self.assertEqual(set_f1([], []), 1.0)

    def test_scripted_policy_scores_perfectly_which_validates_the_plumbing(self):
        report = run_agent_eval(self.service.agent, self.questions, log=lambda *a: None)
        self.assertEqual(report["overall"]["exact"], 1.0, [r for r in report["rows"] if not r["exact"]])
        self.assertEqual(report["overall"]["tool_ok"], 1.0)
        self.assertEqual(set(report["by_type"]), set(EXPECTED_TOOLS))
        self.assertGreater(report["overall"]["prompt_tokens"], 0)

    def test_a_useless_model_scores_low(self):
        bad = Agent(FakeLLM([json.dumps({"final": "no idea", "items": ["no idea"]})]), make_tools(self.store))
        report = run_agent_eval(bad, self.questions, log=lambda *a: None)
        self.assertLess(report["overall"]["exact"], 0.2)

    def test_llm_judge_parses_scores(self):
        self.assertEqual(judge_faithfulness(FakeLLM(['{"score": 0.75, "unsupported": ["x"]}']), "a", ["s"])["score"], 0.75)
        self.assertEqual(judge_faithfulness(FakeLLM(['{"score": 7}']), "a", ["s"])["score"], 1.0)
        self.assertTrue(math.isnan(judge_faithfulness(FakeLLM(["garbage"]), "a", ["s"])["score"]))


class ApiTest(Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = TestClient(create_app(cls.service))

    def test_health_stats_and_request_id(self):
        r = self.client.get("/health", headers={"x-request-id": "abc123"})
        self.assertEqual(r.json(), {"status": "ok"})
        self.assertEqual(r.headers["x-request-id"], "abc123")
        self.assertEqual(self.client.get("/stats").json()["papers"], len(self.papers))

    def test_search_and_paper_endpoints(self):
        pid = self.papers["paper_id"].iloc[3]
        title = self.store.paper(pid)["title"]
        res = self.client.get("/search", params={"q": title, "k": 5}).json()
        self.assertIn(pid, [h["paper_id"] for h in res["results"]])
        self.assertEqual(self.client.get(f"/paper/{pid}").json()["paper_id"], pid)
        self.assertEqual(self.client.get("/paper/W0").status_code, 404)
        self.assertEqual(self.client.get("/search", params={"q": "x", "k": 500}).status_code, 422)

    def test_ask_and_agent_endpoints(self):
        title = self.store.paper(self.papers["paper_id"].iloc[5])["title"]
        r = self.client.post("/agent", json={"question": f'Who wrote the paper titled "{title}"?'}).json()
        self.assertTrue(r["finished"])
        self.assertGreater(len(r["items"]), 0)
        self.assertEqual([s["tool"] for s in r["steps"]], ["search_papers", "get_authors"])
        self.assertEqual(self.client.post("/ask", json={"question": "x"}).status_code, 422)

    def test_metrics_endpoint_reports_traffic(self):
        self.client.get("/health")
        text = self.client.get("/metrics").text
        self.assertIn('scholargraph_requests_total{path="/health",status="200"}', text)
        self.assertIn("scholargraph_request_seconds_count", text)
        self.assertIn("scholargraph_llm_calls_total", text)


if __name__ == "__main__":
    unittest.main()
