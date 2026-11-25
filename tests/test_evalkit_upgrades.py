import copy
import json
import tempfile
import unittest
from pathlib import Path

from scholargraph.agent.agent import AgentResult, Step
from scholargraph.demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold
from scholargraph.api.service import build_service
from scholargraph.evalkit.harness import run_agent_eval
from scholargraph.evalkit.questions import OPTIMAL_TOOLS, Question, generate_questions, load_questions, save_questions
from scholargraph.evalkit.regression import compare_reports
from scholargraph.evalkit.trajectory import is_subsequence, score_trajectory


def result(tools, errors=()):
    steps = [Step(t, {}, "", e) for t, e in zip(tools, list(errors) + [None] * len(tools))]
    return AgentResult("q", "a", ["x"], steps, True, 0.1)


Q = Question("author_of-000", "author_of", "Who wrote ...?", ["ada"])


class TrajectoryTest(unittest.TestCase):
    def test_subsequence(self):
        self.assertTrue(is_subsequence(["a", "b"], ["x", "a", "y", "b"]))
        self.assertFalse(is_subsequence(["a", "b"], ["b", "a"]))
        self.assertTrue(is_subsequence([], ["a"]))

    def test_optimal_path_scores_perfectly(self):
        s = score_trajectory(Q, result(["search_papers", "get_authors"]))
        self.assertEqual((s.precision, s.efficiency, s.order_ok, s.error_steps), (1.0, 1.0, True, 0))

    def test_wasteful_path_is_penalised_by_hand_computed_amounts(self):
        s = score_trajectory(Q, result(["get_paper", "search_papers", "top_cited", "get_authors"], errors=[None, None, "bad_args", None]))
        self.assertAlmostEqual(s.precision, 0.5)                    # 2 of 4 calls were useful
        self.assertAlmostEqual(s.efficiency, 0.5)                   # 2 optimal / 4 actual
        self.assertTrue(s.order_ok)
        self.assertEqual(s.error_steps, 1)

    def test_wrong_order_and_no_calls(self):
        self.assertFalse(score_trajectory(Q, result(["get_authors", "search_papers"])).order_ok)
        s = score_trajectory(Q, result([]))
        self.assertEqual((s.precision, s.efficiency, s.order_ok), (0.0, 0.0, False))

    def test_every_question_type_has_an_optimal_path(self):
        from scholargraph.evalkit.questions import EXPECTED_TOOLS
        self.assertEqual(set(OPTIMAL_TOOLS), set(EXPECTED_TOOLS))
        for t, path in OPTIMAL_TOOLS.items():
            self.assertTrue(set(EXPECTED_TOOLS[t]) <= set(path))


class RegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gold = make_demo_gold(450, 31)
        svc = build_service(gold, make_demo_concepts(gold), ScriptedPolicyLLM())
        cls.questions = generate_questions(svc.store, 4, 0)
        cls.report = run_agent_eval(svc.agent, cls.questions, log=lambda *a: None)

    def test_new_overall_metrics_exist_and_make_sense(self):
        o = self.report["overall"]
        self.assertEqual(o["trajectory_precision"], 1.0)
        self.assertEqual(o["order_ok"], 1.0)
        self.assertGreater(o["efficiency"], 0.95)
        self.assertEqual(o["error_steps_per_question"], 0.0)
        self.assertLessEqual(o["p50_seconds"], o["p95_seconds"])
        self.assertEqual(o["handled_without_llm"], 0.0)
        self.assertEqual(o["routes"], {"agent": len(self.questions)})
        self.assertGreater(o["tokens_per_question"], 0)

    def test_a_crashed_agent_run_is_not_counted_as_handled_without_a_model(self):
        """Found by the real small-model run: a run that crashed reports zero model calls, which used to look like 'no model needed'."""
        class Boom:
            def run(self, question):
                raise TimeoutError("model timed out")
        rep = run_agent_eval(Boom(), self.questions[:3], log=lambda *a: None)
        self.assertEqual(rep["overall"]["handled_without_llm"], 0.0)
        self.assertEqual(rep["overall"]["finished"], 0.0)

    def test_identical_reports_do_not_regress(self):
        r = compare_reports(self.report, copy.deepcopy(self.report))
        self.assertTrue(r.ok, r.summary())
        self.assertEqual(r.flipped, [])

    def test_quality_drop_is_flagged_with_the_flipped_questions(self):
        worse = copy.deepcopy(self.report)
        worse["rows"][0]["exact"] = False
        worse["by_type"]["author_of"]["exact"] = 0.5
        worse["overall"]["exact"] -= 0.1
        r = compare_reports(self.report, worse)
        self.assertFalse(r.ok)
        self.assertIn(self.report["rows"][0]["qid"], r.flipped)
        self.assertTrue(any(x.scope == "author_of" and x.metric == "exact" for x in r.regressions))

    def test_cost_growth_is_flagged_but_small_changes_are_tolerated(self):
        slower = copy.deepcopy(self.report)
        slower["overall"]["mean_steps"] *= 1.1
        self.assertTrue(compare_reports(self.report, slower).ok)
        slower["overall"]["mean_steps"] = self.report["overall"]["mean_steps"] * 2
        slower["overall"]["tokens_per_question"] *= 2
        r = compare_reports(self.report, slower)
        self.assertTrue({"mean_steps", "tokens_per_question"} <= {x.metric for x in r.regressions})

    def test_golden_question_set_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            save_questions(self.questions, Path(d) / "golden.json")
            back = load_questions(Path(d) / "golden.json")
        self.assertEqual([q.qid for q in back], [q.qid for q in self.questions])
        self.assertEqual(back[0].answer, self.questions[0].answer)
        self.assertEqual(back[0].expected_tools, self.questions[0].expected_tools)


if __name__ == "__main__":
    unittest.main()
