import json
import time
import unittest

from scholargraph.agent.agent import Agent
from scholargraph.agent.hooks import HookManager
from scholargraph.agent.multi import (GRAPH_TOOLS, RETRIEVAL_TOOLS, GroundingHook, MultiAgent, Subtask, plan, restrict_tools, verify_items, waves)
from scholargraph.agent.tools import Tool, make_tools
from scholargraph.api.service import build_service
from scholargraph.demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold
from scholargraph.llm import FakeLLM
from scholargraph.evalkit.questions import norm


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = make_demo_gold(450, 31)
        cls.svc = build_service(cls.gold, make_demo_concepts(cls.gold), ScriptedPolicyLLM())
        cls.store = cls.svc.store
        cls.tools = make_tools(cls.store, cls.svc.retriever)
        ids = cls.store.ids
        cls.t1, cls.t2 = cls.store.paper(ids[20])["title"], cls.store.paper(ids[30])["title"]
        cls.a1 = [a["name"] for a in cls.store.authors_of(ids[20])]
        cls.a2 = [a["name"] for a in cls.store.authors_of(ids[30])]

    def graph_agent(self):
        return Agent(ScriptedPolicyLLM(), restrict_tools(self.tools, GRAPH_TOOLS))

    def plan_json(self, subtasks):
        return json.dumps({"subtasks": subtasks})


class PlanningTest(Fixture):
    def test_valid_plan_is_parsed(self):
        llm = FakeLLM([self.plan_json([{"id": "s1", "goal": "goal one", "agent": "graph"}, {"id": "s2", "goal": "goal two", "agent": "retrieval", "depends_on": ["s1"]}])])
        subs, attempts = plan(llm, "q")
        self.assertEqual([(s.id, s.agent, s.depends_on) for s in subs], [("s1", "graph", []), ("s2", "retrieval", ["s1"])])

    def test_invalid_plans_fall_back_to_one_graph_subtask(self):
        subs, attempts = plan(FakeLLM(["no json", self.plan_json([{"id": "s1", "goal": "g", "agent": "wizard"}])]), "my question")
        self.assertEqual([(s.goal, s.agent) for s in subs], [("my question", "graph")])
        self.assertEqual(attempts, 2)

    def test_waves_group_independent_work_and_survive_cycles(self):
        ws = waves([Subtask("a", "x", "graph"), Subtask("b", "x", "graph"), Subtask("c", "x", "graph", ["a", "b"]), Subtask("d", "x", "graph", ["c"])])
        self.assertEqual([[s.id for s in w] for w in ws], [["a", "b"], ["c"], ["d"]])
        cyc = waves([Subtask("a", "x", "graph", ["b"]), Subtask("b", "x", "graph", ["a"])])
        self.assertEqual(sum(len(w) for w in cyc), 2)                      # no deadlock
        self.assertEqual(len(waves([Subtask("a", "x", "graph", ["ghost"])])), 1)     # unknown dependencies are ignored

    def test_specialists_only_see_their_own_tools(self):
        r = restrict_tools(self.tools, RETRIEVAL_TOOLS)
        self.assertEqual(set(r), {"search_papers", "get_paper"})
        agent = Agent(FakeLLM([json.dumps({"tool": "get_citations", "args": {"paper_id": "W1"}}), json.dumps({"final": "x", "items": []})]), r)
        res = agent.run("q")
        self.assertEqual(res.steps[0].error, "unknown_tool")                # a retrieval specialist cannot call a graph tool
        self.assertNotIn("get_citations", agent.system)


class VerificationTest(Fixture):
    def test_verify_items_requires_evidence_not_just_existence(self):
        ev = "OBSERVATION: " + json.dumps({"authors": [{"name": "Ada Lovelace"}]})
        ok, bad = verify_items(["Ada Lovelace", "Alan Turing"], ev, self.store)
        self.assertEqual((ok, bad), (["Ada Lovelace"], ["Alan Turing"]))

    def test_grounding_hook_rejects_unseen_items_then_accepts_the_corrected_answer(self):
        hooks = HookManager()
        GroundingHook(self.store).register(hooks)
        name = self.a1[0]
        replies = [json.dumps({"tool": "get_authors", "args": {"paper_id": self.store.ids[20]}}),
                   json.dumps({"final": "fabricated", "items": ["Totally Invented Person"]}),
                   json.dumps({"final": name, "items": [name]})]
        res = Agent(FakeLLM(replies), self.tools, hooks=hooks, max_steps=6).run("q")
        self.assertEqual([s.error for s in res.steps], [None, "final_rejected"])
        self.assertEqual(res.items, [name])
        self.assertTrue(res.finished)


class MultiAgentTest(Fixture):
    def q_author(self, title):
        return f'Who wrote the paper titled "{title}"?'

    def make(self, synth_reply, plan_subtasks=None):
        plan_subtasks = plan_subtasks or [{"id": "s1", "goal": self.q_author(self.t1), "agent": "graph"}, {"id": "s2", "goal": self.q_author(self.t2), "agent": "graph"}]
        return MultiAgent(FakeLLM([self.plan_json(plan_subtasks)]), FakeLLM(synth_reply if isinstance(synth_reply, list) else [synth_reply]),
                          self.graph_agent(), store=self.store)

    def test_plan_delegate_synthesise_verify(self):
        want = self.a1 + self.a2
        ma = self.make(json.dumps({"final": "authors", "items": want}))
        res = ma.run("Who wrote the two papers?")
        self.assertTrue(res.finished)
        self.assertEqual(res.route, "planner")
        self.assertEqual(sorted(norm(i) for i in res.items), sorted(norm(i) for i in want))
        self.assertEqual(len(ma.last_plan), 2)
        self.assertTrue({"search_papers", "get_authors"} <= {s.tool for s in res.steps})
        self.assertGreater(res.usage["llm_calls"], 0)

    def test_hallucinated_items_get_one_repair_round_then_are_dropped(self):
        good = self.a1[0]
        ma = self.make([json.dumps({"final": "x", "items": [good, "Fabricated Person"]}),                 # first synthesis: one invented item
                        json.dumps({"final": "y", "items": [good, "Still Fabricated"]})])                  # repair does not fix it
        res = ma.run("q")
        self.assertEqual(res.items, [good])                                                               # unsupported items never reach the user
        self.assertTrue(any(s.error == "verifier" for s in res.steps))
        ma2 = self.make([json.dumps({"final": "x", "items": [good, "Fabricated Person"]}), json.dumps({"final": "y", "items": [good]})])
        self.assertEqual(ma2.run("q").items, [good])                                                      # a successful repair keeps everything verified

    def test_independent_subtasks_run_in_parallel_and_dependent_ones_see_results(self):
        seen = []

        class SlowLLM(ScriptedPolicyLLM):
            def chat(self, messages, **kw):
                time.sleep(0.15)
                seen.append([m["content"][:60] for m in messages if m["role"] == "system"])
                return super().chat(messages, **kw)

        sub = [{"id": "s1", "goal": self.q_author(self.t1), "agent": "graph"}, {"id": "s2", "goal": self.q_author(self.t2), "agent": "graph"},
               {"id": "s3", "goal": self.q_author(self.t1), "agent": "graph", "depends_on": ["s1", "s2"]}]
        ma = MultiAgent(FakeLLM([self.plan_json(sub)]), FakeLLM([json.dumps({"final": "f", "items": [self.a1[0]]})]), Agent(SlowLLM(), restrict_tools(self.tools, GRAPH_TOOLS)), store=self.store)
        t0 = time.time()
        res = ma.run("q")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.15 * 2 * 3 * 2 * 0.8)                    # s1 and s2 overlapped: much faster than running all three one after another
        self.assertTrue(any(x.startswith("Results of earlier subtasks") for msgs in seen for x in msgs))
        self.assertEqual(res.items, [self.a1[0]])

    def test_synthesis_failure_is_reported_not_raised(self):
        res = self.make("garbage").run("q")
        self.assertFalse(res.finished)
        self.assertEqual(res.stopped_by, "synthesis failed")


if __name__ == "__main__":
    unittest.main()
