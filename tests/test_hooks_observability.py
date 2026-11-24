import json
import tempfile
import time
import unittest
from pathlib import Path

from scholargraph.agent.agent import Agent
from scholargraph.agent.hooks import (AFTER_TOOL, BEFORE_FINAL, BEFORE_LLM, BEFORE_TOOL, ApprovalHook, BudgetHook, HookContext, HookManager,
                                      HookResult, LoopGuardHook, RedactionHook, ToolAllowlistHook, TracingHook, redact)
from scholargraph.agent.tools import Tool
from scholargraph.llm import FakeLLM, LLMResult
from scholargraph.observability import InMemoryTracer, NullTracer, OTelTracer, TracedLLM
from scholargraph.structured import AGENT_STEP_FLAT_SCHEMA


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


FINAL = json.dumps({"final": "done", "items": ["x"]})


def tools():
    return {"echo": Tool("echo", "echo", {"text": "str"}, lambda text="": {"text": text}),
            "mail": Tool("mail", "sensitive", {}, lambda: {"contact": "ada@example.org, +30 210 123 4567"}),
            "slow": Tool("slow", "sleeps", {"n": "int"}, lambda n=0: (time.sleep(0.25), {"n": n})[1])}


class HookManagerTest(unittest.TestCase):
    def test_priority_block_and_accumulating_modifications(self):
        order = []
        h = HookManager()
        h.register(BEFORE_TOOL, lambda c: order.append("late") or HookResult.modify(args={"text": "B"}), "late", 50)
        h.register(BEFORE_TOOL, lambda c: order.append("early") or HookResult.modify(args={"text": "A"}), "early", 10)
        res = h.fire(BEFORE_TOOL, HookContext(BEFORE_TOOL, tool="echo", args={"text": "orig"}))
        self.assertEqual(order, ["early", "late"])
        self.assertEqual(res.patch["args"], {"text": "B"})
        h.register(BEFORE_TOOL, lambda c: HookResult.block("no"), "blocker", 60)
        self.assertEqual(h.fire(BEFORE_TOOL, HookContext(BEFORE_TOOL)).action, "block")

    def test_a_crashing_hook_is_isolated_and_unknown_events_are_rejected(self):
        h = HookManager()
        h.register(BEFORE_LLM, lambda c: 1 / 0, "bad")
        self.assertEqual(h.fire(BEFORE_LLM, HookContext(BEFORE_LLM)).action, "allow")
        self.assertIn("ZeroDivisionError", h.log[0])
        with self.assertRaises(ValueError):
            h.register("nope", lambda c: None)


class AgentHookTest(unittest.TestCase):
    def run_agent(self, replies, setup, **kw):
        hooks = HookManager()
        setup(hooks)
        llm = FakeLLM(replies)
        return Agent(llm, tools(), hooks=hooks, **kw).run("q"), llm

    def test_budget_stops_the_run(self):
        res, llm = self.run_agent([call("echo", text="a")], lambda h: BudgetHook(max_llm_calls=2).register(h), max_steps=6)
        self.assertFalse(res.finished)
        self.assertIn("model-call budget", res.stopped_by)
        self.assertEqual(len(llm.calls), 2)

    def test_tool_budget_and_token_budget(self):
        res, _ = self.run_agent([call("echo", text="a")], lambda h: BudgetHook(max_tool_calls=1).register(h), max_steps=6)
        self.assertIn("tool-call budget", res.stopped_by)
        res, _ = self.run_agent([call("echo", text="a")], lambda h: BudgetHook(max_tokens=1).register(h), max_steps=6)
        self.assertIn("token budget", res.stopped_by)

    def test_allowlist_blocks_before_the_tool_runs(self):
        ran = []
        t = tools()
        t["echo"] = Tool("echo", "echo", {"text": "str"}, lambda text="": ran.append(1) or {})
        hooks = HookManager()
        ToolAllowlistHook(["mail"]).register(hooks)
        res = Agent(FakeLLM([call("echo", text="x"), FINAL]), t, hooks=hooks).run("q")
        self.assertEqual(res.steps[0].error, "blocked")
        self.assertEqual(ran, [])
        self.assertTrue(res.finished)

    def test_approval_hook(self):
        for approve, expected in ((False, "blocked"), (True, None)):
            res, _ = self.run_agent([call("mail"), FINAL], lambda h: ApprovalHook(["mail"], lambda t, a: approve).register(h))
            self.assertEqual(res.steps[0].error, expected)

    def test_redaction_of_observations_and_answers(self):
        replies = [call("mail"), json.dumps({"final": "write to ada@example.org", "items": ["+30 210 123 4567"]})]
        res, llm = self.run_agent(replies, lambda h: RedactionHook().register(h))
        seen_by_model = llm.calls[1][-1]["content"]
        self.assertNotIn("ada@example.org", seen_by_model)
        self.assertIn("[EMAIL]", seen_by_model)
        self.assertIn("[PHONE]", seen_by_model)
        self.assertEqual(res.answer, "write to [EMAIL]")
        self.assertEqual(res.items, ["[PHONE]"])
        self.assertEqual(redact("call 4111 1111 1111 1111 now"), "call [CARD] now")

    def test_loop_guard_blocks_exact_repeats(self):
        res, _ = self.run_agent([call("echo", text="a"), call("echo", text="a"), FINAL], lambda h: LoopGuardHook().register(h))
        self.assertEqual([s.error for s in res.steps], [None, "blocked"])

    def test_final_answer_can_be_rejected_twice_then_accepted(self):
        hooks = HookManager()
        hooks.register(BEFORE_FINAL, lambda c: HookResult.block("unverified"), "verifier")
        res = Agent(FakeLLM([FINAL]), tools(), hooks=hooks, max_steps=6).run("q")
        self.assertEqual([s.error for s in res.steps], ["final_rejected", "final_rejected"])
        self.assertTrue(res.finished)

    def test_before_llm_can_append_messages(self):
        hooks = HookManager()
        hooks.register(BEFORE_LLM, lambda c: HookResult.modify(append_messages=[{"role": "system", "content": "REMINDER"}]), "reminder")
        llm = FakeLLM([FINAL])
        Agent(llm, tools(), hooks=hooks).run("q")
        self.assertEqual(llm.calls[0][-1]["content"], "REMINDER")

    def test_on_finish_receives_the_result(self):
        seen = []
        hooks = HookManager()
        hooks.register("on_finish", lambda c: seen.append(c.state["result"].answer), "obs")
        Agent(FakeLLM([FINAL]), tools(), hooks=hooks).run("q")
        self.assertEqual(seen, ["done"])


class AgentProtocolTest(unittest.TestCase):
    def test_parallel_calls_run_concurrently_and_keep_order(self):
        reply = json.dumps({"calls": [{"tool": "slow", "args": {"n": 1}}, {"tool": "slow", "args": {"n": 2}}, {"tool": "slow", "args": {"n": 3}}]})
        llm = FakeLLM([reply, FINAL])
        t0 = time.time()
        res = Agent(llm, tools()).run("q")
        self.assertLess(time.time() - t0, 0.6)                          # three 0.25 s calls would take 0.75 s one after another
        obs = json.loads(llm.calls[1][-1]["content"][len("OBSERVATION: "):])
        self.assertEqual([json.loads(o["result"])["n"] for o in obs], [1, 2, 3])
        self.assertEqual(len(res.steps), 3)

    def test_invalid_step_gets_feedback_and_schema_is_sent(self):
        llm = FakeLLM([json.dumps({"hello": "world"}), FINAL])
        res = Agent(llm, tools()).run("q")
        self.assertEqual(res.steps[0].error, "invalid_step")
        self.assertIn("invalid reply", llm.calls[1][-1]["content"])
        self.assertEqual(llm.schemas[0], AGENT_STEP_FLAT_SCHEMA)
        llm2 = FakeLLM([FINAL])
        Agent(llm2, tools(), use_schema=False).run("q")
        self.assertIsNone(llm2.schemas[0])                                      # constrained decoding can be switched off

    def test_result_keeps_the_conversation(self):
        res = Agent(FakeLLM([call("echo", text="a"), FINAL]), tools()).run("q")
        self.assertEqual([m["role"] for m in res.messages], ["system", "user", "assistant", "user", "assistant"])


class ObservabilityTest(unittest.TestCase):
    def test_nested_spans_and_tree(self):
        tr = InMemoryTracer()
        hooks = HookManager()
        TracingHook(tr).register(hooks)
        llm = TracedLLM(FakeLLM([call("echo", text="a"), FINAL]), tr)
        Agent(llm, tools(), hooks=hooks, tracer=tr).run("q")
        root = tr.by_name("agent.run")[0]
        children = [s for s in tr.spans if s.parent_id == root.span_id]
        self.assertTrue({"llm.chat", "tool.echo"} <= {s.name for s in children})
        self.assertEqual(len({s.trace_id for s in tr.spans}), 1)            # one trace for the whole request
        chat = tr.by_name("llm.chat")[0]
        self.assertEqual(chat.attrs["gen_ai.operation.name"], "chat")
        self.assertIn("gen_ai.usage.input_tokens", chat.attrs)
        self.assertIn("agent.run", tr.tree())
        self.assertTrue(root.attrs["finished"])

    def test_errors_are_recorded_and_exported(self):
        tr = InMemoryTracer()
        with self.assertRaises(ValueError):
            with tr.span("boom"):
                raise ValueError("x")
        self.assertEqual((tr.spans[0].status, tr.spans[0].error), ("error", "ValueError: x"))
        with tempfile.TemporaryDirectory() as d:
            tr.export_jsonl(Path(d) / "t.jsonl")
            rec = json.loads((Path(d) / "t.jsonl").read_text().splitlines()[0])
            self.assertEqual(rec["name"], "boom")

    def test_null_tracer_is_a_no_op(self):
        with NullTracer().span("x") as sp:
            sp.set(a=1)

    def test_opentelemetry_bridge(self):
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:
            self.skipTest("opentelemetry-sdk not installed")
        exp = InMemorySpanExporter()
        tr = OTelTracer(exp)
        Agent(TracedLLM(FakeLLM([call("echo", text="a"), FINAL]), tr), tools(), tracer=tr).run("q")
        names = [s.name for s in exp.get_finished_spans()]
        self.assertIn("agent.run", names)
        self.assertIn("llm.chat", names)
        chat = next(s for s in exp.get_finished_spans() if s.name == "llm.chat")
        self.assertEqual(chat.attributes["gen_ai.operation.name"], "chat")


if __name__ == "__main__":
    unittest.main()
