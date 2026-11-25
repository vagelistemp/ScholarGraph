import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import torch

from scholargraph.agent.agent import Agent
from scholargraph.agent.tools import make_tools
from scholargraph.api.service import build_service
from scholargraph.demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold
from scholargraph.efficiency import (MODEL_PRESETS, CachedLLM, CascadeLLM, benchmark_llm, count_tokens_approx, estimate_memory_gb, fits, json_acceptor,
                                     truncate_messages)
from scholargraph.evalkit.questions import generate_questions
from scholargraph.evalkit.traces import collect_traces, load_traces, merge_traces, save_traces
from scholargraph.finetune.dpo import (DPOConfig, PreferencePair, build_preference_pairs, build_sft_from_traces, dpo_loss, render_prompt, sequence_logprob,
                                       train_dpo, trajectory_pairs, _batch)
from scholargraph.finetune.lora_extractor import ByteTokenizer, LoraConfigLite, attach_lora
from scholargraph.llm import FakeLLM, LLMResult
from scholargraph.structured import CONCEPTS_SCHEMA


class CachedLLMTest(unittest.TestCase):
    MSG = [{"role": "user", "content": "hello"}]

    def test_hits_avoid_the_model_and_persist_across_instances(self):
        inner = FakeLLM(["one", "two"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.jsonl"
            c = CachedLLM(inner, path)
            a = c.chat(self.MSG)
            b = c.chat(self.MSG)
            self.assertEqual((a.text, b.text, len(inner.calls)), ("one", "one", 1))
            self.assertEqual(b.seconds, 0.0)
            c2 = CachedLLM(FakeLLM(["other"]), path)                   # a new process reads the file
            self.assertEqual(c2.chat(self.MSG).text, "one")
            self.assertEqual(c2.stats, {"hits": 1, "misses": 0, "bypassed": 0})

    def test_sampled_calls_and_different_settings_are_not_conflated(self):
        inner = FakeLLM(["a", "b", "c", "d"])
        c = CachedLLM(inner)
        c.chat(self.MSG, temperature=0.7)
        c.chat(self.MSG, temperature=0.7)
        self.assertEqual((len(inner.calls), c.stats["bypassed"]), (2, 2))
        c.chat(self.MSG)
        c.chat(self.MSG, schema={"type": "object"})                    # the schema is part of the key
        self.assertEqual(c.stats["misses"], 2)


class CascadeTest(unittest.TestCase):
    def test_small_model_is_used_when_valid_and_the_large_one_only_on_failure(self):
        small, large = FakeLLM(['{"methods": [], "tasks": [], "datasets": []}', "garbage"]), FakeLLM(['{"methods": ["x"], "tasks": [], "datasets": []}'])
        c = CascadeLLM(small, large, json_acceptor(CONCEPTS_SCHEMA))
        r1 = c.chat([{"role": "user", "content": "q"}])
        self.assertEqual(c.stats, {"small_ok": 1, "escalated": 0})
        r2 = c.chat([{"role": "user", "content": "q"}])
        self.assertEqual(c.stats, {"small_ok": 1, "escalated": 1})
        self.assertIn("x", r2.text)
        self.assertEqual(len(large.calls), 1)
        self.assertGreaterEqual(r2.prompt_tokens, 2)                    # the cost of the failed small call is included

    def test_json_acceptor_checks_the_schema(self):
        acc = json_acceptor(CONCEPTS_SCHEMA)
        self.assertTrue(acc(LLMResult('{"methods": [], "tasks": [], "datasets": []}')))
        self.assertFalse(acc(LLMResult('{"methods": "x"}')))
        self.assertFalse(acc(LLMResult("nope")))
        self.assertTrue(json_acceptor()(LLMResult('{"anything": 1}')))


class BudgetTest(unittest.TestCase):
    def test_truncation_keeps_system_and_last_message_and_respects_the_budget(self):
        msgs = [{"role": "system", "content": "S" * 40}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 400} for i in range(10)] + \
               [{"role": "user", "content": "LAST"}]
        out = truncate_messages(msgs, 300)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[-1]["content"], "LAST")
        self.assertLess(len(out), len(msgs))
        self.assertLessEqual(sum(count_tokens_approx(m["content"]) + 4 for m in out), 300 + 110)      # the last-kept message may still be large
        tiny = truncate_messages(msgs, 1)
        self.assertEqual([m["content"] for m in tiny], [msgs[0]["content"], "LAST"])

    def test_memory_estimate_by_hand(self):
        m = estimate_memory_gb(8.0, 4, 8192, 32, 8, 128, kv_bits=16, overhead_gb=1.0)
        self.assertAlmostEqual(m["weights_gb"], 4.0)
        self.assertAlmostEqual(m["kv_cache_gb"], 2 * 32 * 8 * 128 * 8192 * 2 / 1e9)               # 1.07 GB
        self.assertAlmostEqual(m["total_gb"], 4.0 + m["kv_cache_gb"] + 1.0)
        self.assertTrue(fits(m["total_gb"], 16))
        big = estimate_memory_gb(**MODEL_PRESETS["14B-class"], bits=16, context_tokens=8192)
        self.assertFalse(fits(big["total_gb"], 16))                                               # 14B at 16 bit cannot share a 16 GB Mac
        self.assertTrue(fits(estimate_memory_gb(**MODEL_PRESETS["14B-class"], bits=4, context_tokens=4096)["total_gb"], 16))

    def test_benchmark_reports_latency_and_throughput(self):
        class Slow(FakeLLM):
            def chat(self, messages, **kw):
                time.sleep(0.05)
                return LLMResult("word " * 10, 5, 10)
        b = benchmark_llm(Slow(["x"]), ["a", "b", "c"], repeats=2)
        self.assertEqual(b["calls"], 6)
        self.assertGreaterEqual(b["p50_seconds"], 0.05)
        self.assertGreater(b["tokens_per_second"], 100)
        self.assertLessEqual(b["p50_seconds"], b["p95_seconds"])


class DPOMathTest(unittest.TestCase):
    def test_loss_equals_log_two_when_policy_equals_reference(self):
        z = torch.tensor([-5.0, -7.0])
        loss, acc, margin = dpo_loss(z, z - 1, z, z - 1, beta=0.3)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        self.assertEqual(float(margin), 0.0)

    def test_hand_computed_value_and_direction(self):
        # policy margin 2, reference margin 1, beta 0.5 -> -log sigmoid(0.5 * (2 - 1))
        loss, acc, margin = dpo_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-3.0]), beta=0.5)
        self.assertAlmostEqual(float(loss), -math.log(1 / (1 + math.exp(-0.5))), places=6)
        self.assertEqual(float(acc), 1.0)
        worse, wacc, _ = dpo_loss(torch.tensor([-3.0]), torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
        self.assertGreater(float(worse), float(loss))
        self.assertEqual(float(wacc), 0.0)


class TraceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = make_demo_gold(450, 31)
        cls.svc = build_service(cls.gold, make_demo_concepts(cls.gold), ScriptedPolicyLLM())
        cls.tools = make_tools(cls.svc.store, cls.svc.retriever)
        cls.questions = generate_questions(cls.svc.store, 3, 0)
        cls.good = collect_traces(Agent(ScriptedPolicyLLM(), cls.tools), cls.questions)

        class Wrong(ScriptedPolicyLLM):                         # a weaker policy: asks for the paper (not a search) and then gives up
            def chat(self, messages, **kw):
                n = len([m for m in messages if m["role"] == "user" and m["content"].startswith("OBSERVATION")])
                if n == 0:
                    return LLMResult(json.dumps({"tool": "top_cited", "args": {"k": 1}}), 10, 5)
                return LLMResult(json.dumps({"final": "no idea", "items": ["no idea"]}), 10, 5)
        cls.bad = collect_traces(Agent(Wrong(), cls.tools), cls.questions)
        cls.runs = merge_traces(cls.good, cls.bad)


class TraceTest(TraceFixture):
    def test_correctness_flags_and_round_trip(self):
        self.assertTrue(all(ok for att in self.good.values() for _, ok in att))
        self.assertTrue(all(not ok for att in self.bad.values() for _, ok in att))
        with tempfile.TemporaryDirectory() as d:
            save_traces(self.runs, Path(d) / "t.jsonl")
            back = load_traces(Path(d) / "t.jsonl")
        self.assertEqual({k: [ok for _, ok in v] for k, v in back.items()}, {k: [ok for _, ok in v] for k, v in self.runs.items()})
        q = self.questions[0].qid
        self.assertEqual(back[q][0][0].messages, self.runs[q][0][0].messages)
        self.assertEqual(back[q][0][0].steps[0].tool, self.runs[q][0][0].steps[0].tool)

    def test_preference_pairs_come_from_the_first_divergence(self):
        pairs = build_preference_pairs(self.runs)
        self.assertEqual(len(pairs), len(self.questions))
        p = pairs[0]
        self.assertIn('"tool": "search_papers"', p.chosen)
        self.assertIn('"tool": "top_cited"', p.rejected)
        self.assertTrue(p.prompt.startswith("### System:") and "### User:\nQUESTION:" in p.prompt and p.prompt.endswith("### Assistant:\n"))
        self.assertIsNone(trajectory_pairs(self.good[self.questions[0].qid][0][0], self.good[self.questions[0].qid][0][0]))    # identical runs: no pair

    def test_sft_examples_use_only_correct_runs_and_one_example_per_assistant_turn(self):
        sft = build_sft_from_traces(self.runs)
        n_turns = sum(sum(1 for m in r.messages if m["role"] == "assistant") for att in self.good.values() for r, _ in att)
        self.assertEqual(len(sft), n_turns)
        self.assertTrue(all("no idea" not in t for _, t in sft))
        self.assertEqual(build_sft_from_traces(self.bad), [])


class DPOTrainingTest(unittest.TestCase):
    def test_training_raises_the_preferred_reply_relative_to_the_reference_and_freezes_the_base(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=259, n_positions=1024, n_embd=32, n_layer=2, n_head=2, pad_token_id=0, eos_token_id=1, bos_token_id=1)
        base = GPT2LMHeadModel(cfg)
        before = {k: v.clone() for k, v in base.state_dict().items()}
        model = attach_lora(base, LoraConfigLite(r=8, alpha=16, dropout=0.0, target_modules=("c_attn", "c_fc", "c_proj")), fan_in_fan_out=True)     # no dropout: policy == reference at step 1
        tok = ByteTokenizer()
        pairs = [PreferencePair(render_prompt([{"role": "user", "content": f"q{i}"}]), '{"tool": "search_papers"}', '{"tool": "top_cited"}') for i in range(8)]
        hist = train_dpo(model, tok, pairs, DPOConfig(beta=0.5, lr=1e-2, epochs=12, batch_size=4, device="cpu", log_every=10 ** 9), log=lambda *a: None)
        self.assertAlmostEqual(hist[0]["loss"], math.log(2), places=3)         # at the start the policy IS the reference
        self.assertLess(hist[-1]["loss"], hist[0]["loss"])
        self.assertGreater(hist[-1]["margin"], 0.0)
        self.assertGreaterEqual(hist[-1]["reward_accuracy"], 0.75)
        ids_c, att_c, lab_c = _batch(tok, [pairs[0].prompt], [pairs[0].chosen], 1024, "cpu")
        ids_r, att_r, lab_r = _batch(tok, [pairs[0].prompt], [pairs[0].rejected], 1024, "cpu")
        with torch.no_grad():
            pol = float(sequence_logprob(model, ids_c, att_c, lab_c) - sequence_logprob(model, ids_r, att_r, lab_r))
            with model.disable_adapter():
                ref = float(sequence_logprob(model, ids_c, att_c, lab_c) - sequence_logprob(model, ids_r, att_r, lab_r))
        self.assertGreater(pol, ref)                                           # the trained policy prefers the chosen reply more than the reference does
        import re
        after = model.state_dict()
        for k, v in before.items():
            key = "base_model.model." + re.sub(r"(c_attn|c_fc|c_proj)\.(weight|bias)", r"\1.base_layer.\2", k)
            self.assertTrue(torch.equal(v, after[key]), f"frozen weight changed: {k}")


if __name__ == "__main__":
    unittest.main()
