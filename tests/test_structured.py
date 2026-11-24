import json
import unittest
from unittest import mock

from scholargraph.llm import FakeLLM, MeteredLLM, OllamaLLM, OpenAICompatibleLLM
from scholargraph.structured import (AGENT_STEP_FLAT_SCHEMA, AGENT_STEP_SCHEMA, CONCEPTS_SCHEMA, generate_structured, validate)


class ValidatorTest(unittest.TestCase):
    def test_types_required_enum_and_nesting(self):
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer", "minimum": 1, "maximum": 5},
                                                                    "b": {"type": "string", "enum": ["x", "y"]},
                                                                    "c": {"type": "array", "items": {"type": "number"}}}}
        self.assertEqual(validate({"a": 3, "b": "x", "c": [1, 2.5]}, schema), [])
        self.assertTrue(validate({}, schema)[0].endswith("missing required field 'a'"))
        self.assertTrue(any("below the minimum" in e for e in validate({"a": 0}, schema)))
        self.assertTrue(any("not one of" in e for e in validate({"a": 2, "b": "z"}, schema)))
        self.assertTrue(any("$.c[1]" in e for e in validate({"a": 2, "c": [1, "no"]}, schema)))

    def test_booleans_are_not_integers_and_additional_properties(self):
        self.assertTrue(validate(True, {"type": "integer"}))
        self.assertTrue(validate(True, {"type": "number"}))
        self.assertEqual(validate(None, {"type": ["string", "null"]}), [])
        self.assertTrue(validate({"x": 1}, {"type": "object", "additionalProperties": False}))

    def test_any_of_agent_step_shapes(self):
        self.assertEqual(validate({"tool": "get_paper", "args": {"paper_id": "W1"}}, AGENT_STEP_SCHEMA), [])
        self.assertEqual(validate({"final": "done", "items": ["a"]}, AGENT_STEP_SCHEMA), [])
        self.assertEqual(validate({"calls": [{"tool": "a"}, {"tool": "b", "args": {}}]}, AGENT_STEP_SCHEMA), [])
        self.assertTrue(validate({"tool": 5}, AGENT_STEP_SCHEMA))
        self.assertTrue(validate({"hello": "world"}, AGENT_STEP_SCHEMA))
        self.assertTrue(validate({"final": "x", "items": "not a list"}, AGENT_STEP_SCHEMA))


class RepairLoopTest(unittest.TestCase):
    def test_succeeds_after_feedback_and_counts_attempts(self):
        llm = FakeLLM(["I think the answer is 4", '{"methods": "wrong"}', '{"methods": ["a"], "tasks": [], "datasets": []}'])
        res = generate_structured(llm, [{"role": "user", "content": "extract"}], CONCEPTS_SCHEMA, retries=2)
        self.assertTrue(res.ok)
        self.assertEqual(res.attempts, 3)
        self.assertIn("invalid", llm.calls[1][-1]["content"])                # the problems are fed back to the model
        self.assertIn("expected array", llm.calls[2][-1]["content"])

    def test_gives_up_after_the_retry_budget(self):
        res = generate_structured(FakeLLM(["nope"]), [{"role": "user", "content": "x"}], CONCEPTS_SCHEMA, retries=1)
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 2)
        self.assertTrue(res.problems)


class ProviderSchemaTest(unittest.TestCase):
    def _resp(self, payload):
        m = mock.Mock()
        m.raise_for_status = lambda: None
        m.json = lambda: payload
        return m

    def test_ollama_sends_the_schema_for_constrained_decoding(self):
        with mock.patch("scholargraph.llm.httpx.post", return_value=self._resp({"message": {"content": "{}"}})) as post:
            OllamaLLM("m").chat([{"role": "user", "content": "x"}], schema=AGENT_STEP_FLAT_SCHEMA)
            self.assertEqual(post.call_args.kwargs["json"]["format"], AGENT_STEP_FLAT_SCHEMA)
            OllamaLLM("m").chat([{"role": "user", "content": "x"}], json_mode=True)
            self.assertEqual(post.call_args.kwargs["json"]["format"], "json")

    def test_openai_compatible_uses_json_schema_response_format(self):
        payload = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
        with mock.patch("scholargraph.llm.httpx.post", return_value=self._resp(payload)) as post:
            OpenAICompatibleLLM("m").chat([{"role": "user", "content": "x"}], schema=CONCEPTS_SCHEMA)
            rf = post.call_args.kwargs["json"]["response_format"]
            self.assertEqual(rf["type"], "json_schema")
            self.assertEqual(rf["json_schema"]["schema"], CONCEPTS_SCHEMA)

    def test_metered_wrapper_forwards_the_schema(self):
        inner = mock.Mock(name="inner")
        inner.name = "inner"
        inner.chat.return_value = FakeLLM(["{}"]).chat([{"role": "user", "content": "x"}])
        MeteredLLM(inner).chat([{"role": "user", "content": "x"}], schema={"type": "object"})
        self.assertEqual(inner.chat.call_args.kwargs["schema"], {"type": "object"})


if __name__ == "__main__":
    unittest.main()
