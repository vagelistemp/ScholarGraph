"""Structured output: make a language model's reply conform to a schema, and repair it when it does not.

Three layers of defence, from cheapest to most robust:

1. **Constrained decoding** (when the server supports it): the JSON schema is sent with the request and the sampler can only produce
   tokens that keep the output valid (Ollama's ``format`` parameter, OpenAI-style ``response_format``). A small local model then cannot
   emit malformed JSON at all.
2. **Validation**: whatever comes back is parsed and checked against the schema by ``validate`` (a small JSON-Schema subset: type,
   properties, required, enum, items, minimum, maximum, minLength, maxLength, anyOf; no external dependency).
3. **Repair loop**: on failure the exact list of problems is sent back to the model as feedback and it is asked again, up to
   ``retries`` times. Every attempt is counted so the cost of unreliability is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm import LLM, Message, parse_json_object

_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list, "null": type(None)}


def _type_ok(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[t]) and not (t in ("string", "object", "array") and isinstance(value, bool))


def validate(value: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errs: List[str] = []
    if "anyOf" in schema:
        branches = [validate(value, s, path) for s in schema["anyOf"]]
        if not any(not b for b in branches):
            errs.append(f"{path}: matches none of the allowed shapes ({'; '.join(b[0] for b in branches if b)})")
        return errs
    t = schema.get("type")
    if t is not None:
        types = [t] if isinstance(t, str) else list(t)
        if not any(_type_ok(value, x) for x in types):
            return [f"{path}: expected {' or '.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} characters")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}: missing required field '{req}'")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected field '{k}'")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(v, schema["items"], f"{path}[{i}]")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
    return errs


# ------------------------------------------------------------------------------------------------ schemas used by the system
TOOL_CALL_SCHEMA = {"type": "object", "required": ["tool"], "properties": {
    "thought": {"type": "string"}, "tool": {"type": "string", "minLength": 1}, "args": {"type": "object"}}}
PARALLEL_CALLS_SCHEMA = {"type": "object", "required": ["calls"], "properties": {
    "thought": {"type": "string"},
    "calls": {"type": "array", "maxItems": 6, "items": {"type": "object", "required": ["tool"], "properties": {
        "tool": {"type": "string", "minLength": 1}, "args": {"type": "object"}}}}}}
FINAL_SCHEMA = {"type": "object", "required": ["final"], "properties": {
    "final": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}}}}
AGENT_STEP_SCHEMA = {"anyOf": [TOOL_CALL_SCHEMA, PARALLEL_CALLS_SCHEMA, FINAL_SCHEMA]}
# for constrained decoding servers that need a single object schema (no anyOf at the top level)
AGENT_STEP_FLAT_SCHEMA = {"type": "object", "properties": {
    "thought": {"type": "string"}, "tool": {"type": "string"}, "args": {"type": "object"},
    "calls": {"type": "array", "items": {"type": "object", "properties": {"tool": {"type": "string"}, "args": {"type": "object"}}, "required": ["tool"]}},
    "final": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}}}}
CONCEPTS_SCHEMA = {"type": "object", "required": ["methods", "tasks", "datasets"], "properties": {
    k: {"type": "array", "items": {"type": "string"}} for k in ("methods", "tasks", "datasets")}}


@dataclass
class StructuredResult:
    value: Optional[Dict[str, Any]]
    attempts: int
    problems: List[str] = field(default_factory=list)       # problems of the LAST failed attempt (empty on success)
    texts: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


def generate_structured(llm: LLM, messages: Sequence[Message], schema: Dict[str, Any], retries: int = 2, max_tokens: int = 400,
                        constrain: Optional[Dict[str, Any]] = None) -> StructuredResult:
    """Ask for a reply that validates against ``schema``; on failure feed the problems back and retry.

    ``constrain`` is the (single-object) schema sent to the server for constrained decoding; the full ``schema`` is always enforced locally."""
    msgs = list(messages)
    texts: List[str] = []
    problems: List[str] = []
    for attempt in range(1, retries + 2):
        r = llm.chat(msgs, temperature=0.0, max_tokens=max_tokens, json_mode=True, schema=constrain)
        texts.append(r.text)
        obj = parse_json_object(r.text)
        problems = ["the reply is not a JSON object"] if obj is None else validate(obj, schema)
        if not problems:
            return StructuredResult(obj, attempt, [], texts)
        msgs = msgs + [{"role": "assistant", "content": r.text},
                       {"role": "user", "content": "Your reply was invalid: " + "; ".join(problems[:5]) + ". Reply again with one valid JSON object only."}]
    return StructuredResult(None, retries + 1, problems, texts)
