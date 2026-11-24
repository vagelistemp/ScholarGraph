"""A small tool-using agent (the ReAct pattern) over the knowledge graph, with hooks, schema validation, parallel tools and tracing.

The agent talks to any chat model through a *plain JSON protocol* instead of a vendor-specific tool-calling API, so it works with a local
Ollama model, a hosted model or a scripted test double alike. On every turn the model must reply with exactly one JSON object:

    {"thought": "why", "tool": "tool_name", "args": {...}}                  call one tool
    {"thought": "why", "calls": [{"tool": "a", "args": {}}, {"tool": "b"}]}  call several independent tools at once (run in parallel)
    {"final": "the answer text", "items": ["short", "answers"]}              finish; ``items`` is the machine-checkable list of answers

Every reply is validated against ``structured.AGENT_STEP_SCHEMA`` (and, for servers that support it, constrained decoding forces the
shape). The loop is bounded (``max_steps``), tolerant of malformed replies, records a full trace, and fires **lifecycle hooks**
(``hooks.py``) around every model call, tool call and final answer, so budgets, permissions, redaction and memory are plug-ins rather
than changes to the loop.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm import LLM, Message, Usage, parse_json_object
from ..observability import NullTracer
from ..structured import AGENT_STEP_FLAT_SCHEMA, AGENT_STEP_SCHEMA, validate
from .hooks import (AFTER_LLM, AFTER_TOOL, BEFORE_FINAL, BEFORE_LLM, BEFORE_TOOL, ON_ERROR, ON_FINISH, HookContext, HookManager)
from .tools import Tool

SYSTEM_TEMPLATE = """You answer questions about a graph of scientific papers by calling tools. Reply with exactly ONE JSON object per turn and nothing else.

To call a tool:        {{"thought": "...", "tool": "<name>", "args": {{...}}}}
To call several at once (only when they are independent):  {{"thought": "...", "calls": [{{"tool": "<name>", "args": {{...}}}}, ...]}}
To finish:             {{"final": "<answer>", "items": ["<short answer 1>", "<short answer 2>"]}}

Rules: paper ids look like W123. Use search_papers to find a paper's id from its title. Never invent ids; take them from observations. "items" must contain only the concrete answers (titles, names or numbers as text). If a question cannot be answered with the tools, finish with an empty items list.

Tools:
{tools}"""


@dataclass
class Step:
    tool: Optional[str]
    args: Dict[str, Any]
    observation: str
    error: Optional[str] = None


@dataclass
class AgentResult:
    question: str
    answer: str
    items: List[str]
    steps: List[Step]
    finished: bool
    seconds: float
    usage: Dict[str, int] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)       # the full conversation (used to build training data from traces)
    stopped_by: Optional[str] = None                             # reason when a hook ended the run
    route: str = "agent"


def _clip(text: str, n: int = 2500) -> str:
    return text if len(text) <= n else text[:n] + " ...[truncated]"


class Agent:
    def __init__(self, llm: LLM, tools: Dict[str, Tool], max_steps: int = 6, max_obs_chars: int = 2500, hooks: Optional[HookManager] = None,
                 tracer=None, use_schema: bool = True, max_parallel: int = 4, system_extra: str = ""):
        self.llm, self.tools, self.max_steps, self.max_obs_chars = llm, tools, max_steps, max_obs_chars
        self.hooks, self.tracer, self.use_schema, self.max_parallel = hooks or HookManager(), tracer or NullTracer(), use_schema, max_parallel
        self.system = SYSTEM_TEMPLATE.format(tools="\n".join(t.spec() for t in tools.values())) + (("\n\n" + system_extra) if system_extra else "")

    # ------------------------------------------------------------------------------------------ one tool call
    def _call_tool(self, name: Any, args: Any, step_no: int, question: str, state: Dict[str, Any]) -> Tuple[str, Step, Optional[str]]:
        """Returns (observation text, Step, stop_reason)."""
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None or not isinstance(args, dict):
            obs = f"unknown tool {name!r}. Available: {', '.join(self.tools)}"
            self.hooks.fire(ON_ERROR, HookContext(ON_ERROR, question, step_no, tool=str(name), error="unknown_tool", state=state))
            return obs, Step(str(name), {}, obs, "unknown_tool"), None
        ctx = HookContext(BEFORE_TOOL, question, step_no, tool=name, args=dict(args), state=state)
        res = self.hooks.fire(BEFORE_TOOL, ctx)
        if res.action == "stop":
            return f"run stopped: {res.reason}", Step(name, dict(args), res.reason, "stopped"), res.reason
        if res.action == "block":
            obs = f"blocked: {res.reason}"
            return obs, Step(name, dict(args), obs, "blocked"), None
        args = res.patch.get("args", ctx.args) if res.action == "modify" else ctx.args
        error: Optional[str] = None
        with self.tracer.span("tool." + name, **{"gen_ai.tool.name": name}):
            try:
                obs = json.dumps(tool.fn(**args), default=str, ensure_ascii=False)
            except TypeError as e:
                obs, error = f"bad arguments for {name}: {e}", "bad_args"
            except Exception as e:                                           # a tool failure must not crash the agent
                obs, error = f"tool {name} failed: {type(e).__name__}: {e}", "tool_error"
        if error:
            self.hooks.fire(ON_ERROR, HookContext(ON_ERROR, question, step_no, tool=name, args=args, error=error, state=state))
        actx = HookContext(AFTER_TOOL, question, step_no, tool=name, args=args, observation=obs, state=state)
        ares = self.hooks.fire(AFTER_TOOL, actx)
        if ares.action == "modify" and "observation" in ares.patch:
            obs = ares.patch["observation"]
        return obs, Step(name, dict(args), _clip(obs, 200), error), None

    # ------------------------------------------------------------------------------------------ the loop
    def run(self, question: str, extra_messages: Optional[List[Message]] = None) -> AgentResult:
        t0 = time.time()
        state: Dict[str, Any] = {}
        messages: List[Message] = [{"role": "system", "content": self.system}, *(extra_messages or []), {"role": "user", "content": f"QUESTION: {question}"}]
        steps: List[Step] = []
        usage = Usage()
        stopped: Optional[str] = None
        final_rejections = 0
        with self.tracer.span("agent.run", question=question[:200]) as root:
            for step_no in range(1, self.max_steps + 1):
                bctx = HookContext(BEFORE_LLM, question, step_no, messages=list(messages), state=state)
                bres = self.hooks.fire(BEFORE_LLM, bctx)
                if bres.action == "stop":
                    stopped = bres.reason
                    break
                if bres.action == "modify" and "append_messages" in bres.patch:
                    messages = messages + list(bres.patch["append_messages"])
                r = self.llm.chat(messages, temperature=0.0, max_tokens=400, json_mode=True, schema=AGENT_STEP_FLAT_SCHEMA if self.use_schema else None)
                usage.add(r)
                state["last_tokens"] = r.prompt_tokens + r.completion_tokens
                messages.append({"role": "assistant", "content": r.text})
                ares = self.hooks.fire(AFTER_LLM, HookContext(AFTER_LLM, question, step_no, llm_text=r.text, state=state))
                if ares.action == "stop":
                    stopped = ares.reason
                    break
                msg = parse_json_object(r.text)
                problems = ["the reply is not a single JSON object"] if not isinstance(msg, dict) else validate(msg, AGENT_STEP_SCHEMA)
                if problems:
                    kind = "invalid_json" if not isinstance(msg, dict) else "invalid_step"
                    steps.append(Step(None, {}, "", kind))
                    self.hooks.fire(ON_ERROR, HookContext(ON_ERROR, question, step_no, error=kind, llm_text=r.text, state=state))
                    messages.append({"role": "user", "content": "OBSERVATION: " + ("your reply was not a single JSON object. Reply with one JSON object only."
                                                                                if kind == "invalid_json" else "invalid reply: " + "; ".join(problems[:3]))})
                    continue
                if "final" in msg:
                    items = msg.get("items") or []
                    items = [str(i) for i in items] if isinstance(items, list) else [str(items)]
                    fctx = HookContext(BEFORE_FINAL, question, step_no, answer=str(msg["final"]), items=items, messages=list(messages), state=state)
                    fres = self.hooks.fire(BEFORE_FINAL, fctx)
                    if fres.action == "block" and final_rejections < 2:
                        final_rejections += 1
                        steps.append(Step(None, {}, fres.reason, "final_rejected"))
                        messages.append({"role": "user", "content": f"OBSERVATION: your answer was rejected: {fres.reason}. Fix it with the tools or finish with what is verified."})
                        continue
                    answer, items = (fres.patch.get("answer", fctx.answer), fres.patch.get("items", fctx.items)) if fres.action == "modify" else (fctx.answer, fctx.items)
                    result = AgentResult(question, answer, items, steps, True, time.time() - t0, self._usage(usage), messages)
                    root.set(finished=True, steps=len(steps))
                    self.hooks.fire(ON_FINISH, HookContext(ON_FINISH, question, step_no, answer=answer, items=items, state={**state, "result": result}))
                    return result
                calls = msg["calls"][: self.max_parallel] if "calls" in msg else [{"tool": msg.get("tool"), "args": msg.get("args") or {}}]
                if len(calls) == 1:
                    outs = [self._call_tool(calls[0].get("tool"), calls[0].get("args") or {}, step_no, question, state)]
                else:
                    with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                        outs = list(pool.map(lambda c: self._call_tool(c.get("tool"), c.get("args") or {}, step_no, question, state), calls))
                for _, st, _ in outs:
                    steps.append(st)
                stop_reasons = [s for _, _, s in outs if s]
                if stop_reasons:
                    stopped = stop_reasons[0]
                    break
                obs = outs[0][0] if len(outs) == 1 else json.dumps([{"tool": c.get("tool"), "result": o} for c, (o, _, _) in zip(calls, outs)], ensure_ascii=False)
                messages.append({"role": "user", "content": "OBSERVATION: " + _clip(obs, self.max_obs_chars)})
            result = AgentResult(question, "", [], steps, False, time.time() - t0, self._usage(usage), messages, stopped)
            root.set(finished=False, steps=len(steps), stopped_by=stopped or "max_steps")
            self.hooks.fire(ON_FINISH, HookContext(ON_FINISH, question, len(steps), state={**state, "result": result}))
            return result

    @staticmethod
    def _usage(u: Usage) -> Dict[str, int]:
        return {"llm_calls": u.calls, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
