"""A small tool-using agent (the ReAct pattern) over the knowledge graph.

The agent talks to any chat model through a *plain JSON protocol* instead of a vendor-specific tool-calling API, so it works with a local
Ollama model, a hosted model or a scripted test double alike. On every turn the model must reply with exactly one JSON object:

    {"thought": "why", "tool": "tool_name", "args": {...}}          call a tool; the result comes back as an OBSERVATION
    {"final": "the answer text", "items": ["short", "answers"]}     finish; ``items`` is the machine-checkable list of answers

The loop is bounded (``max_steps``), tolerant of malformed replies (it explains the problem to the model and lets it retry) and records
a full trace so that behaviour can be inspected and evaluated.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import LLM, Message, Usage, parse_json_object
from .tools import Tool

SYSTEM_TEMPLATE = """You answer questions about a graph of scientific papers by calling tools. Reply with exactly ONE JSON object per turn and nothing else.

To call a tool:   {{"thought": "...", "tool": "<name>", "args": {{...}}}}
To finish:        {{"final": "<answer>", "items": ["<short answer 1>", "<short answer 2>"]}}

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


def _clip(text: str, n: int = 2500) -> str:
    return text if len(text) <= n else text[:n] + " ...[truncated]"


class Agent:
    def __init__(self, llm: LLM, tools: Dict[str, Tool], max_steps: int = 6, max_obs_chars: int = 2500):
        self.llm, self.tools, self.max_steps, self.max_obs_chars = llm, tools, max_steps, max_obs_chars
        self.system = SYSTEM_TEMPLATE.format(tools="\n".join(t.spec() for t in tools.values()))

    def run(self, question: str) -> AgentResult:
        t0 = time.time()
        messages: List[Message] = [{"role": "system", "content": self.system}, {"role": "user", "content": f"QUESTION: {question}"}]
        steps: List[Step] = []
        usage = Usage()
        for _ in range(self.max_steps):
            r = self.llm.chat(messages, temperature=0.0, max_tokens=400, json_mode=True)
            usage.add(r)
            messages.append({"role": "assistant", "content": r.text})
            msg = parse_json_object(r.text)
            if not isinstance(msg, dict):
                messages.append({"role": "user", "content": "OBSERVATION: your reply was not a single JSON object. Reply with one JSON object only."})
                steps.append(Step(None, {}, "", "invalid_json"))
                continue
            if "final" in msg:
                items = msg.get("items") or []
                items = [str(i) for i in items] if isinstance(items, list) else [str(items)]
                return AgentResult(question, str(msg["final"]), items, steps, True, time.time() - t0, self._usage(usage))
            name, args = msg.get("tool"), msg.get("args") or {}
            tool = self.tools.get(name) if isinstance(name, str) else None
            if tool is None or not isinstance(args, dict):
                obs = f"unknown tool {name!r}. Available: {', '.join(self.tools)}"
                steps.append(Step(str(name), {}, obs, "unknown_tool"))
            else:
                try:
                    obs = json.dumps(tool.fn(**args), default=str, ensure_ascii=False)
                    steps.append(Step(name, args, _clip(obs, 200)))
                except TypeError as e:
                    obs = f"bad arguments for {name}: {e}"
                    steps.append(Step(name, args, obs, "bad_args"))
                except Exception as e:                                       # a tool failure must not crash the agent
                    obs = f"tool {name} failed: {type(e).__name__}: {e}"
                    steps.append(Step(name, args, obs, "tool_error"))
            messages.append({"role": "user", "content": "OBSERVATION: " + _clip(obs, self.max_obs_chars)})
        return AgentResult(question, "", [], steps, False, time.time() - t0, self._usage(usage))

    @staticmethod
    def _usage(u: Usage) -> Dict[str, int]:
        return {"llm_calls": u.calls, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
