"""Multi-agent orchestration: plan, delegate to specialists, synthesise, verify.

For a question that needs several lookups ("who wrote A and who wrote B, and who do they have in common?") one agent loop is slow and
error-prone. Here the work is split:

1. a **planner** turns the question into at most five *subtasks*, each assigned to a specialist and optionally depending on others;
2. **specialist sub-agents** (a graph-query agent and a retrieval agent, each limited to its own tools) execute the subtasks; independent
   subtasks of one wave run in parallel threads, dependent ones receive the earlier results as context;
3. a **synthesiser** call composes the final answer from the sub-results only;
4. a **verifier** checks every concrete item of the answer against what the tools actually returned and against the graph; unsupported
   items trigger one repair round, and anything still unsupported is dropped rather than shown.

The verifier also exists as a ``GroundingHook`` for the single agent: it blocks a final answer whose items were never seen in any observation.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..graph.store import GraphStore
from ..llm import LLM, Usage
from ..structured import FINAL_SCHEMA, generate_structured
from ..system1.verifier import _norm, entity_exists, observations_text
from .agent import Agent, AgentResult, Step
from .hooks import BEFORE_FINAL, HookContext, HookManager, HookResult
from .tools import Tool

GRAPH_TOOLS = ["get_paper", "get_authors", "get_references", "get_citations", "find_author", "author_papers", "coauthors", "papers_with_concept",
               "top_cited", "co_cited", "shortest_path", "search_papers"]
RETRIEVAL_TOOLS = ["search_papers", "get_paper"]

PLAN_SCHEMA = {"type": "object", "required": ["subtasks"], "properties": {"subtasks": {"type": "array", "maxItems": 5, "items": {
    "type": "object", "required": ["id", "goal", "agent"], "properties": {
        "id": {"type": "string", "minLength": 1}, "goal": {"type": "string", "minLength": 3},
        "agent": {"type": "string", "enum": ["graph", "retrieval"]}, "depends_on": {"type": "array", "items": {"type": "string"}}}}}}}

PLANNER_SYSTEM = ("You plan how to answer a question about scientific papers. Split it into at most 5 subtasks. Each subtask is a self-contained "
                  "question for one specialist: 'graph' (exact lookups: authors, citations, references, co-authors, counts) or 'retrieval' "
                  "(finding papers by topic). Use depends_on only when a subtask needs another's result. Reply with one JSON object: "
                  '{"subtasks": [{"id": "s1", "goal": "...", "agent": "graph", "depends_on": []}]}')
SYNTH_SYSTEM = ("You compose the final answer from the sub-results only. Do not add facts that are not in them. Reply with one JSON object: "
                '{"final": "short answer", "items": ["concrete answers as text"]}')


def restrict_tools(tools: Dict[str, Tool], names: Sequence[str]) -> Dict[str, Tool]:
    return {n: tools[n] for n in names if n in tools}


@dataclass
class Subtask:
    id: str
    goal: str
    agent: str
    depends_on: List[str] = field(default_factory=list)


def plan(llm: LLM, question: str) -> Tuple[List[Subtask], int]:
    """Returns (subtasks, number of planner attempts). Falls back to one graph subtask when the plan is invalid."""
    res = generate_structured(llm, [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": question}], PLAN_SCHEMA, retries=1, max_tokens=300)
    if not res.ok or not res.value["subtasks"]:
        return [Subtask("s1", question, "graph")], res.attempts
    seen: Set[str] = set()
    out: List[Subtask] = []
    for s in res.value["subtasks"]:
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        out.append(Subtask(s["id"], s["goal"], s["agent"], [d for d in s.get("depends_on", []) if d != s["id"]]))
    return out, res.attempts


def waves(subtasks: Sequence[Subtask]) -> List[List[Subtask]]:
    """Group subtasks into waves: everything in a wave only depends on earlier waves (so it can run in parallel). Cycles are broken."""
    done: Set[str] = set()
    known = {s.id for s in subtasks}
    pending = list(subtasks)
    out: List[List[Subtask]] = []
    while pending:
        ready = [s for s in pending if all(d in done or d not in known for d in s.depends_on)]
        if not ready:                                       # a dependency cycle: run what is left sequentially rather than deadlock
            ready = [pending[0]]
        out.append(ready)
        done |= {s.id for s in ready}
        pending = [s for s in pending if s not in ready]
    return out


def verify_items(items: Sequence[str], evidence: str, store: Optional[GraphStore]) -> Tuple[List[str], List[str]]:
    """Split items into (supported, unsupported). Supported = seen in the tools' observations; an item that merely exists in the graph but
    was never observed is NOT enough (the model could have recalled it from memory instead of looking it up)."""
    ev = _norm(evidence)
    supported = [i for i in items if _norm(i) in ev]
    return supported, [i for i in items if i not in supported]


class GroundingHook:
    """before_final hook for a single agent: reject an answer whose items never appeared in an observation."""

    def __init__(self, store: Optional[GraphStore] = None):
        self.store = store

    def register(self, hooks: HookManager) -> "GroundingHook":
        hooks.register(BEFORE_FINAL, self, "grounding", 30)
        return self

    def __call__(self, ctx: HookContext) -> Optional[HookResult]:
        if not ctx.items:
            return None
        ev = " ".join(m["content"] for m in ctx.messages if m["role"] == "user" and m["content"].startswith("OBSERVATION"))
        _, bad = verify_items(ctx.items, ev, self.store)
        if bad:
            return HookResult.block("these items were not found in any tool result: " + "; ".join(bad[:5]))
        ctx.state["verified"] = True                           # downstream hooks (e.g. memory) may rely on this flag
        return None


@dataclass
class SubResult:
    subtask: Subtask
    result: AgentResult


class MultiAgent:
    def __init__(self, planner_llm: LLM, synth_llm: LLM, graph_agent: Agent, retrieval_agent: Optional[Agent] = None, store: Optional[GraphStore] = None,
                 max_parallel: int = 3):
        self.planner_llm, self.synth_llm, self.store, self.max_parallel = planner_llm, synth_llm, store, max_parallel
        self.agents = {"graph": graph_agent, "retrieval": retrieval_agent or graph_agent}
        self.last_plan: List[Subtask] = []

    def _run_subtask(self, s: Subtask, results: Dict[str, SubResult]) -> SubResult:
        ctx = [{"role": "system", "content": "Results of earlier subtasks you may need:\n" + "\n".join(
            f"- {results[d].subtask.goal} -> {results[d].result.answer or results[d].result.items}" for d in s.depends_on if d in results)}] if s.depends_on else None
        return SubResult(s, self.agents.get(s.agent, self.agents["graph"]).run(s.goal, extra_messages=ctx))

    def _synthesise(self, question: str, subs: Sequence[SubResult], feedback: str = "") -> Optional[Dict]:
        body = "\n".join(f"[{r.subtask.id}] {r.subtask.goal}\n    answer: {r.result.answer}\n    items: {r.result.items}" for r in subs)
        msgs = [{"role": "system", "content": SYNTH_SYSTEM}, {"role": "user", "content": f"QUESTION: {question}\n\nSUB-RESULTS:\n{body}" + (f"\n\n{feedback}" if feedback else "")}]
        res = generate_structured(self.synth_llm, msgs, FINAL_SCHEMA, retries=1, max_tokens=400)
        return res.value

    def run(self, question: str) -> AgentResult:
        t0 = time.time()
        subtasks, _ = plan(self.planner_llm, question)
        self.last_plan = subtasks
        results: Dict[str, SubResult] = {}
        for wave in waves(subtasks):
            if len(wave) == 1:
                out = [self._run_subtask(wave[0], results)]
            else:
                with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                    out = list(pool.map(lambda s: self._run_subtask(s, results), wave))
            for r in out:
                results[r.subtask.id] = r
        subs = [results[s.id] for s in subtasks]
        evidence = " ".join(observations_text(r.result) for r in subs)
        final = self._synthesise(question, subs)
        steps = [st for r in subs for st in r.result.steps]
        if final is None:
            return AgentResult(question, "", [], steps, False, time.time() - t0, self._usage(subs), [], "synthesis failed", "planner")
        items = [str(i) for i in (final.get("items") or [])]
        ok, bad = verify_items(items, evidence, self.store)
        if bad:                                                              # one repair round, then drop whatever is still unsupported
            retry = self._synthesise(question, subs, "Your previous answer contained items that are not in the sub-results: "
                                     + "; ".join(bad[:5]) + ". Use only items from the sub-results.")
            if retry is not None:
                final = retry
                items = [str(i) for i in (final.get("items") or [])]
                ok, bad = verify_items(items, evidence, self.store)
            items = ok
            steps = steps + [Step(None, {}, "unsupported items dropped: " + "; ".join(bad[:5]), "verifier")] if bad else steps
        return AgentResult(question, str(final.get("final", "")), items, steps, True, time.time() - t0, self._usage(subs), [], None, "planner")

    @staticmethod
    def _usage(subs: Sequence[SubResult]) -> Dict[str, int]:
        return {k: sum(r.result.usage.get(k, 0) for r in subs) for k in ("llm_calls", "prompt_tokens", "completion_tokens")}
