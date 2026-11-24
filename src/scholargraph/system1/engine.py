"""The query engine: System 1 in front, System 2 behind.

A request flows through cheap, deterministic stages first and reaches the language model only if it must:

    request -> safety check -> answer cache -> direct handlers -> intent classifier -> { RAG answer | agent | planner | refusal }
                                                                                        -> verifier -> cache the verified answer

Every result records which stage produced the answer (``route``), how long it took and how many model calls it needed, so the share
of requests that **never touched the model** can be reported next to quality. ``run`` adapts the engine to the evaluation harness,
so the same graph-derived questions measure the whole system, not only the agent.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agent.agent import Agent, AgentResult
from ..graph.store import GraphStore
from ..rag.graphrag import GraphRAG
from ..rag.guardrails import REFUSAL
from .cache import AnswerCache
from .classifier import IntentClassifier
from .handlers import DirectHandlers
from .safety import check_request
from .verifier import AnswerVerifier

OFF_TOPIC_ANSWER = "I can only answer questions about the indexed scientific papers, their authors and citations."


@dataclass
class QueryResult:
    question: str
    answer: str
    items: List[str]
    route: str                     # refused | cache | handler | rag | agent | planner | off_topic
    seconds: float
    llm_calls: int
    verified: bool
    flags: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


class QueryEngine:
    def __init__(self, store: GraphStore, agent: Agent, rag: Optional[GraphRAG] = None, cache: Optional[AnswerCache] = None,
                 classifier: Optional[IntentClassifier] = None, verifier: Optional[AnswerVerifier] = None, llm_usage=None,
                 planner: Optional[Any] = None, max_chars: int = 1000):
        self.store, self.agent, self.rag, self.cache, self.classifier, self.verifier = store, agent, rag, cache, classifier, verifier
        self.handlers = DirectHandlers(store)
        self.usage = llm_usage                 # a MeteredLLM's Usage object, to count model calls per request
        self.planner, self.max_chars = planner, max_chars
        self.route_counts: Counter = Counter()

    # ------------------------------------------------------------------------------------------
    def _calls(self) -> int:
        return self.usage.calls if self.usage is not None else 0

    def _finish(self, q: str, t0: float, calls0: int, answer: str, items: List[str], route: str, verified: bool, flags: List[str], **detail) -> QueryResult:
        self.route_counts[route] += 1
        return QueryResult(q, answer, items, route, time.time() - t0, self._calls() - calls0, verified, flags, detail)

    def _verify_agent(self, res: AgentResult) -> bool:
        if not res.finished:
            return False
        if self.verifier is not None:
            return self.verifier.verify(res, self.store)
        return bool(res.items)

    def answer(self, question: str) -> QueryResult:
        t0, calls0 = time.time(), self._calls()
        verdict = check_request(question, self.max_chars)
        if not verdict.allowed:
            return self._finish(question, t0, calls0, f"Request refused: {verdict.reason}.", [], "refused", True, verdict.flags)
        q = verdict.cleaned
        if self.cache is not None:
            hit = self.cache.get(q)
            if hit is not None:
                return self._finish(question, t0, calls0, hit.entry.answer, hit.entry.items, "cache", True, verdict.flags, cache_kind=hit.kind, score=hit.score)
        h = self.handlers.handle(q)
        if h is not None:
            if self.cache is not None:
                self.cache.put(q, h.answer, h.items, True, route="handler")
            return self._finish(question, t0, calls0, h.answer, h.items, "handler", True, verdict.flags, handler=h.handler)
        intent = self.classifier.predict(q) if self.classifier is not None else None
        label = intent.label if intent else "graph_lookup"
        detail = {"intent": label, "confidence": intent.confidence if intent else None}
        if label == "off_topic":
            return self._finish(question, t0, calls0, OFF_TOPIC_ANSWER, [], "off_topic", True, verdict.flags, **detail)
        if label == "topic_search" and self.rag is not None:
            a = self.rag.answer(q)
            ok = a.report.ok and not a.report.abstained
            if ok and self.cache is not None:
                self.cache.put(q, a.answer, [], True, route="rag")
            return self._finish(question, t0, calls0, a.answer, [], "rag", ok or a.report.abstained, verdict.flags, groundedness=a.report.groundedness, **detail)
        runner, route = (self.planner, "planner") if (label == "analysis" and self.planner is not None) else (self.agent, "agent")
        res = runner.run(q)
        verified = self._verify_agent(res)
        if verified and self.cache is not None and res.items:
            self.cache.put(q, res.answer, res.items, True, route=route)
        return self._finish(question, t0, calls0, res.answer, res.items, route, verified, verdict.flags, steps=len(res.steps), result=res, **detail)

    # ------------------------------------------------------------------------------------------ harness adapter
    def run(self, question: str) -> AgentResult:
        r = self.answer(question)
        res = r.detail.get("result")
        if res is not None:                                      # the agent's own result keeps its tool trace for trajectory scoring
            res.route = r.route
            res.usage = {**res.usage, "llm_calls": r.llm_calls}
            return res
        return AgentResult(question, r.answer, r.items, [], r.route not in ("refused",), r.seconds,
                           {"llm_calls": r.llm_calls, "prompt_tokens": 0, "completion_tokens": 0}, [], None, r.route)

    def stats(self) -> Dict[str, Any]:
        n = sum(self.route_counts.values()) or 1
        no_llm = sum(v for k, v in self.route_counts.items() if k in ("refused", "cache", "handler", "off_topic"))
        return {"requests": sum(self.route_counts.values()), "routes": dict(self.route_counts), "handled_without_llm": no_llm / n,
                "cache": dict(self.cache.stats) if self.cache else {}}
