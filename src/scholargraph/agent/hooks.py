"""Lifecycle hooks: your own code, running at fixed points of the agent loop.

A hook is a small function that receives a ``HookContext`` and may return a ``HookResult`` to steer the agent. The loop fires these events:

=============  ===========================================================  ======================================
event          when                                                         what a hook may do
=============  ===========================================================  ======================================
before_llm     just before every model call                                 add messages, stop the run
after_llm      right after the reply                                        stop the run
before_tool    before each tool call                                        block it, change its arguments
after_tool     after a tool returned                                        rewrite the observation (e.g. redact)
before_final   when the model wants to answer                               reject it (the model must try again), rewrite it
on_error       when a step fails                                            observe
on_finish      once, at the end                                             observe (log, store memories, metrics)
=============  ===========================================================  ======================================

Typical uses: budgets (tokens, steps, seconds), tool permissions, human approval for risky tools, privacy redaction, loop detection,
tracing and logging, fact-checking the final answer. Hooks are plain Python, so every one of them is unit-testable without a model.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

BEFORE_LLM, AFTER_LLM, BEFORE_TOOL, AFTER_TOOL = "before_llm", "after_llm", "before_tool", "after_tool"
BEFORE_FINAL, ON_ERROR, ON_FINISH = "before_final", "on_error", "on_finish"
EVENTS = (BEFORE_LLM, AFTER_LLM, BEFORE_TOOL, AFTER_TOOL, BEFORE_FINAL, ON_ERROR, ON_FINISH)


@dataclass
class HookContext:
    event: str
    question: str = ""
    step: int = 0
    messages: List[Dict[str, str]] = field(default_factory=list)
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None
    answer: Optional[str] = None
    items: List[str] = field(default_factory=list)
    llm_text: Optional[str] = None
    error: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)          # shared by all hooks during one run (budgets, counters, ...)


@dataclass
class HookResult:
    action: str = "allow"                    # allow | block | modify | stop
    reason: str = ""
    patch: Dict[str, Any] = field(default_factory=dict)   # for modify: new args / observation / answer / items / append_messages

    @staticmethod
    def allow() -> "HookResult":
        return HookResult()

    @staticmethod
    def block(reason: str) -> "HookResult":
        return HookResult("block", reason)

    @staticmethod
    def stop(reason: str) -> "HookResult":
        return HookResult("stop", reason)

    @staticmethod
    def modify(**patch: Any) -> "HookResult":
        return HookResult("modify", patch=patch)


@dataclass
class _Entry:
    name: str
    fn: Callable[[HookContext], Optional[HookResult]]
    priority: int


class HookManager:
    """Registry of hooks. Lower ``priority`` runs first. The first ``block`` or ``stop`` ends the chain; ``modify`` patches accumulate."""

    def __init__(self):
        self._hooks: Dict[str, List[_Entry]] = {e: [] for e in EVENTS}
        self.log: List[str] = []

    def register(self, event: str, fn: Callable[[HookContext], Optional[HookResult]], name: Optional[str] = None, priority: int = 100) -> "HookManager":
        if event not in self._hooks:
            raise ValueError(f"unknown event {event!r}; choose from {EVENTS}")
        self._hooks[event].append(_Entry(name or getattr(fn, "__name__", "hook"), fn, priority))
        self._hooks[event].sort(key=lambda e: e.priority)
        return self

    def fire(self, event: str, ctx: HookContext) -> HookResult:
        merged: Dict[str, Any] = {}
        for h in self._hooks[event]:
            try:
                res = h.fn(ctx) or HookResult.allow()
            except Exception as e:                                          # a broken hook must never take the agent down
                self.log.append(f"{h.name} raised {type(e).__name__}: {e}")
                continue
            if res.action in ("block", "stop"):
                self.log.append(f"{h.name}: {res.action} ({res.reason})")
                return res
            if res.action == "modify":
                merged.update(res.patch)
                for k, v in res.patch.items():                              # later hooks see earlier patches
                    if k == "append_messages":
                        ctx.messages = ctx.messages + list(v)
                    elif hasattr(ctx, k):
                        setattr(ctx, k, v)
        return HookResult.modify(**merged) if merged else HookResult.allow()

    def count(self, event: Optional[str] = None) -> int:
        return sum(len(v) for k, v in self._hooks.items() if event in (None, k))


# ------------------------------------------------------------------------------------------------ built-in hooks
class BudgetHook:
    """Stops the run when it spends too much: model calls, tool calls, tokens or wall-clock time."""

    def __init__(self, max_llm_calls: int = 10, max_tool_calls: int = 12, max_tokens: int = 20_000, max_seconds: float = 120.0):
        self.limits = dict(llm=max_llm_calls, tool=max_tool_calls, tokens=max_tokens, seconds=max_seconds)

    def register(self, hooks: HookManager) -> "BudgetHook":
        hooks.register(BEFORE_LLM, self.before_llm, "budget", 10)
        hooks.register(BEFORE_TOOL, self.before_tool, "budget", 10)
        hooks.register(AFTER_LLM, self.after_llm, "budget", 10)
        return self

    @staticmethod
    def _s(ctx: HookContext) -> Dict[str, Any]:
        return ctx.state.setdefault("budget", {"llm": 0, "tool": 0, "tokens": 0, "t0": time.time()})

    def before_llm(self, ctx: HookContext) -> Optional[HookResult]:
        b = self._s(ctx)
        if b["llm"] >= self.limits["llm"]:
            return HookResult.stop(f"model-call budget of {self.limits['llm']} used")
        if b["tokens"] >= self.limits["tokens"]:
            return HookResult.stop(f"token budget of {self.limits['tokens']} used")
        if time.time() - b["t0"] > self.limits["seconds"]:
            return HookResult.stop(f"time budget of {self.limits['seconds']} s used")
        b["llm"] += 1
        return None

    def after_llm(self, ctx: HookContext) -> Optional[HookResult]:
        self._s(ctx)["tokens"] += int(ctx.state.get("last_tokens", 0))
        return None

    def before_tool(self, ctx: HookContext) -> Optional[HookResult]:
        b = self._s(ctx)
        if b["tool"] >= self.limits["tool"]:
            return HookResult.stop(f"tool-call budget of {self.limits['tool']} used")
        b["tool"] += 1
        return None


class ToolAllowlistHook:
    """Only the listed tools may run (a permission system in miniature)."""

    def __init__(self, allowed: Sequence[str]):
        self.allowed = set(allowed)

    def register(self, hooks: HookManager) -> "ToolAllowlistHook":
        hooks.register(BEFORE_TOOL, self, "allowlist", 5)
        return self

    def __call__(self, ctx: HookContext) -> Optional[HookResult]:
        return None if ctx.tool in self.allowed else HookResult.block(f"tool '{ctx.tool}' is not permitted")


class ApprovalHook:
    """Risky tools need a yes from a human (or a policy function) before they run."""

    def __init__(self, needs_approval: Sequence[str], approver: Callable[[str, Dict[str, Any]], bool]):
        self.needs, self.approver = set(needs_approval), approver

    def register(self, hooks: HookManager) -> "ApprovalHook":
        hooks.register(BEFORE_TOOL, self, "approval", 6)
        return self

    def __call__(self, ctx: HookContext) -> Optional[HookResult]:
        if ctx.tool in self.needs and not self.approver(ctx.tool or "", ctx.args):
            return HookResult.block(f"approval for '{ctx.tool}' was denied")
        return None


PII_PATTERNS = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
                (re.compile(r"\b(?:\d[ -]?){12,15}\d\b"), "[CARD]"),                  # before PHONE: a card number must not be taken for a phone
                (re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)"), "[PHONE]")]


def redact(text: str) -> str:
    for pat, rep in PII_PATTERNS:
        text = pat.sub(rep, text)
    return text


class RedactionHook:
    """Removes e-mail addresses, phone numbers and card-like numbers from tool observations and final answers."""

    def register(self, hooks: HookManager) -> "RedactionHook":
        hooks.register(AFTER_TOOL, self.after_tool, "redact", 20)
        hooks.register(BEFORE_FINAL, self.before_final, "redact", 20)
        return self

    @staticmethod
    def after_tool(ctx: HookContext) -> Optional[HookResult]:
        new = redact(ctx.observation or "")
        return HookResult.modify(observation=new) if new != ctx.observation else None

    @staticmethod
    def before_final(ctx: HookContext) -> Optional[HookResult]:
        new = redact(ctx.answer or "")
        items = [redact(i) for i in ctx.items]
        return HookResult.modify(answer=new, items=items) if (new != ctx.answer or items != ctx.items) else None


class LoopGuardHook:
    """Detects an agent that repeats the same tool call with the same arguments and blocks the repeat."""

    def __init__(self, max_repeats: int = 1):
        self.max_repeats = max_repeats

    def register(self, hooks: HookManager) -> "LoopGuardHook":
        hooks.register(BEFORE_TOOL, self, "loopguard", 8)
        return self

    def __call__(self, ctx: HookContext) -> Optional[HookResult]:
        seen = ctx.state.setdefault("calls_seen", {})
        key = (ctx.tool, repr(sorted(ctx.args.items())))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > self.max_repeats:                      # max_repeats = how many times one exact call may run in total
            return HookResult.block("this exact call was already made; use its earlier result or try something different")
        return None


class TracingHook:
    """Records one span per tool call and per final answer on a tracer (see ``observability``)."""

    def __init__(self, tracer):
        self.tracer = tracer

    def register(self, hooks: HookManager) -> "TracingHook":
        hooks.register(AFTER_TOOL, self.after_tool, "tracing", 90)
        hooks.register(BEFORE_FINAL, self.before_final, "tracing", 90)
        hooks.register(ON_ERROR, self.on_error, "tracing", 90)
        return self

    def after_tool(self, ctx: HookContext) -> Optional[HookResult]:
        with self.tracer.span("tool.call", **{"gen_ai.tool.name": ctx.tool, "tool.args": str(ctx.args)[:200],
                                              "tool.observation_chars": len(ctx.observation or "")}):
            pass
        return None

    def before_final(self, ctx: HookContext) -> Optional[HookResult]:
        with self.tracer.span("agent.final", answer_chars=len(ctx.answer or ""), n_items=len(ctx.items)):
            pass
        return None

    def on_error(self, ctx: HookContext) -> Optional[HookResult]:
        with self.tracer.span("agent.error", error=ctx.error or "", tool=ctx.tool or ""):
            pass
        return None
