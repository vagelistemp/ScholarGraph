"""Running the agent on the generated questions and scoring it.

Metrics per question: **exact set F1** between the agent's ``items`` and the graph's answer (numbers and single answers are the one-item
special case), whether the agent called a tool of the expected kind (**tool_ok**), the number of steps, latency and token usage. The
report aggregates by question type, so a weak spot (say, ``co_cited`` questions) is visible instead of averaged away.

An optional LLM judge (``judge_faithfulness``) scores free-text RAG answers for support by their sources; it is the costly check that
complements the lexical groundedness of ``rag/guardrails.py``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..agent.agent import Agent, AgentResult
from ..llm import LLM, parse_json_object
from .questions import Question, norm


def set_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    p, g = {norm(x) for x in pred}, set(gold)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


@dataclass
class Row:
    qid: str
    qtype: str
    f1: float
    exact: bool
    tool_ok: bool
    steps: int
    finished: bool
    seconds: float
    prompt_tokens: int
    completion_tokens: int


def score(q: Question, res: AgentResult) -> Row:
    f1 = set_f1(res.items, q.answer)
    used = {s.tool for s in res.steps if s.tool}
    return Row(q.qid, q.qtype, f1, f1 == 1.0, bool(used & q.expected_tools), len(res.steps), res.finished, res.seconds,
               res.usage.get("prompt_tokens", 0), res.usage.get("completion_tokens", 0))


def run_agent_eval(agent: Agent, questions: Sequence[Question], log: Callable[[str], None] = print) -> Dict:
    rows: List[Row] = []
    for i, q in enumerate(questions, 1):
        try:
            res = agent.run(q.text)
        except Exception as e:                                   # a crashed run is a failed question, not a failed evaluation
            log(f"  {q.qid}: agent crashed ({type(e).__name__}: {e})")
            res = AgentResult(q.text, "", [], [], False, 0.0, {})
        rows.append(score(q, res))
        if i % 10 == 0:
            log(f"  {i}/{len(questions)} questions, running exact-match {np.mean([r.exact for r in rows]):.2f}")
    by_type: Dict[str, Dict[str, float]] = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r.qtype].append(r)
    for t, rs in sorted(groups.items()):
        by_type[t] = {"n": len(rs), "exact": float(np.mean([r.exact for r in rs])), "f1": float(np.mean([r.f1 for r in rs])),
                      "tool_ok": float(np.mean([r.tool_ok for r in rs])), "mean_steps": float(np.mean([r.steps for r in rs]))}
    overall = {"n": len(rows), "exact": float(np.mean([r.exact for r in rows])), "f1": float(np.mean([r.f1 for r in rows])),
               "tool_ok": float(np.mean([r.tool_ok for r in rows])), "finished": float(np.mean([r.finished for r in rows])),
               "mean_steps": float(np.mean([r.steps for r in rows])), "mean_seconds": float(np.mean([r.seconds for r in rows])),
               "prompt_tokens": int(sum(r.prompt_tokens for r in rows)), "completion_tokens": int(sum(r.completion_tokens for r in rows))}
    return {"overall": overall, "by_type": by_type, "rows": [asdict(r) for r in rows]}


def save_report(report: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")


JUDGE_SYSTEM = (
    "You are a strict fact-checker. You are given SOURCES and an ANSWER. Decide what fraction of the ANSWER's factual claims is directly "
    'supported by the SOURCES. Reply with one JSON object: {"score": <number between 0 and 1>, "unsupported": ["claim", ...]}.'
)


def judge_faithfulness(llm: LLM, answer: str, sources: Sequence[str]) -> Dict:
    """LLM-as-judge faithfulness in [0, 1]. Returns ``{"score": nan}`` when the judge's reply cannot be parsed."""
    prompt = "SOURCES:\n" + "\n".join(f"[P{i}] {s[:700]}" for i, s in enumerate(sources, 1)) + f"\n\nANSWER:\n{answer}"
    r = llm.chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], temperature=0.0, max_tokens=250, json_mode=True)
    d = parse_json_object(r.text) or {}
    try:
        return {"score": min(1.0, max(0.0, float(d["score"]))), "unsupported": list(d.get("unsupported", []))}
    except (KeyError, TypeError, ValueError):
        return {"score": float("nan"), "unsupported": []}
