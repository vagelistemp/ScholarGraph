"""Collect and store agent traces: the raw material for regression analysis, DPO preference pairs and distillation."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..agent.agent import Agent, AgentResult, Step
from .harness import set_f1
from .questions import Question


def collect_traces(agent: Agent, questions: Sequence[Question]) -> Dict[str, List[Tuple[AgentResult, bool]]]:
    """Run the agent once per question; returns qid -> [(result, exactly correct)]. Call again (other seed / model) to add attempts."""
    out: Dict[str, List[Tuple[AgentResult, bool]]] = {}
    for q in questions:
        res = agent.run(q.text)
        out.setdefault(q.qid, []).append((res, set_f1(res.items, q.answer) == 1.0))
    return out


def merge_traces(*runs: Dict[str, List[Tuple[AgentResult, bool]]]) -> Dict[str, List[Tuple[AgentResult, bool]]]:
    merged: Dict[str, List[Tuple[AgentResult, bool]]] = {}
    for r in runs:
        for qid, attempts in r.items():
            merged.setdefault(qid, []).extend(attempts)
    return merged


def save_traces(runs: Dict[str, List[Tuple[AgentResult, bool]]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for qid, attempts in runs.items():
            for res, ok in attempts:
                fh.write(json.dumps({"qid": qid, "ok": ok, "result": asdict(res)}, ensure_ascii=False) + "\n")


def load_traces(path: Path) -> Dict[str, List[Tuple[AgentResult, bool]]]:
    out: Dict[str, List[Tuple[AgentResult, bool]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            d = rec["result"]
            d["steps"] = [Step(**s) for s in d["steps"]]
            out.setdefault(rec["qid"], []).append((AgentResult(**d), rec["ok"]))
    return out
