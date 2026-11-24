"""Trajectory evaluation: judging *how* an agent reached its answer, not only whether the answer was right.

Two agents can both answer correctly, one in two tool calls and one in nine calls with three errors. The second is slower, costlier and
fragile. Trajectory metrics make that visible:

* **precision**   share of the agent's tool calls that are of a kind useful for this question type;
* **efficiency**  optimal number of calls divided by the actual number (1.0 = no wasted call);
* **order_ok**    whether the optimal tool sequence occurs, in order, inside the agent's sequence;
* **error_steps** how many steps failed (invalid JSON, unknown tool, bad arguments, blocked...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..agent.agent import AgentResult
from .questions import EXPECTED_TOOLS, OPTIMAL_TOOLS, Question


@dataclass
class TrajectoryScore:
    precision: float
    efficiency: float
    order_ok: bool
    error_steps: int
    n_calls: int


def is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    it = iter(haystack)
    return all(any(x == y for y in it) for x in needle)


def score_trajectory(q: Question, res: AgentResult) -> TrajectoryScore:
    calls: List[str] = [s.tool for s in res.steps if s.tool]
    optimal = OPTIMAL_TOOLS[q.qtype]
    useful = set(optimal) | EXPECTED_TOOLS[q.qtype]
    precision = sum(c in useful for c in calls) / len(calls) if calls else 0.0
    efficiency = min(1.0, len(optimal) / len(calls)) if calls else 0.0
    return TrajectoryScore(precision, efficiency, is_subsequence(optimal, calls), sum(1 for s in res.steps if s.error), len(calls))
