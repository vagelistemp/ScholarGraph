"""Regression testing for AI behaviour: did the last change make the system worse?

A change to a prompt, a model, a retriever or a hook can silently break behaviour that worked yesterday. The remedy is the same as in
ordinary software: keep a **golden set** of questions and a saved **baseline report**, re-run after every change and fail when quality
drops by more than a tolerance. ``compare_reports`` does the comparison; ``flipped`` lists the individual questions that used to be
exactly right and no longer are, which is usually the fastest way to find the cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Regression:
    scope: str            # "overall" or a question type
    metric: str
    baseline: float
    current: float

    def __str__(self) -> str:
        return f"{self.scope}.{self.metric}: {self.baseline:.3f} -> {self.current:.3f}"


@dataclass
class RegressionReport:
    regressions: List[Regression] = field(default_factory=list)
    flipped: List[str] = field(default_factory=list)          # question ids exact before, wrong now
    improved: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.regressions

    def summary(self) -> str:
        if self.ok:
            return f"no regression ({len(self.improved)} questions improved, {len(self.flipped)} flipped)"
        return "REGRESSION: " + "; ".join(str(r) for r in self.regressions) + (f"  [flipped: {', '.join(self.flipped[:8])}]" if self.flipped else "")


def compare_reports(baseline: Dict, current: Dict, tol_quality: float = 0.02, tol_cost: float = 0.25) -> RegressionReport:
    """Quality metrics (exact, f1, tool_ok) may drop by at most ``tol_quality`` (absolute); cost metrics (steps, p95 latency, tokens per
    question) may grow by at most ``tol_cost`` (relative)."""
    out = RegressionReport()
    for scope, b, c in [("overall", baseline["overall"], current["overall"])] + [(t, baseline["by_type"][t], current["by_type"].get(t, {}))
                                                                              for t in baseline["by_type"]]:
        for m in ("exact", "f1", "tool_ok"):
            if m in b and b[m] - c.get(m, 0.0) > tol_quality:
                out.regressions.append(Regression(scope, m, b[m], c.get(m, 0.0)))
        for m in ("mean_steps", "p95_seconds", "tokens_per_question"):
            if m in b and b[m] > 0 and c.get(m, 0.0) > b[m] * (1 + tol_cost):
                out.regressions.append(Regression(scope, m, b[m], c.get(m, 0.0)))
    was = {r["qid"]: r["exact"] for r in baseline.get("rows", [])}
    now = {r["qid"]: r["exact"] for r in current.get("rows", [])}
    out.flipped = sorted(q for q, e in was.items() if e and not now.get(q, False))
    out.improved = sorted(q for q, e in now.items() if e and not was.get(q, True))
    return out
