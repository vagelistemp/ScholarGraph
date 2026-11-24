"""Data-quality checks on the gold tables.

A pipeline that silently produces wrong tables is worse than one that fails. ``run_quality_checks`` returns a list of results
(name, passed, detail); ``tests`` and the ``quality`` command use it, and a failed *error* check should stop the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..schemas import validate_gold_tables


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "error"      # "error" stops the pipeline, "warning" is reported only


def run_quality_checks(t: Dict[str, pd.DataFrame]) -> List[Check]:
    out: List[Check] = []
    problems = validate_gold_tables(t)
    out.append(Check("contract", not problems, "; ".join(problems) or "all tables match schemas.GOLD"))
    if problems:
        return out
    papers, cit, ash = t["papers"], t["citations"], t["authorships"]
    ids = set(papers["paper_id"])

    ref_ok = cit["citing_id"].isin(ids).all() and cit["cited_id"].isin(ids).all()
    out.append(Check("citations reference known papers", bool(ref_ok), f"{len(cit):,} citations"))
    out.append(Check("no self-citations", bool((cit["citing_id"] != cit["cited_id"]).all()), ""))
    out.append(Check("no citation to the future", bool((cit["cited_year"] <= cit["citing_year"]).all()), ""))
    out.append(Check("authorships reference known papers and authors",
                     bool(ash["paper_id"].isin(ids).all() and ash["author_id"].isin(t["authors"]["author_id"]).all()), f"{len(ash):,} rows"))
    out.append(Check("abstracts are present", bool((papers["abstract"].str.len() >= 40).all()), ""))
    out.append(Check("years are plausible", bool(papers["year"].between(1900, 2100).all()), f"{papers['year'].min()}..{papers['year'].max()}"))
    cited = set(cit["cited_id"]) | set(cit["citing_id"])
    iso = 1 - len(cited & ids) / max(len(ids), 1)
    out.append(Check("isolated papers (no citation inside the slice)", iso < 0.5, f"{iso:.1%} of papers", severity="warning"))
    apy = papers["n_authors"].mean()
    out.append(Check("authors per paper is plausible", 0.5 < apy < 50, f"mean {apy:.2f}", severity="warning"))
    return out


def summarise(checks: List[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok  " if c.passed else ("FAIL" if c.severity == "error" else "warn")
        lines.append(f"[{mark}] {c.name}" + (f": {c.detail}" if c.detail else ""))
    return "\n".join(lines)


def has_errors(checks: List[Check]) -> bool:
    return any((not c.passed) and c.severity == "error" for c in checks)
