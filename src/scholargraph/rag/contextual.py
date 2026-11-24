"""Contextual retrieval: index each paper together with its context.

An abstract on its own can be ambiguous ("we propose a new attention mechanism"): which field, which year, which tasks? Prefixing the
indexed text with *where the paper sits* (year, venue, topic, the concepts extracted in Phase 2) gives both BM25 and the embedding model
information that is otherwise only in the graph. This is the idea of "contextual retrieval": enrich the chunk with its context before
indexing, not after retrieval.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import pandas as pd


def build_contextual_texts(papers: pd.DataFrame, concepts: Optional[pd.DataFrame] = None, max_concepts: int = 6) -> List[str]:
    by_paper: Dict[str, List[str]] = {}
    if concepts is not None and len(concepts):
        for pid, g in concepts.groupby("paper_id"):
            by_paper[pid] = list(g["concept"])[:max_concepts]
    out = []
    for r in papers.itertuples(index=False):
        ctx = [f"year {r.year}"]
        if getattr(r, "venue_name", None):
            ctx.append(f"venue {r.venue_name}")
        if getattr(r, "primary_topic_name", None):
            ctx.append(f"topic {r.primary_topic_name}")
        c = by_paper.get(r.paper_id)
        if c:
            ctx.append("concepts " + ", ".join(c))
        out.append(f"{r.title}. [{'; '.join(ctx)}] {r.abstract}")
    return out
