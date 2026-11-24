"""The gold-table contract: names, columns and keys of the curated tables that phases 2-5 read.

Phase 1 writes these tables as Parquet files (``data/gold/<name>.parquet``). Every later phase reads them with pandas / pyarrow
and therefore needs no Spark and no Java. ``validate_gold_tables`` checks a set of tables against this contract.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import pandas as pd

# column -> pandas dtype kind: "s" string, "i" integer, "f" float
GOLD: Dict[str, Dict] = {
    "papers": {
        "key": ["paper_id"],
        "columns": {
            "paper_id": "s", "title": "s", "abstract": "s", "year": "i", "doi": "s", "type": "s", "language": "s",
            "cited_by_count": "i", "venue_id": "s", "venue_name": "s", "primary_topic_id": "s", "primary_topic_name": "s",
            "n_authors": "i", "n_references": "i",
        },
    },
    "authors": {"key": ["author_id"], "columns": {"author_id": "s", "name": "s", "n_papers": "i", "first_year": "i", "last_year": "i"}},
    "authorships": {
        "key": ["paper_id", "author_id"],
        "columns": {"paper_id": "s", "author_id": "s", "position": "i", "institution_id": "s", "institution_name": "s", "country": "s"},
    },
    "institutions": {"key": ["institution_id"], "columns": {"institution_id": "s", "name": "s", "country": "s"}},
    "citations": {
        "key": ["citing_id", "cited_id"],
        "columns": {"citing_id": "s", "cited_id": "s", "citing_year": "i", "cited_year": "i"},
    },
    "topics": {"key": ["topic_id"], "columns": {"topic_id": "s", "name": "s", "subfield": "s", "field": "s"}},
    "paper_topics": {"key": ["paper_id", "topic_id"], "columns": {"paper_id": "s", "topic_id": "s", "score": "f"}},
    "venues": {"key": ["venue_id"], "columns": {"venue_id": "s", "name": "s", "type": "s"}},
    "keywords": {"key": ["paper_id", "keyword"], "columns": {"paper_id": "s", "keyword": "s", "score": "f"}},
}

# Written by Phase 2 (not part of the Phase 1 gold set)
CONCEPTS_TABLE = {
    "key": ["paper_id", "concept"],
    "columns": {"paper_id": "s", "concept": "s", "kind": "s", "source": "s"},
}


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "s":
        return series.map(lambda v: v is None or isinstance(v, str) or (isinstance(v, float) and v != v)).all()
    if kind == "i":
        return pd.api.types.is_integer_dtype(series) or (pd.api.types.is_float_dtype(series) and series.dropna().mod(1).eq(0).all())
    return pd.api.types.is_numeric_dtype(series)


def validate_gold_tables(tables: Mapping[str, pd.DataFrame]) -> List[str]:
    """Return a list of contract violations (empty = the tables satisfy the contract)."""
    problems: List[str] = []
    for name, spec in GOLD.items():
        if name not in tables:
            problems.append(f"missing table: {name}")
            continue
        df = tables[name]
        missing = [c for c in spec["columns"] if c not in df.columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
            continue
        for col, kind in spec["columns"].items():
            if not _kind_ok(df[col], kind):
                problems.append(f"{name}.{col}: expected kind '{kind}', got {df[col].dtype}")
        if df.duplicated(spec["key"]).any():
            problems.append(f"{name}: duplicate keys on {spec['key']}")
        for k in spec["key"]:
            if df[k].isna().any():
                problems.append(f"{name}.{k}: null in key column")
    return problems
