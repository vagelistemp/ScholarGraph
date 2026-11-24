"""Paths and settings shared by every phase.

The data folder is set with the ``SCHOLARGRAPH_DATA`` environment variable (default: ``./data``). Its layout is the *data contract*
between the phases:

    data/
      bronze/works/page_00001.jsonl.gz ...   Phase 1: raw API pages, exactly as received (selected fields only)
      bronze/_state.json                     Phase 1: cursor checkpoint, so downloading can resume
      silver/<table>/                        Phase 1: parsed, flattened tables (Delta Lake when Spark is used)
      gold/<table>.parquet                   Phase 1: curated tables, the ONLY input of phases 2-5 (see schemas.py)
      graph/                                 Phase 2: node / edge exports for Neo4j, concept tables, caches
      embeddings/                            Phases 3-4: vector matrices
      models/                                Phases 3-4: trained weights and metrics
      eval/                                  Phase 5: question sets and reports
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    return Path(os.environ.get("SCHOLARGRAPH_DATA", "data")).resolve()


@dataclass
class Paths:
    root: Path = field(default_factory=default_data_dir)

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def graph(self) -> Path:
        return self.root / "graph"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    def ensure(self) -> "Paths":
        for p in (self.bronze, self.silver, self.gold, self.graph, self.embeddings, self.models, self.eval):
            p.mkdir(parents=True, exist_ok=True)
        return self


# The research slice downloaded by Phase 1 (change with command-line options, not by editing code).
SLICE = {
    "subfield_id": 1702,            # OpenAlex subfield "Artificial Intelligence"
    "year_from": 2015,
    "year_to": 2025,
    "types": ("article", "preprint"),
    "max_works": 300_000,           # about 1-2 GB of compressed raw pages
}

# Time split used by every model (see Phase 3): train on citations made up to TRAIN_END, validate on VAL_YEAR, test afterwards.
TRAIN_END = 2022
VAL_YEAR = 2023
