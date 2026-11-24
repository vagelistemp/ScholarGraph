"""Synthetic data helpers shared by the tests and the ``demo`` command (the full end-to-end demo is at the bottom)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .embeddings import TfidfSvdEncoder
from .ingest.synthetic import make_synthetic_works
from .ingest.transform import normalise_work
from .lake.pandas_pipeline import build_gold


def make_demo_gold(n_papers: int = 600, seed: int = 0) -> Dict[str, pd.DataFrame]:
    works = [w for w in (normalise_work(r) for r in make_synthetic_works(n_papers, seed)) if w]
    return build_gold(iter(works), 2015, 2025)


def make_demo_text(gold: Dict[str, pd.DataFrame], dim: int = 32) -> np.ndarray:
    papers = gold["papers"]
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    return TfidfSvdEncoder(dim=dim).fit(texts).encode(texts)
