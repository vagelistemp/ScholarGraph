"""Reading and writing the gold tables (plain Parquet files, one per table)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from ..schemas import GOLD


def write_gold(tables: Dict[str, pd.DataFrame], gold_dir: Path) -> None:
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(gold_dir / f"{name}.parquet", index=False)


def read_gold(gold_dir: Path, names: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    gold_dir = Path(gold_dir)
    names = list(names) if names is not None else list(GOLD)
    return {n: pd.read_parquet(gold_dir / f"{n}.parquet") for n in names}
