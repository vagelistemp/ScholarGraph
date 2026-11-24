"""Small utilities shared by the deep-learning phases: device choice, seeding, experiment tracking."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def pick_device(prefer: Optional[str] = None) -> str:
    """cuda (NVIDIA / Colab / Kaggle) -> mps (Apple Silicon Mac) -> cpu. ``prefer`` overrides the automatic choice."""
    import torch
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Tracker:
    """Experiment tracking. Uses MLflow when it is installed (``pip install .[track]``), otherwise appends to ``runs/<name>.jsonl``.

    The interface is the small subset every experiment needs: parameters once, metrics over time, and the final summary.
    """

    def __init__(self, experiment: str, run_name: str, root: Path = Path("runs"), use_mlflow: Optional[bool] = None):
        self.experiment, self.run_name = experiment, run_name
        self.path = Path(root) / f"{experiment}-{run_name}.jsonl"
        self._mlflow = None
        if use_mlflow is not False:
            try:
                import mlflow
                mlflow.set_experiment(experiment)
                mlflow.start_run(run_name=run_name)
                self._mlflow = mlflow
            except Exception:                                   # not installed or no tracking backend: fall back to the file
                self._mlflow = None
        if self._mlflow is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, kind: str, payload: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), "kind": kind, **payload}) + "\n")

    def params(self, **p: Any) -> None:
        if self._mlflow:
            self._mlflow.log_params({k: str(v) for k, v in p.items()})
        else:
            self._write("params", p)

    def metrics(self, step: int, **m: float) -> None:
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in m.items()}, step=step)
        else:
            self._write("metrics", {"step": step, **{k: float(v) for k, v in m.items()}})

    def finish(self, **summary: Any) -> None:
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})
            self._mlflow.end_run()
        else:
            self._write("summary", summary)
