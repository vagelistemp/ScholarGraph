"""Training and evaluation of the citation GNN."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from ..common import Tracker, pick_device, set_seed
from .data import CitationData, EvalSet, hide_sources, make_eval_set, sample_negatives
from .metrics import all_metrics
from .model import Adjacency, CitationGNN, count_parameters, node_features


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-3
    weight_decay: float = 1e-5
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    hide_frac: float = 0.2            # share of training papers hidden each epoch (their references are the supervision targets)
    neg_per_pos: int = 1
    hard_frac: float = 0.5            # share of training negatives drawn from the source paper's own topic
    batch_pairs: int = 4096           # supervision pairs per optimizer step
    steps_per_epoch: int = 30         # optimizer steps per epoch (each re-encodes the graph); one giant batch per epoch means far too few updates
    patience: int = 20                # validation MRR can dip before a long climb (seen on real data); a short patience stops far too early
    max_eval_pairs: int = 20_000
    k_neg: int = 99
    seed: int = 0
    device: str = "auto"
    select_metric: str = "mrr"


@dataclass
class TrainResult:
    model: CitationGNN
    history: List[Dict[str, float]]
    best_epoch: int
    val: Dict[str, float]
    test: Dict[str, float]
    seconds: float
    config: TrainConfig


@torch.no_grad()
def score_eval_set(model: CitationGNN, data: CitationData, adj: Adjacency, ev: EvalSet, device: str, chunk: int = 400_000):
    model.eval()
    x = node_features(data, adj, device)
    z = model.encode(x, adj)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    pu, pv = t(ev.pos[:, 0]), t(ev.pos[:, 1])
    pos = model.score(z, x, pu, pv).float().cpu().numpy()
    k = ev.neg.shape[1]
    nu = t(np.repeat(ev.pos[:, 0], k))
    nv = t(ev.neg.reshape(-1))
    neg = np.concatenate([model.score(z, x, nu[i:i + chunk], nv[i:i + chunk]).float().cpu().numpy() for i in range(0, len(nu), chunk)])
    return pos, neg.reshape(len(ev.pos), k)


def evaluate(model, data, adj, ev, device) -> Dict[str, float]:
    pos, neg = score_eval_set(model, data, adj, ev, device)
    return all_metrics(pos, neg)


def train_gnn(data: CitationData, cfg: Optional[TrainConfig] = None, log: Callable[[str], None] = print,
              tracker: Optional[Tracker] = None) -> TrainResult:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    split = data.split()
    if min(len(split["train"]), len(split["val"]), len(split["test"])) == 0:
        raise ValueError("the temporal split has an empty part; check train_end / val_year against the data's years")
    forbidden = data.all_pairs_codes()
    val_set = make_eval_set(data, split["val"], cfg.k_neg, cfg.max_eval_pairs, seed=cfg.seed + 1)
    test_set = make_eval_set(data, split["test"], cfg.k_neg, cfg.max_eval_pairs, seed=cfg.seed + 2)
    val_adj = Adjacency(split["train"], data.n, device)                                   # validate on the past graph only
    test_adj = Adjacency(np.vstack([split["train"], split["val"]]), data.n, device)       # test sees train + val edges

    in_dim = data.text.shape[1] + 3
    model = CitationGNN(in_dim, data.text.shape[1], cfg.hidden, cfg.layers, cfg.dropout, torch.as_tensor(data.topic)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    log(f"device {device}; {count_parameters(model):,} parameters; train/val/test edges "
        f"{len(split['train']):,}/{len(split['val']):,}/{len(split['test']):,}")
    if tracker:
        tracker.params(**{**asdict(cfg), "device": device})            # the resolved device replaces the "auto" in the config

    history, best, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        msg, sup = hide_sources(split["train"], cfg.hide_frac, rng)                      # edge-out-of-fold, on whole papers
        adj = Adjacency(msg, data.n, device)
        x = node_features(data, adj, device)
        order = rng.permutation(len(sup))
        losses = []
        for i in range(0, min(len(order), cfg.batch_pairs * cfg.steps_per_epoch), cfg.batch_pairs):
            pos = sup[order[i:i + cfg.batch_pairs]]
            neg_v = sample_negatives(data, pos[:, 0], cfg.neg_per_pos, rng, forbidden, cfg.hard_frac)
            u = torch.as_tensor(np.concatenate([pos[:, 0], np.repeat(pos[:, 0], cfg.neg_per_pos)]), device=device)
            v = torch.as_tensor(np.concatenate([pos[:, 1], neg_v.reshape(-1)]), device=device)
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(pos) * cfg.neg_per_pos)]).to(device)
            z = model.encode(x, adj)
            loss = bce(model.score(z, x, u, v), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        val = evaluate(model, data, val_adj, val_set, device)
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        if tracker:
            tracker.metrics(epoch, **{k: v for k, v in rec.items() if k != "epoch"})
        log(f"epoch {epoch:3d}  loss {rec['loss']:.4f}  val MRR {val['mrr']:.4f}  hits@10 {val['hits@10']:.4f}  AUC {val['auc']:.4f}")
        if val[cfg.select_metric] > best:
            best, best_epoch, bad = val[cfg.select_metric], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    val_final = evaluate(model, data, val_adj, val_set, device)
    test_final = evaluate(model, data, test_adj, test_set, device)
    if tracker:
        tracker.finish(**{f"test_{k}": v for k, v in test_final.items()})
    return TrainResult(model, history, best_epoch, val_final, test_final, time.time() - t0, cfg)
