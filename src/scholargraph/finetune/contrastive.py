"""Contrastive fine-tuning of a text encoder with in-batch negatives (the InfoNCE loss).

For a batch of B (anchor, positive) pairs the encoder produces anchor vectors ``a_i`` and positive vectors ``p_j``. The similarity
matrix ``S = a @ p.T / temperature`` should be large on the diagonal (own positive) and small elsewhere (the other papers' positives act
as free negatives): the loss is the cross-entropy of every row against its diagonal entry. Hard negatives, if present, are appended as
extra columns. With ``symmetric=True`` the loss is also applied to the transposed matrix (positive -> its anchor).

False negatives: two anchors in a batch may cite the *same* paper, so one anchor's positive is another's "negative". Those cells are
masked out of the softmax (``duplicate_mask``).

The training loop is plain PyTorch (no Trainer API), so it works on CUDA (Colab / Kaggle), Apple Silicon (mps) and CPU alike.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..common import pick_device, set_seed
from .pairs import PairSet


@dataclass
class ContrastiveConfig:
    epochs: int = 1
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    temperature: float = 0.05
    warmup_frac: float = 0.1
    symmetric: bool = True
    use_hard_negatives: bool = True
    max_steps: Optional[int] = None
    grad_clip: float = 1.0
    amp: bool = True                  # bfloat16 autocast on CUDA only
    seed: int = 0
    device: str = "auto"
    log_every: int = 50


def duplicate_mask(anchor_idx: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
    """True where a candidate (column) is the same paper as the row's own positive, but not on the diagonal: a false negative."""
    same = cand_idx[None, :] == cand_idx[: len(anchor_idx)][:, None]
    own = torch.zeros_like(same)
    own[torch.arange(len(anchor_idx)), torch.arange(len(anchor_idx))] = True
    return same & ~own


def info_nce(a: torch.Tensor, p: torch.Tensor, cand_idx: torch.Tensor, temperature: float, symmetric: bool = True,
             n: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``a`` (B, d) anchors; ``p`` (B, d) positives; ``n`` optional (B', d) hard negatives; ``cand_idx`` paper indices of all candidates."""
    a, p = F.normalize(a, dim=-1), F.normalize(p, dim=-1)
    cands = p if n is None else torch.cat([p, F.normalize(n, dim=-1)], dim=0)
    logits = a @ cands.T / temperature
    logits = logits.masked_fill(duplicate_mask(torch.arange(len(a), device=a.device), cand_idx.to(a.device)), float("-inf"))
    labels = torch.arange(len(a), device=a.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        back = (p @ a.T) / temperature
        dup = cand_idx[: len(a)][None, :] == cand_idx[: len(a)][:, None]
        back = back.masked_fill(dup & ~torch.eye(len(a), dtype=torch.bool, device=a.device), float("-inf"))
        loss = 0.5 * (loss + F.cross_entropy(back, labels))
    return loss


def train_contrastive(encoder, texts: Sequence[str], pairs: PairSet, cfg: Optional[ContrastiveConfig] = None,
                      log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    cfg = cfg or ContrastiveConfig()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    encoder.to(device)
    rng = np.random.default_rng(cfg.seed)
    steps_per_epoch = max(1, len(pairs) // cfg.batch_size)
    total = min(cfg.epochs * steps_per_epoch, cfg.max_steps or 10 ** 12)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    use_amp = cfg.amp and device == "cuda"
    history: List[Dict[str, float]] = []
    step, t0 = 0, time.time()
    encoder.train()
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(pairs))
        for i in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            if step >= total:
                break
            b = order[i:i + cfg.batch_size]
            anchors = [texts[j] for j in pairs.anchor[b]]
            pos_idx = pairs.positive[b]
            positives = [texts[j] for j in pos_idx]
            hard_idx = pairs.hard_negative[b]
            has_hard = cfg.use_hard_negatives and (hard_idx >= 0).any()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                a = encoder(anchors)
                p = encoder(positives)
                n = encoder([texts[j] for j in hard_idx[hard_idx >= 0]]) if has_hard else None
                cand = np.concatenate([pos_idx, hard_idx[hard_idx >= 0]]) if has_hard else pos_idx
                loss = info_nce(a.float(), p.float(), torch.as_tensor(cand), cfg.temperature, cfg.symmetric, None if n is None else n.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            history.append({"step": step, "loss": float(loss), "lr": sched.get_last_lr()[0]})
            if step % cfg.log_every == 0 or step == 1:
                log(f"step {step}/{total}  loss {float(loss):.4f}  ({time.time() - t0:.0f}s)")
    encoder.eval()
    return history
