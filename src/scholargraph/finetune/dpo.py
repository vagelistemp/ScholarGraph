"""Learning from preferences: Direct Preference Optimization (DPO) and distillation from agent traces.

After supervised fine-tuning teaches *what a good answer looks like*, preference tuning teaches *which of two answers is better*. For a
prompt x with a preferred reply y_w and a rejected reply y_l, DPO minimises

    loss = - log sigmoid( beta * [ (log pi(y_w|x) - log pi(y_l|x))  -  (log pi_ref(y_w|x) - log pi_ref(y_l|x)) ] )

where ``pi`` is the model being trained and ``pi_ref`` the frozen starting model. The model is rewarded for raising the probability of the
preferred reply *relative to the reference*; ``beta`` limits how far it may drift. No separate reward model and no reinforcement-learning
loop are needed, which makes it practical on a laptop-sized model with LoRA (the reference log-probabilities are obtained for free by
switching the adapter off).

Where do preferences come from here? From **agent traces**: run the agent on graph-derived questions with answers known exactly; at the first
step where a successful run and a failed run of the *same* question diverge, the successful step is "chosen" and the failed step is "rejected".
``build_sft_from_traces`` uses only the successful runs as supervised examples (distillation of good behaviour into a small model).
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..agent.agent import AgentResult
from ..common import pick_device, set_seed
from .lora_extractor import _encode


def render_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """A plain-text rendering of a chat prefix (the same format is used in training and inference)."""
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n".join(f"### {names.get(m['role'], m['role'])}:\n{m['content']}" for m in messages) + "\n### Assistant:\n"


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


def trajectory_pairs(chosen: AgentResult, rejected: AgentResult) -> Optional[PreferencePair]:
    """The first assistant turn where two runs of the same question differ gives one (prompt, chosen, rejected) example."""
    a = [(i, m["content"]) for i, m in enumerate(chosen.messages) if m["role"] == "assistant"]
    b = [(i, m["content"]) for i, m in enumerate(rejected.messages) if m["role"] == "assistant"]
    for (i, ca), (j, cb) in zip(a, b):
        if chosen.messages[:i] != rejected.messages[:j]:
            return None                                              # the contexts already diverged earlier (different observations): skip
        if ca != cb:
            return PreferencePair(render_prompt(chosen.messages[:i]), ca, cb)
    return None


def build_preference_pairs(runs: Dict[str, List[Tuple[AgentResult, bool]]]) -> List[PreferencePair]:
    """``runs``: question id -> [(agent result, was exactly correct), ...] from several attempts (other seeds, models or prompts)."""
    pairs: List[PreferencePair] = []
    for qid, attempts in runs.items():
        good = [r for r, ok in attempts if ok]
        bad = [r for r, ok in attempts if not ok]
        if good and bad:
            best = min(good, key=lambda r: len(r.steps))                 # among correct runs prefer the most efficient
            for b in bad:
                p = trajectory_pairs(best, b)
                if p:
                    pairs.append(p)
    return pairs


def build_sft_from_traces(runs: Dict[str, List[Tuple[AgentResult, bool]]]) -> List[Tuple[str, str]]:
    """(prompt, target) examples from every assistant turn of every *correct* run: distilling good agent behaviour."""
    out: List[Tuple[str, str]] = []
    for attempts in runs.values():
        for res, ok in attempts:
            if not ok:
                continue
            for i, m in enumerate(res.messages):
                if m["role"] == "assistant":
                    out.append((render_prompt(res.messages[:i]), m["content"]))
    return out


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1):
    """Returns (loss, reward accuracy, mean margin). All inputs are summed log-probabilities of whole replies, shape (batch,)."""
    margin = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    loss = -F.logsigmoid(margin).mean()
    return loss, (margin > 0).float().mean(), margin.mean()


def sequence_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probabilities of the label tokens (positions with label -100 are ignored), one number per sequence."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    tgt = labels[:, 1:]
    mask = tgt != -100
    logp = torch.gather(F.log_softmax(logits.float(), dim=-1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def _batch(tok, prompts: Sequence[str], replies: Sequence[str], max_len: int, device: str):
    ids_l, lab_l = [], []
    for p, r in zip(prompts, replies):
        pi, ri = _encode(tok, p), _encode(tok, r) + [tok.eos_token_id]
        pi = pi[-(max_len - len(ri)):] if len(pi) + len(ri) > max_len else pi
        ids_l.append(pi + ri)
        lab_l.append([-100] * len(pi) + ri)
    w = max(len(x) for x in ids_l)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(ids_l), w), pad, dtype=torch.long)
    lab = torch.full((len(ids_l), w), -100, dtype=torch.long)
    att = torch.zeros((len(ids_l), w), dtype=torch.long)
    for i, (a, b) in enumerate(zip(ids_l, lab_l)):
        ids[i, :len(a)], lab[i, :len(b)], att[i, :len(a)] = torch.tensor(a), torch.tensor(b), 1
    return ids.to(device), att.to(device), lab.to(device)


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 5e-5
    epochs: int = 1
    batch_size: int = 4
    max_len: int = 768
    seed: int = 0
    device: str = "auto"
    log_every: int = 10


def train_dpo(model, tok, pairs: Sequence[PreferencePair], cfg: Optional[DPOConfig] = None, log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    """``model`` must be a PEFT (LoRA) model: the reference model is the same network with the adapter switched off."""
    cfg = cfg or DPOConfig()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    model.to(device).eval()      # dropout off: the policy and the reference must see the same network, otherwise random masks add noise to the margin
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    hist: List[Dict[str, float]] = []
    step, t0 = 0, time.time()
    for _ in range(cfg.epochs):
        order = rng.permutation(len(pairs))
        for i in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            b = [pairs[j] for j in order[i:i + cfg.batch_size]]
            ids_c, att_c, lab_c = _batch(tok, [p.prompt for p in b], [p.chosen for p in b], cfg.max_len, device)
            ids_r, att_r, lab_r = _batch(tok, [p.prompt for p in b], [p.rejected for p in b], cfg.max_len, device)
            with torch.no_grad():
                with model.disable_adapter():                                   # reference model = the base model without the adapter
                    ref_c, ref_r = (sequence_logprob(model, ids_c, att_c, lab_c), sequence_logprob(model, ids_r, att_r, lab_r))
            pol_c, pol_r = sequence_logprob(model, ids_c, att_c, lab_c), sequence_logprob(model, ids_r, att_r, lab_r)
            loss, acc, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, cfg.beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            hist.append({"step": step, "loss": float(loss), "reward_accuracy": float(acc), "margin": float(margin)})
            if step % cfg.log_every == 0 or step == 1:
                log(f"step {step}  loss {float(loss):.4f}  reward acc {float(acc):.2f}  margin {float(margin):.3f}  ({time.time() - t0:.0f}s)")
    model.eval()
    return hist
