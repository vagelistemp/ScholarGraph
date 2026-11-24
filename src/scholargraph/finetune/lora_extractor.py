"""LoRA fine-tuning of a small causal language model to extract concepts from abstracts.

Why: in Phase 2 a large prompt-based LLM extracts methods, tasks and datasets. That is accurate but slow and costly when run over
hundreds of thousands of abstracts. Here its outputs (and the rule-based lexicon) become *training labels* for a small model that
learns the same JSON-extraction behaviour (knowledge distillation by fine-tuning), which is then cheap to run at scale.

LoRA (low-rank adaptation) freezes the pre-trained weights W and learns a small update ``W + (alpha / r) * B A`` with ``A`` of shape
(r, d) and ``B`` of shape (d, r): a few million trainable parameters instead of hundreds of millions, so it fits on a free Colab GPU or
a Mac. Plain LoRA is used on purpose (no 4-bit QLoRA, which needs CUDA-only libraries); on a Mac, Apple's MLX library does the same
job (see the lecture notes).

The training loop is written out in plain PyTorch: tokenise ``prompt + target``, mask the prompt tokens in the labels (we only train the
model to produce the answer), pad, and minimise the next-token cross-entropy.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..common import pick_device, set_seed
from ..llm import parse_json_object

PROMPT = "### Abstract:\n{abstract}\n\n### Concepts (JSON):\n"


def format_target(methods: Sequence[str], tasks: Sequence[str], datasets: Sequence[str]) -> str:
    return json.dumps({"methods": sorted(methods), "tasks": sorted(tasks), "datasets": sorted(datasets)})


def build_sft_examples(papers: pd.DataFrame, concepts: pd.DataFrame, max_examples: Optional[int] = None) -> List[Tuple[str, str]]:
    """(prompt, target) pairs from the concept table of Phase 2; papers without any concept are kept (the model must learn to say none)."""
    by_paper = {pid: g for pid, g in concepts.groupby("paper_id")}
    out = []
    for pid, title, abstract in zip(papers["paper_id"], papers["title"], papers["abstract"]):
        g = by_paper.get(pid)
        pick = lambda kind: sorted(g[g["kind"] == kind]["concept"]) if g is not None else []  # noqa: E731
        out.append((PROMPT.format(abstract=f"{title}. {abstract}"), format_target(pick("method"), pick("task"), pick("dataset"))))
        if max_examples and len(out) >= max_examples:
            break
    return out


class ByteTokenizer:
    """A dependency-free tokenizer (one token per UTF-8 byte) used by the tests; real runs use the model's own Hugging Face tokenizer."""
    pad_token_id, eos_token_id = 0, 1
    vocab_size = 259

    def encode(self, text: str) -> List[int]:
        return [b + 3 for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i - 3 for i in ids if i >= 3).decode("utf-8", errors="ignore")


def _encode(tok, text: str) -> List[int]:
    return list(tok.encode(text)) if isinstance(tok, ByteTokenizer) else tok(text, add_special_tokens=False)["input_ids"]


def make_batch(tok, examples: Sequence[Tuple[str, str]], max_len: int, device: str):
    ids_list, lab_list = [], []
    for prompt, target in examples:
        p, t = _encode(tok, prompt), _encode(tok, target) + [tok.eos_token_id]
        p = p[-(max_len - len(t)):] if len(p) + len(t) > max_len else p             # keep the end of a long prompt, never cut the answer
        ids_list.append(p + t)
        lab_list.append([-100] * len(p) + t)                                         # loss only on the answer tokens
    width = max(len(x) for x in ids_list)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(ids_list), width), pad, dtype=torch.long)
    lab = torch.full((len(ids_list), width), -100, dtype=torch.long)
    att = torch.zeros((len(ids_list), width), dtype=torch.long)
    for i, (a, b) in enumerate(zip(ids_list, lab_list)):
        ids[i, :len(a)], lab[i, :len(b)], att[i, :len(a)] = torch.tensor(a), torch.tensor(b), 1
    return ids.to(device), att.to(device), lab.to(device)


@dataclass
class LoraConfigLite:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 8
    max_len: int = 640
    seed: int = 0
    device: str = "auto"
    log_every: int = 20
    max_steps: Optional[int] = None


def attach_lora(model, cfg: LoraConfigLite, fan_in_fan_out: bool = False):
    from peft import LoraConfig, get_peft_model
    pc = LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=list(cfg.target_modules),
                    task_type="CAUSAL_LM", fan_in_fan_out=fan_in_fan_out)
    return get_peft_model(model, pc)


def train_lora(model, tok, examples: Sequence[Tuple[str, str]], cfg: Optional[LoraConfigLite] = None,
               log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    """``model`` must already be a PEFT model (see ``attach_lora``). Only the LoRA parameters receive gradients."""
    cfg = cfg or LoraConfigLite()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    hist, step, t0 = [], 0, time.time()
    total = cfg.epochs * max(1, len(examples) // cfg.batch_size)
    for _ in range(cfg.epochs):
        order = rng.permutation(len(examples))
        for i in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            if cfg.max_steps and step >= cfg.max_steps:
                break
            ids, att, lab = make_batch(tok, [examples[j] for j in order[i:i + cfg.batch_size]], cfg.max_len, device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            hist.append({"step": step, "loss": float(out.loss)})
            if step % cfg.log_every == 0 or step == 1:
                log(f"step {step}/{total}  loss {float(out.loss):.4f}  ({time.time() - t0:.0f}s)")
    model.eval()
    return hist


@torch.no_grad()
def generate_concepts(model, tok, abstract: str, max_new_tokens: int = 160, device: Optional[str] = None) -> str:
    """Greedy decoding of the JSON answer for one abstract."""
    device = device or next(model.parameters()).device
    ids = torch.tensor([_encode(tok, PROMPT.format(abstract=abstract))], device=device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:].tolist())


def evaluate_extractor(model, tok, examples: Sequence[Tuple[str, str]], max_new_tokens: int = 160) -> Dict[str, float]:
    """JSON validity rate and micro-averaged F1 of the predicted concepts against the reference targets."""
    valid, tp, fp, fn = 0, 0, 0, 0
    for prompt, target in examples:
        abstract = prompt.split("### Abstract:\n", 1)[1].rsplit("\n\n### Concepts (JSON):\n", 1)[0]
        pred = parse_json_object(generate_concepts(model, tok, abstract, max_new_tokens)) or {}
        gold = json.loads(target)
        ok = isinstance(pred, dict) and all(isinstance(pred.get(k, []), list) for k in ("methods", "tasks", "datasets"))
        valid += int(bool(pred) and ok)
        for k in ("methods", "tasks", "datasets"):
            p = set(map(str, pred.get(k, []))) if ok else set()
            g = set(gold[k])
            tp, fp, fn = tp + len(p & g), fp + len(p - g), fn + len(g - p)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"json_valid": valid / max(len(examples), 1), "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def load_base_model(name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: Optional[str] = None):
    """Download a small instruct model and its tokenizer (about 1 GB; run this on Colab or the Mac, not on a slow laptop)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return model.to(device or pick_device()), tok
