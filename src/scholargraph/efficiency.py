"""Making the system fast and cheap enough for a 16 GB laptop: caching, cascades, context budgets, memory estimates, benchmarks.

* **CachedLLM**       a disk cache of model replies for deterministic calls (temperature 0): repeated evaluations and repeated questions cost
                      nothing, and a regression run is reproducible.
* **CascadeLLM**      try a small, fast model first and escalate to a larger one only when the small model's reply is unusable (invalid JSON,
                      schema violation...). Most requests are easy; the expensive model is paid for only when needed.
* **truncate_messages** keep a conversation inside a token budget by dropping the oldest middle turns, never the system prompt or the last message.
* **estimate_memory_gb** how much RAM a model needs (weights + KV cache + overhead), to plan what fits next to the rest of the system.
* **MLXLLM**          Apple's MLX backend for Apple Silicon (needs ``pip install mlx-lm`` on a Mac; not exercised by the tests).
* **benchmark_llm**   latency percentiles and tokens per second for any model.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .llm import LLM, LLMResult, Message, parse_json_object
from .structured import validate


class CachedLLM:
    def __init__(self, inner: LLM, path: Optional[Path] = None):
        self.inner, self.name, self.path = inner, inner.name, Path(path) if path else None
        self._mem: Dict[str, Dict] = {}
        self.stats = {"hits": 0, "misses": 0, "bypassed": 0}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._mem[rec["k"]] = rec

    def _key(self, messages, temperature, max_tokens, json_mode, schema) -> str:
        blob = json.dumps({"m": self.inner.name, "msgs": list(messages), "t": temperature, "n": max_tokens, "j": json_mode, "s": schema}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 512, json_mode: bool = False, schema: Optional[dict] = None) -> LLMResult:
        if temperature > 0:                                           # sampled replies must differ from call to call: never cache them
            self.stats["bypassed"] += 1
            return self.inner.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema)
        key = self._key(messages, temperature, max_tokens, json_mode, schema)
        if key in self._mem:
            self.stats["hits"] += 1
            r = self._mem[key]
            return LLMResult(r["text"], r["p"], r["c"], 0.0)
        self.stats["misses"] += 1
        res = self.inner.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema)
        rec = {"k": key, "text": res.text, "p": res.prompt_tokens, "c": res.completion_tokens}
        self._mem[key] = rec
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return res


def json_acceptor(schema: Optional[dict] = None) -> Callable[[LLMResult], bool]:
    """Accept a reply only if it is a JSON object (that validates against ``schema`` when given)."""
    def accept(r: LLMResult) -> bool:
        obj = parse_json_object(r.text)
        return obj is not None and (schema is None or not validate(obj, schema))
    return accept


class CascadeLLM:
    def __init__(self, small: LLM, large: LLM, accept: Callable[[LLMResult], bool]):
        self.small, self.large, self.accept, self.name = small, large, accept, f"cascade:{small.name}->{large.name}"
        self.stats = {"small_ok": 0, "escalated": 0}

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        r = self.small.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema)
        if self.accept(r):
            self.stats["small_ok"] += 1
            return r
        self.stats["escalated"] += 1
        big = self.large.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema)
        return LLMResult(big.text, big.prompt_tokens + r.prompt_tokens, big.completion_tokens + r.completion_tokens, big.seconds + r.seconds)   # both calls were paid for


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)                                      # a rule of thumb for English: about four characters per token


def truncate_messages(messages: Sequence[Message], max_tokens: int) -> List[Message]:
    """Drop the oldest messages after the system prompt until the estimate fits; the system prompt and the last message are never dropped."""
    msgs = list(messages)
    cost = lambda ms: sum(count_tokens_approx(m["content"]) + 4 for m in ms)  # noqa: E731
    while cost(msgs) > max_tokens and len(msgs) > 2:
        drop = 1 if msgs[0]["role"] == "system" else 0
        if drop >= len(msgs) - 1:
            break
        del msgs[drop]
    return msgs


# approximate architectures (public model families of these sizes); use your actual model's config for real planning
MODEL_PRESETS = {
    "3B-class": dict(params_b=3.2, n_layers=28, n_kv_heads=8, head_dim=128),
    "8B-class": dict(params_b=8.0, n_layers=32, n_kv_heads=8, head_dim=128),
    "14B-class": dict(params_b=14.0, n_layers=40, n_kv_heads=8, head_dim=128),
}


def estimate_memory_gb(params_b: float, bits: int, context_tokens: int, n_layers: int, n_kv_heads: int, head_dim: int, kv_bits: int = 16,
                       overhead_gb: float = 1.0) -> Dict[str, float]:
    """RAM needed = weights + KV cache + runtime overhead.

    weights  = parameters x bits / 8
    KV cache = 2 (keys and values) x layers x kv-heads x head-dim x context x bytes per value"""
    weights = params_b * 1e9 * bits / 8 / 1e9
    kv = 2 * n_layers * n_kv_heads * head_dim * context_tokens * kv_bits / 8 / 1e9
    return {"weights_gb": weights, "kv_cache_gb": kv, "overhead_gb": overhead_gb, "total_gb": weights + kv + overhead_gb}


def fits(total_gb: float, ram_gb: float = 16.0, reserve_for_os_and_app_gb: float = 6.0) -> bool:
    """Does a model fit next to macOS and the rest of the system (graph in memory, encoder, API)?"""
    return total_gb <= ram_gb - reserve_for_os_and_app_gb


class MLXLLM:
    """Apple Silicon backend (``pip install mlx-lm``). Untested here: it needs a Mac; the interface is identical to the other providers."""

    def __init__(self, model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        from mlx_lm import load
        self.model, self.tokenizer = load(model)
        self.name = f"mlx:{model}"

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        from mlx_lm import generate
        prompt = self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)
        return LLMResult(text, len(self.tokenizer.encode(prompt)), len(self.tokenizer.encode(text)), time.time() - t0)


def benchmark_llm(llm: LLM, prompts: Sequence[str], max_tokens: int = 128, repeats: int = 1) -> Dict[str, float]:
    lat, toks = [], 0
    for _ in range(repeats):
        for p in prompts:
            t0 = time.time()
            r = llm.chat([{"role": "user", "content": p}], max_tokens=max_tokens)
            lat.append(time.time() - t0)
            toks += r.completion_tokens
    total = sum(lat)
    return {"calls": len(lat), "p50_seconds": float(np.percentile(lat, 50)), "p95_seconds": float(np.percentile(lat, 95)),
            "tokens_per_second": toks / total if total > 0 else 0.0, "mean_seconds": float(np.mean(lat))}
