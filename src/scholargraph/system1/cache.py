"""An answer cache with an exact layer and a semantic layer.

* **Exact layer**: the normalised question text (lower case, no punctuation, single spaces) is the key. Free and always safe.
* **Semantic layer**: a paraphrase ("who wrote X" versus "authors of X") is found by cosine similarity of embeddings. Semantic caches are
  dangerous when a small difference changes the right answer: "most cited paper of 2021" and "...of 2022" are almost identical as
  vectors. Two guards make the layer safe: a high similarity threshold, and a **signature check**: the numbers and the quoted strings of
  the two questions must be identical, otherwise the hit is rejected.

Only **verified** answers are stored (see ``verifier.py``), entries expire after a time-to-live and the cache is bounded (least recently
used entries are evicted). Personal data never enters the cache because the safety layer redacts it before the question gets here.
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def normalise(q: str) -> str:
    return " ".join(re.sub(r"[^\w\s\"']", " ", q.lower()).split())


def signature(q: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(all numbers, all quoted strings) of a question: two questions that differ here must never share an answer."""
    nums = tuple(sorted(re.findall(r"\d+", q)))
    quoted = tuple(sorted(s.strip().lower() for s in re.findall(r"\"([^\"]+)\"", q)))
    return nums, quoted


@dataclass
class CacheEntry:
    question: str
    answer: str
    items: List[str]
    created: float
    meta: Dict = field(default_factory=dict)
    hits: int = 0


@dataclass
class CacheHit:
    entry: CacheEntry
    kind: str            # "exact" | "semantic"
    score: float


class AnswerCache:
    def __init__(self, encoder=None, threshold: float = 0.93, ttl_seconds: float = 24 * 3600, max_entries: int = 5000,
                 clock: Callable[[], float] = time.time):
        self.encoder, self.threshold, self.ttl, self.max_entries, self.clock = encoder, threshold, ttl_seconds, max_entries, clock
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._vecs: Dict[str, np.ndarray] = {}
        self.stats = {"exact": 0, "semantic": 0, "miss": 0, "rejected_semantic": 0}

    def __len__(self) -> int:
        return len(self._entries)

    def _expire(self) -> None:
        now = self.clock()
        for k in [k for k, e in self._entries.items() if now - e.created > self.ttl]:
            self._drop(k)

    def _drop(self, key: str) -> None:
        self._entries.pop(key, None)
        self._vecs.pop(key, None)

    def put(self, question: str, answer: str, items: List[str], verified: bool, **meta) -> bool:
        if not verified:
            return False                                              # unverified answers are never cached
        key = normalise(question)
        self._entries[key] = CacheEntry(question, answer, list(items), self.clock(), dict(meta))
        self._entries.move_to_end(key)
        if self.encoder is not None:
            self._vecs[key] = np.asarray(self.encoder.encode([question])[0], dtype=np.float32)
        while len(self._entries) > self.max_entries:
            self._drop(next(iter(self._entries)))
        return True

    def get(self, question: str) -> Optional[CacheHit]:
        self._expire()
        key = normalise(question)
        e = self._entries.get(key)
        if e is not None:
            e.hits += 1
            self._entries.move_to_end(key)
            self.stats["exact"] += 1
            return CacheHit(e, "exact", 1.0)
        if self.encoder is not None and self._vecs:
            q = np.asarray(self.encoder.encode([question])[0], dtype=np.float32)
            keys = list(self._vecs)
            sims = np.stack([self._vecs[k] for k in keys]) @ q
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                cand = self._entries[keys[best]]
                if signature(cand.question) == signature(question):
                    cand.hits += 1
                    self.stats["semantic"] += 1
                    return CacheHit(cand, "semantic", float(sims[best]))
                self.stats["rejected_semantic"] += 1
        self.stats["miss"] += 1
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._vecs.clear()
