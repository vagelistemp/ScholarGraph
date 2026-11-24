"""Long-term memory: a store of short notes with provenance, trust, importance, recency and forgetting.

Memory types (borrowed from cognitive science, as used in agent design):

* **semantic**   stable facts about the user or the domain ("The user works on graph neural networks.")
* **episodic**   what happened ("Asked who wrote X; answer: A, B; checked against the graph."), expires after a while
* **procedural** how to do things well (not written automatically here; reserved for curated rules)

Retrieval score of an item for a query::

    score = cosine(query, item) * (0.5 + 0.5 * importance) * 0.5 ** (age_days / half_life) + 0.05 * log(1 + uses)

so a relevant, important, recent, often-used note wins; old unused notes fade and are evicted first when the store is full.
Items with trust ``unverified`` are never returned unless asked for explicitly.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..embeddings import BowHashEncoder
from .safety import screen_memory

TTL_DAYS = {"episodic": 30.0, "semantic": None, "procedural": None}
TRUST_ORDER = {"unverified": 0, "verified": 1, "user": 2}
PAPER_ID = re.compile(r"\bW\d{3,}\b")


@dataclass
class MemoryItem:
    id: str
    kind: str
    text: str
    created: float
    last_used: float
    importance: float = 0.5
    uses: int = 0
    trust: str = "verified"
    source: Dict = field(default_factory=dict)          # provenance: {"conversation": ..., "turn": ..., "route": ...}
    paper_ids: List[str] = field(default_factory=list)  # links into the knowledge graph


class MemoryStore:
    def __init__(self, encoder=None, clock: Callable[[], float] = time.time, max_items: int = 500, half_life_days: float = 30.0,
                 dedupe_threshold: float = 0.95):
        self.encoder = encoder or BowHashEncoder()
        self.clock, self.max_items, self.half_life, self.dedupe_threshold = clock, max_items, half_life_days, dedupe_threshold
        self.items: Dict[str, MemoryItem] = {}
        self._vec: Dict[str, np.ndarray] = {}
        self.rejected: List[str] = []                    # reasons for rejected writes (for audit and tests)

    def __len__(self) -> int:
        return len(self.items)

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self.encoder.encode([text])[0], dtype=np.float32)

    # ------------------------------------------------------------------------------------------ write
    def add(self, text: str, kind: str = "semantic", importance: float = 0.5, trust: str = "verified", source: Optional[Dict] = None,
            paper_ids: Optional[Sequence[str]] = None) -> Optional[MemoryItem]:
        """Store a note. Returns ``None`` when the safety screen rejects it. Near-duplicates are merged instead of stored twice."""
        scr = screen_memory(text)
        if not scr.ok:
            self.rejected.append(scr.reason)
            return None
        now = self.clock()
        vec = self._embed(scr.text)
        for iid, v in self._vec.items():                                   # merge near-duplicates
            if float(v @ vec) >= self.dedupe_threshold and self.items[iid].kind == kind:
                it = self.items[iid]
                it.importance, it.last_used, it.uses = max(it.importance, importance), now, it.uses + 1
                if TRUST_ORDER[trust] > TRUST_ORDER[it.trust]:
                    it.trust = trust
                return it
        item = MemoryItem(uuid.uuid4().hex[:10], kind, scr.text, now, now, float(importance), 0, trust, dict(source or {}),
                          list(paper_ids) if paper_ids else PAPER_ID.findall(scr.text))
        self.items[item.id], self._vec[item.id] = item, vec
        self.prune()
        return item

    # ------------------------------------------------------------------------------------------ read
    def _decay(self, it: MemoryItem) -> float:
        age_days = max(0.0, (self.clock() - it.last_used) / 86400.0)
        return 0.5 ** (age_days / self.half_life)

    def _score(self, it: MemoryItem, cos: float) -> float:
        return cos * (0.5 + 0.5 * it.importance) * self._decay(it) + 0.05 * math.log1p(it.uses)

    def search(self, query: str, k: int = 5, min_trust: str = "verified", kinds: Optional[Sequence[str]] = None, min_cosine: float = 0.15,
               touch: bool = True) -> List[MemoryItem]:
        if not self.items:
            return []
        q = self._embed(query)
        ranked = []
        for iid, it in self.items.items():
            if TRUST_ORDER[it.trust] < TRUST_ORDER[min_trust] or (kinds and it.kind not in kinds):
                continue
            cos = float(self._vec[iid] @ q)
            if cos >= min_cosine:
                ranked.append((self._score(it, cos), it))
        ranked.sort(key=lambda x: -x[0])
        top = [it for _, it in ranked[:k]]
        if touch:
            now = self.clock()
            for it in top:
                it.uses, it.last_used = it.uses + 1, now
        return top

    # ------------------------------------------------------------------------------------------ forget
    def prune(self) -> int:
        """Drop expired episodic notes, then the lowest-valued items beyond ``max_items``. Returns how many were removed."""
        now, removed = self.clock(), 0
        for iid in [i for i, it in self.items.items() if TTL_DAYS.get(it.kind) and (now - it.created) / 86400.0 > TTL_DAYS[it.kind]]:
            self.delete(iid)
            removed += 1
        while len(self.items) > self.max_items:
            worst = min(self.items.values(), key=lambda it: self._score(it, 1.0))
            self.delete(worst.id)
            removed += 1
        return removed

    def delete(self, item_id: str) -> None:
        self.items.pop(item_id, None)
        self._vec.pop(item_id, None)

    def forget_matching(self, topic: str, min_cosine: float = 0.3) -> int:
        """Delete every item related to ``topic`` (word overlap or embedding similarity)."""
        words = {w for w in re.findall(r"\w+", topic.lower()) if len(w) > 2}
        q = self._embed(topic)
        hit = [iid for iid, it in self.items.items()
               if (words and words & set(re.findall(r"\w+", it.text.lower()))) or float(self._vec[iid] @ q) >= min_cosine]
        for iid in hit:
            self.delete(iid)
        return len(hit)

    def wipe(self) -> int:
        n = len(self.items)
        self.items.clear()
        self._vec.clear()
        return n

    def describe(self) -> List[str]:
        return [f"[{it.kind}, {it.trust}] {it.text}" for it in sorted(self.items.values(), key=lambda i: -i.created)]

    # ------------------------------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for it in self.items.values():
                fh.write(json.dumps(asdict(it)) + "\n")

    def load(self, path: Path) -> int:
        n = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = MemoryItem(**json.loads(line))
                self.items[it.id], self._vec[it.id] = it, self._embed(it.text)
                n += 1
        return n

    def to_cypher_rows(self) -> List[Dict]:
        """Rows for loading memories into the knowledge graph as (:Memory)-[:ABOUT]->(:Paper) (see graph/schema.py conventions)."""
        return [{"id": it.id, "kind": it.kind, "text": it.text, "trust": it.trust, "paper_ids": it.paper_ids} for it in self.items.values()]
