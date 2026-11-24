"""Working memory: keep a long conversation inside a small context window.

A small local model (16 GB machine) has a short context, and every token of history costs time. ``ConversationMemory`` keeps the most
recent turns verbatim and replaces older turns by a running **summary**, so the prompt stays within a word budget however long the
conversation gets. Summaries come from a summarizer function (a model call in production, an extractive fallback otherwise).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Turn = Tuple[str, str]      # (user text, assistant text)


def words(text: str) -> int:
    return len(text.split())


def extractive_summary(turns: Sequence[Turn], previous: str = "", max_words: int = 80) -> str:
    """Fallback summariser: the first sentence of each older question plus the first words of the answer."""
    parts = [previous] if previous else []
    for u, a in turns:
        q = re.split(r"(?<=[.?!])\s", u.strip())[0]
        parts.append(f"User asked: {q} Answer: {' '.join(a.split()[:12])}")
    out = " | ".join(parts)
    w = out.split()
    return out if len(w) <= max_words else " ".join(w[-max_words:])


class ConversationMemory:
    def __init__(self, max_recent_turns: int = 4, max_words: int = 600, summarizer: Optional[Callable[[Sequence[Turn], str], str]] = None):
        self.max_recent, self.max_words = max_recent_turns, max_words
        self.summarizer = summarizer or extractive_summary
        self.summary = ""
        self.turns: List[Turn] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))
        self._compact()

    def _size(self) -> int:
        return words(self.summary) + sum(words(u) + words(a) for u, a in self.turns)

    def _compact(self) -> None:
        while len(self.turns) > self.max_recent or (self._size() > self.max_words and len(self.turns) > 1):
            oldest = self.turns[:1]
            self.turns = self.turns[1:]
            self.summary = self.summarizer(oldest, self.summary)

    def messages(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.summary:
            msgs.append({"role": "system", "content": "Summary of the earlier conversation: " + self.summary})
        for u, a in self.turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        return msgs

    def clear(self) -> None:
        self.summary, self.turns = "", []
