"""Concept extraction: turning abstracts into knowledge-graph nodes (methods, tasks, datasets).

Two extractors with the same interface:

* ``RuleConceptExtractor``   a curated lexicon (``resources/concept_lexicon.json``) matched against the text. Fast, free,
                             precise but limited to what the lexicon contains. It is the baseline and the offline fallback.
* ``LLMConceptExtractor``    asks a language model for JSON and then **validates the answer against the text**: a concept the model
                             invents but the abstract does not mention is discarded (a cheap, effective guard against hallucination).

``extract_all`` runs an extractor over the papers with a resumable on-disk cache, because LLM calls are slow and may be interrupted.
Its output, the ``paper_concepts`` table, becomes the ``Concept`` nodes and ``MENTIONS`` edges of the knowledge graph.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..llm import LLM, parse_json_object

KINDS = ("method", "task", "dataset")
PROMPT_VERSION = "v1"

SYSTEM = (
    "You extract scientific concepts from a paper abstract. Reply with ONE JSON object and nothing else, with exactly these keys: "
    '"methods" (techniques or models the paper uses or proposes), "tasks" (problems it addresses) and "datasets" (benchmark datasets '
    "it uses). Each value is a list of short lowercase noun phrases (1 to 4 words). Only include a concept that is explicitly mentioned in "
    "the abstract. Use empty lists when there is none."
)


def load_lexicon() -> Dict[str, Dict[str, List[str]]]:
    with resources.files("scholargraph.resources").joinpath("concept_lexicon.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalise_phrase(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+\-\. ]", " ", s.lower())).strip(" .-")


class RuleConceptExtractor:
    name = "rules"

    def __init__(self, lexicon: Optional[Dict[str, Dict[str, List[str]]]] = None):
        lexicon = lexicon or load_lexicon()
        self.alias: Dict[str, Tuple[str, str]] = {}                 # surface form -> (canonical concept, kind)
        for kind, entries in lexicon.items():
            for canonical, aliases in entries.items():
                for surface in [canonical, *aliases]:
                    self.alias[normalise_phrase(surface)] = (canonical, kind)
        # one regex over all surface forms, longest first, so "graph neural network" wins over "neural network"
        forms = sorted(self.alias, key=len, reverse=True)
        self._re = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(f) for f in forms) + r")(?![a-z0-9])")

    def extract(self, text: str) -> List[Tuple[str, str]]:
        found: Dict[str, str] = {}
        for m in self._re.finditer(normalise_phrase(text)):
            canonical, kind = self.alias[m.group(1)]
            found.setdefault(canonical, kind)
        return sorted(found.items())


class LLMConceptExtractor:
    def __init__(self, llm: LLM, lexicon: Optional[Dict[str, Dict[str, List[str]]]] = None, max_per_kind: int = 6):
        self.llm, self.max_per_kind, self.name = llm, max_per_kind, f"llm:{llm.name}"
        rules = RuleConceptExtractor(lexicon)
        self._alias = rules.alias                                    # maps surface forms to canonical names where known

    def extract(self, text: str) -> List[Tuple[str, str]]:
        reply = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Abstract:\n{text}"}],
                              temperature=0.0, max_tokens=300, json_mode=True)
        data = parse_json_object(reply.text) or {}
        hay = normalise_phrase(text)
        out: Dict[str, str] = {}
        for key, kind in (("methods", "method"), ("tasks", "task"), ("datasets", "dataset")):
            items = data.get(key) or []
            if not isinstance(items, list):
                continue
            kept = 0
            for raw in items:
                if not isinstance(raw, str):
                    continue
                phrase = normalise_phrase(raw)
                if not phrase or len(phrase.split()) > 5:
                    continue
                if phrase not in hay and not self._grounded_alias(phrase, hay):         # drop hallucinated concepts
                    continue
                canonical = self._alias.get(phrase, (phrase, kind))[0]
                out.setdefault(canonical, kind)
                kept += 1
                if kept >= self.max_per_kind:
                    break
        return sorted(out.items())

    def _grounded_alias(self, phrase: str, hay: str) -> bool:
        """The model used a canonical name ("graph neural network") while the text says "gnn": accept if any known surface form occurs."""
        canon = self._alias.get(phrase)
        if not canon:
            return False
        return any(c == canon and (s in hay) for s, c in self._alias.items())


def _cache_key(paper_id: str, extractor_name: str) -> str:
    return hashlib.sha1(f"{PROMPT_VERSION}|{extractor_name}|{paper_id}".encode()).hexdigest()[:16]


def extract_all(papers: pd.DataFrame, extractor, cache_path: Optional[Path] = None, limit: Optional[int] = None,
                log: Callable[[str], None] = print, flush_every: int = 200) -> pd.DataFrame:
    """Run ``extractor`` over ``papers`` (columns paper_id, title, abstract). Resumable through a JSON-lines cache."""
    cache: Dict[str, List[List[str]]] = {}
    if cache_path and Path(cache_path).exists():
        for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["k"]] = rec["c"]
    fh = open(cache_path, "a", encoding="utf-8") if cache_path else None
    rows: List[Dict[str, str]] = []
    todo = papers if limit is None else papers.head(limit)
    try:
        for n, (pid, title, abstract) in enumerate(zip(todo["paper_id"], todo["title"], todo["abstract"]), 1):
            key = _cache_key(pid, extractor.name)
            if key not in cache:
                cache[key] = [list(c) for c in extractor.extract(f"{title}. {abstract}")]
                if fh:
                    fh.write(json.dumps({"k": key, "c": cache[key]}) + "\n")
                    if n % flush_every == 0:
                        fh.flush()
                        log(f"  {n:,} / {len(todo):,} papers")
            rows.extend({"paper_id": pid, "concept": c, "kind": k, "source": extractor.name.split(":")[0]} for c, k in cache[key])
    finally:
        if fh:
            fh.close()
    return pd.DataFrame(rows, columns=["paper_id", "concept", "kind", "source"]).drop_duplicates(["paper_id", "concept"])


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, float]:
    """Micro precision / recall / F1 of concept table ``a`` against reference table ``b`` (same paper_id, concept keys)."""
    A = set(zip(a["paper_id"], a["concept"]))
    B = set(zip(b["paper_id"], b["concept"]))
    tp = len(A & B)
    p, r = tp / max(len(A), 1), tp / max(len(B), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "n_pred": len(A), "n_ref": len(B)}
