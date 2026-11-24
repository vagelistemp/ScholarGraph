"""Synthetic data helpers shared by the tests and the ``demo`` command, and a scripted LLM stand-in for testing the agent loop."""
from __future__ import annotations

import json
import re
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .embeddings import TfidfSvdEncoder
from .graph.concepts import RuleConceptExtractor, extract_all
from .ingest.synthetic import make_synthetic_works
from .ingest.transform import normalise_work
from .lake.pandas_pipeline import build_gold
from .llm import LLMResult


def make_demo_gold(n_papers: int = 600, seed: int = 0) -> Dict[str, pd.DataFrame]:
    works = [w for w in (normalise_work(r) for r in make_synthetic_works(n_papers, seed)) if w]
    return build_gold(iter(works), 2015, 2025)


def make_demo_text(gold: Dict[str, pd.DataFrame], dim: int = 32) -> np.ndarray:
    papers = gold["papers"]
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    return TfidfSvdEncoder(dim=dim).fit(texts).encode(texts)


def make_demo_concepts(gold: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return extract_all(gold["papers"], RuleConceptExtractor(), None, log=lambda *a: None)


class ScriptedPolicyLLM:
    """A deterministic stand-in for a language model that *knows the question templates* of ``evalkit.questions``.

    It is **not** an AI: it reads the question with regular expressions and emits the right sequence of tool calls. It exists to test the
    machinery around a model (the JSON protocol, the tools, the question generator, the scoring) without any network or GPU: if this
    policy scores 100% the plumbing is correct, and any score a real model gets below that measures the *model*, not the plumbing.
    """
    name = "scripted-policy"

    @staticmethod
    def _obs(messages: Sequence[Dict[str, str]]):
        return [json.loads(m["content"][len("OBSERVATION: "):]) for m in messages if m["role"] == "user" and m["content"].startswith("OBSERVATION: ")]

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        q = next(m["content"] for m in messages if m["role"] == "user")[len("QUESTION: "):]
        obs = self._obs(messages)
        call = lambda tool, **args: json.dumps({"thought": "scripted", "tool": tool, "args": args})  # noqa: E731
        final = lambda items: json.dumps({"final": ", ".join(items), "items": list(items)})  # noqa: E731
        first = lambda o: o["results"][0]["paper_id"]  # noqa: E731
        titles = lambda o: [r["title"] for r in o["results"]]  # noqa: E731
        s = len(obs)
        if m := re.fullmatch(r'Who wrote the paper titled "(.+)"\?', q):
            out = call("search_papers", query=m.group(1), k=1) if s == 0 else call("get_authors", paper_id=first(obs[0])) if s == 1 else final([a["name"] for a in obs[1]["authors"]])
        elif m := re.fullmatch(r'How many papers in the graph cite the paper titled "(.+)"\?', q):
            out = call("search_papers", query=m.group(1), k=1) if s == 0 else call("get_citations", paper_id=first(obs[0])) if s == 1 else final([str(obs[1]["count"])])
        elif m := re.fullmatch(r'Which papers does the paper titled "(.+)" cite\?', q):
            out = call("search_papers", query=m.group(1), k=1) if s == 0 else call("get_references", paper_id=first(obs[0])) if s == 1 else final(titles(obs[1]))
        elif m := re.fullmatch(r'Which papers cite both the paper titled "(.+)" and the paper titled "(.+)"\?', q):
            out = (call("search_papers", query=m.group(1), k=1) if s == 0 else call("search_papers", query=m.group(2), k=1) if s == 1
                   else call("co_cited", paper_a=first(obs[0]), paper_b=first(obs[1])) if s == 2 else final(titles(obs[2])))
        elif m := re.fullmatch(r'Which papers published in (\d+) or later mention the concept "(.+)"\?', q):
            out = call("papers_with_concept", concept=m.group(2), year_from=int(m.group(1))) if s == 0 else final(titles(obs[0]))
        elif m := re.fullmatch(r"What is the most cited paper published in (\d+)\?", q):
            out = call("top_cited", k=1, year=int(m.group(1))) if s == 0 else final(titles(obs[0]))
        elif m := re.fullmatch(r"Who is the most frequent co-author of (.+)\?", q):
            name = m.group(1)
            out = (call("find_author", name=name) if s == 0
                   else call("coauthors", author_id=next(a["author_id"] for a in obs[0]["authors"] if a["name"] == name), k=1) if s == 1
                   else final([obs[1]["coauthors"][0]["name"]]))
        else:
            out = final([])
        return LLMResult(out, prompt_tokens=sum(len(x["content"].split()) for x in messages), completion_tokens=len(out.split()))
