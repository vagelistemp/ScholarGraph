"""Evaluate the agent and the System-1 engine with a REAL local language model (Ollama) on questions generated from the real graph.

    python scripts/real_llm_eval.py --model llama3.2:1b --per-type 1 --out results/real_llm_llama3.2-1b.json

It compares, on the same questions: (a) the agent with schema-constrained decoding, (b) the agent without it (free-form JSON), and
(c) the full engine (direct handlers in front). Everything is measured, nothing is assumed: a 1-billion-parameter model is expected to be
a weak tool user, and this script shows by how much, and what constrained decoding and System 1 change.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from scholargraph.agent.agent import Agent
from scholargraph.agent.tools import make_tools
from scholargraph.api.service import build_engine, build_service
from scholargraph.config import Paths
from scholargraph.evalkit.harness import run_agent_eval
from scholargraph.evalkit.questions import generate_questions
from scholargraph.lake.gold import read_gold
from scholargraph.llm import MeteredLLM, OllamaLLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--per-type", type=int, default=1)
    ap.add_argument("--out", default="results/real_llm_eval.json")
    a = ap.parse_args()
    paths = Paths(Path(a.data))
    gold = read_gold(paths.gold)
    cpath = paths.graph / "paper_concepts.parquet"
    concepts = pd.read_parquet(cpath) if cpath.exists() else None
    llm = MeteredLLM(OllamaLLM(a.model, timeout=240))
    svc = build_service(gold, concepts, llm)
    qs = generate_questions(svc.store, a.per_type, 0)
    print(f"{len(qs)} questions from the real graph; model {a.model}", flush=True)
    keep = ("n", "exact", "f1", "tool_ok", "finished", "mean_steps", "mean_seconds", "p95_seconds", "llm_calls_per_question",
            "handled_without_llm", "error_steps_per_question", "tokens_per_question")
    out = {"model": a.model, "questions": len(qs), "types": sorted({q.qtype for q in qs})}
    tools = make_tools(svc.store, svc.retriever)
    configs = {"agent_schema_constrained": Agent(llm, tools, use_schema=True), "agent_free_form_json": Agent(llm, tools, use_schema=False),
               "engine_system1": build_engine(svc)}
    for name, runner in configs.items():
        t0 = time.time()
        rep = run_agent_eval(runner, qs, log=lambda *x: None)
        out[name] = {k: rep["overall"].get(k) for k in keep}
        out[name]["wall_seconds"] = round(time.time() - t0, 1)
        out[name]["by_type_exact"] = {t: v["exact"] for t, v in rep["by_type"].items()}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if k != "by_type_exact"}, flush=True)
    # an open question through the RAG route, checked by the guardrails
    rag = []
    for q in ("What are the main approaches to link prediction?", "Which methods are used for question answering?"):
        r = svc.rag.answer(q)
        rag.append({"question": q, "seconds": round(r.seconds, 1), "n_sources": len(r.sources), "answer": r.answer[:300], "guardrails_ok": r.report.ok,
                    "groundedness": round(r.report.groundedness, 3), "invalid_citations": r.report.invalid_citations, "has_citation": r.report.has_citation})
        print(rag[-1], flush=True)
    out["rag"] = rag
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
