"""A real-data smoke run: everything that can be measured on a 10,000-paper slice without a GPU or a language model.

    python scripts/real_smoke.py --data data --out results/real_smoke_10k.json

It collects (1) the lakehouse statistics, (2) the citation-prediction baseline and GNN results written by the CLI commands, (3) a
retrieval benchmark of several configurations on test-period papers, and (4) the System-1 engine on questions generated from the real graph,
compared with the bare agent driven by the scripted policy (which stands in for a model and costs one call per step).

These are results on a *small, highly cited slice with TF-IDF text features*; they are not the final results of the project.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scholargraph.agent.agent import Agent
from scholargraph.agent.tools import make_tools
from scholargraph.api.service import build_engine, build_service
from scholargraph.config import TRAIN_END, VAL_YEAR, Paths
from scholargraph.demo import ScriptedPolicyLLM
from scholargraph.evalkit.harness import run_agent_eval
from scholargraph.evalkit.questions import generate_questions
from scholargraph.evalkit.retrieval_bench import evaluate_search_fn, leak_free_tables
from scholargraph.gnn.data import load_citation_data
from scholargraph.lake.gold import read_gold
from scholargraph.llm import MeteredLLM
from scholargraph.rag.query import expand_with_lexicon, multi_query_search
from scholargraph.rag.rerank import LexicalReranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/real_smoke_10k.json")
    ap.add_argument("--max-queries", type=int, default=300)
    a = ap.parse_args()
    paths = Paths(Path(a.data))
    gold = read_gold(paths.gold)
    cpath = paths.graph / "paper_concepts.parquet"
    concepts = pd.read_parquet(cpath) if cpath.exists() else None
    out = {"slice": {k: int(len(v)) for k, v in gold.items()}, "note": "10,000 most cited AI works 2015-2025 (keyless page download); TF-IDF text features; CPU only"}
    out["concept_links"] = int(len(concepts)) if concepts is not None else 0
    for name in ("baseline_results", "gnn_results"):
        f = paths.models / f"{name}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if name == "gnn_results":
                r = {k: r[k] for k in ("best_epoch", "val", "test", "seconds")}
            else:
                r = {k: {"val": v["val"], "test": v["test"]} for k, v in r.items()}
            out[name] = r

    svc = build_service(gold, concepts, MeteredLLM(ScriptedPolicyLLM()))
    papers = gold["papers"]
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    data = load_citation_data(gold, svc.retriever.emb, TRAIN_END, VAL_YEAR)
    # the retrieval benchmark must not see the test-period citations (they are the answers): use a service built on the leak-free graph
    bsvc = build_service(leak_free_tables(gold, data), concepts, MeteredLLM(ScriptedPolicyLLM()))
    r = bsvc.retriever
    idx = lambda hits: [r.idx[h.paper_id] for h in hits]  # noqa: E731
    configs = {
        "bm25": lambda q, i: idx(r.search(q, k=20, use_dense=False)),
        "dense (tf-idf+svd)": lambda q, i: idx(r.search(q, k=20, use_bm25=False)),
        "hybrid": lambda q, i: idx(r.search(q, k=20)),
        "hybrid + graph bonus": lambda q, i: idx(r.search(q, k=20, graph_weight=0.5)),
        "hybrid + lexical rerank": lambda q, i: idx(r.search(q, k=20, reranker=LexicalReranker(), rerank_pool=40)),
        "multi-query (lexicon expansion)": lambda q, i: idx(multi_query_search(r, [q, expand_with_lexicon(q)], k=20)),
    }
    out["retrieval"] = {}
    for name, fn in configs.items():
        t0 = time.time()
        out["retrieval"][name] = {**evaluate_search_fn(fn, data, texts, max_queries=a.max_queries, store=bsvc.store), "seconds": round(time.time() - t0, 1)}
        print(name, {k: round(v, 3) for k, v in out["retrieval"][name].items()}, flush=True)

    store = svc.store
    qs = generate_questions(store, 10, 0)
    engine = build_engine(svc)
    rep_engine = run_agent_eval(engine, qs, log=lambda *x: None)
    base_llm = MeteredLLM(ScriptedPolicyLLM())
    rep_agent = run_agent_eval(Agent(base_llm, make_tools(store, r)), qs, log=lambda *x: None)
    keep = ("n", "exact", "f1", "handled_without_llm", "llm_calls_per_question", "mean_seconds", "p95_seconds", "routes")
    out["system1"] = {"questions": len(qs), "types": sorted({q.qtype for q in qs}),
                      "engine": {k: rep_engine["overall"][k] for k in keep}, "agent_only_scripted": {k: rep_agent["overall"][k] for k in keep}}
    unhandled = [row["qid"] for row in rep_engine["rows"] if row["route"] != "handler"]
    out["system1"]["questions_not_handled_directly"] = unhandled
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
