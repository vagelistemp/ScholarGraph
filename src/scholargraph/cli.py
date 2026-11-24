"""Command line: ``python -m scholargraph <command>``  (or ``scholargraph <command>`` after ``pip install -e .``)

Phase 1   ingest          download the research slice from OpenAlex (resumable)
          lake            bronze -> silver -> gold tables (--engine pandas | spark), then quality checks
          quality         re-run the data-quality checks on the gold tables
Phase 2   concepts        extract methods / tasks / datasets from abstracts (--extractor rules | llm:ollama:llama3.2:3b)
          neo4j-load      load the graph into a running Neo4j server
Phase 3   baseline        feature-based citation-prediction baseline (boosted trees + heuristics)
          train-gnn       train and evaluate the graph neural network
Phase 4   finetune        contrastive fine-tuning of the text encoder, with before / after retrieval evaluation
          finetune-lora   LoRA fine-tuning of a small LLM for concept extraction (GPU recommended)
Phase 5   ask             GraphRAG answer to a question          agent   the tool-using agent
          eval-agent      run the graph-derived question set     serve   start the HTTP API
          demo            everything above on a small synthetic network (a few minutes, no downloads)

All commands read and write under the data folder (``--data`` or ``SCHOLARGRAPH_DATA``, default ./data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SLICE, TRAIN_END, VAL_YEAR, Paths


def _paths(args) -> Paths:
    return Paths(Path(args.data)).ensure() if args.data else Paths().ensure()


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print("wrote", path)


def _gold(paths: Paths):
    from .lake.gold import read_gold
    return read_gold(paths.gold)


def _encoder(spec: str, gold, device: str = "auto"):
    """``tfidf`` (no download) or a Hugging Face model name / local folder (``sentence-transformers/all-MiniLM-L6-v2``)."""
    from .embeddings import HFEncoder, TfidfSvdEncoder
    papers = gold["papers"]
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    if spec == "tfidf":
        return TfidfSvdEncoder(128).fit(texts), texts
    from .common import pick_device
    return HFEncoder(spec, device=pick_device(device)), texts


# ---------------------------------------------------------------------------------------------------- phase 1
def cmd_ingest(a):
    from .ingest.openalex import SliceSpec, download
    p = _paths(a)
    state = download(p.bronze, SliceSpec(max_works=a.max_works, year_from=a.year_from, year_to=a.year_to))
    print(f"bronze: {state.works:,} works in {state.pages} pages; done={state.done}")


def cmd_lake(a):
    p = _paths(a)
    if a.engine == "spark":
        from .lake import spark_pipeline as sp
        spark = sp.spark_session(driver_memory=a.memory)
        print("silver:", sp.build_silver(spark, p.bronze, p.silver))
        print("gold:  ", sp.build_gold(spark, p.silver, p.gold, a.year_from, a.year_to))
    else:
        from .ingest.transform import iter_normalised
        from .lake.gold import write_gold
        from .lake.pandas_pipeline import build_gold
        write_gold(build_gold(iter_normalised(p.bronze), a.year_from, a.year_to), p.gold)
    cmd_quality(a)


def cmd_quality(a):
    from .lake.quality import has_errors, run_quality_checks, summarise
    checks = run_quality_checks(_gold(_paths(a)))
    print(summarise(checks))
    if has_errors(checks):
        sys.exit(1)


# ---------------------------------------------------------------------------------------------------- phase 2
def cmd_concepts(a):
    from .graph.concepts import LLMConceptExtractor, RuleConceptExtractor, extract_all
    from .llm import make_llm
    p = _paths(a)
    papers = _gold(p)["papers"]
    ex = RuleConceptExtractor() if a.extractor == "rules" else LLMConceptExtractor(make_llm(a.extractor.split(":", 1)[1] if a.extractor.startswith("llm:") else a.extractor))
    cache = p.graph / f"concept_cache_{ex.name.replace(':', '_')}.jsonl"
    out = extract_all(papers, ex, cache, limit=a.limit)
    out.to_parquet(p.graph / "paper_concepts.parquet", index=False)
    print(f"{len(out):,} concept links, {out['concept'].nunique():,} distinct concepts")


def cmd_neo4j_load(a):
    from .graph.neo4j_store import connect, load_graph
    p = _paths(a)
    cpath = p.graph / "paper_concepts.parquet"
    load_graph(connect(a.uri, a.user, a.password), _gold(p), pd.read_parquet(cpath) if cpath.exists() else None)


# ---------------------------------------------------------------------------------------------------- phase 3
def _citation_data(a, p: Paths):
    from .embeddings import embed_corpus
    from .gnn.data import load_citation_data
    gold = _gold(p)
    enc, texts = _encoder(a.encoder, gold, a.device)
    slug = a.encoder.replace("/", "_").replace("\\", "_").replace(":", "_")
    emb = embed_corpus(enc, texts, p.embeddings / f"papers_{slug}.npy", batch_size=a.batch_size, tag=a.encoder)
    return load_citation_data(gold, emb, TRAIN_END, VAL_YEAR)


def cmd_baseline(a):
    from .gnn.baseline import run_baseline
    from .gnn.data import make_eval_set
    p = _paths(a)
    data = _citation_data(a, p)
    sp = data.split()
    out = run_baseline(data, make_eval_set(data, sp["val"], 99, a.max_eval, 1), make_eval_set(data, sp["test"], 99, a.max_eval, 2))
    _save(out, p.models / "baseline_results.json")


def cmd_train_gnn(a):
    from .common import Tracker
    from .gnn.train import TrainConfig, train_gnn
    p = _paths(a)
    data = _citation_data(a, p)
    cfg = TrainConfig(epochs=a.epochs, hidden=a.hidden, device=a.device, max_eval_pairs=a.max_eval)
    res = train_gnn(data, cfg, tracker=Tracker("gnn", "run", p.root / "runs"))
    _save({"config": cfg.__dict__, "best_epoch": res.best_epoch, "val": res.val, "test": res.test, "seconds": res.seconds, "history": res.history}, p.models / "gnn_results.json")
    import torch
    torch.save(res.model.state_dict(), p.models / "gnn.pt")


# ---------------------------------------------------------------------------------------------------- phase 4
def cmd_finetune(a):
    from .embeddings import HFEncoder, TfidfSvdEncoder, embed_corpus
    from .finetune.contrastive import ContrastiveConfig, train_contrastive
    from .finetune.pairs import build_pairs, mine_hard_negatives
    from .finetune.retrieval_eval import evaluate_retrieval
    from .gnn.data import load_citation_data
    from .common import pick_device
    p = _paths(a)
    gold = _gold(p)
    papers = gold["papers"]
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    tfidf = TfidfSvdEncoder(128).fit(texts)
    tf_emb = tfidf.encode(texts)
    data = load_citation_data(gold, tf_emb, TRAIN_END, VAL_YEAR)
    enc = HFEncoder(a.model, device=pick_device(a.device))
    base_emb = embed_corpus(enc, texts, None, batch_size=a.batch_size)
    results = {"tfidf": evaluate_retrieval(tf_emb, data, max_queries=a.max_queries),
               "base_encoder": evaluate_retrieval(base_emb, data, max_queries=a.max_queries)}
    print("before:", json.dumps(results, indent=1))
    pairs = build_pairs(data, a.max_pairs)
    if a.hard_negatives:
        pairs = mine_hard_negatives(data, pairs, base_emb)
    cfg = ContrastiveConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=a.device, use_hard_negatives=a.hard_negatives)
    train_contrastive(enc, texts, pairs, cfg)
    out_dir = p.models / "embedder"
    enc.model.save_pretrained(out_dir)
    enc.tok.save_pretrained(out_dir)
    tuned = embed_corpus(enc, texts, p.embeddings / "papers_finetuned.npy", batch_size=a.batch_size, tag="finetuned:" + a.model)
    results["finetuned_encoder"] = evaluate_retrieval(tuned, data, max_queries=a.max_queries)
    results["config"] = {**cfg.__dict__, "model": a.model, "pairs": len(pairs)}
    _save(results, p.models / "embedder_results.json")


def cmd_finetune_lora(a):
    from .finetune.lora_extractor import (LoraConfigLite, attach_lora, build_sft_examples, evaluate_extractor, load_base_model, train_lora)
    p = _paths(a)
    gold = _gold(p)
    concepts = pd.read_parquet(p.graph / "paper_concepts.parquet")
    papers = gold["papers"].sort_values("year")
    train_p, test_p = papers[papers["year"] <= TRAIN_END], papers[papers["year"] > VAL_YEAR]
    train_ex = build_sft_examples(train_p.sample(min(a.max_examples, len(train_p)), random_state=0), concepts)
    test_ex = build_sft_examples(test_p.sample(min(200, len(test_p)), random_state=0), concepts)
    model, tok = load_base_model(a.base)
    model = attach_lora(model, LoraConfigLite())
    model.print_trainable_parameters()
    before = evaluate_extractor(model, tok, test_ex[:50])
    train_lora(model, tok, train_ex, LoraConfigLite(epochs=a.epochs, device=a.device))
    after = evaluate_extractor(model, tok, test_ex[:50])
    model.save_pretrained(p.models / "lora_extractor")
    _save({"before": before, "after": after, "n_train": len(train_ex)}, p.models / "lora_results.json")


# ---------------------------------------------------------------------------------------------------- phase 5
def _service(a):
    from .api.service import build_service
    from .llm import make_llm
    p = _paths(a)
    gold = _gold(p)
    cpath = p.graph / "paper_concepts.parquet"
    from .embeddings import embed_corpus
    enc, texts = _encoder(a.encoder, gold, "auto")
    slug = a.encoder.replace("/", "_").replace("\\", "_").replace(":", "_")
    emb = embed_corpus(enc, texts, p.embeddings / f"papers_{slug}.npy", tag=a.encoder)           # cached: no re-embedding on every start
    return build_service(gold, pd.read_parquet(cpath) if cpath.exists() else None, make_llm(a.llm), enc, emb)


def cmd_ask(a):
    r = _service(a).rag.answer(a.question)
    print(r.answer, "\n")
    for s in r.sources:
        print(f"  [P{s.marker}] {s.title} ({s.year})")
    print(f"\nguardrails: ok={r.report.ok} groundedness={r.report.groundedness:.2f} invalid_citations={r.report.invalid_citations}")


def cmd_agent(a):
    r = _service(a).agent.run(a.question)
    for s in r.steps:
        print(f"  tool {s.tool}({s.args}) {'ERROR ' + s.error if s.error else ''}")
    print("answer:", r.answer, "| items:", r.items)


def cmd_eval_agent(a):
    from .evalkit.harness import run_agent_eval, save_report
    from .evalkit.questions import generate_questions
    svc = _service(a)
    qs = generate_questions(svc.store, a.per_type, a.seed)
    rep = run_agent_eval(svc.agent, qs)
    save_report(rep, _paths(a).eval / "agent_report.json")
    print(json.dumps({"overall": rep["overall"], "by_type": rep["by_type"]}, indent=1))


def cmd_serve(a):
    import uvicorn
    from .api.app import create_app
    uvicorn.run(create_app(_service(a)), host=a.host, port=a.port)


# ---------------------------------------------------------------------------------------------------- demo
def cmd_demo(a):
    """Every phase on a small synthetic network. A smoke test of the whole system, not a source of results."""
    import tempfile
    from .api.service import build_service
    from .demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold, make_demo_text
    from .embeddings import HashingBagEncoder, embed_corpus
    from .evalkit.harness import run_agent_eval
    from .evalkit.questions import generate_questions
    from .finetune.contrastive import ContrastiveConfig, train_contrastive
    from .finetune.pairs import build_pairs
    from .finetune.retrieval_eval import evaluate_retrieval
    from .gnn.baseline import run_baseline
    from .gnn.data import load_citation_data, make_eval_set
    from .gnn.train import TrainConfig, train_gnn
    from .lake.quality import run_quality_checks, summarise

    quiet = lambda *x: None  # noqa: E731
    print("Phase 1: synthetic OpenAlex records -> gold tables")
    gold = make_demo_gold(a.papers, 0)
    print(summarise(run_quality_checks(gold)))
    print(f"\nPhase 2: knowledge graph + concepts ({len(gold['papers'])} papers, {len(gold['citations'])} citations)")
    concepts = make_demo_concepts(gold)
    print(f"  {len(concepts)} concept links, {concepts['concept'].nunique()} distinct concepts")
    text = make_demo_text(gold, 32)
    data = load_citation_data(gold, text, TRAIN_END, VAL_YEAR)
    sp = data.split()
    va, te = make_eval_set(data, sp["val"], 49, 400, 1), make_eval_set(data, sp["test"], 49, 400, 2)
    print("\nPhase 3: baseline vs GNN (test years, 50 candidates per query)")
    base = run_baseline(data, va, te, log=quiet)
    gnn = train_gnn(data, TrainConfig(epochs=15, hidden=32, k_neg=49, max_eval_pairs=400, device="cpu", patience=5), log=quiet)
    for name, m in [("popularity", base["popularity"]["test"]), ("boosted trees", base["boosted_trees"]["test"]), ("GNN", gnn.test)]:
        print(f"  {name:14s} hits@10 {m['hits@10']:.3f}  MRR {m['mrr']:.3f}  AUC {m['auc']:.3f}")
    print("\nPhase 4: contrastive fine-tuning of a tiny encoder")
    papers = gold["papers"]
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    enc = HashingBagEncoder(64)
    before = evaluate_retrieval(embed_corpus(enc, texts, log=quiet), data, ks=(10,), max_queries=200)
    train_contrastive(enc, texts, build_pairs(data), ContrastiveConfig(epochs=3, lr=5e-2, temperature=0.1, device="cpu", log_every=10 ** 9), log=quiet)
    emb = embed_corpus(enc, texts, log=quiet)
    after = evaluate_retrieval(emb, data, ks=(10,), max_queries=200)
    print(f"  recall@10 {before['recall@10']:.3f} -> {after['recall@10']:.3f}   MRR {before['mrr']:.3f} -> {after['mrr']:.3f}")
    print("\nPhase 5: GraphRAG service and agent evaluation (scripted policy standing in for the LLM)")
    svc = build_service(gold, concepts, ScriptedPolicyLLM(), enc, emb)
    rep = run_agent_eval(svc.agent, generate_questions(svc.store, 5, 0), log=quiet)
    print(f"  {rep['overall']['n']} graph-derived questions, exact match {rep['overall']['exact']:.2f}, tool choice {rep['overall']['tool_ok']:.2f}")
    print("\nDemo finished. These numbers come from synthetic data and only show that the pipeline runs end to end.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scholargraph", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data folder (default: $SCHOLARGRAPH_DATA or ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, help=fn.__doc__)
        sp.set_defaults(fn=fn)
        return sp

    s = add("ingest", cmd_ingest); s.add_argument("--max-works", type=int, default=SLICE["max_works"])
    s.add_argument("--year-from", type=int, default=SLICE["year_from"]); s.add_argument("--year-to", type=int, default=SLICE["year_to"])
    s = add("lake", cmd_lake); s.add_argument("--engine", choices=["pandas", "spark"], default="pandas"); s.add_argument("--memory", default="4g")
    s.add_argument("--year-from", type=int, default=SLICE["year_from"]); s.add_argument("--year-to", type=int, default=SLICE["year_to"])
    add("quality", cmd_quality)
    s = add("concepts", cmd_concepts); s.add_argument("--extractor", default="rules", help="rules | llm:ollama:llama3.2:3b | llm:anthropic:claude-haiku-5-5")
    s.add_argument("--limit", type=int, default=None)
    s = add("neo4j-load", cmd_neo4j_load); s.add_argument("--uri", default="bolt://localhost:7687"); s.add_argument("--user", default="neo4j")
    s.add_argument("--password", default="scholargraph")
    for name, fn in (("baseline", cmd_baseline), ("train-gnn", cmd_train_gnn)):
        s = add(name, fn); s.add_argument("--encoder", default="tfidf"); s.add_argument("--device", default="auto"); s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--max-eval", type=int, default=20000); s.add_argument("--epochs", type=int, default=40); s.add_argument("--hidden", type=int, default=128)
    s = add("finetune", cmd_finetune); s.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2"); s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--batch-size", type=int, default=32); s.add_argument("--lr", type=float, default=2e-5); s.add_argument("--max-pairs", type=int, default=200_000)
    s.add_argument("--max-queries", type=int, default=2000); s.add_argument("--device", default="auto"); s.add_argument("--hard-negatives", action="store_true")
    s = add("finetune-lora", cmd_finetune_lora); s.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct"); s.add_argument("--epochs", type=int, default=1)
    s.add_argument("--max-examples", type=int, default=4000); s.add_argument("--device", default="auto")
    for name, fn in (("ask", cmd_ask), ("agent", cmd_agent)):
        s = add(name, fn); s.add_argument("question")
    s = add("eval-agent", cmd_eval_agent); s.add_argument("--per-type", type=int, default=10); s.add_argument("--seed", type=int, default=0)
    s = add("serve", cmd_serve); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    for name in ("ask", "agent", "eval-agent", "serve"):
        sp = sub.choices[name]
        sp.add_argument("--llm", default="ollama:llama3.2:3b"); sp.add_argument("--encoder", default="tfidf")
    s = add("demo", cmd_demo); s.add_argument("--papers", type=int, default=600)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
