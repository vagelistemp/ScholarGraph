# ScholarGraph

A scholarly **knowledge-graph platform** over [OpenAlex](https://openalex.org): a Spark/Delta Lake lakehouse, a property-graph knowledge graph, a citation-prediction graph neural network, contrastively fine-tuned embeddings and a LoRA-fine-tuned concept extractor, and a **GraphRAG question-answering agent** with guardrails, graph-derived evaluation and an HTTP API.

**Tags:** `knowledge-graph` · `graph-neural-network` · `rag` · `graphrag` · `llm-agents` · `fine-tuning` · `lora` · `contrastive-learning` · `pyspark` · `delta-lake` · `neo4j` · `pytorch` · `fastapi` · `openalex`

![Architecture](docs/figures/architecture.png)

## Status (read this first)

The system is **implemented and unit-tested, but not yet run at full scale on the real data**. This table says exactly what has been verified.

| Phase | What it is | Implemented | Unit-tested | Real-data run |
|---|---|---|---|---|
| 1. Lakehouse | resumable OpenAlex download, bronze / silver (Delta) / gold (Parquet), data-quality checks | yes | yes (Spark = pandas, verified in a Linux container) | pending |
| 2. Knowledge graph | store interface, in-memory and Neo4j stores, rule and LLM concept extraction with grounding | yes | yes (Neo4j loader tested with a recording driver) | pending; live Neo4j check pending |
| 3. Graph neural network | temporal split, hidden-paper protocol, directed GraphSAGE in plain PyTorch, boosted-tree baseline | yes | yes | pending |
| 4. Fine-tuning | contrastive encoder (InfoNCE, hard negatives), LoRA concept extractor, retrieval evaluation | yes | yes (wiring, leakage, metrics) | pending (GPU or Apple Silicon) |
| 5. GraphRAG agent | BM25 + dense + graph retrieval, guardrails, 12-tool agent, evaluation harness, FastAPI + metrics | yes | yes (scripted policy scores 100% on 35 graph-derived questions) | pending (needs an LLM) |

No benchmark number is reported anywhere in this repository because none has been measured on real data yet. The documents describe the design, the evaluation protocol and the verification done so far; results will be added after the full run.

## Documentation

| File | Content |
|---|---|
| [`docs/Project_Scope_and_Aims.pdf`](docs/Project_Scope_and_Aims.pdf) | aims with success criteria, data, evaluation plan, scope, risks, status |
| [`docs/Implementation_Guide.pdf`](docs/Implementation_Guide.pdf) | how each phase works, the data contract, tests, how to run, limitations |

## Quick start

```bash
pip install -e ".[dl,serve,baseline]"        # Python 3.10+; add ",spark" (Java 17/21) or ",graph" (Neo4j) as needed
python -m unittest discover -s tests -t .    # 85 tests, about a minute
python -m scholargraph demo                  # all five phases on a synthetic network, no downloads
```

On the real data (see the Implementation Guide for details):

```bash
export OPENALEX_API_KEY=...                  # optional free key from openalex.org/settings/api (never commit it)
python -m scholargraph ingest                # resumable; default slice: 300,000 most cited AI papers, 2015-2025
python -m scholargraph lake --engine pandas  # or --engine spark (Linux / macOS / Docker)
python -m scholargraph concepts --extractor rules
python -m scholargraph baseline && python -m scholargraph train-gnn
python -m scholargraph finetune --hard-negatives
python -m scholargraph serve --llm ollama:llama3.2:3b
python -m scholargraph eval-agent --llm ollama:llama3.2:3b
```

Spark on native Windows needs the Hadoop `winutils` helper; use the Docker image instead:
`docker build -f docker/Dockerfile.spark -t scholargraph-spark .`

## Data and privacy

OpenAlex data is CC0. The downloader reads an optional API key from the environment, never writes it to disk, and never asks for or transmits an e-mail address. Downloaded data, models and run logs are git-ignored; the repository contains only code, four real API records used as a test fixture, tests and documents.

## Design principles

- **One data contract.** Phase 1 writes nine gold Parquet tables; phases 2 to 5 read only those, with pandas, so they need neither Spark nor Java.
- **Leakage-free evaluation.** Temporal split by the citing paper's year; hidden-paper training so that training looks like cold-start testing; no future text in fine-tuning pairs; none of it is assumed, all of it is tested.
- **A baseline for every model.** The GNN is compared with a boosted-tree model on the same pairs and candidates.
- **Interfaces over vendors.** `GraphStore` (memory or Neo4j) and `LLM` (Ollama, OpenAI-compatible, Anthropic, or a scripted fake) keep tests free and the system portable.
- **Honest reporting.** Tested facts are separated from pending measurements.

## Layout

```
src/scholargraph/   ingest, lake, graph, gnn, finetune, rag, agent, evalkit, api (+ cli, llm, embeddings, schemas)
tests/              85 unit tests (lake, graph, gnn, finetune, rag_agent, embeddings)
docker/             Dockerfile.spark, neo4j.yml
docs/               the two PDFs, figures, and the sources that build them (docs/build)
scripts/            make_fixture.py, make_figures.py
```

## License

MIT, see [`LICENSE`](LICENSE). © 2026 Evangelos Tempelopoulos.
