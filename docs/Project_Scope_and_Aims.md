% ScholarGraph: Project Scope and Aims | A scholarly knowledge-graph platform over OpenAlex: lakehouse, knowledge graph, graph neural network, fine-tuned embeddings and a GraphRAG agent | Evangelos Tempelopoulos  ·  companion document: Implementation Guide

:::note The project in brief
- **What it is.** An end-to-end system that turns a large open bibliographic dataset (OpenAlex, about 300,000 papers on artificial intelligence, 2015 to 2025) into a **knowledge graph** of papers, authors, institutions, venues, topics and scientific concepts, and builds four machine-learning components on top of it: a citation-prediction **graph neural network**, a **fine-tuned embedding model**, a **fine-tuned small language model** for concept extraction, and a **GraphRAG question-answering agent** with guardrails, evaluation and an HTTP API.
- **Why it exists.** To practise, in one coherent system rather than five disconnected demos, the skills that applied AI roles ask for: data engineering at scale (Spark, Delta Lake), graph data modelling, deep learning in PyTorch, fine-tuning (contrastive learning and LoRA), retrieval-augmented generation, tool-using agents, and the engineering around them (tests, evaluation, serving, honest reporting).
- **How it is built.** Five phases with one data contract between them (the *gold* Parquet tables). Each phase is a Python sub-package with its own unit tests; everything runs on a laptop for small data and on a free Colab GPU or an Apple Silicon Mac for the full slice.
- **Status.** Phases 1 to 5 and the 5b layer are implemented and unit-tested (226 tests). A first run on **10,000 real papers** (CPU, no API key) has been done and is reported with its caveats in the Implementation Guide; the full-scale run on the 300,000-paper slice, neural encoders, fine-tuning and stronger language models are **pending**.
:::

# 1. Motivation

Scientific literature is a graph: papers cite papers, authors write papers, institutions employ authors, papers belong to topics and mention methods, tasks and datasets. Most software that works with papers treats them as isolated pieces of text (search, summarisation, plain RAG). Treating the literature as a **graph plus text** unlocks questions that text alone answers badly: *Which papers cite both of these two? Who collaborates most with this author? Which methods appeared in this subfield after 2022? What should this new paper probably cite?*

These questions also exercise exactly the techniques that modern applied-AI teams use together: structured data pipelines, a graph store, learned representations, retrieval, language models and evaluation. The project therefore doubles as a portfolio piece: every component is something an interviewer can probe, and every claim in it is backed by a test or a measurement.

# 2. Aims

Each aim has a measurable success criterion, defined **before** the full run so that results cannot be tuned after the fact.

| Aim | Success criterion |
|---|---|
| A1. A reproducible lakehouse | Bronze, silver and gold layers rebuild from the raw pages with one command; the Spark and pandas pipelines produce identical gold tables (tested); all data-quality checks pass |
| A2. A queryable knowledge graph | The 15 queries of the GraphStore interface are answered identically by the in-memory store and by Neo4j; concept extraction is evaluated against labelled references, with hallucinated concepts filtered |
| A3. Citation prediction with a leakage-free protocol | A graph neural network and a boosted-tree baseline are compared on the same temporal split (train to 2022, validate 2023, test 2024 and 2025) using Hits@k, MRR and AUC; the GNN must be reported honestly even if it does not win |
| A4. Fine-tuned retrieval | A contrastively fine-tuned encoder is compared with its base model and with TF-IDF on held-out citing papers using Recall@k, MRR and nDCG; no text of a test-period paper is used in training (tested) |
| A5. Distilled concept extractor | A LoRA-fine-tuned 0.5-billion-parameter model is compared with its un-tuned self on JSON validity and concept F1 against the teacher labels, on test-period abstracts |
| A6. A trustworthy QA system | A tool-using agent is evaluated on graph-derived questions with exactly known answers; GraphRAG answers carry citations that are checked automatically; the system abstains when evidence is missing; latency and token cost are reported |
| A8. Dependable agent behaviour | Trajectory evaluation, a frozen golden set with automatic regression checks, tracing with the OpenTelemetry GenAI conventions, lifecycle hooks for budgets, permissions and redaction, schema-validated structured output |
| A9. A System 1 / System 2 split | A safety check, cache, direct handlers and trained classifiers in front of the language model; the share of requests answered without a model is measured and reported next to quality |
| A10. Advanced agents with safe memory | A planner with specialist sub-agents and a verifier; long-term memory whose write policy rejects poisoned content (tested end to end); an MCP server and client |
| A11. Stronger retrieval and global questions | Rerankers, query rewriting, contextual indexing, communities with map-reduce answers and multi-hop retrieval, each compared on one retrieval benchmark |
| A12. Local efficiency and preference learning | Memory budgeting for a 16 GB machine, caching and cascades, DPO with LoRA and trace-based preference data |
| A7. Engineering quality | Unit tests for every module, a data contract validated in code, no secrets or personal data in the repository, and documentation that distinguishes tested facts from pending results |

# 3. System overview

![The five phases and the data contract between them](architecture.png|640)

| Phase | Input | Output | Key techniques |
|---|---|---|---|
| 1. Lakehouse | OpenAlex API | gold Parquet tables | cursor-paged download with checkpointing, medallion architecture, Spark SQL window functions, Delta Lake, data-quality checks |
| 2. Knowledge graph | gold tables | graph store, concept table | property-graph modelling, Cypher, store abstraction, rule-based and LLM concept extraction with grounding |
| 3. Graph neural network | gold tables, text embeddings | trained model, metrics | temporal split, hidden-paper training protocol, directed GraphSAGE in plain PyTorch, boosted-tree baseline |
| 4. Fine-tuning | gold tables | fine-tuned encoder and LoRA adapter | InfoNCE with in-batch and hard negatives, false-negative masking, LoRA, leakage tests |
| 5. GraphRAG agent | graph, retriever, LLM | API, evaluation report | BM25 and dense retrieval with reciprocal rank fusion, graph expansion, JSON tool protocol, guardrails, graph-derived evaluation, FastAPI and metrics |
| 5b. Reliability and advanced agents | the Phase 5 agent | query engine, memory, MCP server, advanced retrieval | evaluation upgrades, tracing, hooks, structured output, System 1 layer, planner and verifier, memory safety, MCP, reranking, communities, DPO |

# 4. Data

The data comes from **OpenAlex** (openalex.org), an open catalogue of scholarly works released under the CC0 licence. The slice is defined by one filter and is easy to change:

- subfield **Artificial Intelligence** (OpenAlex subfield 1702), years **2015 to 2025**, types article and preprint, English, abstract present;
- the **300,000 most cited** works of that filter (the default; the filter matches about 468,000), which gives a dense citation core: highly cited papers cite one another far more often than a random sample would;
- about **0.7 GB** of compressed raw pages (measured: 10,000 works take 24 MB), well inside the 10 GB budget.

OpenAlex meters API usage in dollars per day (a list page costs about $0.0001). Without a key the API serves plain page numbers only up to the first **10,000 results**; larger slices need cursor paging, which **requires a free personal key** (obtained by you at openalex.org/settings/api-key and passed in the environment variable `OPENALEX_API_KEY`). The downloader runs in either mode, **checkpoints after every page** and resumes where it stopped. The key is never stored in the repository, and the program never asks for, stores or transmits an e-mail address.

A synthetic generator of OpenAlex-format records (with planted topics, concept phrases, author communities, preferential attachment and a few deliberate data errors) supplies the tests and the demo; four real records saved from the API form a format-fidelity fixture.

# 5. Evaluation plan

The evaluation protocol is fixed in code and described here so that it can be audited.

- **Temporal split.** Train on citations made by papers published up to 2022, validate on 2023, test on 2024 and later. Splitting by the citing paper's year prevents the future from leaking into the past.
- **Cold start.** A test paper has no outgoing edges in the message graph (its references are the targets). During training a random 20% of the papers are hidden each epoch in the same way, so the model learns under the conditions it will be tested in.
- **Negatives.** For each positive pair (u, v), 99 negatives are papers that u could have cited at its publication date (published no later) but did not. Training negatives are half uniform, half drawn from the same topic.
- **Metrics.** Hits@1/5/10/50 and MRR over the 100 candidates, plus AUC and average precision against one random negative.
- **Baselines.** Popularity (citations of the candidate), text cosine, author prior, and a boosted-tree model on eight hand-crafted pair features. A neural model must beat the strongest of these to justify its cost.
- **Retrieval.** Queries are test-period papers; the relevant documents are the papers they cite; the candidate pool is every paper that existed at the query's date. Recall@k, MRR and nDCG@k are compared for TF-IDF, the base encoder and the fine-tuned encoder.
- **Question answering.** Seven question types are generated from the graph so the correct answers are known exactly; the agent is scored by set F1, whether it chose a sensible tool, steps, latency and tokens. A scripted policy that knows the templates must score 100% (it does, in the test suite); this validates the plumbing, so that a real model's score measures the model.
- **Reporting.** Every result states its protocol, its sample size and its limitations; results that are weak or negative are reported as they are.

# 6. Scope boundaries

| In scope | Out of scope |
|---|---|
| One research subfield, 2015 to 2025, English abstracts | The whole of OpenAlex; full-text PDFs; other languages |
| Citation prediction with text, structure and author signals | Temporal graph networks, link-prediction leaderboards (OGB) |
| Fine-tuning a small encoder and a 0.5-billion-parameter LLM | Training any model from scratch; models above a few billion parameters |
| A tool-using agent with a bounded number of steps | Autonomous long-horizon agents, multi-agent systems |
| A local HTTP API with metrics and tracing | Cloud deployment and monitoring (a planned sixth phase) |

# 7. Planned sixth phase

A deployment phase is planned but intentionally not part of this implementation: containerise the API, deploy it on AWS or Azure with infrastructure as code (Terraform), automate tests and builds with CI, track experiments with MLflow and add dashboards over the metrics endpoint. The API already exposes health, metrics and request-ID tracing so that this phase only adds infrastructure.

# 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The API budget stops the download | Checkpoint after every page; resumable across days; optional personal key |
| Spark is awkward on native Windows | Pure-Python gold pipeline with a test that proves equivalence; Docker image for Spark; Spark tests skip when unusable |
| Slow hardware | Heavy steps are separate commands with small defaults; a free Colab GPU or an Apple Silicon Mac handles training; tests run on a laptop in about a minute |
| A GNN that does not beat a strong baseline | The baseline is built in from the start and the result is reported as it is |
| LLM-extracted concepts are hallucinated | Concepts must be grounded in the abstract text; a rule-based extractor is the fallback and the reference |
| Overclaiming | Documentation separates unit-tested facts from pending measurements; no number appears before it has been measured |

# 9. Status

| Component | Implemented | Unit-tested | Full run on real data |
|---|---|---|---|
| Phase 1 lakehouse | yes | yes (Spark equals pandas, verified in a Linux container) | 10,000-paper slice done; full slice pending |
| Phase 2 graph, concepts | yes | yes (live Neo4j parity test passed) | rule-based concepts on the 10,000-paper slice done; LLM extraction pending |
| Phase 3 GNN and baseline | yes | yes | 10,000-paper slice done (3 seeds); full slice pending |
| Phase 4 fine-tuning | yes | yes (wiring, leakage, retrieval metrics) | pending (needs a GPU or the Mac) |
| Phase 5 GraphRAG, agent, API, evaluation | yes | yes | retrieval benchmark and System 1 on the 10,000-paper slice done; stronger-model evaluation pending |
| Phase 5b reliability and advanced agents | yes | yes (real MCP subprocess, live Neo4j) | System 1 measured on the real graph; the rest depends on a language model |
