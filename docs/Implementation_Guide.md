% ScholarGraph: Implementation Guide | How the system is built, how it is tested, and what is still to be measured | Evangelos Tempelopoulos  ·  package scholargraph (Python, Spark, PyTorch)  ·  companion document: Project Scope and Aims

:::note Summary
- **Size.** 79 Python files (about 7,400 lines) and 224 tests (about 2,750 lines) across five phases. The tests run in a few minutes on a laptop; the Spark test skips on native Windows and passes in the provided Linux container; the live Neo4j test runs when a server is available (it passed).
- **Verified so far.** (1) The Spark and pandas pipelines produce **identical** gold tables on the same input, verified inside a Linux container with Spark 4.0 and Delta Lake. (2) The agent machinery scores 100% on 35 graph-derived questions when driven by a scripted policy, which validates the tools, the JSON protocol, the question generator and the scoring. (3) The temporal-split and no-leakage properties hold (tested directly). (4) An end-to-end synthetic demo runs through all five phases.
- **Not yet measured.** No result on the real OpenAlex slice exists yet: the full run needs a GPU or a faster machine and a language model. This guide therefore reports **design and verification**, and marks every real-data result as pending.
:::

# 1. Repository layout

```
src/scholargraph/
  config.py  schemas.py  common.py  embeddings.py  llm.py  demo.py  cli.py
  ingest/     openalex.py (download)  transform.py (parsing)  synthetic.py (test data)
  lake/       pandas_pipeline.py  spark_pipeline.py  quality.py  gold.py
  graph/      store.py  neo4j_store.py  schema.py  concepts.py
  gnn/        data.py  model.py  train.py  baseline.py  metrics.py
  finetune/   pairs.py  contrastive.py  retrieval_eval.py  lora_extractor.py
  rag/        bm25.py  retrieval.py  graphrag.py  guardrails.py
  agent/      tools.py  agent.py
  evalkit/    questions.py  harness.py
  api/        service.py  app.py
tests/        test_lake, test_graph, test_gnn, test_finetune, test_rag_agent, test_embeddings
docker/       Dockerfile.spark   (Neo4j compose file; the API image arrives with phase 6)
docs/         these PDFs, figures, and the sources that build them
```

Command line: `python -m scholargraph <command>` with `ingest`, `lake`, `quality`, `concepts`, `neo4j-load`, `baseline`, `train-gnn`, `finetune`, `finetune-lora`, `ask`, `agent`, `eval-agent`, `serve` and `demo`.

# 2. The data contract

Phase 1 writes nine Parquet tables to `data/gold/`; phases 2 to 5 read only these, with pandas, so they need neither Spark nor Java. `schemas.py` states the contract in code and `validate_gold_tables` checks column names, types and key uniqueness.

| Table | Key | Main columns |
|---|---|---|
| papers | paper_id | title, abstract, year, doi, type, language, cited_by_count, venue_id, primary_topic_id, n_authors, n_references |
| authors | author_id | name, n_papers, first_year, last_year |
| authorships | paper_id, author_id | position, institution_id, institution_name, country |
| institutions | institution_id | name, country |
| citations | citing_id, cited_id | citing_year, cited_year |
| topics, paper_topics | topic_id; paper_id, topic_id | name, subfield, field; score |
| venues | venue_id | name, type |
| keywords | paper_id, keyword | score |

**Row order is part of the contract.** Later phases identify a paper by its row position (graph store, retriever, GNN node features, cached embedding matrices), but Spark writes Parquet with no guaranteed row order. `read_gold` therefore sorts every table by its key, and `embed_corpus` stores a fingerprint (a hash of the texts in order plus an encoder tag) next to each cached matrix and refuses to reuse it on any mismatch. Both behaviours are tested; without them a rebuilt gold layer could silently misalign every cached vector.

The `cited_by_count` column comes from OpenAlex and **includes citations from the future** relative to any training cut-off; no model uses it as a feature (degrees are recomputed from the message graph).

# 3. Phase 1: the lakehouse

![Medallion architecture](medallion.png|600)

**Download.** `ingest/openalex.py` pages through the API with a cursor, sorted by citation count, filtered to the slice. Every page is written atomically (temporary file, then rename) as gzip JSON lines; `_state.json` stores the cursor and counters after each page, so the program can be stopped at any time and resumed, also on another day if the daily budget runs out. Retries use exponential back-off; refusals caused by the budget raise `BudgetExhausted` and end the run cleanly. **Bronze** is the API's JSON exactly as received for the selected fields; nothing is altered.

**Parsing.** `ingest/transform.py` is plain Python: it reconstructs abstracts from OpenAlex's inverted index, strips identifier prefixes, flattens authorships (author, position, first institution), topics, keywords and references, and applies the usability rule (title present, abstract of at least 40 characters, year in range). Because it has no dependencies it is unit-tested directly, including on four real records saved from the API.

**Two engines, one specification.** `lake/pandas_pipeline.py` states the gold rules in readable pandas (duplicates keep the highest citation count; a citation is kept only if both papers are in the slice, it is not a self-citation and the cited paper is not newer than the citing paper). `lake/spark_pipeline.py` implements the same rules with Spark: partitions parse the JSON with the same Python function, **silver** tables are written as **Delta Lake** (ACID writes, schema enforcement, time travel; `merge_works` demonstrates an incremental `MERGE`), and **gold** is produced with Spark SQL window functions and a join of the reference table with the papers table. `tests/test_lake.py` runs both on the same bronze pages and asserts that all nine tables are equal; this passed in the Linux container.

**Quality gate.** `lake/quality.py` checks the contract, referential integrity, absence of self-citations and citations to the future, abstract presence, year range, and reports soft warnings (share of isolated papers, authors per paper). The pipeline stops on an error-level failure.

# 4. Phase 2: the knowledge graph

![The property-graph model](kg_schema.png|520)

**One interface, two stores.** `graph/store.py` defines `GraphStore` with 15 queries (paper lookup, references, citers, authors, co-authors, concepts, co-citation, shortest path, top cited, title search...). `MemoryGraphStore` implements them over scipy sparse matrices built from the gold tables; `Neo4jGraphStore` implements the same methods in Cypher. The agent and the evaluation depend only on the interface, so they run without a server in tests and on Neo4j in production. `graph/schema.py` defines the labels, relationship types, uniqueness constraints and batched `UNWIND ... MERGE` loaders; a test with a recording driver checks the load order (nodes before the relationships that need them), batch sizes and counts.

**Concept extraction.** Concept nodes (methods, tasks, datasets) come from two extractors with one interface. The rule-based extractor matches a curated lexicon of 71 concepts and their aliases with a longest-match regular expression (so "graph neural network" wins over "neural network", and matching respects word boundaries). The LLM extractor asks a model for JSON and then **grounds** every answer in the abstract: a concept that is neither present in the text nor an alias of something present is discarded; aliases are mapped to canonical names. Malformed or non-JSON replies yield an empty result instead of an error. Extraction is resumable through a JSON-lines cache keyed by paper, extractor and prompt version.

# 5. Phase 3: citation prediction with a graph neural network

![The temporal split and the hidden-paper protocol](temporal_split.png|640)

**Task.** For a new paper, rank the older papers it cites. This is a cold-start problem: the new paper has no outgoing edges in the graph the model sees.

**Protocol (gnn/data.py).** Edges are split by the citing paper's year. Negatives for a source paper u are papers that existed at u's date (year no later), are not u, and are not real citations of u; half of the training negatives come from u's topic. Each epoch hides a random 20% of the training papers: **all** their outgoing edges leave the message graph and become the supervision targets, so the hidden papers look like the test papers. Tests verify that the split is temporal and disjoint, that hidden papers have no outgoing edge in the message graph, that no test edge appears in any training or validation graph, and that negatives obey all constraints.

![Message passing](gnn_message.png|600)

**Model (gnn/model.py).** A directed GraphSAGE-style network in plain PyTorch (no PyTorch Geometric dependency): each layer combines a paper's own vector with the mean over the papers it cites and the mean over the papers that cite it, with layer normalisation and a residual connection. Node inputs are the text embedding, the scaled year and the log in- and out-degree of the message graph. A pair is scored by an MLP over the two vectors, their product, the text cosine and the year gap. Sparse aggregation runs on the CPU when the device is Apple's `mps`, which lacks some sparse operations; the transfers are differentiable.

**Baseline (gnn/baseline.py).** Eight pair features (text cosine, candidate popularity, year gap, same topic, same venue, shared authors, an author-citation prior and its normalised form) feed a boosted-tree classifier trained on the same pairs with the same hidden-paper protocol, and are evaluated on the same candidates. Three single-feature heuristics (popularity, text cosine, author prior) are reported alongside.

**Metrics (gnn/metrics.py).** Hits@k and MRR over 100 candidates, and AUC / AP against one random negative; ties are broken by the mean rank so that a constant scorer cannot look good (tested).

**A cautionary observation.** On the *synthetic* network used by the demo, the popularity heuristic is as strong as the GNN, because the generator plants preferential attachment. This is not a result about citation prediction; it illustrates why the baselines are part of the protocol: a model must be compared with the simplest strong alternative.

# 6. Phase 4: fine-tuning

![InfoNCE](contrastive.png|620)

**Pairs (finetune/pairs.py).** Each citation made by a paper up to 2022 yields a pair (citing text, cited text). Because a cited paper is older than its citer, no paper published after 2022 appears in any training pair (tested, including that no test edge is a training pair). Hard negatives are mined with an embedding: among the 30 nearest older training-period papers that the anchor does not cite, one is chosen at random.

**Contrastive training (finetune/contrastive.py).** The loop is plain PyTorch, so it runs on CUDA, `mps` and CPU. The loss is InfoNCE with in-batch negatives (and hard negatives as extra columns), a temperature of 0.05 and an optional symmetric term. Two anchors that cite the same paper would treat each other's positive as a negative; those cells are **masked** out of the softmax (tested). The learning rate is warmed up and decayed linearly; bfloat16 autocast is used on CUDA only.

**Retrieval evaluation (finetune/retrieval_eval.py).** Queries are test-period papers; relevant documents are the papers they cite; candidates are all papers published no later than the query. Recall@k, MRR and nDCG@k are computed identically for dense and sparse (TF-IDF) representations. A hand-computed example is part of the tests.

![LoRA](lora.png|560)

**LoRA concept extractor (finetune/lora_extractor.py).** A small causal language model learns to output the concept JSON for an abstract, with the Phase 2 concept table as training labels. The prompt tokens are masked from the loss so only the answer is learned. Plain LoRA is used (4-bit QLoRA depends on CUDA-only libraries). Tests with a tiny randomly initialised GPT-2 check that the prompt is masked, that long prompts are truncated from the left and never cut the answer, that only the adapter weights change (the base weights are bit-identical after training) and that the loss falls. Evaluation reports JSON validity and micro-averaged concept F1.

# 7. Phase 5: GraphRAG, agent, evaluation and API

![GraphRAG and the agent loop](rag_agent.png|660)

**Retrieval (rag/).** BM25 is implemented from scratch with sparse matrices and verified against the formula on a small corpus; dense retrieval uses any encoder; the two ranked lists are fused with reciprocal rank fusion, and an optional graph bonus raises papers that are cited by, or cite, several of the best hits.

**GraphRAG.** The prompt contains numbered sources (title, year, authors, abstract) and **graph facts** (which source cites which, citation counts, concepts), and instructs the model to answer only from them with `[P1]`-style citations. Guardrails then check that every cited marker exists, measure the lexical **groundedness** of each sentence in its cited sources, and require either a citation or an explicit abstention. When retrieval returns nothing the system abstains without calling the model (tested).

**Agent.** A plain-JSON protocol (one object per turn: a tool call or a final answer with machine-checkable items) works with any chat model, local or hosted. The loop is bounded, recovers from invalid JSON, unknown tools, bad arguments and tool exceptions (each tested), and records a trace. Twelve tools wrap the graph and the retriever; the `search_papers` tool returns an exact title match first, because looking up a pasted title must never depend on fuzzy ranking.

**Evaluation (evalkit/).** Seven question types are generated from the graph so that answers are known exactly (author of a paper, number of citers, references of a paper, papers citing both of two papers, papers mentioning a concept after a year, most cited paper of a year, most frequent co-author). The answers are re-derived from the raw tables in the tests. The harness reports set F1, exact match, tool choice, steps, latency and tokens per type. An LLM judge can score the faithfulness of free-text answers.

**API (api/).** FastAPI endpoints `/health`, `/stats`, `/search`, `/paper/{id}`, `/ask`, `/agent` and `/metrics` (Prometheus text format with request counts, latency histograms and LLM token counters). Every response carries an `X-Request-ID`, and each request is logged as one JSON line.

# 8. Phase 5b: reliability, control and advanced agents

![The five stages of the query engine](system1_flow.png|640)

This layer wraps the Phase 5 agent in the engineering that makes a language-model system dependable. Every item is unit-tested; the model-dependent parts are exercised with scripted stand-ins, so no result below depends on a real model.

**Evaluation upgrades.** Beyond the final answer the harness scores the *trajectory* (precision of tool calls against the optimal sequence for the question type, efficiency, whether the optimal order occurs, error steps), reports p50 and p95 latency, tokens and model calls per question and the **share of requests handled without a model**. A golden question set can be frozen and `compare_reports` fails a change that lowers quality (absolute tolerance 0.02) or raises cost (relative tolerance 25%), listing the questions that flipped from right to wrong.

**Observability.** A tracer interface with a no-op, an in-memory and an OpenTelemetry implementation records nested spans for requests, model calls and tool calls with the OpenTelemetry GenAI attribute names (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...). The OpenTelemetry bridge is tested against the SDK's in-memory exporter.

**Lifecycle hooks.** Seven events (before/after model call, before/after tool call, before final answer, on error, on finish) let plug-ins allow, block, stop or modify. Built-in hooks: budgets (model calls, tool calls, tokens, seconds), tool allow-list, human approval, redaction of e-mail, phone and card numbers, loop guard, tracing, grounding check and memory. A crashing hook is isolated.

**Structured output.** Schemas are validated locally (a dependency-free JSON-Schema subset); Ollama and OpenAI-compatible providers receive the schema for constrained decoding; a repair loop feeds validation errors back to the model. The agent protocol also supports several independent tool calls in one turn, executed in parallel threads.

**System 1 layer.** A request passes through a safety check (length, prompt-injection patterns, PII redaction), an answer cache (exact and semantic, with a numbers-and-quotes signature guard, TTL and LRU eviction, verified answers only), direct handlers (regular expressions and graph lookups for the seven exact question families, no model), a trained intent classifier (route: lookup, topic search, analysis, off-topic) and a trained answer verifier. On unseen phrasings the lexical (TF-IDF) classifier reaches 0.60 accuracy and the semantic (MiniLM embedding) classifier 0.91; both numbers are measured on template-generated questions and will be lower on real users.

**Better agents.** A planner produces at most five subtasks (schema-validated, with a single-task fallback), grouped into dependency waves that run in parallel on specialist sub-agents with restricted tool sets; a synthesiser writes the answer from the sub-results and a verifier requires every item to appear in the tools' observations, repairing once and then dropping unsupported items. The same check exists as a before-final hook for a single agent.

**Memory.** Working memory keeps recent turns and a rolling summary; long-term memory stores semantic and episodic notes with provenance, trust, importance, recency decay (half-life), de-duplication and eviction. The write policy only stores user statements about themselves and verified answers, rejects instruction-like or sensitive text, redacts personal data and honours "what do you remember", "forget about X" and "forget everything". An end-to-end test shows that a poisoned abstract read by the agent never reaches memory while a user preference carries over to a new session.

**Interoperability (MCP).** The twelve read-only graph tools are exposed as an MCP server over stdio (JSON-RPC 2.0, protocol version negotiation, `tools/list` with JSON Schemas derived from the tool signatures, `tools/call` with `isError` semantics, standard error codes). A client wraps remote tools as local ones; the integration test starts a real server subprocess and lets the agent answer a question using only remote tools.

**Retrieval and GraphRAG upgrades.** Rerankers (lexical, bi-encoder, cross-encoder, late-interaction MaxSim), lexicon expansion, LLM query rewriting, HyDE, multi-query fusion, contextual indexing (year, venue, topic and concepts prepended to the indexed text), Louvain communities with reports and a map-reduce global answerer that cites communities and abstains when none knows, and multi-hop retrieval that returns the citation path of every result (the test verifies that each explained edge exists).

**Efficiency and learning.** A disk cache for deterministic model calls, a small-to-large cascade, context truncation, a memory estimator (weights plus KV cache) and an MLX wrapper for Apple Silicon (not testable here). Direct Preference Optimization with LoRA (the reference model is the adapter switched off) and builders that turn agent traces into preference pairs at the first divergence of a correct and a wrong run, and into supervised data from correct runs.

# 9. Testing

| Test module | Tests | What it covers |
|---|---|---|
| test_lake | 14 | parsing, abstract reconstruction, real-record fixture, gold rules and planted errors, contract violations, Parquet round trip, canonical row order, Spark equals pandas |
| test_ingest_openalex | 6 | keyless page mode and its 10,000 cap, cursor mode with a key, helpful 401 error, resume after a budget stop, slice protection (mock OpenAlex server) |
| test_graph | 19 | store queries against raw tables and networkx, Neo4j loader order and batching, concept extraction, grounding, caching, agreement metric |
| test_neo4j_live | 6 | every store method against a live Neo4j server (skipped unless `NEO4J_URI` is set) |
| test_gnn | 13 | temporal split, hidden papers are isolated, no test edge in any graph, negatives, metrics and ties, aggregation by hand, training beats chance, tracker integration, baseline features by hand |
| test_finetune | 10 | no future text in training pairs, hard negatives, loss masking, retrieval metrics by hand, contrastive training improves retrieval, LoRA freezing and masking |
| test_rag_agent | 25 | BM25 formula, fusion, graph bonus, guardrails, abstention, agent robustness, question answers against raw tables, scripted policy at 100%, judge parsing, API endpoints and metrics |
| test_embeddings | 5 | embedding cache reused only for the same texts and encoder, rejected on reordering, unit-length vectors |
| test_structured | 8 | schema validator, repair loop, provider request bodies for constrained decoding |
| test_hooks_observability | 18 | hook ordering and isolation, budgets, allow-list, approval, redaction, loop guard, final-answer rejection, parallel tool calls, nested spans, OpenTelemetry bridge |
| test_evalkit_upgrades | 10 | trajectory metrics by hand, regression comparison, golden-set round trip |
| test_system1 | 21 | safety, cache (TTL, LRU, semantic guards), direct handlers vs the graph, classifier generalisation (lexical and semantic), verifier, engine routes, harness integration |
| test_api_query | 4 | the `/query` endpoint, route counters in the metrics |
| test_multi_agent | 10 | plans, waves, restricted specialists, grounding hook, repair and drop of unsupported items, parallel subtasks |
| test_memory | 16 | scoring and decay, merge, eviction, expiry, trust, poisoning attempts, user commands, cross-session recall, working memory |
| test_retrieval_upgrades | 17 | rerankers, MaxSim, query rewriting, multi-query fusion, contextual text, retrieval benchmark, communities, global answers, multi-hop paths |
| test_mcp | 9 | handshake, schemas, call results, error codes, stdio loop, real subprocess integration |
| test_efficiency_learning | 13 | cache, cascade, truncation, memory estimate, benchmark, DPO loss by hand, preference pairs from traces, DPO training |

224 tests in total (218 run on a machine without Neo4j; the Spark equivalence test also needs Linux or Docker).

Run all with `python -m unittest discover -s tests -t .`.

# 10. How to run it

```
pip install -e ".[dl,serve,baseline]"      # add ",spark" on Linux or in the container, ",graph" for Neo4j
python -m scholargraph demo                # synthetic end-to-end run, a few minutes, no downloads
export OPENALEX_API_KEY=...                # optional, from openalex.org/settings/api
python -m scholargraph ingest              # resumable download of the slice (default 300,000 works)
python -m scholargraph lake --engine pandas        # or: --engine spark (Linux, macOS or the container)
python -m scholargraph concepts --extractor rules
python -m scholargraph baseline && python -m scholargraph train-gnn
python -m scholargraph finetune --hard-negatives   # GPU (Colab) or Apple Silicon
python -m scholargraph serve --llm ollama:llama3.2:3b
python -m scholargraph eval-agent --llm ollama:llama3.2:3b
```

# 11. Results

## 11.1 Verification (no real data needed)

| Experiment | Outcome |
|---|---|
| Spark equals pandas on the same input | verified (synthetic bronze pages, Linux container) |
| Live Neo4j store equals the in-memory store on all 15 methods | verified (Neo4j 5 in Docker, synthetic graph) |
| MCP server and client over real pipes | verified (server subprocess; agent answers using only remote tools) |
| Agent plumbing validated by a scripted policy | verified (100% exact match on the graph-derived questions, on synthetic **and** on the real graph) |

## 11.2 First real-data run (10,000 papers, CPU only, no API key)

These are measured results on a **small, highly cited slice** with TF-IDF text features and no language model. They are real, but they are *not* the project's final numbers.

**Data.** 10,000 most cited English AI works 2015 to 2025 downloaded in page mode (24 MB of compressed raw pages, so the 300,000-paper slice would need about 0.7 GB). The gold layer has 9,993 papers, 35,747 authors, 54,646 authorships, 80,915 citations *between papers of the slice* (3.8% of papers have none), 808 topics and 1,405 venues; all quality checks pass; the rule-based extractor finds 12,749 concept links of 71 distinct concepts. The lakehouse step takes about 10 seconds.

**Citation prediction.** The test period is the 18,072 citations made by 2024 and 2025 papers; each of 3,000 sampled test citations is ranked among 99 time-consistent negatives; the model only sees the graph up to 2023 (cold start). Hits@10 / MRR / AUC:

| Method | Hits@10 | MRR | AUC |
|---|---|---|---|
| popularity of the candidate | 0.365 | 0.206 | 0.643 |
| authors cited it before | 0.134 | 0.125 | 0.563 |
| text cosine (TF-IDF + SVD) alone | 0.698 | 0.405 | 0.883 |
| boosted trees on eight hand-made features (one run) | 0.703 | 0.385 | 0.908 |
| graph neural network, mean of 3 seeds (standard deviation) | **0.760** (0.007) | **0.458** (0.017) | **0.923** (0.001) |

The GNN beats the strongest baseline consistently across seeds (training takes about 2 minutes on a CPU), but the margin is moderate and the baseline was run once, so its own variability is unknown. The GNN's pair scorer also receives the baseline's hand features, so its gain is what the graph adds on top of them. The development of this result exposed six problems that unit tests had missed (a leak through incoming citations of hidden papers, one optimizer step per epoch, a too-short early-stopping patience, among others); they are listed in the study notes and each now has a test.

**Retrieval.** 300 test-period papers are used as queries (title plus abstract); relevant documents are the papers they cite; the retrieval graph contains no test-period citations (a guard enforces it, see below). Dense retrieval here is TF-IDF + SVD, not a neural encoder:

| Configuration | MRR | Recall@10 | Recall@20 |
|---|---|---|---|
| BM25 | 0.369 | 0.115 | 0.151 |
| dense (TF-IDF + SVD) | 0.360 | 0.111 | 0.153 |
| hybrid (BM25 + dense, reciprocal rank fusion) | 0.393 | 0.118 | 0.159 |
| hybrid + graph bonus | **0.449** | **0.153** | **0.202** |
| hybrid + lexical reranker | 0.273 | 0.086 | 0.132 |
| multi-query with lexicon expansion | 0.386 | 0.117 | 0.163 |

Fusion beats either list alone and the graph bonus adds a further clear gain. The simple lexical reranker **hurts** (it discards the first-stage signals), which is reported as measured; a real cross-encoder has not been run. A first version of this benchmark reported an MRR of 0.756 for the graph bonus. That number was a **leak**: the query paper is itself in the index, so its own text ranks it first and the graph bonus boosted its references, which are the answers. The benchmark now rejects any graph store that contains test-period citations (`BenchmarkLeak`) and a test shows the inflation.

**System 1.** On 70 questions generated from the real graph (ten of each of the seven types) the engine answered all of them exactly, 69 through direct handlers and one from the cache, with **zero model calls** and a mean of 11 ms per question. The same questions through the agent, driven by the scripted policy (the cheapest possible stand-in for a model), need 2.9 model calls per question.

[[LLM]]

## 11.3 Still pending

| Experiment | Status |
|---|---|
| The full 300,000-paper slice (needs a free OpenAlex key) | pending |
| Neural text encoder, contrastive fine-tuning, LoRA extractor | pending (needs a GPU or the Apple Silicon Mac) |
| Cross-encoder reranking, HyDE and query rewriting with a model | pending |
| Agent and GraphRAG with a stronger language model, DPO on real traces | pending |

# 12. Limitations

- **Real-data results are preliminary.** They come from a 10,000-paper slice of the most cited papers (a dense citation core), TF-IDF text features, one run of the baseline and three seeds of the GNN; the full-scale run is pending.
- **Time-consistency inside training.** The message graph of a training paper contains edges from papers published after it (within the training period). This is a distribution shift, not leakage of the target, and test-time graphs are strictly in the past; a stricter per-paper time cut would remove it at a large cost.
- **Citation counts are future-contaminated** and are therefore not used as features.
- **Concept extraction is bounded by its labels.** The LoRA model learns from the teacher's labels and inherits their mistakes; the rule lexicon is small.
- **Grounding is lexical.** The groundedness measure is a proxy; the LLM judge is stronger but costs more and has its own biases.
- **Scope.** One subfield, English abstracts, a bounded agent; the Neo4j store was checked against a live server on synthetic data only.
- **Native Windows.** Spark needs the Hadoop `winutils` helper there; the Docker image avoids this.
