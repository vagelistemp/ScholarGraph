"""ScholarGraph: a scholarly knowledge-graph platform over OpenAlex.

Phases (each is a sub-package and a lecture in ``learning/``):

    ingest, lake   1  Spark lakehouse: download, bronze / silver / gold tables, data quality
    graph          2  knowledge graph (Neo4j or in-memory), LLM concept extraction
    gnn            3  graph neural network for citation prediction, with a feature-based baseline
    finetune       4  contrastive fine-tuning of an embedding model, LoRA fine-tuning of a small LLM
    rag, agent,    5  hybrid + graph retrieval, a tool-using agent, guardrails, evaluation, HTTP API
    evalkit, api
"""
__version__ = "0.1.0"
