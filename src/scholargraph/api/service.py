"""Assembling the running system (graph store + retriever + GraphRAG + agent) from the data folder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..agent.agent import Agent
from ..agent.tools import make_tools
from ..config import Paths
from ..embeddings import TfidfSvdEncoder, embed_corpus
from ..graph.store import GraphStore, MemoryGraphStore
from ..lake.gold import read_gold
from ..llm import LLM, MeteredLLM, make_llm
from ..rag.graphrag import GraphRAG
from ..rag.retrieval import HybridRetriever


@dataclass
class Service:
    store: GraphStore
    retriever: HybridRetriever
    rag: GraphRAG
    agent: Agent
    llm: MeteredLLM


def build_service(tables, concepts: Optional[pd.DataFrame] = None, llm: Optional[LLM] = None, encoder=None, embeddings: Optional[np.ndarray] = None,
                  k: int = 5) -> Service:
    """``tables`` are the gold tables. With no ``encoder`` a TF-IDF/SVD encoder is fitted on the corpus (no downloads needed)."""
    store = MemoryGraphStore(tables, concepts)
    papers = tables["papers"].reset_index(drop=True)
    texts = (papers["title"] + ". " + papers["abstract"]).tolist()
    if encoder is None:
        encoder = TfidfSvdEncoder(dim=128).fit(texts)
    if embeddings is None:
        embeddings = encoder.encode(texts)
    retriever = HybridRetriever(papers["paper_id"].tolist(), texts, embeddings, encoder, papers["year"].to_numpy(), store)
    metered = MeteredLLM(llm or make_llm("ollama:llama3.2:3b"))
    rag = GraphRAG(retriever, store, metered, k=k)
    agent = Agent(metered, make_tools(store, retriever))
    return Service(store, retriever, rag, agent, metered)
