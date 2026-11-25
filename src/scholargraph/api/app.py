"""The HTTP API (FastAPI): search, paper lookup, GraphRAG answers and the tool-using agent, with request tracing and metrics.

Endpoints
---------
GET  /health            liveness probe (used by Docker and load balancers)
GET  /stats             size of the graph
GET  /search?q=&k=      hybrid retrieval
GET  /paper/{id}        one paper with its authors, references and concepts
POST /ask               GraphRAG answer with sources and guardrail report
POST /agent             the tool-using agent, with its trace
GET  /metrics           Prometheus text format: request counts, latency, LLM usage

Every response carries an ``X-Request-ID`` header and each request is logged as one JSON line with its latency, so a slow or failing
call can be traced. ``create_app`` takes an assembled ``Service`` so tests can inject fakes.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .service import Service

log = logging.getLogger("scholargraph.api")
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self):
        self.requests: Dict[tuple, int] = defaultdict(int)
        self.latency_sum: Dict[str, float] = defaultdict(float)
        self.latency_count: Dict[str, int] = defaultdict(int)
        self.buckets: Dict[tuple, int] = defaultdict(int)

    def observe(self, path: str, status: int, seconds: float) -> None:
        self.requests[(path, status)] += 1
        self.latency_sum[path] += seconds
        self.latency_count[path] += 1
        for b in BUCKETS:
            if seconds <= b:
                self.buckets[(path, b)] += 1

    def render(self, llm_usage=None) -> str:
        lines = ["# TYPE scholargraph_requests_total counter"]
        lines += [f'scholargraph_requests_total{{path="{p}",status="{s}"}} {n}' for (p, s), n in sorted(self.requests.items())]
        lines.append("# TYPE scholargraph_request_seconds histogram")
        for p in sorted(self.latency_count):
            cum = 0
            for b in BUCKETS:
                lines.append(f'scholargraph_request_seconds_bucket{{path="{p}",le="{b}"}} {self.buckets.get((p, b), 0)}')
            lines.append(f'scholargraph_request_seconds_bucket{{path="{p}",le="+Inf"}} {self.latency_count[p]}')
            lines.append(f'scholargraph_request_seconds_sum{{path="{p}"}} {self.latency_sum[p]:.6f}')
            lines.append(f'scholargraph_request_seconds_count{{path="{p}"}} {self.latency_count[p]}')
        if llm_usage is not None:
            lines += ["# TYPE scholargraph_llm_calls_total counter", f"scholargraph_llm_calls_total {llm_usage.calls}",
                      "# TYPE scholargraph_llm_tokens_total counter",
                      f'scholargraph_llm_tokens_total{{kind="prompt"}} {llm_usage.prompt_tokens}',
                      f'scholargraph_llm_tokens_total{{kind="completion"}} {llm_usage.completion_tokens}']
        return "\n".join(lines) + "\n"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    year_from: Optional[int] = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(service: Service, engine=None) -> FastAPI:
    app = FastAPI(title="ScholarGraph", version="0.1.0")
    metrics = Metrics()

    @app.middleware("http")
    async def trace(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.time()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            dt = time.time() - t0
            path = request.scope.get("route").path if request.scope.get("route") else request.url.path
            metrics.observe(path, status, dt)
            log.info(json.dumps({"request_id": rid, "method": request.method, "path": path, "status": status, "seconds": round(dt, 4)}))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stats")
    def stats():
        return service.store.stats()

    @app.get("/search")
    def search(q: str, k: int = 5, year_from: Optional[int] = None):
        if not 1 <= k <= 50:
            raise HTTPException(422, "k must be between 1 and 50")
        hits = service.retriever.search(q, k=k, year_from=year_from, graph_weight=0.2)
        out = []
        for h in hits:
            p = service.store.paper(h.paper_id) or {}
            out.append({"paper_id": h.paper_id, "title": p.get("title"), "year": p.get("year"), "score": round(h.score, 5), "components": h.sources})
        return {"query": q, "results": out}

    @app.get("/paper/{paper_id}")
    def paper(paper_id: str):
        p = service.store.paper(paper_id)
        if p is None:
            raise HTTPException(404, f"unknown paper {paper_id}")
        return {**p, "authors": service.store.authors_of(paper_id), "references": service.store.references(paper_id)[:50],
                "concepts": service.store.concepts_of(paper_id)}

    @app.post("/ask")
    def ask(req: AskRequest):
        a = service.rag.answer(req.question, req.year_from)
        return {"answer": a.answer, "seconds": round(a.seconds, 3),
                "sources": [{"marker": f"P{s.marker}", "paper_id": s.paper_id, "title": s.title, "year": s.year} for s in a.sources],
                "guardrails": {"ok": a.report.ok, "groundedness": round(a.report.groundedness, 3), "invalid_citations": a.report.invalid_citations,
                               "abstained": a.report.abstained}}

    @app.post("/agent")
    def agent(req: AgentRequest):
        r = service.agent.run(req.question)
        return {"answer": r.answer, "items": r.items, "finished": r.finished, "seconds": round(r.seconds, 3), "usage": r.usage,
                "steps": [{"tool": s.tool, "args": s.args, "observation": s.observation, "error": s.error} for s in r.steps]}

    @app.post("/query")
    def query(req: AgentRequest):
        """The full System-1 / System-2 pipeline: safety -> cache -> direct handlers -> intent classifier -> RAG or agent -> verifier."""
        if engine is None:
            raise HTTPException(501, "the query engine is not configured on this server")
        r = engine.answer(req.question)
        return {"answer": r.answer, "items": r.items, "route": r.route, "verified": r.verified, "seconds": round(r.seconds, 4),
                "llm_calls": r.llm_calls, "flags": r.flags}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus():
        text = metrics.render(getattr(service.llm, "usage", None))
        if engine is not None:
            text += "# TYPE scholargraph_route_total counter\n" + "".join(
                f'scholargraph_route_total{{route="{k}"}} {v}\n' for k, v in sorted(engine.route_counts.items()))
        return text

    return app
