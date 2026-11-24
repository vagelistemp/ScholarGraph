"""Tracing for LLM applications: who called what, with which model, how long it took and how many tokens it cost.

A **trace** is the story of one request; it is made of **spans** (one per step: the request, each model call, each tool call), nested
like a call tree. This module follows the OpenTelemetry model and the OpenTelemetry *GenAI semantic conventions* (attribute names such
as ``gen_ai.request.model`` and ``gen_ai.usage.input_tokens``), so the same data can be shown in any OpenTelemetry-compatible viewer.

Three tracers share one interface (``tracer.span(name, **attributes)`` as a context manager):

* ``NullTracer``      does nothing (the default; zero overhead).
* ``InMemoryTracer``  records finished spans in a list; can print the call tree or write JSON lines. Used by tests and local debugging.
* ``OTelTracer``      forwards to the real OpenTelemetry SDK (``pip install .[otel]``), so spans can go to Jaeger, Grafana Tempo, etc.
"""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .llm import LLM, LLMResult, Message

_current: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar("scholargraph_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    start: float
    end: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def seconds(self) -> float:
        return (self.end or time.time()) - self.start

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


class NullSpan:
    def set(self, **attrs: Any) -> None:
        pass


class NullTracer:
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        yield NullSpan()


class InMemoryTracer:
    def __init__(self):
        self.spans: List[Span] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        sp = Span(name, parent.trace_id if parent else uuid.uuid4().hex, uuid.uuid4().hex[:16], parent.span_id if parent else None,
                  time.time(), attrs=dict(attrs))
        token = _current.set(sp)
        try:
            yield sp
        except BaseException as e:
            sp.status, sp.error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            sp.end = time.time()
            _current.reset(token)
            self.spans.append(sp)

    def by_name(self, prefix: str) -> List[Span]:
        return [s for s in self.spans if s.name.startswith(prefix)]

    def tree(self) -> str:
        """Indented call tree of the recorded spans (children under their parent, in start order)."""
        kids: Dict[Optional[str], List[Span]] = {}
        for s in sorted(self.spans, key=lambda x: x.start):
            kids.setdefault(s.parent_id, []).append(s)
        lines: List[str] = []

        def walk(parent: Optional[str], depth: int) -> None:
            for s in kids.get(parent, []):
                extra = " ".join(f"{k}={v}" for k, v in s.attrs.items() if k in ("gen_ai.request.model", "gen_ai.tool.name", "route"))
                lines.append(f"{'  ' * depth}{s.name} {s.seconds * 1000:.1f} ms {extra}".rstrip() + ("  ERROR" if s.status == "error" else ""))
                walk(s.span_id, depth + 1)
        walk(None, 0)
        return "\n".join(lines)

    def export_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for s in self.spans:
                fh.write(json.dumps({"name": s.name, "trace_id": s.trace_id, "span_id": s.span_id, "parent_id": s.parent_id,
                                     "start": s.start, "seconds": s.seconds, "status": s.status, "error": s.error, "attrs": s.attrs}, default=str) + "\n")


class OTelTracer:
    """Bridge to the OpenTelemetry SDK. Pass an exporter (e.g. an OTLP exporter or ``InMemorySpanExporter`` in tests)."""

    def __init__(self, exporter=None, service_name: str = "scholargraph"):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        self.provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        self.provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
        self._tracer = self.provider.get_tracer("scholargraph")

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        from opentelemetry.trace import Status, StatusCode
        with self._tracer.start_as_current_span(name) as sp:
            for k, v in attrs.items():
                sp.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
            wrapper = _OTelSpan(sp)
            try:
                yield wrapper
            except BaseException as e:
                sp.set_status(Status(StatusCode.ERROR, str(e)))
                raise


class _OTelSpan:
    def __init__(self, sp):
        self._sp = sp

    def set(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            self._sp.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))


class TracedLLM:
    """Wraps any LLM so that every call becomes a span carrying the GenAI semantic-convention attributes."""

    def __init__(self, inner: LLM, tracer):
        self.inner, self.tracer, self.name = inner, tracer, inner.name

    def chat(self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 512, json_mode: bool = False,
             schema: Optional[dict] = None) -> LLMResult:
        with self.tracer.span("llm.chat", **{"gen_ai.operation.name": "chat", "gen_ai.request.model": self.inner.name,
                                              "gen_ai.request.temperature": temperature, "gen_ai.request.max_tokens": max_tokens}) as sp:
            r = self.inner.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema)
            sp.set(**{"gen_ai.usage.input_tokens": r.prompt_tokens, "gen_ai.usage.output_tokens": r.completion_tokens})
            return r
