"""A tiny, provider-independent LLM interface used by Phases 2, 4 and 5.

Every component that needs a language model (concept extraction, the RAG answerer, the agent, the LLM judge) depends on the
``LLM`` protocol below, never on a vendor SDK. That gives three things: tests run with ``FakeLLM`` (no network, no GPU), real runs
use a local model through Ollama (free), and a paid API can be swapped in with one line.

All providers are plain HTTP calls through ``httpx``; the only dependency is the standard library plus httpx.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import httpx

Message = Dict[str, str]          # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLM(Protocol):
    name: str

    def chat(self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 512, json_mode: bool = False,
             schema: Optional[dict] = None) -> LLMResult: ...


@dataclass
class Usage:
    """Running totals, used by the evaluation harness to report latency and token cost."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.seconds += r.seconds


class MeteredLLM:
    """Wraps any LLM and records usage."""

    def __init__(self, inner: LLM):
        self.inner, self.usage, self.name = inner, Usage(), inner.name

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        r = self.inner.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema)
        self.usage.add(r)
        return r


class OllamaLLM:
    """A local model served by Ollama (https://ollama.com): ``ollama pull llama3.2:3b``, then this class talks to localhost:11434."""

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434", timeout: float = 300.0):
        self.model, self.host, self.timeout, self.name = model, host.rstrip("/"), timeout, f"ollama:{model}"

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        body = {"model": self.model, "messages": list(messages), "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if schema:
            body["format"] = schema                      # Ollama structured outputs: the sampler is constrained to the JSON schema
        elif json_mode:
            body["format"] = "json"
        t0 = time.time()
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return LLMResult(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0), time.time() - t0)


class OpenAICompatibleLLM:
    """Any server that speaks the OpenAI chat-completions protocol (vLLM, LM Studio, many cloud gateways)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY", timeout: float = 120.0):
        self.model, self.base_url, self.api_key_env, self.timeout, self.name = model, base_url.rstrip("/"), api_key_env, timeout, f"openai-compat:{model}"

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        body = {"model": self.model, "messages": list(messages), "temperature": temperature, "max_tokens": max_tokens}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                       headers={"Authorization": f"Bearer {os.environ.get(self.api_key_env, '')}"})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return LLMResult(d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0)


class AnthropicLLM:
    """Claude through the Anthropic Messages API (set ``ANTHROPIC_API_KEY``)."""

    def __init__(self, model: str = "claude-haiku-5-5", timeout: float = 120.0):
        self.model, self.timeout, self.name = model, timeout, f"anthropic:{model}"

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "messages": turns}
        if system:
            body["system"] = system
        t0 = time.time()
        r = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                       headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return LLMResult("".join(b.get("text", "") for b in d["content"]), u.get("input_tokens", 0), u.get("output_tokens", 0), time.time() - t0)


class FakeLLM:
    """A scripted model for tests: returns canned replies in order (or computes them from the messages with ``fn``)."""

    def __init__(self, replies: Optional[List[str]] = None, fn: Optional[Callable[[Sequence[Message]], str]] = None):
        self.replies, self.fn, self.name, self.calls, self.schemas = list(replies or []), fn, "fake", [], []
        self._i = 0

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, schema=None) -> LLMResult:
        self.calls.append(list(messages))
        self.schemas.append(schema)
        text = self.fn(messages) if self.fn else self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return LLMResult(text, prompt_tokens=sum(len(m["content"].split()) for m in messages), completion_tokens=len(text.split()))


def make_llm(spec: str) -> LLM:
    """``ollama:llama3.2:3b`` | ``anthropic:claude-haiku-5-5`` | ``openai:gpt-4o-mini`` | ``fake``."""
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaLLM(model or "llama3.2:3b")
    if kind == "anthropic":
        return AnthropicLLM(model or "claude-haiku-5-5")
    if kind == "openai":
        return OpenAICompatibleLLM(model or "gpt-4o-mini")
    if kind == "fake":
        return FakeLLM(["{}"])
    raise ValueError(f"unknown LLM spec {spec!r}")


def parse_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from a model reply (models often wrap JSON in prose or code fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
