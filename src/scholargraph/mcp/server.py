"""A Model Context Protocol (MCP) server over stdio, implemented directly on JSON-RPC 2.0.

MCP is an open standard that lets AI applications (Claude, IDE assistants, agent frameworks) discover and call tools offered by other
programs. This server exposes ScholarGraph's twelve **read-only** graph tools, so any MCP client can ask questions of the knowledge graph.

Protocol subset implemented (MCP specification 2025-06-18, tools capability):

* transport: newline-delimited JSON-RPC messages on stdin / stdout (nothing else may be written to stdout; logs go to stderr);
* lifecycle: ``initialize`` request (version and capability negotiation) -> ``notifications/initialized`` notification;
* ``ping``; ``tools/list`` (name, description, JSON-Schema ``inputSchema``); ``tools/call`` (result ``content`` blocks and ``isError``);
* JSON-RPC errors: -32700 parse error, -32600 invalid request, -32601 unknown method, -32602 invalid params.

Run it with ``python -m scholargraph.mcp.server --demo`` (synthetic data) or ``--data DIR`` (gold tables).
Tool failures are reported inside the result (``isError: true``) so the calling model can see and correct them; protocol problems use
JSON-RPC errors, as the specification requires.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, TextIO

from ..agent.tools import Tool

SUPPORTED_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
MAX_RESULT_CHARS = 20_000


def _json_type(spec: str) -> Dict[str, Any]:
    base = spec.split("=")[0].split("|")[0].strip()
    return {"str": {"type": "string"}, "int": {"type": "integer"}, "float": {"type": "number"}, "bool": {"type": "boolean"}}.get(base, {"type": "string"})


def tool_to_schema(tool: Tool) -> Dict[str, Any]:
    """Derive a JSON Schema from the compact parameter notation of ``agent.tools`` ("str", "int=5", "int|null")."""
    props, required = {}, []
    for name, spec in tool.params.items():
        props[name] = _json_type(spec)
        if "=" not in spec and "null" not in spec:
            required.append(name)
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _error(id_: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


class MCPServer:
    def __init__(self, tools: Dict[str, Tool], name: str = "scholargraph", version: str = "0.1.0"):
        self.tools, self.name, self.version = tools, name, version
        self.initialized = False

    # ------------------------------------------------------------------------------------------ message handling
    def handle(self, msg: Any) -> Optional[Dict[str, Any]]:
        """Process one JSON-RPC message; returns the response, or ``None`` for notifications."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
        method, id_, params = msg["method"], msg.get("id"), msg.get("params") or {}
        is_notification = "id" not in msg
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method.startswith("notifications/"):
            return None
        if is_notification:                                          # unknown notifications are ignored, never answered
            return None
        if method == "initialize":
            client_version = params.get("protocolVersion")
            version = client_version if client_version in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
            return {"jsonrpc": "2.0", "id": id_, "result": {
                "protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": self.version},
                "instructions": "Read-only tools over a scholarly knowledge graph: find papers, authors, citations, concepts. Paper ids look like W123."}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": id_, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": id_, "result": {"tools": [
                {"name": t.name, "description": t.description, "inputSchema": tool_to_schema(t)} for t in self.tools.values()]}}
        if method == "tools/call":
            return self._call(id_, params)
        return _error(id_, -32601, f"method not found: {method}")

    def _call(self, id_: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or name not in self.tools:
            return _error(id_, -32602, f"unknown tool: {name!r}")
        if not isinstance(args, dict):
            return _error(id_, -32602, "arguments must be an object")
        try:
            text = json.dumps(self.tools[name].fn(**args), default=str, ensure_ascii=False)
            is_error = False
        except TypeError as e:
            text, is_error = f"bad arguments for {name}: {e}", True
        except Exception as e:
            text, is_error = f"tool {name} failed: {type(e).__name__}: {e}", True
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + " ...[truncated]"
        return {"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": text}], "isError": is_error}}

    # ------------------------------------------------------------------------------------------ stdio transport
    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                resp: Optional[Dict[str, Any]] = _error(None, -32700, "parse error")
            else:
                if isinstance(msg, list):                              # batches are not part of this subset: answer once, clearly
                    resp = _error(None, -32600, "batch requests are not supported")
                else:
                    resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")           # one message per line, no embedded newlines
                stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ScholarGraph MCP server (stdio)")
    ap.add_argument("--demo", action="store_true", help="serve a small synthetic graph")
    ap.add_argument("--data", default=None, help="data folder with gold tables")
    args = ap.parse_args(argv)
    from ..api.service import build_service
    from ..agent.tools import make_tools
    from ..llm import FakeLLM
    if args.demo or not args.data:
        from ..demo import make_demo_concepts, make_demo_gold
        gold = make_demo_gold(300, 0)
        concepts = make_demo_concepts(gold)
    else:
        import pandas as pd
        from pathlib import Path
        from ..config import Paths
        from ..lake.gold import read_gold
        p = Paths(Path(args.data))
        gold = read_gold(p.gold)
        cp = p.graph / "paper_concepts.parquet"
        concepts = pd.read_parquet(cp) if cp.exists() else None
    svc = build_service(gold, concepts, FakeLLM(["{}"]))
    print("scholargraph MCP server ready on stdio", file=sys.stderr)
    MCPServer(make_tools(svc.store, svc.retriever)).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
