"""A minimal MCP client over stdio: start a server process, discover its tools and call them.

``as_agent_tools`` wraps every remote tool as a local ``Tool``, so the agent of this project can use tools from *any* MCP server (a
filesystem server, a database server, another ScholarGraph) next to its own, through the same loop, hooks and tracing.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence

from ..agent.tools import Tool

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, command: Sequence[str], env: Optional[Dict[str, str]] = None, timeout: float = 60.0):
        self.proc = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                     encoding="utf-8", env=env)
        self.timeout, self._id = timeout, 0
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.server_info: Dict[str, Any] = {}

    def _reader(self) -> None:
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put(None)

    def _send(self, msg: Dict[str, Any]) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, **({"params": params} if params is not None else {})})
        while True:
            try:
                line = self._q.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPError(f"timeout waiting for {method}")
            if line is None:
                raise MCPError("server closed the connection")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise MCPError(f"{msg['error']['code']}: {msg['error']['message']}")
                return msg["result"]

    def initialize(self) -> Dict[str, Any]:
        res = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "scholargraph-client", "version": "0.1.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.server_info = res
        return res

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.request("tools/list")["tools"]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> "MCPClient":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def as_agent_tools(client: MCPClient) -> Dict[str, Tool]:
    """Wrap the server's tools as local ``Tool`` objects (the result text is parsed back into JSON when possible)."""
    out: Dict[str, Tool] = {}
    for t in client.list_tools():
        props = (t.get("inputSchema") or {}).get("properties", {})
        required = set((t.get("inputSchema") or {}).get("required", []))
        params = {k: (v.get("type", "str") if k in required else v.get("type", "str") + "|null") for k, v in props.items()}

        def fn(_name=t["name"], **kwargs):
            res = client.call_tool(_name, kwargs)
            text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
            if res.get("isError"):
                raise MCPError(text)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        out[t["name"]] = Tool(t["name"], t.get("description", ""), params, fn)
    return out
