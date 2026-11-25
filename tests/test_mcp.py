import io
import json
import os
import sys
import unittest

from scholargraph.agent.agent import Agent
from scholargraph.agent.tools import Tool, make_tools
from scholargraph.api.service import build_service
from scholargraph.demo import ScriptedPolicyLLM, make_demo_concepts, make_demo_gold
from scholargraph.llm import FakeLLM
from scholargraph.mcp.client import MCPClient, MCPError, as_agent_tools
from scholargraph.mcp.server import MCPServer, tool_to_schema


def rpc(method, id_=1, **params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, **({"params": params} if params else {})}


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gold = make_demo_gold(200, 4)
        cls.svc = build_service(gold, make_demo_concepts(gold), FakeLLM(["{}"]))
        cls.tools = make_tools(cls.svc.store, cls.svc.retriever)
        cls.server = MCPServer(cls.tools)

    def test_initialize_negotiates_the_version_and_declares_the_tools_capability(self):
        r = self.server.handle(rpc("initialize", protocolVersion="2025-06-18", capabilities={}, clientInfo={"name": "t", "version": "1"}))
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", r["result"]["capabilities"])
        self.assertEqual(r["result"]["serverInfo"]["name"], "scholargraph")
        old = self.server.handle(rpc("initialize", protocolVersion="2024-11-05"))
        self.assertEqual(old["result"]["protocolVersion"], "2024-11-05")
        future = self.server.handle(rpc("initialize", protocolVersion="2099-01-01"))
        self.assertEqual(future["result"]["protocolVersion"], "2025-06-18")                    # unsupported: the server answers with its own

    def test_notifications_get_no_response_and_ping_works(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertTrue(self.server.initialized)
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/whatever"}))
        self.assertEqual(self.server.handle(rpc("ping"))["result"], {})

    def test_tools_list_has_schemas_derived_from_the_parameter_notation(self):
        tools = self.server.handle(rpc("tools/list"))["result"]["tools"]
        self.assertEqual(len(tools), 12)
        by = {t["name"]: t for t in tools}
        s = by["search_papers"]["inputSchema"]
        self.assertEqual(s["required"], ["query"])
        self.assertEqual(s["properties"]["k"], {"type": "integer"})
        self.assertEqual(by["get_paper"]["inputSchema"], {"type": "object", "properties": {"paper_id": {"type": "string"}}, "required": ["paper_id"]})
        self.assertTrue(all(t["description"] for t in tools))
        self.assertEqual(tool_to_schema(Tool("x", "d", {"a": "int|null", "b": "str=5"}, lambda **k: 1)), {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}})

    def test_tools_call_returns_content_blocks(self):
        pid = self.svc.store.ids[3]
        r = self.server.handle(rpc("tools/call", name="get_paper", arguments={"paper_id": pid}))["result"]
        self.assertFalse(r["isError"])
        self.assertEqual(r["content"][0]["type"], "text")
        self.assertEqual(json.loads(r["content"][0]["text"])["paper_id"], pid)

    def test_tool_failures_are_results_and_protocol_failures_are_errors(self):
        bad_args = self.server.handle(rpc("tools/call", name="get_paper", arguments={"wrong": 1}))["result"]
        self.assertTrue(bad_args["isError"])
        self.assertIn("bad arguments", bad_args["content"][0]["text"])
        self.assertEqual(self.server.handle(rpc("tools/call", name="nope"))["error"]["code"], -32602)
        self.assertEqual(self.server.handle(rpc("tools/call", name="get_paper", arguments="x"))["error"]["code"], -32602)
        self.assertEqual(self.server.handle(rpc("resources/list"))["error"]["code"], -32601)
        self.assertEqual(self.server.handle({"id": 5, "method": "ping"})["error"]["code"], -32600)         # missing "jsonrpc"
        crash = MCPServer({"boom": Tool("boom", "d", {}, lambda: 1 / 0)})
        self.assertTrue(crash.handle(rpc("tools/call", name="boom"))["result"]["isError"])

    def test_the_stdio_loop_is_newline_delimited_and_survives_garbage(self):
        stdin = io.StringIO("\n".join([json.dumps(rpc("ping", 1)), "{not json", json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                                       json.dumps([rpc("ping", 2)]), json.dumps(rpc("tools/list", 3)), ""]))
        out = io.StringIO()
        MCPServer(self.tools).serve(stdin, out)
        lines = out.getvalue().splitlines()
        msgs = [json.loads(l) for l in lines]                                 # every line is a complete JSON message (nothing else on stdout)
        self.assertEqual([m.get("id") for m in msgs], [1, None, None, 3])
        self.assertEqual(msgs[1]["error"]["code"], -32700)
        self.assertEqual(msgs[2]["error"]["code"], -32600)
        self.assertEqual(len(msgs[3]["result"]["tools"]), 12)


class SubprocessIntegrationTest(unittest.TestCase):
    """A real client talks to a real server process over pipes, and the agent uses the remote tools through the same loop as local ones."""

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ)
        cls.client = MCPClient([sys.executable, "-m", "scholargraph.mcp.server", "--demo"], env=env, timeout=90)
        cls.info = cls.client.initialize()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_handshake_and_discovery(self):
        self.assertEqual(self.info["serverInfo"]["name"], "scholargraph")
        names = {t["name"] for t in self.client.list_tools()}
        self.assertTrue({"search_papers", "get_authors", "co_cited"} <= names)

    def test_remote_tool_call_and_error_propagation(self):
        top = json.loads(self.client.call_tool("top_cited", {"k": 1})["content"][0]["text"])["results"][0]["paper_id"]
        paper = json.loads(self.client.call_tool("get_paper", {"paper_id": top})["content"][0]["text"])
        self.assertEqual(paper["paper_id"], top)
        with self.assertRaises(MCPError):
            self.client.call_tool("no_such_tool", {})

    def test_the_agent_answers_a_question_using_only_remote_tools(self):
        remote = as_agent_tools(self.client)
        top = json.loads(self.client.call_tool("top_cited", {"k": 1})["content"][0]["text"])["results"][0]
        res = Agent(ScriptedPolicyLLM(), remote).run(f'Who wrote the paper titled "{top["title"]}"?')
        self.assertTrue(res.finished)
        self.assertGreater(len(res.items), 0)
        self.assertEqual([s.tool for s in res.steps], ["search_papers", "get_authors"])
        self.assertTrue(all(s.error is None for s in res.steps))


if __name__ == "__main__":
    unittest.main()
