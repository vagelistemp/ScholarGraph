"""Diagrams for the agent-engineering features (phase 5b).   python scripts/make_figures_agents.py"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_figures", HERE / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)
canvas, box, arrow, save = mf.canvas, mf.box, mf.arrow, mf.save
INK, GREY, BLUE, GREEN, RED, AMBER, PURPLE = mf.INK, mf.GREY, mf.BLUE, mf.GREEN, mf.RED, mf.AMBER, mf.PURPLE


def system1_flow():
    fig, ax = canvas(11.5, 4.6, (0, 23), (0, 9))
    stages = [("safety check", "length, injection,\nredact PII", RED), ("cache", "exact + semantic\nverified only", AMBER), ("direct handlers", "regex + graph\nno model", GREEN),
              ("intent classifier", "trained, small,\nfallback = agent", BLUE), ("System 2", "RAG / agent /\nplanner + verifier", PURPLE)]
    for i, (t, d, c) in enumerate(stages):
        x = 0.3 + i * 4.5
        box(ax, x, 4.3, 3.9, 2.4, t, c, 9.5, True)
        ax.text(x + 1.95, 3.5, d, ha="center", va="top", fontsize=8, color=INK, linespacing=1.3)
        if i < 4:
            arrow(ax, (x + 3.95, 5.5), (x + 4.45, 5.5))
            ax.text(x + 1.95, 7.2, "answers here?" if i else "blocked?", ha="center", fontsize=8, color=GREY, style="italic")
    ax.text(11.5, 8.4, "A request leaves at the first stage that can answer it; only the last stage calls a language model", ha="center", fontsize=10, fontweight="bold")
    ax.text(11.5, 0.6, "each answer records its route, latency and model calls, so the share handled without a model is measured", ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "system1_flow.png")


def hook_events():
    fig, ax = canvas(11.5, 4.4, (0, 23), (0, 9))
    steps = [("before_llm", BLUE), ("model call", GREY), ("after_llm", BLUE), ("before_tool", GREEN), ("tool call", GREY), ("after_tool", GREEN)]
    for i, (t, c) in enumerate(steps):
        x = 0.2 + i * 3.8
        box(ax, x, 5.2, 3.3, 1.5, t, c, 9, c != GREY)
        if i < 5:
            arrow(ax, (x + 3.35, 5.95), (x + 3.75, 5.95))
    box(ax, 6.8, 2.2, 4.4, 1.5, "before_final  (reject / rewrite)", RED, 9, True)
    box(ax, 12.2, 2.2, 3.6, 1.5, "on_finish", PURPLE, 9, True)
    arrow(ax, (11.25, 2.95), (12.15, 2.95))
    ax.text(0.3, 7.6, "hooks may: allow, block, stop the run, or modify arguments / observations / answers", fontsize=9.5, fontweight="bold")
    ax.text(0.3, 0.7, "e.g. BudgetHook stops a run, ToolAllowlistHook blocks a tool, RedactionHook rewrites an observation, GroundingHook rejects an answer", fontsize=8.5, color=GREY, style="italic")
    save(fig, "hook_events.png")


def memory_arch():
    fig, ax = canvas(11.5, 4.8, (0, 23), (0, 9.6))
    box(ax, 0.3, 6.6, 5.0, 1.8, "working memory\nrecent turns + rolling summary", BLUE, 9)
    box(ax, 0.3, 3.9, 5.0, 1.8, "long-term memory\nsemantic / episodic / procedural", GREEN, 9)
    box(ax, 0.3, 1.2, 5.0, 1.8, "knowledge graph\n(:Memory)-[:ABOUT]->(:Paper)", AMBER, 9)
    box(ax, 9.0, 3.9, 5.0, 1.8, "agent loop", PURPLE, 10, True)
    box(ax, 17.5, 6.6, 5.0, 1.8, "write policy: only user facts\nand VERIFIED answers", RED, 9)
    box(ax, 17.5, 3.9, 5.0, 1.8, "screen: injection, orders,\nPII, length", RED, 9)
    box(ax, 17.5, 1.2, 5.0, 1.8, "user control: show / forget\n/ forget everything", GREY, 9)
    arrow(ax, (5.35, 7.5), (8.95, 5.2))
    arrow(ax, (5.35, 4.8), (8.95, 4.8))
    arrow(ax, (14.05, 4.8), (17.45, 7.4))
    arrow(ax, (17.45, 4.8), (14.05, 4.6))
    ax.text(11.5, 9.0, "Memory is read before the model call and written only after a verified finish", ha="center", fontsize=10.5, fontweight="bold")
    save(fig, "memory_arch.png")


def mcp_flow():
    fig, ax = canvas(11.5, 4.0, (0, 23), (0, 8))
    box(ax, 0.3, 3.0, 6.0, 2.6, "MCP client\n(any AI app, or our agent)", BLUE, 10, True)
    box(ax, 16.7, 3.0, 6.0, 2.6, "MCP server\n(ScholarGraph: 12 read-only tools)", GREEN, 10, True)
    arrow(ax, (6.4, 5.0), (16.6, 5.0))
    arrow(ax, (16.6, 3.6), (6.4, 3.6))
    ax.text(11.5, 5.5, "initialize  ->  tools/list  ->  tools/call", ha="center", fontsize=9, color=INK)
    ax.text(11.5, 2.9, "JSON-RPC 2.0, one message per line over stdin / stdout", ha="center", fontsize=9, color=GREY, style="italic")
    ax.text(11.5, 7.2, "Model Context Protocol: one standard way for AI apps to use tools", ha="center", fontsize=11, fontweight="bold")
    save(fig, "mcp_flow.png")


def retrieval_stack():
    fig, ax = canvas(11.5, 4.4, (0, 23), (0, 9))
    stages = [("query", "rewrite / expand\nHyDE", GREY), ("first stage", "BM25 + dense\n+ RRF + graph bonus", BLUE), ("second stage", "cross-encoder or\nlate interaction", PURPLE), ("top k", "to the LLM\nwith citations", GREEN)]
    for i, (t, d, c) in enumerate(stages):
        x = 0.3 + i * 5.7
        box(ax, x, 4.2, 4.8, 2.4, t, c, 10, True)
        ax.text(x + 2.4, 3.6, d, ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.3)
        if i < 3:
            arrow(ax, (x + 4.85, 5.4), (x + 5.65, 5.4))
    ax.text(11.5, 8.0, "Fast and broad first, slow and precise on the few survivors", ha="center", fontsize=11, fontweight="bold")
    ax.text(11.5, 0.8, "contextual indexing adds year, venue, topic and concepts to every document before the first stage", ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "retrieval_stack.png")


def graphrag_global():
    fig, ax = canvas(11.5, 4.6, (0, 23), (0, 9.2))
    box(ax, 0.3, 3.6, 4.2, 2.2, "citation graph", BLUE, 10, True)
    box(ax, 5.8, 3.6, 4.4, 2.2, "communities\n(Louvain)", GREEN, 10, True)
    box(ax, 11.5, 3.6, 4.4, 2.2, "one report each\nsize, years, top papers,\nconcepts", AMBER, 9, True)
    box(ax, 17.2, 5.0, 5.5, 1.8, "map: each relevant community\nanswers from its report", PURPLE, 8.5)
    box(ax, 17.2, 2.2, 5.5, 1.8, "reduce: combine partial answers,\ncite [C#], abstain if none", RED, 8.5)
    arrow(ax, (4.55, 4.7), (5.75, 4.7))
    arrow(ax, (10.25, 4.7), (11.45, 4.7))
    arrow(ax, (15.95, 4.9), (17.15, 5.8))
    arrow(ax, (19.9, 4.95), (19.9, 4.05))
    ax.text(11.5, 8.3, "Global questions: summarise the structure first, then answer from the summaries", ha="center", fontsize=10.5, fontweight="bold")
    save(fig, "graphrag_global.png")


if __name__ == "__main__":
    for f in (system1_flow, hook_events, memory_arch, mcp_flow, retrieval_stack, graphrag_global):
        f()
    print("agent figures written to", mf.OUT)
