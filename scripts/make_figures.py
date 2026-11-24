"""Draw the explanatory diagrams used in the PDFs (matplotlib only).   python scripts/make_figures.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, GREY, BLUE, GREEN, RED, AMBER, PURPLE = "#1F2933", "#8895A4", "#2B6CB0", "#2F855A", "#C53030", "#B7791F", "#6B46C1"
FILL = {BLUE: "#E4EEF8", GREEN: "#E1F2E8", RED: "#FBE3E3", AMBER: "#FAF0D9", PURPLE: "#EBE4F7", GREY: "#EEF1F4"}


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=FILL[color], ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p, q, color=INK, rad=0.0, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas(11, 4.2, (0, 22), (0, 8.4))
    titles = ["1  Lakehouse", "2  Knowledge graph", "3  Graph neural\nnetwork", "4  Fine-tuning", "5  GraphRAG agent"]
    subs = ["OpenAlex download\nSpark + Delta Lake\nbronze / silver / gold", "Neo4j or in-memory\nconcept extraction\n(rules and LLM)", "temporal split\nGNN vs boosted trees", "contrastive encoder\nLoRA extractor", "hybrid retrieval\ntools, guardrails\nAPI + evaluation"]
    cols = [BLUE, GREEN, AMBER, PURPLE, RED]
    for i, (t, s, c) in enumerate(zip(titles, subs, cols)):
        x = 0.3 + i * 4.3
        box(ax, x, 4.4, 3.8, 2.6, t, c, 10, True)
        box(ax, x, 1.2, 3.8, 2.6, s, c, 8)
        if i < 4:
            arrow(ax, (x + 3.85, 5.7), (x + 4.25, 5.7))
    ax.text(11, 0.3, "gold Parquet tables (data contract) feed every later phase; each phase has its own tests", ha="center", fontsize=8.5, color=GREY, style="italic")
    ax.text(11, 7.9, "ScholarGraph: five phases, one data contract", ha="center", fontsize=12, fontweight="bold", color=INK)
    save(fig, "architecture.png")


def medallion():
    fig, ax = canvas(11, 3.6, (0, 22), (0, 7))
    spec = [("BRONZE", "raw API pages\n(gzip JSON lines)\nnever modified", "#B7791F", AMBER), ("SILVER", "parsed and flattened\nworks, refs,\nauthorships, topics\n(Delta Lake tables)", "#718096", GREY),
            ("GOLD", "curated tables\npapers, citations,\nauthors, ...\n(Parquet, the contract)", "#B7791F", AMBER)]
    for i, (n, d, ec, _) in enumerate(spec):
        x = 0.4 + i * 6.2
        color = [AMBER, GREY, BLUE][i]
        box(ax, x, 1.0, 5.2, 4.2, "", color)
        ax.text(x + 2.6, 4.5, n, ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + 2.6, 2.7, d, ha="center", va="center", fontsize=8.5, color=INK, linespacing=1.35)
        if i < 2:
            arrow(ax, (x + 5.25, 3.1), (x + 6.15, 3.1))
    ax.text(5.85, 5.75, "parse", ha="center", fontsize=8, color=GREY); ax.text(12.05, 5.75, "deduplicate, join, validate", ha="center", fontsize=8, color=GREY)
    arrow(ax, (19.1, 3.1), (21.4, 3.1)); ax.text(20.4, 3.6, "phases\n2 - 5", ha="center", fontsize=8.5, color=INK)
    save(fig, "medallion.png")


def kg_schema():
    fig, ax = canvas(9, 5.2, (0, 18), (0, 10.4))
    nodes = {"Author": (3, 5.2, GREEN), "Paper": (9, 5.2, BLUE), "Concept": (15, 7.6, PURPLE), "Topic": (15, 5.2, AMBER), "Venue": (15, 2.8, RED), "Institution": (3, 1.4, GREY)}
    for n, (x, y, c) in nodes.items():
        ax.add_patch(Ellipse((x, y), 3.6, 1.7, fc=FILL[c], ec=c, lw=1.6))
        ax.text(x, y, n, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    arrow(ax, (4.85, 5.2), (7.15, 5.2), GREEN); ax.text(6, 5.55, "WROTE", ha="center", fontsize=8, color=GREEN)
    ax.add_patch(Ellipse((9, 8.4), 3.0, 1.3, fc="white", ec=BLUE, lw=1.4, ls="--")); ax.text(9, 8.4, "another Paper", ha="center", va="center", fontsize=8.5)
    arrow(ax, (9, 6.1), (9, 7.7), BLUE); ax.text(9.25, 6.9, "CITES", fontsize=8.5, color=BLUE)
    arrow(ax, (10.85, 5.9), (13.2, 7.3), PURPLE); ax.text(11.2, 6.75, "MENTIONS", fontsize=8, color=PURPLE, rotation=33)
    arrow(ax, (10.85, 5.2), (13.15, 5.2), AMBER); ax.text(12, 5.5, "ABOUT", ha="center", fontsize=8, color=AMBER)
    arrow(ax, (10.85, 4.5), (13.2, 3.1), RED); ax.text(11.0, 3.65, "PUBLISHED_IN", fontsize=8, color=RED, rotation=-33)
    arrow(ax, (3, 4.3), (3, 2.3), GREY); ax.text(3.3, 3.3, "AFFILIATED_WITH", fontsize=8, color=GREY)
    ax.text(9, 10.0, "The property-graph model (Neo4j labels and relationship types)", ha="center", fontsize=11, fontweight="bold", color=INK)
    save(fig, "kg_schema.png")


def temporal_split():
    fig, ax = canvas(11, 4.6, (2013.8, 2026.2), (0, 9.4))
    ax.add_patch(FancyBboxPatch((2015, 6.3), 7.95, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc=FILL[BLUE], ec=BLUE, lw=1.4))
    ax.add_patch(FancyBboxPatch((2023, 6.3), 0.95, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc=FILL[AMBER], ec=AMBER, lw=1.4))
    ax.add_patch(FancyBboxPatch((2024, 6.3), 2.0, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc=FILL[RED], ec=RED, lw=1.4))
    ax.text(2019, 6.85, "train: citations made by papers up to 2022", ha="center", va="center", fontsize=9)
    ax.text(2023.5, 6.85, "val", ha="center", va="center", fontsize=9); ax.text(2025, 6.85, "test: 2024 - 2025", ha="center", va="center", fontsize=9)
    for y in range(2015, 2026):
        ax.plot([y, y], [6.1, 6.3], color=GREY, lw=0.8); ax.text(y + 0.5, 5.7, str(y), ha="center", fontsize=7.5, color=GREY)
    ax.text(2019, 8.3, "Split by the CITING paper's year: the future never leaks into the past", ha="center", fontsize=11, fontweight="bold")
    ax.text(2019, 4.6, "Inside training, every epoch hides a random 20% of the training papers:", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 2014.4, 1.0, 3.6, 2.8, "message graph\nkeeps the references of\nthe other 80% of papers", BLUE, 8)
    box(ax, 2018.6, 1.0, 3.6, 2.8, "hidden papers: ALL their\nreferences are removed\nand become the targets", RED, 8)
    box(ax, 2022.8, 1.0, 3.2, 2.8, "so they look like the\nbrand-new papers of\nthe test years", GREEN, 8)
    arrow(ax, (2018.05, 2.4), (2018.55, 2.4)); arrow(ax, (2022.25, 2.4), (2022.75, 2.4))
    save(fig, "temporal_split.png")


def gnn_message():
    fig, ax = canvas(10, 4.9, (0, 20), (-1.6, 8.8))
    ax.add_patch(Ellipse((10, 4.2), 3.2, 1.8, fc=FILL[BLUE], ec=BLUE, lw=2)); ax.text(10, 4.2, "paper i", ha="center", va="center", fontsize=11, fontweight="bold")
    refs = [(3, 6.8), (3, 4.2), (3, 1.6)]; cits = [(17, 6.8), (17, 4.2), (17, 1.6)]
    for (x, y) in refs:
        ax.add_patch(Ellipse((x, y), 2.6, 1.3, fc=FILL[GREEN], ec=GREEN, lw=1.4)); ax.text(x, y, "cited by i", ha="center", va="center", fontsize=8.5)
        arrow(ax, (x + 1.35, y - 0.05 if y != 4.2 else y), (8.35, 4.2 + (y - 4.2) * 0.25), GREEN)
    for (x, y) in cits:
        ax.add_patch(Ellipse((x, y), 2.6, 1.3, fc=FILL[AMBER], ec=AMBER, lw=1.4)); ax.text(x, y, "cites i", ha="center", va="center", fontsize=8.5)
        arrow(ax, (x - 1.35, y), (11.65, 4.2 + (y - 4.2) * 0.25), AMBER)
    ax.text(5.8, 7.9, "mean of the papers i cites\n(its intellectual ancestors)", ha="center", fontsize=8.5, color=GREEN)
    ax.text(14.2, 7.9, "mean of the papers that cite i\n(its impact)", ha="center", fontsize=8.5, color=AMBER)
    ax.text(10, 0.0, "new vector of i  =  ReLU( W_self h_i  +  W_ref mean(refs)  +  W_cit mean(citers) )", ha="center", fontsize=9, color=INK, family="monospace")
    ax.text(10, -0.9, "stack two layers and every paper also sees its neighbours' neighbours", ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "gnn_message.png")


def contrastive():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    rng = np.random.default_rng(3)
    S = rng.uniform(-0.2, 0.35, (5, 5)); np.fill_diagonal(S, rng.uniform(0.75, 0.95, 5))
    ax = axes[0]
    ax.imshow(S, cmap="Blues", vmin=-0.3, vmax=1.0)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if S[i, j] > 0.55 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels([f"cited {j + 1}" for j in range(5)], fontsize=7.5, rotation=30)
    ax.set_yticklabels([f"paper {i + 1}" for i in range(5)], fontsize=7.5)
    ax.set_title("similarity of each paper to each cited paper in the batch", fontsize=8.5)
    ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 9.0, "InfoNCE loss", fontsize=11, fontweight="bold")
    ax.text(0, 7.4, "Row i: softmax over the row,\nthe diagonal entry is the right answer.", fontsize=9, va="top")
    ax.text(0, 5.0, "loss = - log  exp(s_ii / T)\n            -----------------\n             sum_j exp(s_ij / T)", fontsize=9, family="monospace", va="top")
    ax.text(0, 1.6, "Pull the diagonal up, push the rest\ndown. T = 0.05 sharpens the contrast.\nTwo papers citing the same target\nare masked out (false negatives).", fontsize=8.5, va="top", color=GREY)
    fig.tight_layout()
    save(fig, "contrastive.png")


def lora():
    fig, ax = canvas(9.6, 3.6, (0, 20), (0, 7.4))
    box(ax, 0.5, 1.6, 5.0, 4.2, "W  (frozen)\nd x d\nmillions of weights", GREY, 9.5, True)
    ax.text(6.6, 3.7, "+", fontsize=20, ha="center", va="center")
    box(ax, 7.8, 1.6, 1.7, 4.2, "B\nd x r", PURPLE, 9.5, True); ax.text(10.0, 3.7, "x", fontsize=14, ha="center", va="center")
    box(ax, 10.6, 3.3, 4.6, 1.4, "A   r x d", PURPLE, 9.5, True)
    ax.text(16.4, 3.7, "r = 16,  d = 896", fontsize=9, color=INK, va="center")
    ax.text(10, 6.8, "LoRA: learn a low-rank update  W' = W + (alpha / r) * B A", ha="center", fontsize=11, fontweight="bold")
    ax.text(10, 0.5, "d x d = 802,816 weights per matrix   versus   2 x d x r = 28,672 trainable weights (3.6%)", ha="center", fontsize=9, color=GREY, style="italic")
    save(fig, "lora.png")


def rag_agent():
    fig, ax = canvas(11.5, 5.6, (0, 23), (0, 11.2))
    ax.text(5.5, 10.6, "GraphRAG: answer from evidence", ha="center", fontsize=11, fontweight="bold")
    steps = [("question", BLUE), ("BM25 + dense\n+ graph bonus", GREEN), ("numbered sources\n+ graph facts", AMBER), ("LLM answers\nwith [P1] markers", PURPLE), ("guardrails:\ncitations, grounding,\nabstain", RED)]
    for i, (t, c) in enumerate(steps):
        box(ax, 0.2, 8.4 - i * 1.95, 5.6, 1.45, t, c, 8.5)
        if i < 4:
            arrow(ax, (3.0, 8.38 - i * 1.95), (3.0, 7.88 - i * 1.95))
    ax.text(17.2, 10.6, "Agent: decide, call a tool, observe, repeat", ha="center", fontsize=11, fontweight="bold")
    box(ax, 11.2, 7.6, 5.0, 1.8, "LLM replies with\nONE JSON object", PURPLE, 8.5)
    box(ax, 18.2, 7.6, 4.4, 1.8, "tool call\n{tool, args}", GREEN, 8.5)
    box(ax, 18.2, 4.3, 4.4, 1.8, "run on the graph\n(12 tools)", AMBER, 8.5)
    box(ax, 11.2, 4.3, 5.0, 1.8, "OBSERVATION\nappended to the chat", BLUE, 8.5)
    box(ax, 11.2, 0.9, 11.4, 1.8, "{final, items}   ->   scored against the graph-derived answer", RED, 8.5)
    arrow(ax, (16.25, 8.5), (18.15, 8.5)); arrow(ax, (20.4, 7.55), (20.4, 6.15)); arrow(ax, (18.15, 5.2), (16.25, 5.2)); arrow(ax, (14.2, 6.15), (14.2, 7.55))
    arrow(ax, (11.7, 7.55), (11.7, 2.75), RED, rad=0.0)
    ax.text(11.5, 5.2, "when it\nfinishes", fontsize=8, color=RED, ha="right", va="center")
    save(fig, "rag_agent.png")


if __name__ == "__main__":
    for f in (architecture, medallion, kg_schema, temporal_split, gnn_message, contrastive, lora, rag_agent):
        f()
    print("figures written to", OUT)
