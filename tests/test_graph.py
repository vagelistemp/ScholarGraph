import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from scholargraph.graph.concepts import (LLMConceptExtractor, RuleConceptExtractor, agreement, extract_all, normalise_phrase)
from scholargraph.graph.neo4j_store import load_graph
from scholargraph.graph.schema import CONCEPT_LOADERS, LOADERS, NODE_KEYS, batches, constraint_statements
from scholargraph.graph.store import MemoryGraphStore
from scholargraph.ingest.synthetic import make_synthetic_works
from scholargraph.ingest.transform import normalise_work
from scholargraph.lake.pandas_pipeline import build_gold
from scholargraph.llm import FakeLLM, parse_json_object


def gold(n=250, seed=4):
    raw = make_synthetic_works(n, seed)
    works = [normalise_work(r) for r in raw]
    return build_gold(iter(works), 2015, 2025), raw


class StoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables, cls.raw = gold()
        cls.store = MemoryGraphStore(cls.tables)
        cls.cit = cls.tables["citations"]

    def test_references_and_cited_by_are_inverse(self):
        s = self.store
        for pid in s.ids[:40]:
            for q in s.references(pid):
                self.assertIn(pid, s.cited_by(q))
        self.assertEqual(sum(len(s.references(p)) for p in s.ids), len(self.cit))

    def test_references_match_table(self):
        pid = self.cit["citing_id"].iloc[0]
        self.assertEqual(sorted(self.store.references(pid)), sorted(self.cit[self.cit["citing_id"] == pid]["cited_id"]))

    def test_authors_and_coauthors(self):
        s, ash = self.store, self.tables["authorships"]
        pid = ash["paper_id"].iloc[0]
        self.assertEqual({a["author_id"] for a in s.authors_of(pid)}, set(ash[ash["paper_id"] == pid]["author_id"]))
        aid = ash["author_id"].iloc[0]
        want = {}
        for p in s.papers_of_author(aid):
            for a in ash[ash["paper_id"] == p]["author_id"]:
                if a != aid:
                    want[a] = want.get(a, 0) + 1
        self.assertEqual(dict(s.coauthors(aid)), want)
        self.assertEqual(s.find_author("author 1")[0]["name"].lower().startswith("author 1"), True)

    def test_co_cited_is_set_intersection(self):
        s = self.store
        a, b = s.top_cited(2)
        self.assertEqual(s.co_cited(a, b), sorted(set(s.cited_by(a)) & set(s.cited_by(b))))

    def test_shortest_path_matches_networkx(self):
        s = self.store
        g = nx.Graph()
        g.add_edges_from(zip(self.cit["citing_id"], self.cit["cited_id"]))
        rng = np.random.default_rng(0)
        nodes = list(g.nodes)
        for _ in range(25):
            a, b = rng.choice(nodes, 2, replace=False)
            path = s.shortest_path(a, b, max_hops=6)
            want = nx.shortest_path_length(g, a, b) if nx.has_path(g, a, b) else None
            if want is None or want > 6:
                self.assertIsNone(path)
            else:
                self.assertEqual(len(path) - 1, want)
                for u, v in zip(path, path[1:]):
                    self.assertTrue(g.has_edge(u, v))
        self.assertEqual(s.shortest_path(nodes[0], nodes[0]), [nodes[0]])

    def test_top_cited_is_sorted_and_filtered(self):
        s = self.store
        top = s.top_cited(10)
        counts = [len(s.cited_by(p)) for p in top]
        self.assertEqual(counts, sorted(counts, reverse=True))
        y = int(self.tables["papers"]["year"].mode()[0])
        self.assertTrue(all(s.paper(p)["year"] == y for p in s.top_cited(5, year=y)))

    def test_paper_lookup_and_title_search(self):
        s = self.store
        pid = s.ids[7]
        self.assertEqual(s.paper(pid)["paper_id"], pid)
        self.assertIsNone(s.paper("W0"))
        self.assertEqual(s.find_papers_by_title(s.paper(pid)["title"], 3)[0], pid)

    def test_concepts_in_store(self):
        c = pd.DataFrame({"paper_id": [self.store.ids[0], self.store.ids[1]], "concept": ["transformer", "transformer"], "kind": ["method"] * 2, "source": ["t"] * 2})
        s = MemoryGraphStore(self.tables, c)
        self.assertEqual(s.concepts_of(self.store.ids[0]), ["transformer"])
        self.assertEqual(set(s.papers_with_concept("Transformer")), set(c["paper_id"]))
        self.assertEqual(s.stats()["concept_links"], 2)


class Neo4jLoaderTest(unittest.TestCase):
    """The loader is tested with a recording fake driver (a real Neo4j server is not needed)."""

    class Session:
        def __init__(self, log): self.log = log
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, cypher, **params): self.log.append((cypher, params))

    class Driver:
        def __init__(self): self.log = []
        def session(self, **kw): return Neo4jLoaderTest.Session(self.log)

    def test_constraints_cover_every_label(self):
        stm = constraint_statements()
        self.assertEqual(len(stm), len(NODE_KEYS))
        self.assertTrue(all("IF NOT EXISTS" in s for s in stm))

    def test_nodes_are_loaded_before_relationships(self):
        tables, _ = gold(120, 5)
        conc = pd.DataFrame({"paper_id": tables["papers"]["paper_id"][:5], "concept": ["transformer"] * 5, "kind": ["method"] * 5, "source": ["rules"] * 5})
        d = self.Driver()
        counts = load_graph(d, tables, conc, batch_size=50, log=lambda *a: None)
        order = list(counts)                                              # loaders run in this order
        needs = {"cites": ["papers"], "wrote": ["authors", "papers"], "affiliated": ["authors", "institutions"],
                 "published_in": ["papers", "venues"], "about": ["papers", "topics"], "mentions": ["papers", "concepts"]}
        for rel, nodes in needs.items():
            for n in nodes:
                self.assertLess(order.index(n), order.index(rel), f"{n} must be loaded before {rel}")
        self.assertEqual(counts["papers"], len(tables["papers"]))
        self.assertEqual(counts["cites"], len(tables["citations"]))
        self.assertEqual(counts["mentions"], 5)
        sent = [len(p["rows"]) for c, p in d.log if "rows" in p]
        self.assertLessEqual(max(sent), 50)

    def test_batches_drop_nulls_and_convert_numpy(self):
        tables, _ = gold(80, 6)
        loader = next(l for l in LOADERS if l["name"] == "published_in")
        rows = [r for b in batches(tables, loader, 1000) for r in b]
        self.assertTrue(all(r["venue_id"] is not None for r in rows))
        json.dumps(rows)                                                                   # plain Python types only
        self.assertEqual([l["name"] for l in CONCEPT_LOADERS], ["concepts", "mentions"])


class ConceptTest(unittest.TestCase):
    def test_rules_find_longest_phrase_and_aliases(self):
        ex = RuleConceptExtractor()
        got = dict(ex.extract("We train a Graph Neural Network (GNN) for link prediction on Cora and ImageNet with LoRA."))
        self.assertEqual(got["graph neural network"], "method")
        self.assertEqual(got["link prediction"], "task")
        self.assertEqual(got["cora"], "dataset")
        self.assertEqual(got["low-rank adaptation"], "method")
        self.assertNotIn("neural network", got)

    def test_rules_do_not_match_inside_words(self):
        self.assertEqual(RuleConceptExtractor().extract("The organisation of nanotransformers and scoranking."), [])

    def test_rules_recover_planted_synthetic_concepts(self):
        ex = RuleConceptExtractor()
        for r in make_synthetic_works(60, 1)[:60]:
            w = normalise_work(r)
            found = {c for c, _ in ex.extract(w["title"] + ". " + w["abstract"])}
            for kw in w["keywords"]:
                self.assertIn(kw["keyword"], found)

    def test_llm_extractor_drops_hallucinations_and_maps_aliases(self):
        reply = '```json\n{"methods": ["gnn", "quantum annealing"], "tasks": ["Link Prediction"], "datasets": ["Cora", "madeup-set"]}\n```'
        ex = LLMConceptExtractor(FakeLLM([reply]))
        got = dict(ex.extract("A GNN approach to link prediction evaluated on the Cora benchmark."))
        self.assertEqual(got, {"graph neural network": "method", "link prediction": "task", "cora": "dataset"})

    def test_llm_extractor_survives_garbage(self):
        self.assertEqual(LLMConceptExtractor(FakeLLM(["I cannot help with that"])).extract("some text about transformers"), [])
        self.assertEqual(LLMConceptExtractor(FakeLLM(['{"methods": "transformer"}'])).extract("transformer text"), [])

    def test_parse_json_object_variants(self):
        self.assertEqual(parse_json_object('Sure! {"a": 1} hope that helps'), {"a": 1})
        self.assertEqual(parse_json_object('{"a": "}"}'), {"a": "}"})
        self.assertIsNone(parse_json_object("no json"))

    def test_extract_all_is_resumable(self):
        tables, _ = gold(60, 7)
        papers = tables["papers"].head(30)
        llm = FakeLLM(fn=lambda m: '{"methods": ["transformer"], "tasks": [], "datasets": []}')
        ex = LLMConceptExtractor(llm)
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "c.jsonl"
            extract_all(papers, ex, cache, limit=10, log=lambda *a: None)
            self.assertEqual(len(llm.calls), 10)
            out = extract_all(papers, ex, cache, log=lambda *a: None)
            self.assertEqual(len(llm.calls), 30)                      # only the 20 missing papers were sent to the model
            extract_all(papers, ex, cache, log=lambda *a: None)
            self.assertEqual(len(llm.calls), 30)                      # fully cached now
            self.assertEqual(set(out.columns), {"paper_id", "concept", "kind", "source"})

    def test_agreement_metric(self):
        a = pd.DataFrame({"paper_id": ["1", "1", "2"], "concept": ["x", "y", "x"]})
        b = pd.DataFrame({"paper_id": ["1", "2", "2"], "concept": ["x", "x", "z"]})
        m = agreement(a, b)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertEqual(normalise_phrase("  Graph-Neural   Networks! "), "graph-neural networks")


if __name__ == "__main__":
    unittest.main()
