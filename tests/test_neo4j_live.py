"""Live parity test: the Neo4j store must answer every GraphStore method like the in-memory store.

Skipped unless a server is reachable: start one with ``docker compose -f docker/neo4j.yml up -d`` and run with
``NEO4J_URI=bolt://localhost:7687 python -m unittest tests.test_neo4j_live``.
"""
import os
import unittest

import numpy as np

from scholargraph.demo import make_demo_concepts, make_demo_gold
from scholargraph.graph.store import MemoryGraphStore

URI = os.environ.get("NEO4J_URI")


@unittest.skipUnless(URI, "set NEO4J_URI to run the live Neo4j parity test")
class LiveParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scholargraph.graph.neo4j_store import Neo4jGraphStore, connect, load_graph
        cls.gold = make_demo_gold(220, 8)
        cls.concepts = make_demo_concepts(cls.gold)
        cls.driver = connect(URI, os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "scholargraph"))
        with cls.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")                              # a clean database for the test
        cls.counts = load_graph(cls.driver, cls.gold, cls.concepts, batch_size=500, log=lambda *a: None)
        cls.mem = MemoryGraphStore(cls.gold, cls.concepts)
        cls.neo = Neo4jGraphStore(cls.driver)
        cls.ids = cls.mem.ids

    @classmethod
    def tearDownClass(cls):
        with cls.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        cls.driver.close()

    def test_loaded_counts_and_stats(self):
        self.assertEqual(self.neo.stats(), self.mem.stats())
        self.assertEqual(self.counts["papers"], len(self.gold["papers"]))

    def test_paper_references_citers(self):
        for pid in self.ids[:60]:
            m, n = self.mem.paper(pid), self.neo.paper(pid)
            for key in ("title", "year", "doi", "citations_in_graph"):
                self.assertEqual(m[key], n[key], (pid, key))
            self.assertEqual(set(self.mem.references(pid)), set(self.neo.references(pid)))
            self.assertEqual(set(self.mem.cited_by(pid)), set(self.neo.cited_by(pid)))
        self.assertIsNone(self.neo.paper("W0"))

    def test_authors_and_coauthors(self):
        for pid in self.ids[:40]:
            m = [(a["author_id"], a["name"], a["position"]) for a in self.mem.authors_of(pid)]
            n = [(a["author_id"], a["name"], a["position"]) for a in self.neo.authors_of(pid)]
            self.assertEqual(sorted(m), sorted(n), pid)
        for aid in list(self.mem.authors.index[:25]):
            self.assertEqual(set(self.mem.papers_of_author(aid)), set(self.neo.papers_of_author(aid)))
            self.assertEqual(self.mem.author_name(aid), self.neo.author_name(aid))
            self.assertEqual(dict(self.mem.coauthors(aid)), dict(self.neo.coauthors(aid)))
        self.assertEqual({a["author_id"] for a in self.mem.find_author("author 1")}, {a["author_id"] for a in self.neo.find_author("author 1")})

    def test_concepts(self):
        for pid in self.ids[:40]:
            self.assertEqual(set(self.mem.concepts_of(pid)), set(self.neo.concepts_of(pid)))
        for c in sorted(self.mem._by_concept)[:6]:
            for y in (None, 2020):
                self.assertEqual(set(self.mem.papers_with_concept(c, y)), set(self.neo.papers_with_concept(c, y)), (c, y))

    def test_co_citation_and_shortest_paths(self):
        top = self.mem.top_cited(12)
        for a, b in zip(top[:-1], top[1:]):
            self.assertEqual(set(self.mem.co_cited(a, b)), set(self.neo.co_cited(a, b)))
        rng = np.random.default_rng(0)
        for _ in range(15):
            a, b = rng.choice(self.ids, 2, replace=False)
            pm, pn = self.mem.shortest_path(a, b, 3), self.neo.shortest_path(a, b, 3)
            self.assertEqual(pm is None, pn is None, (a, b))
            if pm is not None:
                self.assertEqual(len(pm), len(pn))                          # equal length (several shortest paths may exist)
                for u, v in zip(pn, pn[1:]):
                    self.assertTrue(v in self.mem.references(u) or u in self.mem.references(v))

    def test_top_cited_and_title_search(self):
        for year in (None, 2020):
            m, n = self.mem.top_cited(10, year), self.neo.top_cited(10, year)
            cm = [len(self.mem.cited_by(p)) for p in m]
            cn = [len(self.mem.cited_by(p)) for p in n]
            self.assertEqual(cm, cn)                                        # same citation counts in the same order (ties may be ordered differently)
        for pid in self.ids[:10]:
            title = self.mem.paper(pid)["title"]
            self.assertIn(pid, self.neo.find_papers_by_title(title, 5))


if __name__ == "__main__":
    unittest.main()
