"""Neo4j implementation of the ``GraphStore`` interface, plus the loader that fills a Neo4j database from the gold tables.

Requires the ``neo4j`` Python driver (``pip install .[graph]``) and a running server, e.g. ``docker compose -f docker/neo4j.yml up``.
Nothing in this file is imported unless you ask for Neo4j, so the rest of the project never needs it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from .schema import CONCEPT_LOADERS, LOADERS, batches, constraint_statements
from .store import GraphStore


def connect(uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "scholargraph"):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri, auth=(user, password))


def load_graph(driver, tables: Mapping[str, pd.DataFrame], concepts: Optional[pd.DataFrame] = None, batch_size: int = 5000,
               log: Callable[[str], None] = print) -> Dict[str, int]:
    """Create constraints, then load nodes before relationships (relationship loaders MATCH existing nodes)."""
    counts: Dict[str, int] = {}
    with driver.session() as s:
        for stmt in constraint_statements():
            s.run(stmt)
        data = dict(tables)
        loaders = list(LOADERS)
        if concepts is not None and len(concepts):
            data["concepts"] = concepts
            loaders += CONCEPT_LOADERS
        for ld in loaders:
            n = 0
            for rows in batches(data, ld, batch_size):
                s.run(ld["cypher"], rows=rows)
                n += len(rows)
            counts[ld["name"]] = n
            log(f"  loaded {ld['name']}: {n:,} rows")
    return counts


class Neo4jGraphStore(GraphStore):
    def __init__(self, driver, database: Optional[str] = None):
        self.driver, self.database = driver, database

    def _q(self, cypher: str, **params) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as s:
            return [dict(r) for r in s.run(cypher, **params)]

    def stats(self) -> Dict[str, int]:
        one = lambda q: self._q(q)[0]["n"]  # noqa: E731
        return {"papers": one("MATCH (p:Paper) RETURN count(p) AS n"), "authors": one("MATCH (a:Author) RETURN count(a) AS n"),
                "citations": one("MATCH ()-[r:CITES]->() RETURN count(r) AS n"), "authorships": one("MATCH ()-[r:WROTE]->() RETURN count(r) AS n"),
                "concept_links": one("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n")}

    def paper(self, paper_id: str):
        r = self._q("MATCH (p:Paper {paper_id: $id}) OPTIONAL MATCH (p)<-[c:CITES]-() RETURN p AS p, count(c) AS k", id=paper_id)
        if not r or r[0]["p"] is None:
            return None
        d = dict(r[0]["p"])
        d["citations_in_graph"] = r[0]["k"]
        return d

    def references(self, paper_id: str) -> List[str]:
        return [r["id"] for r in self._q("MATCH (:Paper {paper_id: $id})-[:CITES]->(q:Paper) RETURN q.paper_id AS id", id=paper_id)]

    def cited_by(self, paper_id: str) -> List[str]:
        return [r["id"] for r in self._q("MATCH (q:Paper)-[:CITES]->(:Paper {paper_id: $id}) RETURN q.paper_id AS id", id=paper_id)]

    def authors_of(self, paper_id: str) -> List[Dict[str, Any]]:
        return self._q("MATCH (a:Author)-[w:WROTE]->(:Paper {paper_id: $id}) RETURN a.author_id AS author_id, a.name AS name, w.position AS position ORDER BY w.position", id=paper_id)

    def papers_of_author(self, author_id: str) -> List[str]:
        return [r["id"] for r in self._q("MATCH (:Author {author_id: $id})-[:WROTE]->(p:Paper) RETURN p.paper_id AS id", id=author_id)]

    def find_author(self, name: str) -> List[Dict[str, Any]]:
        return self._q("MATCH (a:Author) WHERE toLower(a.name) CONTAINS toLower($n) RETURN a.author_id AS author_id, a.name AS name, a.n_papers AS n_papers ORDER BY a.n_papers DESC", n=name)

    def coauthors(self, author_id: str) -> List[Tuple[str, int]]:
        rows = self._q("MATCH (a:Author {author_id: $id})-[:WROTE]->(p)<-[:WROTE]-(b:Author) WHERE b <> a RETURN b.author_id AS id, count(p) AS n ORDER BY n DESC, id", id=author_id)
        return [(r["id"], r["n"]) for r in rows]

    def concepts_of(self, paper_id: str) -> List[str]:
        return [r["c"] for r in self._q("MATCH (:Paper {paper_id: $id})-[:MENTIONS]->(c:Concept) RETURN c.name AS c", id=paper_id)]

    def papers_with_concept(self, concept: str, year_from: Optional[int] = None) -> List[str]:
        return [r["id"] for r in self._q("MATCH (p:Paper)-[:MENTIONS]->(:Concept {name: $c}) WHERE $y IS NULL OR p.year >= $y RETURN p.paper_id AS id", c=concept.lower(), y=year_from)]

    def co_cited(self, a: str, b: str) -> List[str]:
        return [r["id"] for r in self._q("MATCH (q:Paper)-[:CITES]->(:Paper {paper_id: $a}), (q)-[:CITES]->(:Paper {paper_id: $b}) RETURN DISTINCT q.paper_id AS id ORDER BY id", a=a, b=b)]

    def shortest_path(self, a: str, b: str, max_hops: int = 4) -> Optional[List[str]]:
        if a == b:
            return [a]
        rows = self._q(f"MATCH (x:Paper {{paper_id: $a}}), (y:Paper {{paper_id: $b}}) MATCH path = shortestPath((x)-[:CITES*..{int(max_hops)}]-(y)) "
                       "RETURN [n IN nodes(path) | n.paper_id] AS ids", a=a, b=b)
        return rows[0]["ids"] if rows else None

    def top_cited(self, k: int = 10, year: Optional[int] = None, topic: Optional[str] = None) -> List[str]:
        rows = self._q("MATCH (p:Paper) WHERE ($y IS NULL OR p.year = $y) AND ($t IS NULL OR EXISTS { (p)-[:ABOUT]->(:Topic {name: $t}) }) "
                       "OPTIONAL MATCH (p)<-[c:CITES]-() WITH p, count(c) AS n ORDER BY n DESC, p.paper_id LIMIT $k RETURN p.paper_id AS id", y=year, t=topic, k=k)
        return [r["id"] for r in rows]

    def find_papers_by_title(self, text: str, k: int = 5) -> List[str]:
        words = [w for w in text.lower().split() if len(w) > 2]
        if not words:
            return []
        rows = self._q("MATCH (p:Paper) WITH p, [w IN $words WHERE toLower(p.title) CONTAINS w] AS hit WHERE size(hit) > 0 "
                       "RETURN p.paper_id AS id ORDER BY size(hit) DESC, id LIMIT $k", words=words, k=k)
        return [r["id"] for r in rows]
