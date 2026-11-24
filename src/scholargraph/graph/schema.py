"""The property-graph schema in Neo4j terms: labels, relationship types, constraints and the Cypher used to load the gold tables."""
from __future__ import annotations

from typing import Dict, Iterator, List, Mapping

import pandas as pd

NODE_KEYS = {"Paper": "paper_id", "Author": "author_id", "Institution": "institution_id", "Venue": "venue_id", "Topic": "topic_id", "Concept": "name"}
RELATIONSHIPS = ["CITES", "WROTE", "AFFILIATED_WITH", "PUBLISHED_IN", "ABOUT", "MENTIONS"]


def constraint_statements() -> List[str]:
    """Uniqueness constraints (they also create the indexes that make MERGE and lookups fast)."""
    return [f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE" for label, key in NODE_KEYS.items()]


# Each loader is (cypher, table name, column list). Rows are sent in batches as the $rows parameter.
LOADERS: List[Dict] = [
    {"name": "papers", "table": "papers", "cols": ["paper_id", "title", "abstract", "year", "doi", "type", "cited_by_count"],
     "cypher": "UNWIND $rows AS r MERGE (p:Paper {paper_id: r.paper_id}) SET p.title = r.title, p.abstract = r.abstract, p.year = r.year, "
               "p.doi = r.doi, p.type = r.type, p.cited_by_count = r.cited_by_count"},
    {"name": "authors", "table": "authors", "cols": ["author_id", "name", "n_papers"],
     "cypher": "UNWIND $rows AS r MERGE (a:Author {author_id: r.author_id}) SET a.name = r.name, a.n_papers = r.n_papers"},
    {"name": "institutions", "table": "institutions", "cols": ["institution_id", "name", "country"],
     "cypher": "UNWIND $rows AS r MERGE (i:Institution {institution_id: r.institution_id}) SET i.name = r.name, i.country = r.country"},
    {"name": "venues", "table": "venues", "cols": ["venue_id", "name", "type"],
     "cypher": "UNWIND $rows AS r MERGE (v:Venue {venue_id: r.venue_id}) SET v.name = r.name, v.type = r.type"},
    {"name": "topics", "table": "topics", "cols": ["topic_id", "name", "subfield", "field"],
     "cypher": "UNWIND $rows AS r MERGE (t:Topic {topic_id: r.topic_id}) SET t.name = r.name, t.subfield = r.subfield, t.field = r.field"},
    {"name": "cites", "table": "citations", "cols": ["citing_id", "cited_id"],
     "cypher": "UNWIND $rows AS r MATCH (a:Paper {paper_id: r.citing_id}), (b:Paper {paper_id: r.cited_id}) MERGE (a)-[:CITES]->(b)"},
    {"name": "wrote", "table": "authorships", "cols": ["author_id", "paper_id", "position"],
     "cypher": "UNWIND $rows AS r MATCH (a:Author {author_id: r.author_id}), (p:Paper {paper_id: r.paper_id}) "
               "MERGE (a)-[w:WROTE]->(p) SET w.position = r.position"},
    {"name": "affiliated", "table": "authorships", "cols": ["author_id", "institution_id"], "dropna": ["institution_id"], "distinct": True,
     "cypher": "UNWIND $rows AS r MATCH (a:Author {author_id: r.author_id}), (i:Institution {institution_id: r.institution_id}) "
               "MERGE (a)-[:AFFILIATED_WITH]->(i)"},
    {"name": "published_in", "table": "papers", "cols": ["paper_id", "venue_id"], "dropna": ["venue_id"],
     "cypher": "UNWIND $rows AS r MATCH (p:Paper {paper_id: r.paper_id}), (v:Venue {venue_id: r.venue_id}) MERGE (p)-[:PUBLISHED_IN]->(v)"},
    {"name": "about", "table": "paper_topics", "cols": ["paper_id", "topic_id", "score"],
     "cypher": "UNWIND $rows AS r MATCH (p:Paper {paper_id: r.paper_id}), (t:Topic {topic_id: r.topic_id}) MERGE (p)-[a:ABOUT]->(t) SET a.score = r.score"},
]
CONCEPT_LOADERS: List[Dict] = [
    {"name": "concepts", "table": "concepts", "cols": ["concept", "kind"], "distinct": True, "rename": {"concept": "name"},
     "cypher": "UNWIND $rows AS r MERGE (c:Concept {name: r.name}) SET c.kind = r.kind"},
    {"name": "mentions", "table": "concepts", "cols": ["paper_id", "concept", "source"], "rename": {"concept": "name"},
     "cypher": "UNWIND $rows AS r MATCH (p:Paper {paper_id: r.paper_id}), (c:Concept {name: r.name}) MERGE (p)-[m:MENTIONS]->(c) SET m.source = r.source"},
]


def _py(v):
    if isinstance(v, float) and v != v:
        return None
    return v.item() if hasattr(v, "item") else v


def batches(tables: Mapping[str, pd.DataFrame], loader: Dict, size: int = 5000) -> Iterator[List[dict]]:
    df = tables[loader["table"]][loader["cols"]]
    if loader.get("dropna"):
        df = df.dropna(subset=loader["dropna"])
    if loader.get("distinct"):
        df = df.drop_duplicates()
    if loader.get("rename"):
        df = df.rename(columns=loader["rename"])
    records = df.to_dict("records")
    for i in range(0, len(records), size):
        yield [{k: _py(v) for k, v in r.items()} for r in records[i:i + size]]
