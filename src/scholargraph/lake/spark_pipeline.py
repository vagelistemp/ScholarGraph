"""Bronze -> silver -> gold with Apache Spark (and Delta Lake for silver).

Medallion layers
----------------
* **bronze**  raw API pages (see ``ingest/openalex.py``).
* **silver**  parsed and flattened, but not yet curated: ``works``, ``refs``, ``authorships``, ``topics``, ``keywords``. Stored as
              **Delta Lake** tables when ``delta-spark`` is installed (ACID writes, schema enforcement, time travel, MERGE), else Parquet.
* **gold**    the curated tables of ``schemas.GOLD`` written as plain Parquet, ready for pandas.

Parsing happens in Python inside Spark partitions with the same ``normalise_work`` function the pandas pipeline uses; the heavy
relational part (deduplication with window functions, the 12-million-row citation join, aggregations) is Spark SQL.

Windows note: Spark needs the Hadoop ``winutils`` helper to write files on native Windows. The easiest fix is to run in Docker, WSL
or on a Mac / Linux machine. Without Spark, use ``--engine pandas``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..ingest.transform import MIN_ABSTRACT_CHARS, iter_bronze_files, normalise_work


def delta_available() -> bool:
    try:
        import delta  # noqa: F401
        return True
    except ImportError:
        return False


def spark_session(app: str = "scholargraph", master: str = "local[*]", driver_memory: str = "4g", use_delta: Optional[bool] = None):
    from pyspark.sql import SparkSession
    use_delta = delta_available() if use_delta is None else use_delta
    b = (SparkSession.builder.appName(app).master(master).config("spark.driver.memory", driver_memory)
         .config("spark.sql.shuffle.partitions", "16").config("spark.sql.session.timeZone", "UTC")
         .config("spark.ui.showConsoleProgress", "false"))
    if use_delta:
        from delta import configure_spark_with_delta_pip
        b = (b.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
             .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
        return configure_spark_with_delta_pip(b).getOrCreate()
    return b.getOrCreate()


def _schemas():
    from pyspark.sql.types import (ArrayType, DoubleType, IntegerType, LongType, StringType, StructField, StructType)
    S, L, I, D = StringType(), LongType(), IntegerType(), DoubleType()
    f = lambda n, t: StructField(n, t, True)  # noqa: E731
    return {
        "works": StructType([f("paper_id", S), f("title", S), f("abstract", S), f("year", I), f("doi", S), f("type", S), f("language", S),
                             f("cited_by_count", L), f("venue_id", S), f("venue_name", S), f("venue_type", S), f("primary_topic_id", S),
                             f("primary_topic_name", S), f("n_authors", L), f("n_references", L)]),
        "refs": StructType([f("paper_id", S), f("ref_id", S)]),
        "authorships": StructType([f("paper_id", S), f("author_id", S), f("name", S), f("position", L), f("institution_id", S),
                                   f("institution_name", S), f("country", S)]),
        "topics": StructType([f("paper_id", S), f("topic_id", S), f("name", S), f("subfield", S), f("field", S), f("score", D)]),
        "keywords": StructType([f("paper_id", S), f("keyword", S), f("score", D)]),
    }


_WORK_COLS = ["paper_id", "title", "abstract", "year", "doi", "type", "language", "cited_by_count", "venue_id", "venue_name", "venue_type",
              "primary_topic_id", "primary_topic_name", "n_authors", "n_references"]


def _parse_partition(lines):
    """Runs inside the Spark executors: JSON line -> flat rows for each silver table."""
    for line in lines:
        w = normalise_work(json.loads(line))
        if w is None:
            continue
        yield ("works", tuple(w[c] for c in _WORK_COLS))
        for r in w["refs"]:
            yield ("refs", (w["paper_id"], r))
        for a in w["authorships"]:
            yield ("authorships", (w["paper_id"], a["author_id"], a["name"], a["position"], a["institution_id"], a["institution_name"], a["country"]))
        for t in w["topics"]:
            yield ("topics", (w["paper_id"], t["topic_id"], t["name"], t["subfield"], t["field"], t["score"]))
        for k in w["keywords"]:
            yield ("keywords", (w["paper_id"], k["keyword"], k["score"]))


def _write(df, path: Path, fmt: str) -> None:
    w = df.write.mode("overwrite")
    (w.format("delta") if fmt == "delta" else w.format("parquet")).save(str(path))


def _read(spark, path: Path, fmt: str):
    return spark.read.format("delta" if fmt == "delta" else "parquet").load(str(path))


def build_silver(spark, bronze_dir: Path, silver_dir: Path, fmt: Optional[str] = None) -> Dict[str, int]:
    fmt = fmt or ("delta" if delta_available() else "parquet")
    files = iter_bronze_files(bronze_dir)
    if not files:
        raise FileNotFoundError(f"no bronze pages under {bronze_dir}/works; run the ingest step first")
    sc = spark.sparkContext
    parsed = sc.textFile(",".join(str(f) for f in files)).mapPartitions(_parse_partition).cache()
    schemas = _schemas()
    counts: Dict[str, int] = {}
    for name in ("works", "refs", "authorships", "topics", "keywords"):
        rdd = parsed.filter(lambda kv, n=name: kv[0] == n).map(lambda kv: kv[1])
        df = spark.createDataFrame(rdd, schemas[name])
        _write(df, Path(silver_dir) / name, fmt)
        counts[name] = df.count()
    parsed.unpersist()
    return counts


def merge_works(spark, new_works_df, silver_dir: Path) -> None:
    """Incremental update of silver ``works`` with Delta Lake MERGE (insert new papers, update changed ones)."""
    from delta.tables import DeltaTable
    target = DeltaTable.forPath(spark, str(Path(silver_dir) / "works"))
    (target.alias("t").merge(new_works_df.alias("s"), "t.paper_id = s.paper_id")
     .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())


def build_gold(spark, silver_dir: Path, gold_dir: Path, year_from: int = 0, year_to: int = 9999, fmt: Optional[str] = None) -> Dict[str, int]:
    fmt = fmt or ("delta" if delta_available() else "parquet")
    silver_dir, gold_dir = Path(silver_dir), Path(gold_dir)
    for n in ("works", "refs", "authorships", "topics", "keywords"):
        _read(spark, silver_dir / n, fmt).createOrReplaceTempView("s_" + n)
    sql = spark.sql
    sql(f"""
        CREATE OR REPLACE TEMP VIEW papers AS
        SELECT paper_id, title, abstract, year, doi, type, language, cited_by_count, venue_id, venue_name,
               primary_topic_id, primary_topic_name, n_authors, n_references
        FROM (SELECT *, row_number() OVER (PARTITION BY paper_id ORDER BY cited_by_count DESC) AS rn FROM s_works)
        WHERE rn = 1 AND title IS NOT NULL AND length(title) > 0 AND abstract IS NOT NULL AND length(abstract) >= {MIN_ABSTRACT_CHARS}
          AND year IS NOT NULL AND year BETWEEN {year_from} AND {year_to}""")
    sql("""
        CREATE OR REPLACE TEMP VIEW citations AS
        SELECT DISTINCT r.paper_id AS citing_id, r.ref_id AS cited_id, a.year AS citing_year, b.year AS cited_year
        FROM s_refs r JOIN papers a ON r.paper_id = a.paper_id JOIN papers b ON r.ref_id = b.paper_id
        WHERE r.paper_id <> r.ref_id AND b.year <= a.year""")
    sql("""
        CREATE OR REPLACE TEMP VIEW authorships AS
        SELECT paper_id, author_id, position, institution_id, institution_name, country, name
        FROM (SELECT s.*, row_number() OVER (PARTITION BY s.paper_id, s.author_id ORDER BY s.position ASC) AS rn
              FROM s_authorships s JOIN papers p ON s.paper_id = p.paper_id) WHERE rn = 1""")
    sql("""
        CREATE OR REPLACE TEMP VIEW authors AS
        SELECT x.author_id, x.name, g.n_papers, g.first_year, g.last_year
        FROM (SELECT author_id, name, row_number() OVER (PARTITION BY author_id ORDER BY name ASC NULLS LAST) AS rn FROM authorships) x
        JOIN (SELECT a.author_id, count(DISTINCT a.paper_id) AS n_papers, min(p.year) AS first_year, max(p.year) AS last_year
              FROM authorships a JOIN papers p ON a.paper_id = p.paper_id GROUP BY a.author_id) g ON x.author_id = g.author_id
        WHERE x.rn = 1""")
    sql("""
        CREATE OR REPLACE TEMP VIEW institutions AS
        SELECT institution_id, institution_name AS name, country
        FROM (SELECT institution_id, institution_name, country,
                     row_number() OVER (PARTITION BY institution_id ORDER BY institution_name ASC NULLS LAST) AS rn
              FROM authorships WHERE institution_id IS NOT NULL) WHERE rn = 1""")
    sql("""
        CREATE OR REPLACE TEMP VIEW venues AS
        SELECT venue_id, name, type FROM (
          SELECT w.venue_id, w.venue_name AS name, w.venue_type AS type,
                 row_number() OVER (PARTITION BY w.venue_id ORDER BY w.venue_name ASC NULLS LAST) AS rn
          FROM s_works w JOIN papers p ON w.paper_id = p.paper_id WHERE w.venue_id IS NOT NULL) WHERE rn = 1""")
    sql("""
        CREATE OR REPLACE TEMP VIEW topics AS
        SELECT topic_id, name, subfield, field FROM (
          SELECT t.topic_id, t.name, t.subfield, t.field, row_number() OVER (PARTITION BY t.topic_id ORDER BY t.name ASC NULLS LAST) AS rn
          FROM s_topics t JOIN papers p ON t.paper_id = p.paper_id) WHERE rn = 1""")
    sql("""
        CREATE OR REPLACE TEMP VIEW paper_topics AS
        SELECT paper_id, topic_id, score FROM (
          SELECT t.paper_id, t.topic_id, t.score, row_number() OVER (PARTITION BY t.paper_id, t.topic_id ORDER BY t.score DESC) AS rn
          FROM s_topics t JOIN papers p ON t.paper_id = p.paper_id) WHERE rn = 1""")
    sql("""
        CREATE OR REPLACE TEMP VIEW keywords AS
        SELECT paper_id, keyword, score FROM (
          SELECT k.paper_id, k.keyword, k.score, row_number() OVER (PARTITION BY k.paper_id, k.keyword ORDER BY k.score DESC) AS rn
          FROM s_keywords k JOIN papers p ON k.paper_id = p.paper_id) WHERE rn = 1""")
    out_cols = {
        "papers": None, "authors": None, "institutions": None, "citations": None, "topics": None, "paper_topics": None,
        "venues": None, "keywords": None,
        "authorships": ["paper_id", "author_id", "position", "institution_id", "institution_name", "country"],
    }
    counts: Dict[str, int] = {}
    gold_dir.mkdir(parents=True, exist_ok=True)
    for name, cols in out_cols.items():
        df = spark.table(name)
        if cols:
            df = df.select(*cols)
        df.write.mode("overwrite").parquet(str(gold_dir / f"{name}.parquet"))
        counts[name] = df.count()
    return counts
