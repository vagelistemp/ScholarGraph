import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scholargraph.ingest.synthetic import make_synthetic_works
from scholargraph.ingest.transform import (is_usable, iter_normalised, normalise_work, reconstruct_abstract, strip_id,
                                           write_bronze_page)
from scholargraph.lake.gold import read_gold, write_gold
from scholargraph.lake.pandas_pipeline import build_gold
from scholargraph.lake.quality import has_errors, run_quality_checks
from scholargraph.schemas import GOLD, validate_gold_tables


def synthetic_gold(n=300, seed=1):
    works = [w for w in (normalise_work(r) for r in make_synthetic_works(n, seed)) if w]
    return build_gold(iter(works), 2015, 2025), works


class TransformTest(unittest.TestCase):
    def test_strip_id(self):
        self.assertEqual(strip_id("https://openalex.org/W123"), "W123")
        self.assertEqual(strip_id("https://openalex.org/subfields/1702"), "1702")
        self.assertIsNone(strip_id(None))

    def test_reconstruct_abstract_orders_by_position_and_repeats(self):
        inv = {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}
        self.assertEqual(reconstruct_abstract(inv), "the cat sat the mat")
        self.assertEqual(reconstruct_abstract(None), "")
        self.assertEqual(reconstruct_abstract({}), "")

    def test_normalise_work_fields(self):
        raw = make_synthetic_works(30, 0)[5]
        w = normalise_work(raw)
        self.assertEqual(w["paper_id"], "W10005")
        self.assertEqual(w["n_authors"], len(w["authorships"]))
        self.assertEqual(w["n_references"], len(w["refs"]))
        self.assertTrue(all(a["position"] == i for i, a in enumerate(w["authorships"])))
        self.assertGreater(len(w["abstract"]), 40)
        self.assertIsNone(normalise_work({"title": "no id"}))

    def test_is_usable_rules(self):
        base = {"title": "t", "abstract": "x" * 50, "year": 2020}
        self.assertTrue(is_usable(base, 2015, 2025))
        self.assertFalse(is_usable({**base, "abstract": "short"}, 2015, 2025))
        self.assertFalse(is_usable({**base, "year": 2010}, 2015, 2025))
        self.assertFalse(is_usable({**base, "title": ""}, 2015, 2025))

    def test_real_openalex_fixture_parses(self):
        """Four real records saved from the API (tests/fixtures): the parser must handle the real format."""
        raw = [json.loads(l) for l in gzip.open(Path(__file__).parent / "fixtures" / "openalex_sample.jsonl.gz", "rt", encoding="utf-8")]
        works = [normalise_work(r) for r in raw]
        self.assertEqual(len(works), 4)
        for w in works:
            self.assertTrue(w["paper_id"].startswith("W"))
            self.assertTrue(is_usable(w, 2015, 2025))
            self.assertGreater(w["n_authors"], 0)
            self.assertGreater(w["n_references"], 0)
        gold = build_gold(iter(works), 2015, 2025)
        self.assertEqual(validate_gold_tables(gold), [])
        self.assertEqual(len(gold["papers"]), 4)

    def test_bronze_roundtrip(self):
        recs = make_synthetic_works(25, 2)
        with tempfile.TemporaryDirectory() as d:
            write_bronze_page(Path(d), 1, recs[:10])
            write_bronze_page(Path(d), 2, recs[10:])
            back = list(iter_normalised(Path(d)))
            self.assertEqual(len(back), len(recs))
            self.assertEqual(back[0]["paper_id"], strip_id(recs[0]["id"]))


class GoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold, cls.works = synthetic_gold()

    def test_contract_holds(self):
        self.assertEqual(validate_gold_tables(self.gold), [])

    def test_quality_checks_pass(self):
        checks = run_quality_checks(self.gold)
        self.assertFalse(has_errors(checks), [c for c in checks if not c.passed])

    def test_planted_errors_are_removed(self):
        papers, cit = self.gold["papers"], self.gold["citations"]
        self.assertEqual(papers["paper_id"].is_unique, True)                    # the duplicate record
        self.assertNotIn("W77777", set(papers["paper_id"]))                      # no abstract
        self.assertTrue((cit["cited_year"] <= cit["citing_year"]).all())         # W77778 cited a later paper
        self.assertNotIn("W77778", set(cit["citing_id"]))
        self.assertTrue(cit["cited_id"].isin(papers["paper_id"]).all())          # references outside the slice are dropped

    def test_counts_are_consistent(self):
        papers, ash, authors = self.gold["papers"], self.gold["authorships"], self.gold["authors"]
        per_paper = ash.groupby("paper_id").size()
        merged = papers.set_index("paper_id").join(per_paper.rename("k"), how="left").fillna({"k": 0})
        self.assertTrue((merged["n_authors"] == merged["k"]).all())
        self.assertEqual(int(authors["n_papers"].sum()), len(ash))

    def test_parquet_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            write_gold(self.gold, Path(d))
            back = read_gold(Path(d))
            self.assertEqual(set(back), set(GOLD))
            pd.testing.assert_frame_equal(back["papers"], self.gold["papers"], check_dtype=False)

    def test_contract_violations_are_reported(self):
        broken = dict(self.gold)
        broken["papers"] = self.gold["papers"].drop(columns=["year"])
        self.assertTrue(any("missing columns" in p for p in validate_gold_tables(broken)))
        dup = dict(self.gold)
        dup["papers"] = pd.concat([self.gold["papers"], self.gold["papers"].iloc[[0]]])
        self.assertTrue(any("duplicate keys" in p for p in validate_gold_tables(dup)))


def _spark_or_skip():
    try:
        from scholargraph.lake import spark_pipeline as sp
        spark = sp.spark_session(driver_memory="2g")
        with tempfile.TemporaryDirectory() as d:                                  # can this machine write files with Spark?
            spark.range(3).write.mode("overwrite").parquet(str(Path(d) / "probe"))
        return sp, spark
    except Exception as e:                                                         # no Java, no winutils on Windows, no pyspark ...
        raise unittest.SkipTest(f"Spark is not usable here: {type(e).__name__}: {str(e)[:120]}")


class SparkEquivalenceTest(unittest.TestCase):
    """The Spark pipeline and the pandas pipeline must produce the same gold tables from the same bronze pages."""

    def test_spark_equals_pandas(self):
        sp, spark = _spark_or_skip()
        raw = make_synthetic_works(150, 3)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_bronze_page(d / "bronze", 1, raw[:80])
            write_bronze_page(d / "bronze", 2, raw[80:])
            fmt = "delta" if sp.delta_available() else "parquet"
            sp.build_silver(spark, d / "bronze", d / "silver", fmt)
            sp.build_gold(spark, d / "silver", d / "gold", 2015, 2025, fmt)
            got = {n: pd.read_parquet(d / "gold" / f"{n}.parquet") for n in GOLD}
        want = build_gold(iter_normalised_from(raw), 2015, 2025)
        for name in GOLD:
            cols = list(want[name].columns)
            a = got[name][cols].sort_values(cols[:2]).reset_index(drop=True)
            b = want[name][cols].sort_values(cols[:2]).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False, obj=name)


def iter_normalised_from(raw):
    return iter([w for w in (normalise_work(r) for r in raw) if w])


if __name__ == "__main__":
    unittest.main()
