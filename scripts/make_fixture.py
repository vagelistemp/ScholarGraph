"""Save a few real OpenAlex records as a test fixture (OpenAlex data is CC0). One tiny API call, about $0.0001.

    python scripts/make_fixture.py            # writes tests/fixtures/openalex_sample.jsonl.gz
"""
import gzip
import json
import os
from pathlib import Path

import httpx

from scholargraph.ingest.openalex import SELECT, SliceSpec

params = {"filter": SliceSpec(year_from=2023, year_to=2023).filter_string(), "sort": "cited_by_count:desc", "per-page": 4, "select": SELECT}
if os.environ.get("OPENALEX_API_KEY"):
    params["api_key"] = os.environ["OPENALEX_API_KEY"]
r = httpx.get("https://api.openalex.org/works", params=params, timeout=60, headers={"User-Agent": "scholargraph-research-pipeline"})
r.raise_for_status()
out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "openalex_sample.jsonl.gz"
out.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out, "wt", encoding="utf-8") as fh:
    for rec in r.json()["results"]:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", out, out.stat().st_size, "bytes")
