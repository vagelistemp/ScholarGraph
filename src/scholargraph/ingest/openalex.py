"""Resumable download of a research slice from the OpenAlex API (CC0 data, https://openalex.org).

* **Cursor paging** over a filtered, ``cited_by_count``-sorted query: each request returns up to 200 works plus a cursor for the next page.
* **Bronze pages** are the API's JSON records exactly as received (only the ``select`` fields are requested), one gzip-compressed
  JSON-lines file per page. Nothing is modified at this stage, so every later step can be rebuilt from bronze.
* **Checkpointing**: ``bronze/_state.json`` stores the cursor and the number of works saved. Stop the program at any time (or hit
  the daily budget) and start it again: it continues where it stopped.
* **Budget**: OpenAlex meters usage in dollars per day (a list page costs about $0.0001). Without an API key the budget is tiny and
  shared; with a free personal key (https://openalex.org/settings/api) it is larger. The key is read from the environment variable
  ``OPENALEX_API_KEY``. This program never asks for your e-mail address and never sends it anywhere.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from ..config import SLICE
from .transform import write_bronze_page

API = "https://api.openalex.org/works"
SELECT = ",".join([
    "id", "doi", "title", "publication_year", "type", "language", "cited_by_count", "referenced_works", "authorships",
    "primary_topic", "topics", "keywords", "primary_location", "abstract_inverted_index",
])
PER_PAGE = 200


@dataclass
class SliceSpec:
    subfield_id: int = SLICE["subfield_id"]
    year_from: int = SLICE["year_from"]
    year_to: int = SLICE["year_to"]
    types: tuple = SLICE["types"]
    max_works: int = SLICE["max_works"]

    def filter_string(self) -> str:
        return ",".join([
            f"primary_topic.subfield.id:{self.subfield_id}",
            f"publication_year:{self.year_from}-{self.year_to}",
            "has_abstract:true",
            "language:en",
            "type:" + "|".join(self.types),
        ])


@dataclass
class DownloadState:
    cursor: Optional[str] = "*"
    pages: int = 0
    works: int = 0
    spec_filter: str = ""
    done: bool = False

    @staticmethod
    def load(path: Path) -> "DownloadState":
        if path.exists():
            return DownloadState(**json.loads(path.read_text(encoding="utf-8")))
        return DownloadState()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")
        tmp.replace(path)


class BudgetExhausted(RuntimeError):
    """The API refused the request because the daily budget is used up; run the command again tomorrow."""


def _request_params(spec: SliceSpec, cursor: str, api_key: Optional[str]) -> Dict[str, Any]:
    params = {"filter": spec.filter_string(), "sort": "cited_by_count:desc", "per-page": PER_PAGE, "cursor": cursor, "select": SELECT}
    if api_key:
        params["api_key"] = api_key
    return params


def fetch_page(client: httpx.Client, spec: SliceSpec, cursor: str, api_key: Optional[str], retries: int = 5) -> Dict[str, Any]:
    delay = 2.0
    for attempt in range(retries):
        r = client.get(API, params=_request_params(spec, cursor, api_key), timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (402, 403) or (r.status_code == 429 and "budget" in r.text.lower()):
            raise BudgetExhausted(f"OpenAlex refused the request ({r.status_code}): {r.text[:200]}")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up after {retries} attempts (last status {r.status_code})")


def download(bronze_dir: Path, spec: Optional[SliceSpec] = None, api_key: Optional[str] = None,
             client: Optional[httpx.Client] = None, log: Callable[[str], None] = print,
             sleep_between: float = 0.1) -> DownloadState:
    """Download until ``spec.max_works`` works are saved, the data runs out, or the budget is exhausted. Resumable."""
    spec = spec or SliceSpec()
    api_key = api_key or os.environ.get("OPENALEX_API_KEY")
    bronze_dir = Path(bronze_dir)
    state_path = bronze_dir / "_state.json"
    state = DownloadState.load(state_path)
    if state.spec_filter and state.spec_filter != spec.filter_string():
        raise RuntimeError("the bronze folder was started with a different slice; use another data folder or delete bronze/")
    state.spec_filter = spec.filter_string()
    own_client = client is None
    client = client or httpx.Client(headers={"User-Agent": "scholargraph-research-pipeline"})
    try:
        while not state.done and state.works < spec.max_works:
            try:
                page = fetch_page(client, spec, state.cursor or "*", api_key)
            except BudgetExhausted as e:
                log(f"budget exhausted after {state.works} works: {e}\nRun the same command again later; it resumes from the checkpoint.")
                break
            results: List[Dict[str, Any]] = page.get("results") or []
            if not results:
                state.done = True
                break
            room = spec.max_works - state.works
            results = results[:room]
            state.pages += 1
            write_bronze_page(bronze_dir, state.pages, results)
            state.works += len(results)
            state.cursor = (page.get("meta") or {}).get("next_cursor")
            if not state.cursor:
                state.done = True
            state.save(state_path)                      # checkpoint after every page
            if state.pages % 10 == 0:
                log(f"  {state.works:,} works saved ({state.pages} pages)")
            time.sleep(sleep_between)
        state.save(state_path)
    finally:
        if own_client:
            client.close()
    return state
