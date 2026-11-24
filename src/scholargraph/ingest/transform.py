"""Pure-Python parsing of OpenAlex ``work`` records.

Everything here is plain Python with no Spark and no pandas, so it is trivial to unit-test. The Spark pipeline reuses these exact
functions inside its partitions (``lake/spark_pipeline.py``), which guarantees that the Spark and the pandas pipelines parse
records identically.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

OPENALEX_PREFIX = "https://openalex.org/"
MIN_ABSTRACT_CHARS = 40


def strip_id(value: Optional[str]) -> Optional[str]:
    """``https://openalex.org/W123`` -> ``W123`` (other prefixes such as /topics/ and /subfields/ are removed too)."""
    if not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex stores abstracts as an inverted index ``{word: [positions]}``; rebuild the running text."""
    if not inverted_index:
        return ""
    slots: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        for p in positions:
            slots[p] = word
    if not slots:
        return ""
    return " ".join(slots[i] for i in sorted(slots))


def _first_institution(authorship: Dict[str, Any]) -> Dict[str, Optional[str]]:
    insts = authorship.get("institutions") or []
    if not insts:
        return {"institution_id": None, "institution_name": None, "country": None}
    i = insts[0]
    return {"institution_id": strip_id(i.get("id")), "institution_name": i.get("display_name"), "country": i.get("country_code")}


def normalise_work(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Flatten one raw record. Returns ``None`` when the record is unusable (no id)."""
    paper_id = strip_id(raw.get("id"))
    if not paper_id:
        return None
    loc = ((raw.get("primary_location") or {}).get("source")) or {}
    topic = raw.get("primary_topic") or {}
    authorships = []
    for pos, a in enumerate(raw.get("authorships") or []):
        author = a.get("author") or {}
        aid = strip_id(author.get("id"))
        if not aid:
            continue
        authorships.append({"author_id": aid, "name": author.get("display_name"), "position": pos, **_first_institution(a)})
    topics = []
    for t in raw.get("topics") or []:
        tid = strip_id(t.get("id"))
        if tid:
            topics.append({"topic_id": tid, "name": t.get("display_name"), "subfield": (t.get("subfield") or {}).get("display_name"),
                           "field": (t.get("field") or {}).get("display_name"), "score": float(t.get("score") or 0.0)})
    keywords = [{"keyword": (k.get("display_name") or "").strip().lower(), "score": float(k.get("score") or 0.0)}
                for k in (raw.get("keywords") or []) if (k.get("display_name") or "").strip()]
    refs = [r for r in (strip_id(x) for x in (raw.get("referenced_works") or [])) if r]
    return {
        "paper_id": paper_id,
        "doi": raw.get("doi"),
        "title": (raw.get("title") or raw.get("display_name") or "").strip(),
        "abstract": reconstruct_abstract(raw.get("abstract_inverted_index")),
        "year": raw.get("publication_year"),
        "type": raw.get("type"),
        "language": raw.get("language"),
        "cited_by_count": int(raw.get("cited_by_count") or 0),
        "venue_id": strip_id(loc.get("id")),
        "venue_name": loc.get("display_name"),
        "venue_type": loc.get("type"),
        "primary_topic_id": strip_id(topic.get("id")),
        "primary_topic_name": topic.get("display_name"),
        "n_authors": len(authorships),
        "n_references": len(refs),
        "authorships": authorships,
        "topics": topics,
        "keywords": keywords,
        "refs": refs,
    }


def is_usable(work: Dict[str, Any], year_from: int = 0, year_to: int = 9999) -> bool:
    """The quality rule that decides whether a work enters the gold ``papers`` table."""
    return bool(
        work.get("title")
        and len(work.get("abstract") or "") >= MIN_ABSTRACT_CHARS
        and work.get("year") is not None
        and year_from <= int(work["year"]) <= year_to
    )


def iter_bronze_files(bronze_dir: Path) -> List[Path]:
    return sorted(Path(bronze_dir).glob("works/page_*.jsonl.gz"))


def iter_raw_works(bronze_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield raw records from all bronze pages (one JSON object per line, gzip-compressed)."""
    for f in iter_bronze_files(bronze_dir):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def iter_normalised(bronze_dir: Path) -> Iterator[Dict[str, Any]]:
    for raw in iter_raw_works(bronze_dir):
        w = normalise_work(raw)
        if w is not None:
            yield w


def write_bronze_page(bronze_dir: Path, page_no: int, records: Iterable[Dict[str, Any]]) -> Path:
    out = Path(bronze_dir) / "works"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"page_{page_no:06d}.jsonl.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)                      # atomic: a half-written page never looks complete
    return path
