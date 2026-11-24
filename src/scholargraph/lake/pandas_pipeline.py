"""Reference implementation of the silver -> gold logic with pandas.

It defines, in the most readable form, what the gold tables contain. The Spark pipeline (``spark_pipeline.py``) implements the same
rules for large data, and ``tests/test_lake.py`` checks that both produce identical tables on the same input. It is also the
engine to use when Spark / Java is not available (a 300,000-paper slice fits in the memory of a laptop with 16 GB).

Gold rules (the data contract):

1. ``papers``: one row per ``paper_id`` (duplicates: keep the record with the highest ``cited_by_count``); a usable title, an
   abstract of at least 40 characters, and a publication year inside the requested range.
2. ``citations``: a reference (citing -> cited) is kept only when *both* papers are in ``papers``, it is not a self-citation, and
   the cited paper is not published after the citing paper (a paper cannot cite the future; such rows are data errors).
3. ``authorships``: one row per (paper, author), keeping the smallest author position; only for kept papers.
4. ``authors``, ``institutions``, ``venues``, ``topics``, ``paper_topics``, ``keywords``: distinct values derived from the kept papers.
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator

import pandas as pd

from ..ingest.transform import MIN_ABSTRACT_CHARS


def _papers_frame(works: Iterable[dict], year_from: int, year_to: int) -> pd.DataFrame:
    cols = ["paper_id", "title", "abstract", "year", "doi", "type", "language", "cited_by_count", "venue_id", "venue_name",
            "primary_topic_id", "primary_topic_name", "n_authors", "n_references"]
    df = pd.DataFrame([{c: w.get(c) for c in cols} for w in works], columns=cols)
    if df.empty:
        return df
    ok = (df["title"].fillna("").str.len() > 0) & (df["abstract"].fillna("").str.len() >= MIN_ABSTRACT_CHARS) \
        & df["year"].notna() & (df["year"] >= year_from) & (df["year"] <= year_to)
    df = df[ok].copy()
    df = df.sort_values(["paper_id", "cited_by_count"], ascending=[True, False]).drop_duplicates("paper_id", keep="first")
    for c in ["year", "cited_by_count", "n_authors", "n_references"]:
        df[c] = df[c].astype("int64")
    return df.sort_values("paper_id").reset_index(drop=True)


def build_gold(works: Iterator[dict], year_from: int = 0, year_to: int = 9999) -> Dict[str, pd.DataFrame]:
    best: Dict[str, dict] = {}
    for w in works:                                   # duplicates: keep the record with the highest cited_by_count (first wins a tie)
        cur = best.get(w["paper_id"])
        if cur is None or w["cited_by_count"] > cur["cited_by_count"]:
            best[w["paper_id"]] = w
    works = list(best.values())
    papers = _papers_frame(works, year_from, year_to)
    keep = set(papers["paper_id"])
    year_of = dict(zip(papers["paper_id"], papers["year"]))
    # a duplicate record is identical in every field the tables use, so iterating the raw list and de-duplicating the output is safe
    seen: set = set()
    cit_rows, auth_rows, topic_rows, pt_rows, kw_rows, venue_rows = [], [], [], [], [], []
    for w in works:
        pid = w["paper_id"]
        if pid not in keep or pid in seen:
            continue
        seen.add(pid)
        for ref in w["refs"]:
            if ref in keep and ref != pid and year_of[ref] <= year_of[pid]:
                cit_rows.append((pid, ref, int(year_of[pid]), int(year_of[ref])))
        for a in w["authorships"]:
            auth_rows.append({"paper_id": pid, **a})
        for t in w["topics"]:
            topic_rows.append({k: t[k] for k in ("topic_id", "name", "subfield", "field")})
            pt_rows.append({"paper_id": pid, "topic_id": t["topic_id"], "score": t["score"]})
        for k in w["keywords"]:
            kw_rows.append({"paper_id": pid, **k})
        if w.get("venue_id"):
            venue_rows.append({"venue_id": w["venue_id"], "name": w.get("venue_name"), "type": w.get("venue_type")})

    citations = pd.DataFrame(cit_rows, columns=["citing_id", "cited_id", "citing_year", "cited_year"]).drop_duplicates(["citing_id", "cited_id"])
    ash = pd.DataFrame(auth_rows, columns=["paper_id", "author_id", "name", "position", "institution_id", "institution_name", "country"])
    ash = ash.sort_values(["paper_id", "author_id", "position"]).drop_duplicates(["paper_id", "author_id"], keep="first")
    ash["position"] = ash["position"].astype("int64")
    ash = ash.merge(papers[["paper_id", "year"]], on="paper_id")
    names = ash.sort_values(["author_id", "name"]).drop_duplicates("author_id")[["author_id", "name"]]
    stats = ash.groupby("author_id").agg(n_papers=("paper_id", "nunique"), first_year=("year", "min"), last_year=("year", "max")).reset_index()
    authors = names.merge(stats, on="author_id").astype({"n_papers": "int64", "first_year": "int64", "last_year": "int64"})
    institutions = ash.dropna(subset=["institution_id"]).sort_values(["institution_id", "institution_name"]) \
        .drop_duplicates("institution_id")[["institution_id", "institution_name", "country"]].rename(columns={"institution_name": "name"})
    authorships = ash[["paper_id", "author_id", "position", "institution_id", "institution_name", "country"]]
    topics = pd.DataFrame(topic_rows, columns=["topic_id", "name", "subfield", "field"]).sort_values(["topic_id", "name"]).drop_duplicates("topic_id")
    paper_topics = pd.DataFrame(pt_rows, columns=["paper_id", "topic_id", "score"]).sort_values(["paper_id", "topic_id", "score"], ascending=[True, True, False]).drop_duplicates(["paper_id", "topic_id"])
    keywords = pd.DataFrame(kw_rows, columns=["paper_id", "keyword", "score"]).sort_values(["paper_id", "keyword", "score"], ascending=[True, True, False]).drop_duplicates(["paper_id", "keyword"])
    venues = pd.DataFrame(venue_rows, columns=["venue_id", "name", "type"]).sort_values(["venue_id", "name"]).drop_duplicates("venue_id")
    out = {"papers": papers, "authors": authors, "authorships": authorships, "institutions": institutions,
           "citations": citations, "topics": topics, "paper_topics": paper_topics, "venues": venues, "keywords": keywords}
    return {k: v.sort_values(list(v.columns[:2]) if len(v.columns) > 1 else list(v.columns)).reset_index(drop=True) for k, v in out.items()}
