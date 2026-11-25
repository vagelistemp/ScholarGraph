import gzip
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from scholargraph.ingest.openalex import (KEYLESS_MAX_WORKS, PER_PAGE, BudgetExhausted, DownloadState, KeyRequired, SliceSpec, download)
from scholargraph.ingest.synthetic import make_synthetic_works


def server(calls, total=1_000_000, budget_after=None, require_key_for_cursor=True):
    """A fake OpenAlex: serves pages of synthetic works, enforces the real API's key rule for cursor paging."""
    recs = make_synthetic_works(60, 0)

    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        calls.append(q)
        if "cursor" in q and "api_key" not in q and require_key_for_cursor:
            return httpx.Response(401, json={"error": "API key required for cursor paging"})
        if budget_after is not None and len(calls) > budget_after:
            return httpx.Response(429, text="daily budget exhausted")
        page_no = int(q["page"]) if "page" in q else (1 if q["cursor"] == "*" else int(q["cursor"]))
        n = min(PER_PAGE, max(0, total - (page_no - 1) * PER_PAGE))
        results = [dict(recs[i % len(recs)], id=f"https://openalex.org/W{(page_no - 1) * PER_PAGE + i}") for i in range(n)]
        nxt = str(page_no + 1) if (page_no * PER_PAGE < total and "cursor" in q) else None
        return httpx.Response(200, json={"meta": {"next_cursor": nxt}, "results": results})
    return httpx.Client(transport=httpx.MockTransport(handler))


class DownloadTest(unittest.TestCase):
    def count_works(self, d):
        return sum(1 for f in (Path(d) / "works").glob("*.jsonl.gz") for _ in gzip.open(f, "rt", encoding="utf-8"))

    def test_keyless_mode_uses_page_numbers_and_stops_at_ten_thousand(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            logs = []
            st = download(Path(d), SliceSpec(max_works=300_000), api_key="", client=server(calls), log=logs.append, sleep_between=0)
            self.assertEqual(st.works, KEYLESS_MAX_WORKS)
            self.assertEqual(self.count_works(d), KEYLESS_MAX_WORKS)
        self.assertTrue(all("page" in c and "cursor" not in c for c in calls))
        self.assertEqual([c["page"] for c in calls][:3], ["1", "2", "3"])
        self.assertTrue(any("OPENALEX_API_KEY" in l for l in logs))

    def test_with_a_key_the_cursor_is_used_and_the_key_is_sent(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            st = download(Path(d), SliceSpec(max_works=1000), api_key="k", client=server(calls), log=lambda *a: None, sleep_between=0)
            self.assertEqual(st.works, 1000)
        self.assertTrue(all("cursor" in c and c["api_key"] == "k" for c in calls))
        self.assertEqual([c["cursor"] for c in calls], ["*", "2", "3", "4", "5"])

    def test_a_401_without_a_key_raises_a_helpful_error(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(KeyRequired) as cm:
                from scholargraph.ingest.openalex import fetch_page
                fetch_page(server(calls), SliceSpec(), "*", None)
            self.assertIn("OPENALEX_API_KEY", str(cm.exception))

    def test_resume_continues_where_the_budget_stopped_it(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            st1 = download(Path(d), SliceSpec(max_works=1000), api_key="k", client=server(calls, budget_after=2), log=lambda *a: None, sleep_between=0)
            self.assertEqual(st1.pages, 2)
            self.assertFalse(st1.done)
            calls2 = []
            st2 = download(Path(d), SliceSpec(max_works=1000), api_key="k", client=server(calls2), log=lambda *a: None, sleep_between=0)
            self.assertEqual(st2.works, 1000)
            self.assertEqual(calls2[0]["cursor"], "3")                    # picked up at the saved cursor, not from the start
            self.assertEqual(self.count_works(d), 1000)                    # no page lost, none duplicated

    def test_the_download_stops_at_the_requested_size_and_refuses_a_different_slice(self):
        with tempfile.TemporaryDirectory() as d:
            st = download(Path(d), SliceSpec(max_works=250), api_key="k", client=server([]), log=lambda *a: None, sleep_between=0)
            self.assertEqual(st.works, 250)
            self.assertEqual(self.count_works(d), 250)
            with self.assertRaises(RuntimeError):
                download(Path(d), SliceSpec(max_works=250, year_from=2020), api_key="k", client=server([]), log=lambda *a: None, sleep_between=0)

    def test_state_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            DownloadState("abc", 3, 600, "f", False).save(Path(d) / "s.json")
            self.assertEqual(DownloadState.load(Path(d) / "s.json").cursor, "abc")


if __name__ == "__main__":
    unittest.main()
