"""Checks for web_search / fetch_url (Bionic mirror over a frozen corpus). No model, no GPU,
no real corpus: a tiny in-memory fake with corpus.py's interface.

  ..\\.venv\\Scripts\\python.exe -m unittest test_web_tools -v
"""

import tempfile
import unittest
from pathlib import Path

import tools as T
import web_tools as W
from agent import run_action
from runner import Runner
from schema import Action
from workspace import Workspace

DOCS = {
    "https://example.org/alpha": ("Alpha Town",
                                  "Alpha Town sits on the Blue River.\n It is the capital of the north."),
    "https://example.org/beta": ("Beta City", "Beta City is a river port."),
    "https://example.org/gamma": ("Gamma", "Gamma is the capital of the south."),
}
HITS = {   # query -> ranked urls (what BM25 would return); a utm/case/slash variant of alpha
    "river city": ["https://example.org/alpha", "https://example.org/beta"],
    "capital": ["https://Example.org/alpha/?utm_source=x", "https://example.org/gamma"],
}


class FakeCorpus:
    def __init__(self, docs=DOCS, hits=HITS):
        self.docs, self.hits = docs, hits

    def search(self, query, k):
        out = []
        for u in self.hits.get(query, [])[:k]:
            title, text = self.docs[W.norm_url(u)]
            out.append({"docid": W.norm_url(u), "url": u, "title": title, "snippet": text})
        return out

    def get(self, url_or_docid):
        d = self.docs.get(url_or_docid)
        return {"url": url_or_docid, "title": d[0], "text": d[1]} if d else None


GOLDEN_SEARCH = (
    "Results for objective: capital rivers\n\n"
    "[1] Alpha Town — https://example.org/alpha\n"
    "    Alpha Town sits on the Blue River. It is the capital of the north.\n"
    "[2] Beta City — https://example.org/beta\n"
    "    Beta City is a river port.\n"
    "[3] Gamma — https://example.org/gamma\n"
    "    Gamma is the capital of the south.\n"
    "(3 results from 2 queries; 1 duplicates merged)\n\n"
    "NEXT STEP: call fetch_url with the 1-3 most promising URLs above to read the full pages. "
    "Do not answer from snippets alone.")


class WebSearchTests(unittest.TestCase):
    def test_golden(self):
        out = W.web_search(FakeCorpus(), {"objective": "capital rivers",
                                          "searchQueries": ["river city", "capital"]},
                           "Qwen/Qwen3.5-4B")
        self.assertEqual(out, GOLDEN_SEARCH)

    def test_deterministic(self):
        a = {"objective": "x", "searchQueries": ["capital", "river city"]}
        self.assertEqual(W.web_search(FakeCorpus(), a), W.web_search(FakeCorpus(), a))

    def test_hint_selection(self):
        self.assertEqual(W.hint_for("Qwen/Qwen3.5-4B"), W.HINTS["h2"])
        self.assertEqual(W.hint_for(""), W.HINTS["h2"])
        self.assertEqual(W.hint_for("gemma-4-E4B"), "")
        out = W.web_search(FakeCorpus(), {"objective": "o", "searchQueries": ["capital"]},
                           "Qwen/Qwen3.5-9B")
        self.assertTrue(out.endswith("(2 results from 1 query; 0 duplicates merged)"))

    def test_legacy_query_and_errors(self):
        out = W.web_search(FakeCorpus(), {"query": "capital"})
        self.assertTrue(out.startswith("Results for objective: capital\n\n[1] Alpha Town"))
        self.assertEqual(W.web_search(FakeCorpus(), {"searchQueries": ["  ", ""]}),
                         'web_search needs searchQueries: ["…"] — 1-3 keyword queries.')
        self.assertEqual(W.web_search(FakeCorpus(), {"objective": "o",
                                                     "searchQueries": ["nothing here"]}),
                         W.NO_RESULTS)
        self.assertEqual(W.web_search(None, {"searchQueries": ["capital"]}),
                         "Search failed: no corpus attached to this workspace")

    def test_query_caps(self):
        seen = []

        class Spy(FakeCorpus):
            def search(self, q, k):
                seen.append((q, k))
                return []
        W.web_search(Spy(), {"objective": "o", "searchQueries": ["a" * 250] + list("bcdefg")})
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(seen[0][0]), 200)
        self.assertTrue(all(k == 10 for _, k in seen))

    def test_snippet_cut_and_top8(self):
        docs = {f"https://e.org/{i}": (f"T{i}", "word " * 100) for i in range(12)}
        out = W.web_search(FakeCorpus(docs, {"q": list(docs)}), {"objective": "o",
                                                                 "searchQueries": ["q"]})
        self.assertIn("[8] T7 — https://e.org/7", out)
        self.assertNotIn("[9]", out)
        snippet = out.split("\n")[3]
        self.assertEqual(len(snippet), 4 + 200)
        self.assertIn("(8 results from 1 query; 0 duplicates merged)", out)

    def test_failing_query_is_empty_list(self):
        class Half(FakeCorpus):
            def search(self, q, k):
                if q == "boom":
                    raise RuntimeError("x")
                return super().search(q, k)
        out = W.web_search(Half(), {"objective": "o", "searchQueries": ["boom", "capital"]})
        self.assertIn("(2 results from 2 queries; 0 duplicates merged)", out)


class FetchTests(unittest.TestCase):
    def test_missing_golden(self):
        self.assertEqual(W.fetch_url(FakeCorpus(), {"urls": ["https://example.org/missing"]}),
                         "## https://example.org/missing\n(couldn't fetch — HTTP 404)")

    def test_found_short_page_untrimmed(self):
        out = W.fetch_url(FakeCorpus(), {"url": " https://example.org/beta "})   # legacy + trim
        self.assertEqual(out, "## https://example.org/beta\nBeta City is a river port.")

    def test_url_filter_and_error(self):
        self.assertEqual(W.fetch_url(FakeCorpus(), {"urls": ["example.org/beta", "ftp://x"]}),
                         'fetch_url needs urls: ["https://…"] — 1-3 full http(s) URLs.')
        out = W.fetch_url(FakeCorpus(), {"urls": ["https://example.org/beta"] * 5})
        self.assertEqual(out.count("## https://example.org/beta"), 3)

    def test_budget_split(self):
        long = "\n\n".join(f"para {i} " + "x" * 300 for i in range(100))
        c = FakeCorpus({"https://e.org/a": ("A", long), "https://e.org/b": ("B", long)}, {})
        one = W.fetch_url(c, {"urls": ["https://e.org/a"]})
        self.assertTrue(one.endswith(" …[truncated]"))                 # no objective
        self.assertEqual(len(one), len("## https://e.org/a\n") + 8000 + len(" …[truncated]"))
        two = W.fetch_url(c, {"urls": ["https://e.org/a", "https://e.org/b"], "objective": "para"})
        part = two.split("\n\n## https://e.org/b\n")[0][len("## https://e.org/a\n"):]
        self.assertTrue(part.endswith("\n\n…[trimmed to the parts relevant to the objective]"))
        self.assertLessEqual(len(part) - len("\n\n…[trimmed to the parts relevant to the objective]"),
                             4000)


LONG = "\n\n".join(f"para {i} " + "x" * 300 for i in range(100))    # 30,968 chars


class OffsetTests(unittest.TestCase):
    """fetch_url `offset`: no offset = legacy trim; any offset = raw window + continuation line."""

    def corpus(self):
        return FakeCorpus({"https://e.org/a": ("A", LONG),
                           "https://e.org/s": ("S", "short page")}, {})

    def test_continuation_line_text(self):
        self.assertEqual(W.continuation_line(12, 8000),
                         "[truncated — 12 chars left; call fetch_url again with offset=8000]")

    def test_absent_and_null_offset_are_legacy(self):
        legacy = W.fetch_url(self.corpus(), {"urls": ["https://e.org/a"], "objective": "para"})
        self.assertEqual(W.fetch_url(self.corpus(), {"urls": ["https://e.org/a"],
                                                     "objective": "para", "offset": None}), legacy)
        self.assertNotIn("call fetch_url again", legacy)

    def test_offset_zero_is_raw_window(self):
        out = W.fetch_url(self.corpus(), {"urls": ["https://e.org/a"], "objective": "para",
                                          "offset": 0})
        self.assertEqual(out, "## https://e.org/a\n" + LONG[:8000] + "\n" +
                         W.continuation_line(len(LONG) - 8000, 8000))

    def test_reading_on_reconstructs_page(self):
        got, offset = "", 0
        for _ in range(10):
            body = W.fetch_url(self.corpus(), {"urls": ["https://e.org/a"], "offset": offset})
            body = body[len("## https://e.org/a\n"):]
            head, sep, last = body.rpartition("\n[truncated — ")
            if not sep:
                got += body
                break
            got += head
            offset = int(last.split("offset=")[1].rstrip("]"))
        self.assertEqual(got, LONG)

    def test_per_url_budget_and_past_end(self):
        out = W.fetch_url(self.corpus(), {"urls": ["https://e.org/a", "https://e.org/s",
                                                   "https://e.org/x"], "offset": "100"})
        self.assertEqual(out, "\n\n".join([
            "## https://e.org/a\n" + LONG[100:2766] + "\n" +
            W.continuation_line(len(LONG) - 2766, 2766),
            "## https://e.org/s\n(offset 100 is past the end — the page has 10 chars)",
            "## https://e.org/x\n(couldn't fetch — HTTP 404)"]))

    def test_last_window_has_no_continuation(self):
        out = W.fetch_url(self.corpus(), {"urls": ["https://e.org/a"], "offset": len(LONG) - 5})
        self.assertEqual(out, "## https://e.org/a\n" + LONG[-5:])

    def test_bad_offsets(self):
        for bad in (-1, 1.5, "abc", " 5", True, [], {}):
            self.assertEqual(W.fetch_url(self.corpus(), {"urls": ["https://e.org/a"],
                                                         "offset": bad}), W.OFFSET_ERROR)
        self.assertEqual(W.parse_offset(3.0), 3)
        self.assertTrue(W.fetch_url(self.corpus(), {"urls": [], "offset": -1})
                        .startswith("fetch_url needs urls"))          # url check comes first

    def test_window_counts_utf16_units(self):
        self.assertEqual(W.window_text("ab\U0001F600cd", 0, 3),
                         "ab�\n" + W.continuation_line(3, 3))       # split emoji -> U+FFFD
        self.assertEqual(W.window_text("ab\U0001F600cd", 3, 10), "�cd")

    def test_schema_advertises_offset(self):
        p = T.TOOLS["fetch_url"]["schema"]["properties"]["offset"]
        self.assertEqual((p["type"], p["minimum"]), ("integer", 0))
        self.assertNotIn("offset", T.TOOLS["fetch_url"]["schema"]["required"])


class TrimTests(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(W.trim_to_objective("abc", "x", 10), "abc")

    def test_relevance_order_and_lead(self):
        text = "Lead para.\n\nAbout cats.\n\nAbout rivers and towns.\n\nMore rivers."
        out = W.trim_to_objective(text, "rivers towns", 50)
        # lead kept, then the highest-overlap paragraphs that fit, in document order
        self.assertEqual(out, "Lead para.\n\nAbout rivers and towns.\n\nMore rivers.\n\n"
                              "…[trimmed to the parts relevant to the objective]")

    def test_terms_list_counts_duplicates(self):
        self.assertEqual(W.terms("The rivers of the River city, river!"),
                         ["rivers", "river", "city", "river"])

    def test_js_slice_utf16(self):
        self.assertEqual(W.js_slice("ab\U0001F600cd", 3), "ab�")  # split emoji: JS lone
        #                                                surrogate, U+FFFD once UTF-8 encoded
        self.assertEqual(W.js_slice("ab\U0001F600cd", 4), "ab\U0001F600")
        self.assertEqual(W.js_slice("hello", 3), "hel")

    def test_norm_url(self):
        self.assertEqual(W.norm_url("https://Example.ORG/a/?utm_source=x#frag"),
                         "https://example.org/a")
        self.assertEqual(W.norm_url("https://e.org/a?b=1&utm_x=2"), "https://e.org/a?b=1")
        self.assertEqual(W.norm_url("not a url"), "not a url")


class HarnessParityTests(unittest.TestCase):
    """The tools go through the ONE dispatch (agent.run_action) that eval, collect and the
    GRPO rollout all use, and appear in the shared system prompt."""

    def test_run_action_and_ring(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(Path(td), Runner("host"))
            ws.corpus = FakeCorpus()
            act = T.parse_action('<tool_call>{"name": "web_search", "arguments": '
                                 '{"objective": "capital rivers", "searchQueries": '
                                 '["river city", "capital"]}}</tool_call>')
            self.assertEqual(act.arguments["searchQueries"], ["river city", "capital"])
            self.assertEqual(run_action(ws, act, ["web_search", "fetch_url"]), GOLDEN_SEARCH)
            self.assertTrue(run_action(ws, act, ["read_file"]).startswith(
                "error: no tool named 'web_search'"))
            f = run_action(ws, Action("fetch_url", {"urls": ["https://example.org/gamma"]}),
                           ["fetch_url"])
            self.assertEqual(f, "## https://example.org/gamma\nGamma is the capital of the south.")

    def test_no_corpus_is_an_observation(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(Path(td), Runner("host"))
            out = run_action(ws, Action("fetch_url", {"urls": ["https://e.org/x"]}), ["fetch_url"])
            self.assertEqual(out, "## https://e.org/x\n(couldn't fetch — no corpus attached to "
                                  "this workspace)")

    def test_system_prompt_lists_bionic_text(self):
        s = T.describe_tools(["web_search", "fetch_url"])
        self.assertIn("  web_search(objective, searchQueries) -- Search the PUBLIC INTERNET", s)
        self.assertIn("  fetch_url(urls, objective?, offset?) -- Fetch 1-3 web pages", s)
        self.assertIn("searchQueries (array of 1-5 strings): Concise keyword search queries", s)
        other = T.describe_tools(["list_dir", "read_file", "search"])
        self.assertNotIn("web_search", other)             # other families' prompts unchanged


if __name__ == "__main__":
    unittest.main()
