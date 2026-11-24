"""Tests for corpus.py (needs data/corpus built).  Run: .venv\\Scripts\\python.exe sandbox-gym\\test_corpus.py"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

QS = ["university cultural activities Yemen 2002", "football player born 1990 scored hat-trick",
      "novel published 1967 author award nobel"]


def test_determinism():
    for q in QS:
        a, b = corpus.search(q, 10), corpus.search(q, 10)
        assert a == b and len(a) == 10, q
        assert len({r["docid"] for r in a}) == 10


def test_prefix_consistency():
    assert [r["docid"] for r in corpus.search(QS[0], 5)] == [r["docid"] for r in corpus.search(QS[0], 10)][:5]


def test_get_roundtrip():
    r = corpus.search(QS[0], 3)[0]
    assert set(r) == {"docid", "url", "title", "snippet"}
    d = corpus.get(r["docid"])
    assert d and d["url"] == r["url"] and d["title"] == r["title"] and d["text"]
    d2 = corpus.get(r["url"])
    assert d2 and d2["text"] and d2["url"] == r["url"]
    assert corpus.get("nonexistent-doc-id-xyz") is None and corpus.get("") is None


def test_snippet_cap():
    for q in QS + ["the of and", "zzzqqqxxx", "a"]:
        for r in corpus.search(q, 10):
            assert 0 < len(r["snippet"]) <= 300, (q, len(r["snippet"]))


def test_empty_and_k0():
    assert corpus.search("", 5) == [] and corpus.search("zzzqqqxxx", 5) == [] and corpus.search(QS[0], 0) == []


def test_threads():
    base = {q: corpus.search(q, 5) for q in QS}
    bad = []

    def w():
        for _ in range(5):
            for q in QS:
                if corpus.search(q, 5) != base[q]:
                    bad.append(q)
    ts = [threading.Thread(target=w) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not bad


if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try:
                f()
                print("PASS", n)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", n, repr(e))
    sys.exit(1 if fails else 0)
