"""Differential check: web_tools.py vs the Bionic JS functions themselves (trimToObjective,
normUrl, RRF merge + formatResults, and the whole fetch_url tool body incl. `offset` paging,
lifted from the websearch MCP source and run in node). Any byte difference is a parity bug.

Bionic is referenced by path, never copied: env BIONIC_DIR (default
C:\\Users\\juxxs\\bionic\\llama-studio; point it at a worktree to check a branch).

  ..\\.venv\\Scripts\\python.exe web_tools_parity.py      exit 0 = 0 diffs, 1 = diffs/error,
                                                          3 = skipped (no node / no BIONIC_DIR)
Also run by selftest.py and by the PostToolUse hook in C:\\Users\\juxxs\\vllm\\.claude\\settings.json.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import web_tools as W  # noqa: E402

DEFAULT_BIONIC = r"C:\Users\juxxs\bionic\llama-studio"
SKIPPED = 3
W_TOOLS = ("web_search", "fetch_url")


def bionic_dir() -> Path:
    return Path(os.environ.get("BIONIC_DIR") or DEFAULT_BIONIC)


def node_exe() -> str | None:
    for cand in (shutil.which("node"), "C:/Program Files/nodejs/node.exe"):
        if cand and Path(cand).exists():
            return cand
    return None


def skip_reason() -> str | None:
    if not node_exe():
        return "node not found on PATH"
    src = bionic_dir() / "reference" / "websearch-mcp-index.js"
    if not src.exists():
        return f"BIONIC_DIR has no {src.relative_to(bionic_dir())} ({bionic_dir()})"
    return None


def make_cases(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    words = ("river city capital the of town Blue north port south mentor company founder "
             "alpha beta \U0001F600 caf\u00e9 na\u00efve").split()

    def para():
        return " ".join(rng.choice(words) for _ in range(rng.randint(1, 40)))

    cases = []
    for _ in range(300):
        sep = "\n\n" if rng.random() < .8 else "\n\n\n"
        cases.append({"kind": "trim", "text": sep.join(para() for _ in range(rng.randint(1, 30))),
                      "objective": rng.choice(["", "river capital", "the mentor of the company",
                                               "blue blue river"]),
                      "budget": rng.choice([50, 200, 1000, 2666, 8000])})
    for u in ["https://Example.ORG/a/?utm_source=x#f", "https://e.org/a?b=1&utm_x=2",
              "http://e.org", "https://e.org/x/y/", "https://e.org:8080/P?q=a%20b"]:
        cases.append({"kind": "norm", "url": u})
    for _ in range(100):
        pool = [{"url": f"https://e.org/{i}" + rng.choice(["", "/", "?utm_a=1", "#h"]),
                 "title": f"T{i} {para()[:20]}", "content": para() * rng.randint(1, 4)}
                for i in range(15)]
        lists = [rng.sample(pool, rng.randint(0, 10)) for _ in range(rng.randint(1, 5))]
        if not any(lists):
            continue
        cases.append({"kind": "fmt", "objective": para(),
                      "queries": [f"q{i}" for i in range(len(lists))], "lists": lists,
                      "hint": rng.choice(["", W.HINTS["h2"]])})
    # the whole fetch_url tool: arg salvage, per-URL budget, legacy trim vs offset windows
    for _ in range(300):
        pages = {}
        for i in range(4):
            r = rng.random()
            pages[f"https://e.org/p{i}"] = (
                None if r < .08 else "" if r < .12 else
                ("\n\n" if rng.random() < .8 else "\n").join(
                    para() for _ in range(rng.randint(1, rng.choice([3, 30, 120])))))
        urls = [rng.choice(list(pages) + ["https://e.org/missing", " https://e.org/p1 ",
                                          "e.org/p2", "ftp://e.org/x"])
                for _ in range(rng.choice([0, 1, 1, 2, 3, 5]))]
        args = {"urls": urls} if rng.random() < .85 else ({"url": urls[0]} if urls else {})
        if rng.random() < .5:
            args["objective"] = rng.choice(["", "river capital", "the mentor", para()])
        longest = max((len(t) for t in pages.values() if t), default=0)
        r = rng.random()
        if r < .2:
            pass                                             # legacy: no offset
        elif r < .25:
            args["offset"] = None
        elif r < .8:
            args["offset"] = rng.choice([0, 0, rng.randint(0, longest + 10), 1000, 2666, 8000,
                                         longest, longest + rng.randint(0, 3)])
        else:
            args["offset"] = rng.choice(["2000", "0", "-3", " 5", "abc", -1, 1.5, 3.0, True,
                                         [], {}])
        cases.append({"kind": "fetch", "pages": pages, "args": args})
    return cases


class _Lists:
    def __init__(self, lists):
        self.lists = iter(lists)

    def search(self, q, k):
        return [{"url": r["url"], "title": r["title"], "snippet": r["content"]}
                for r in next(self.lists)]


class _Pages:
    def __init__(self, pages):
        self.pages = pages

    def get(self, u):
        return {"url": u, "title": "", "text": self.pages[u]} if u in self.pages else None


def python_out(c: dict) -> str:
    if c["kind"] == "trim":
        return W.trim_to_objective(c["text"], c["objective"], c["budget"])
    if c["kind"] == "norm":
        return W.norm_url(c["url"])
    if c["kind"] == "fetch":
        return W.fetch_url(_Pages(c["pages"]), c["args"])
    merged, dupes = W.fan_out_search(_Lists(c["lists"]), c["queries"])
    return W.format_results(c["objective"], c["queries"], merged, dupes, c["hint"])


def run(verbose: bool = True) -> tuple[int, int, dict]:
    """(cases, mismatches, mismatches-by-kind). Raises on node/IO failure."""
    cases = make_cases()
    tmp = Path(tempfile.mkdtemp(prefix="webparity-"))
    (tmp / "cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, BIONIC_DIR=str(bionic_dir()))
    proc = subprocess.run([node_exe(), str(HERE / "web_tools_parity.mjs"), str(tmp / "cases.json"),
                           str(tmp / "js.json")], env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"node failed ({proc.returncode}): {proc.stderr.strip()[-600:]}")
    out = json.loads((tmp / "js.json").read_text(encoding="utf-8"))
    shutil.rmtree(tmp, ignore_errors=True)
    js = out["cases"]
    bad, by_kind = 0, {}
    # tools.py must advertise Bionic's names, descriptions and input schemas verbatim
    import tools as T  # noqa: PLC0415
    js_tools = {t["name"]: t for t in out["tools"]}
    for name in W_TOOLS:
        t = js_tools.get(name)
        if not t or T.TOOLS[name]["help"] != t["description"] or \
                T.TOOLS[name]["schema"] != t["inputSchema"]:
            bad += 1
            by_kind["schema"] = by_kind.get("schema", 0) + 1
            if verbose:
                print("MISMATCH schema", name, "\nPY", json.dumps(
                    {"help": T.TOOLS[name]["help"], "schema": T.TOOLS[name]["schema"]},
                    ensure_ascii=False)[:600], "\nJS", json.dumps(t, ensure_ascii=False)[:600])
    for c, j in zip(cases, js):
        p = python_out(c)
        # JS may hold a lone surrogate; on the wire (UTF-8) it is U+FFFD, which is what we emit
        j = j.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        if p != j:
            bad += 1
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
            if verbose and bad <= 3:
                print("MISMATCH", c["kind"], ascii(c.get("args", ""))[:160],
                      "\nPY", ascii(p[-160:]), "\nJS", ascii(j[-160:]))
    return len(cases) + len(W_TOOLS), bad, by_kind


def main() -> int:
    why = skip_reason()
    if why:
        print(f"web_tools parity SKIPPED: {why}")
        return SKIPPED
    try:
        n, bad, by_kind = run()
    except Exception as e:  # noqa: BLE001
        print(f"web_tools parity ERROR (BIONIC_DIR={bionic_dir()}): {e}")
        return 1
    print(f"web_tools parity vs {bionic_dir()}: {n} cases, {bad} mismatches"
          + (f" {by_kind}" if by_kind else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
