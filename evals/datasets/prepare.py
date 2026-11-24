"""Build post-cutoff eval sets, sealed holdouts and training pools for Qwen3.5-4B.

  .venv\\Scripts\\python.exe evals\\datasets\\prepare.py [--datasets math,code,mhqa]

Rules live in evals/datasets/RULES.md; this script enforces them:
  cutoff 2026-03-02 | eval+holdout = post-cutoff roots only | train = pre-cutoff
  roots only | split by root task | holdout sealed + hashed before any training |
  endpoint-only graders (evals/graders.py).

Per dataset: load sources -> date every item -> cluster into root tasks ->
classify each root (pre / post / ambiguous) -> post roots split eval/holdout by
hash(root) -> seal holdout (evals/holdout/seal.py) -> write pools -> leak gates.
Log: evals/datasets/logs/prepare-<UTC>.log + .json. RULES.md table regenerated.
"""

from __future__ import annotations

import argparse
import calendar
import collections
import datetime as dt
import hashlib
import html
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
EVALS = HERE.parent
REPO = EVALS.parent
sys.path[:0] = [str(HERE), str(EVALS), str(REPO)]
import graders  # noqa: E402
import roots as R  # noqa: E402
import registry as DR  # noqa: E402 (stdlib-only public identity helpers)

MODEL = "Qwen/Qwen3.5-4B"
CUTOFF = "2026-03-02"            # Qwen3.5 public release date; no official training cutoff exists
LEETCODE_MARGIN_DAYS = 7         # Historical metadata only; upper bounds never prove post-cutoff publication
VERSION = "v1"
SEED = "grpo-local:qwen3.5-4b:2026-03-02"
HOLDOUT_FRAC = 0.4               # share of post-cutoff roots sealed as holdout
TEST_BYTES_CAP = 5_000_000       # per-problem test payload cap in training pools
BASELINE = {"math": 5, "code": 5, "mhqa": 5}   # baseline: 5 per bucket

RAW = EVALS / "data" / "raw"
HF = RAW / "hf"
POOLS = EVALS / "data" / "pools"
SETS = EVALS / "data" / "sets"
SEAL_PY = EVALS / "holdout" / "seal.py"
SEALED = EVALS / "holdout" / "sealed"
LOGS = HERE / "logs"
RULES = HERE / "RULES.md"

log = logging.getLogger("prepare")
PROV: dict[str, dict] = {}       # source -> {revision, files...}


# --- helpers -----------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_snapshot(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import HfApi, snapshot_download
    sha = HfApi().dataset_info(repo).sha
    d = HF / repo.replace("/", "__")
    snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=d, allow_patterns=patterns)
    PROV[repo] = {"revision": sha}
    return d


def read_parquets(d: Path) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(d.rglob("*.parquet")):
        rows += pq.read_table(f).to_pylist()
    return rows


def month_end(yymm: str) -> str:
    y, m = 2000 + int(yymm[:2]), int(yymm[2:4])
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def add_days(d: str, n: int) -> str:
    return (dt.date.fromisoformat(d) + dt.timedelta(days=n)).isoformat()


def html_to_text(s: str) -> str:
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s or "", flags=re.S)
    s = re.sub(r"<li>", "- ", s)
    s = re.sub(r"</(p|li|pre|div|ul|ol)>|<br\s*/?>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def date_record(r: dict, date: str | None, kind: str) -> None:
    """kind: exact | upper (true date <= date) | leetcode (first-UGC upper bound).
    pre  = proven before cutoff; post = proven on/after cutoff; else ambiguous."""
    r["first_published"], r["date_basis"] = date, kind
    if date is None:
        r["_when"] = "ambiguous"
    elif date < CUTOFF:
        r["_when"] = "pre"
    elif kind == "exact":
        r["_when"] = "post"
    else:
        r["_when"] = "ambiguous"


# --- math ----------------------------------------------------------------------

COMP_DATES = {  # exam dates; all pre-cutoff
    "aime_2023_I": "2023-02-07", "aime_2023_II": "2023-02-15", "aime_2024_I": "2024-01-31",
    "aime_2024_II": "2024-02-07", "aime_2025_I": "2025-02-06", "aime_2025_II": "2025-02-12",
    "aime_2025": "2025-02-12", "aime_2026_I": "2026-02-05", "aime_2026": "2026-02-11",
    "hmmt_feb_2023": "2023-02-18", "hmmt_feb_2024": "2024-02-17", "hmmt_feb_2025": "2025-02-15",
    "hmmt_feb_2026": "2026-02-14", "hmmt_nov_2025": "2025-11-08", "brumo_2025": "2025-04-05",
    "cmimc_2025": "2025-03-01", "smt_2025": "2025-04-12", "apex_2025": "2025-12-31",
}
ARXIV_SETS = ["arxivmath-1225", "arxivmath-0126", "arxivmath-0226", "arxivmath-0326", "arxivmath-0426",
              "arxivmath-0526", "arxivmath-0626", "arxivmath-0826", "arxivmath"]
ARXIV_CACHE = RAW / "arxiv_v1_dates.json"


def arxiv_v1_dates(ids: set[str]) -> dict[str, str]:
    cache = json.loads(ARXIV_CACHE.read_text()) if ARXIV_CACHE.exists() else {}
    todo = sorted(i for i in ids if i not in cache)
    for k in range(0, len(todo), 50):
        batch = todo[k : k + 50]
        r = requests.get("https://export.arxiv.org/api/query",
                         params={"id_list": ",".join(batch), "max_results": len(batch)}, timeout=60)
        r.raise_for_status()
        for entry in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
            m_id = re.search(r"<id>https?://arxiv.org/abs/([^<]+?)(v\d+)?</id>", entry)
            m_pub = re.search(r"<published>(\d{4}-\d{2}-\d{2})", entry)
            if m_id and m_pub:
                cache[m_id.group(1)] = m_pub.group(1)
        time.sleep(3)  # arXiv API etiquette
    ARXIV_CACHE.write_bytes(json.dumps(cache, indent=0, sort_keys=True).encode("utf-8"))
    return cache


def load_math() -> list[dict]:
    recs = []

    def add(uid, lineage, problem, answer, date, kind, source):
        if answer is None or str(answer).strip() == "":
            return
        r = {"uid": uid, "dataset": "math", "source": source, "lineage": lineage, "skel_text": problem,
             "prompt": problem.strip(), "answer": str(answer).strip()}
        date_record(r, date, kind)
        recs.append(r)

    for comp, date in COMP_DATES.items():
        for row in read_parquets(hf_snapshot(f"MathArena/{comp}", ["*.parquet", "data/*"])):
            key = f"math:comp:{comp.lower()}:{row['problem_idx']}"  # cross-set copies merge via skeleton
            add(f"math:{comp}:{row['problem_idx']}", [key], row["problem"], row["answer"], date, "upper",
                f"MathArena/{comp}")
    for row in read_parquets(hf_snapshot("MathArena/apex-shortlist", ["*.parquet", "data/*"])):
        years = re.findall(r"20\d\d", str(row.get("source") or ""))
        date = f"{years[0]}-12-31" if years and years[0] <= "2025" else None
        add(f"math:apex-shortlist:{row['problem_idx']}", [f"math:src:{str(row.get('source')).lower()}"],
            row["problem"], row["answer"], date, "upper", "MathArena/apex-shortlist")

    arx_rows = []
    for s in ARXIV_SETS:
        for row in read_parquets(hf_snapshot(f"MathArena/{s}", ["*.parquet", "data/*"])):
            arx_rows.append((s, row, str(row["source"]).strip()))
    for row in read_parquets(hf_snapshot("MathArena/arxivmath-training", ["*.parquet", "data/*"])):
        arx_rows.append(("arxivmath-training", {"problem_idx": None, "problem": row["question"],
                                                "answer": row["answer"]}, str(row["paper_id"]).strip()))
    need_api = {pid for _, _, pid in arx_rows if not re.match(r"^\d{4}\.", pid) or pid[:4] >= "2603"}
    v1 = arxiv_v1_dates(need_api)
    for s, row, pid in arx_rows:
        if pid in v1:
            date, kind = v1[pid], "exact"
        elif re.match(r"^\d{4}\.", pid) and pid[:4] < "2603":
            date, kind = month_end(pid[:4]), "upper"      # yymm id => submitted by month end
        else:
            date, kind = None, "exact"
        idx = row["problem_idx"] if row["problem_idx"] is not None else hashlib.sha1(row["problem"].encode()).hexdigest()[:8]
        add(f"math:{s}:{idx}", [f"math:arxiv:{pid}"], row["problem"], row["answer"], date, kind, f"MathArena/{s}")

    # Gold must be machine-checkable: math-verify has to parse it.
    from math_verify import parse
    kept = []
    for r in recs:
        if parse(f"${r['answer']}$", parsing_timeout=None):
            kept.append(r)
        else:
            DROPS["math"]["gold-unparseable"] += 1
    return kept


# --- code ----------------------------------------------------------------------

def slug_key(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return "code:lc:" + re.sub(r"-(i|ii|iii|iv|v)$", "", s)       # I/II variants share a root


def parse_signature(snippet: str):
    if "class Solution" not in snippet:
        return None
    defs = re.findall(r"def (\w+)\(self(?:,\s*)?(.*?)\)\s*(?:->\s*([^:]+))?:", snippet)
    if len(defs) != 1:
        return None
    name, params, ret = defs[0]
    if re.search(r"ListNode|TreeNode|\bNode\b|Optional\[", params + (ret or "")):
        return None
    names = [p.split(":")[0].strip() for p in params.split(",") if p.strip()]
    return name, names


NON_UNIQUE = re.compile(r"return any|any of them|any valid|in any order|multiple (valid )?(answers|solutions)|"
                        r"any such|if there are multiple|any one of", re.I)


def statement_outputs(text: str) -> list | None:
    outs = []
    for v in re.findall(r"^\s*Output:\s*(.+?)\s*$", text, re.M):
        try:
            outs.append(json.loads(v))
        except Exception:
            return None
    return outs


def load_code() -> list[dict]:
    recs = []
    wl_dir = hf_snapshot("whiskwhite/leetcode-complete", ["*.jsonl"])
    wl = []
    for f in ("train.jsonl", "validation.jsonl", "test.jsonl", "unsolved.jsonl"):
        wl += [json.loads(l) for l in open(wl_dir / f, encoding="utf-8")]
    lc_date = {slug_key(w["title"]): w["created_at_approx"][:10] for w in wl}

    # LeetCode, whiskwhite: every problem joins clustering; post-cutoff ones become eval candidates.
    max_pre_fid = max(int(w["frontend_id"]) for w in wl if w["frontend_id"].isdigit()
                      and w["created_at_approx"][:10] < CUTOFF)
    for w in wl:
        date = w["created_at_approx"][:10]
        text = html_to_text(w["content"])
        snippet = next((c["code"] for c in w.get("code_snippets") or [] if c["lang"] == "python3"), "")
        sig = parse_signature(snippet)
        r = {"uid": f"code:leetcode:{w['title_slug']}", "dataset": "code", "source": "whiskwhite/leetcode-complete",
             "lineage": [slug_key(w["title"])], "skel_text": text, "idents": sig[1] if sig else [], "emit": False}
        date_record(r, date, "leetcode")
        recs.append(r)
        if r["_when"] != "post":
            continue  # upper bounds cannot qualify evaluation; pre-cutoff tests come from neulab
        if not w["frontend_id"].isdigit() or int(w["frontend_id"]) <= max_pre_fid:
            DROPS["code"]["lc-id-not-monotonic"] += 1
            continue
        oracle = next((s["typed_code"] for s in w.get("solutions") or [] if s["lang"] == "python3"), None)
        why = ("no-accepted-oracle" if not oracle else "not-single-method-json-signature" if not sig
               else "non-unique-answer" if NON_UNIQUE.search(text) else None)
        if why:
            DROPS["code"][why] += 1
            continue
        lines = [l for l in w["example_test_cases"].split("\n") if l.strip()]
        if not lines or len(lines) % max(1, len(sig[1])) != 0:
            DROPS["code"]["examples-unparseable"] += 1
            continue
        k = max(1, len(sig[1]))
        cases = [{"input": "\n".join(lines[i : i + k]), "output": None} for i in range(0, len(lines), k)]
        res = graders.run_functional(oracle, f"Solution.{sig[0]}", cases, record=True)
        if "error" in res:
            DROPS["code"][f"oracle-{res['error'][:20]}"] += 1
            continue
        stated = statement_outputs(text)
        run_ok = [json.dumps(o) for o in res["outputs"]]
        if stated is not None and len(stated) == len(cases):
            if any(not _same(a, b) for a, b in zip(stated, res["outputs"])):
                DROPS["code"]["oracle-disagrees-with-statement"] += 1
                continue
            r["tests_basis"] = "statement examples; oracle output == stated output"
        else:
            r["tests_basis"] = "statement examples; oracle output (statement outputs unparseable)"
        for c, o in zip(cases, run_ok):
            c["output"] = o
        r.update(emit=True, prompt=f"{text}\n\nStarter code:\n```python\n{snippet.strip()}\n```",
                 tests={"kind": "functional", "entry": f"Solution.{sig[0]}", "cases": cases})

    # neulab/leetcode: pre-cutoff LeetCode with canonical assert tests (train pool).
    nl = hf_snapshot("neulab/leetcode", ["data/*/python*.jsonl"])
    for f in sorted(nl.rglob("python*.jsonl")):
        for row in map(json.loads, open(f, encoding="utf-8")):
            itf = row["interface"]
            if itf.get("container") != "Solution":
                DROPS["code"]["neulab-non-solution"] += 1
                continue
            key = slug_key(row["task_id"])
            date = lc_date.get(key, "2025-04-30")      # else: LeetCodeDataset release bound
            starter = f"class Solution:\n    {itf['raw_signature']}\n        "
            r = {"uid": f"code:neulab:{row['task_id']}", "dataset": "code", "source": "neulab/leetcode",
                 "lineage": [key], "skel_text": row["problem_description"],
                 "idents": [p["name"] for p in itf["parameters"]], "emit": True,
                 "prompt": f"{row['problem_description'].strip()}\n\nStarter code:\n```python\n{starter}\n```",
                 "tests": {"kind": "assert", "entry": f"Solution.{itf['callable']}",
                           "source": row["canonical_tests"]["source"]}}
            date_record(r, date, "upper")
            recs.append(r)

    # LiveCodeBench (all releases, 2023-05..2025-04): pre-cutoff, train pool.
    from build_set import decode_lcb_tests
    PROV["livecodebench/code_generation_lite"] = {"files": {}}
    for f in sorted((RAW / "lcb").glob("test*.jsonl")):
        PROV["livecodebench/code_generation_lite"]["files"][f.name] = sha256_file(f)
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            tests = json.loads(it["public_test_cases"]) + decode_lcb_tests(it["private_test_cases"])
            meta = json.loads(it.get("metadata") or "{}")
            functional = tests and tests[0].get("testtype") == "functional"
            cases, size = [], 0
            for t in sorted(tests, key=lambda t: len(t["input"]) + len(t["output"])):
                size += len(t["input"]) + len(t["output"])
                if size > TEST_BYTES_CAP:
                    break
                cases.append({"input": t["input"], "output": t["output"]})
            if it["platform"] == "leetcode":
                lineage = [slug_key(it["question_title"])]
            else:
                lineage = [f"code:{it['platform']}:{it['question_id'].lower()}"]
            prompt = it["question_content"].strip()
            if functional:
                prompt += f"\n\nStarter code:\n```python\n{it['starter_code'].strip()}\n```"
                tst = {"kind": "functional", "entry": f"Solution.{meta.get('func_name')}", "cases": cases}
            else:
                prompt += "\n\nRead the input from stdin and write the answer to stdout."
                tst = {"kind": "stdin", "cases": cases}
            r = {"uid": f"code:lcb:{it['question_id']}", "dataset": "code", "source": f"livecodebench/{f.name}",
                 "lineage": lineage, "skel_text": it["question_content"], "emit": True, "prompt": prompt,
                 "tests": tst, "tests_truncated": len(cases) < len(tests)}
            date_record(r, it["contest_date"][:10], "exact")
            recs.append(r)
    return recs


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 + 1e-5 * abs(float(b))
        except Exception:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# --- mhqa ----------------------------------------------------------------------

ANTILEAK = HERE / "antileak" / "items.jsonl"


def render_paragraphs(paras: list[dict], question: str) -> str:
    body = "\n\n".join(f"[{i + 1}] {p['title']}\n{p['text'].strip()}" for i, p in enumerate(paras))
    return f"{body}\n\nQuestion: {question.strip()}"


def load_mhqa() -> list[dict]:
    recs = []
    for f in ("musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl"):
        p = RAW / "musique" / f
        PROV.setdefault("dgslibisey/MuSiQue", {"files": {}})["files"][f] = sha256_file(p)
        for m in map(json.loads, open(p, encoding="utf-8")):
            if not m.get("answerable", True):
                continue
            paras = [{"title": x["title"], "text": x["paragraph_text"]} for x in m["paragraphs"]]
            r = {"uid": f"mhqa:musique:{m['id']}", "dataset": "mhqa", "source": f"MuSiQue/{f}",
                 "lineage": [f"mhqa:musique-hop:{d['id']}" for d in m["question_decomposition"]],
                 "skel_text": m["question"], "emit": True, "prompt": render_paragraphs(paras, m["question"]),
                 "answer": {"answer": m["answer"], "aliases": m.get("answer_aliases", [])}}
            date_record(r, "2021-08-31", "upper")       # MuSiQue v1.0 release (arXiv 2108.00573)
            recs.append(r)
    if ANTILEAK.exists():
        PROV["antileak-sparql"] = {"file_sha256": sha256_file(ANTILEAK)}
        for it in map(json.loads, open(ANTILEAK, encoding="utf-8")):
            r = {"uid": it["uid"], "dataset": "mhqa", "source": it["source"], "lineage": it["lineage"],
                 "skel_text": it["question"], "emit": True,
                 "prompt": render_paragraphs(it["paragraphs"], it["question"]),
                 "answer": {"answer": it["answer"], "aliases": it.get("aliases", [])}}
            date_record(r, it["first_published"], "exact")
            recs.append(r)
    else:
        log.warning("  mhqa: %s missing -> no post-cutoff multi-hop items this run", ANTILEAK)
    return recs


LOADERS = {"math": load_math, "code": load_code, "mhqa": load_mhqa}
SKEL = {"math": (5, 0.7), "code": (5, 0.7), "mhqa": (3, 0.8)}
DROPS: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)


# --- split, seal, gate -----------------------------------------------------------

def hfrac(root: str) -> float:
    return int(hashlib.sha256(f"{SEED}:{root}".encode()).hexdigest()[:12], 16) / 16**12


PUBLIC = ("uid", "original_uid", "dataset", "source", "lineage", "root_id", "first_published", "date_basis", "prompt", "answer", "tests",
          "tests_basis", "tests_truncated")


def public(r: dict) -> dict:
    return {k: r[k] for k in PUBLIC if k in r}


def write_jsonl(p: Path, rows: list[dict]) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    raw = body.encode("utf-8")
    p.write_bytes(raw)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assign_content_ids(records: list[dict]) -> list[dict]:
    """Source IDs can repeat for different questions; never key clustering on them."""
    unique = {}
    for record in records:
        record["original_uid"] = record.get("original_uid", record["uid"])
        record["uid"] = DR.stable_uid(record)
        previous = unique.get(record["uid"])
        if previous is None:
            unique[record["uid"]] = record
        elif {previous["_when"], record["_when"]} != {previous["_when"]}:
            # Do not silently pick one of contradictory date annotations.
            raise ValueError("Identical task identity has conflicting publication classification")
    return list(unique.values())


def merge_provenance(previous: dict, current: dict) -> dict:
    """Retain source evidence when only a subset of datasets is rebuilt."""
    return {**previous, **current}


def sh(*args) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(SEAL_PY), *map(str, args)], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    for line in out.splitlines():
        log.info("    seal.py| %s", line)
    return r.returncode, out


def process(ds: str) -> dict:
    t0 = time.time()
    log.info("== %s: loading sources", ds)
    recs = assign_content_ids(LOADERS[ds]())
    log.info("  %s: %d records loaded", ds, len(recs))
    k, tau = SKEL[ds]
    root_of = R.cluster(recs, k, tau, log)
    by_root = collections.defaultdict(list)
    for r in recs:
        r["root_id"] = root_of[r["uid"]]
        by_root[r["root_id"]].append(r)

    split = {"train": [], "eval": [], "holdout": []}
    root_class = collections.Counter()
    for rid, members in by_root.items():
        whens = {m["_when"] for m in members}
        if "pre" in whens:
            cls = "pre"                      # any pre-cutoff member => whole lineage predates the model
        elif whens == {"post"}:
            cls = "post"
        else:
            cls = "ambiguous"
        root_class[cls] += 1
        emit = [m for m in members if m.get("emit", True)]
        if cls == "ambiguous":
            DROPS[ds]["ambiguous-date-root"] += len(emit)
            continue
        if cls == "pre":
            proven = [m for m in emit if m["_when"] == "pre"]
            DROPS[ds]["post-variant-of-pre-root"] += len(emit) - len(proven)
            split["train"] += proven
        else:
            split["holdout" if hfrac(rid) < HOLDOUT_FRAC else "eval"] += emit
    for part in split.values():
        part.sort(key=lambda r: r["uid"])

    # hard invariants
    rs = {k: {r["root_id"] for r in v} for k, v in split.items()}
    assert not (rs["train"] & rs["eval"]) and not (rs["train"] & rs["holdout"]) and not (rs["eval"] & rs["holdout"])
    assert all(r["first_published"] >= CUTOFF for r in split["eval"] + split["holdout"])
    assert all(r["first_published"] < CUTOFF for r in split["train"])

    out = {"root_classes": dict(root_class), "counts": {k: len(v) for k, v in split.items()},
           "roots": {k: len(v) for k, v in rs.items()}, "drops": dict(DROPS[ds])}
    name = f"{ds}-{VERSION}"
    work = HERE / ".work"
    # seal holdout first: nothing downstream exists until the holdout is locked
    if split["holdout"]:
        cand = work / f"{name}.holdout.jsonl"
        write_jsonl(cand, [public(r) for r in split["holdout"]])
        sdir = SEALED / name
        if sdir.exists():
            sealed_roots = set(json.loads((sdir / "roots.json").read_text())["root_ids"])
            fresh = {R_canon(r["root_id"]) for r in split["holdout"]}
            if sealed_roots != fresh:
                raise SystemExit(f"{name} already sealed with different roots -- data changed; bump VERSION")
            log.info("  %s: holdout already sealed with identical roots; verifying", name)
        else:
            code, _ = sh("seal", cand, "--name", name, "--min-date", CUTOFF, "--role", "eval")
            if code:
                raise SystemExit(f"sealing {name} failed")
        code, _ = sh("verify", sdir)
        if code:
            raise SystemExit(f"seal verification failed for {name}; no pools published")
        out["seal"] = json.loads((sdir / "manifest.json").read_text())["merkle_root"]
        out["seal_verified"] = True
    else:
        log.warning("  %s: no post-cutoff holdout items", ds)

    out["train_sha256"] = write_jsonl(POOLS / f"{ds}.train.jsonl", [public(r) for r in split["train"]])
    out["eval_sha256"] = write_jsonl(work / f"{ds}.eval.jsonl", [public(r) for r in split["eval"]])
    if split["holdout"]:
        for f in (POOLS / f"{ds}.train.jsonl", work / f"{ds}.eval.jsonl"):
            code, _ = sh("check", f, "--against", SEALED / name)
            out[f"leak_gate_{f.name}"] = "CLEAN" if code == 0 else "LEAKED"
            if code:
                raise SystemExit(f"leak gate failed: {f} vs {name}")
    out["eval_items"] = [public(r) for r in split["eval"]]
    log.info("  %s: train %d | eval %d | holdout %d (roots %s) | %.0fs", ds, len(split["train"]),
             len(split["eval"]), len(split["holdout"]), out["roots"], time.time() - t0)
    log.info("  %s drops: %s", ds, dict(DROPS[ds]))
    return out


def R_canon(v: str) -> str:
    return re.sub(r"[^a-z0-9]+", ":", v.strip().lower()).strip(":")


def write_rules_table(results: dict, stamp: str) -> None:
    rows = ["| dataset | eval source (post-cutoff) | train pool (pre-cutoff) | train | eval | holdout | holdout merkle root |",
            "|---|---|---|---|---|---|---|"]
    src = {"math": ("MathArena ArXivMath 03–08/2026, arXiv v1 date >= cutoff",
                    "AIME/HMMT/BRUMO/CMIMC/SMT/Apex 2023–Feb 2026, ArXivMath <= Feb 2026, arxivmath-training"),
           "code": ("Only independently proven exact publication dates qualify; first-activity upper bounds excluded",
                    "LiveCodeBench v1–v6 (2023-05..2025-04), neulab/leetcode"),
           "mhqa": ("AntiLeakBench method (Wikidata updates with start date >= cutoff) via SPARQL",
                    "MuSiQue-Ans v1.0 train+dev (2021)")}
    for ds, r in results.items():
        c = r["counts"]
        rows.append(f"| {ds} | {src[ds][0]} | {src[ds][1]} | {c['train']} | {c['eval']} | {c['holdout']} | "
                    f"`{r.get('seal', '-')[:16]}` |")
    table = "\n".join(rows) + f"\n\nGenerated by prepare.py at {stamp}; full log: `logs/prepare-{stamp}.log`.\n"
    text = RULES.read_text(encoding="utf-8")
    text = re.sub(r"(<!-- BEGIN GENERATED -->\n).*?(<!-- END GENERATED -->)", lambda m: m.group(1) + table + m.group(2),
                  text, flags=re.S)
    RULES.write_bytes(text.encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="math,code,mhqa")
    a = ap.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LOGS.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOGS / f"prepare-{stamp}.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("prepare %s | model %s | cutoff %s | leetcode margin %dd | holdout frac %.2f | seed %s",
             VERSION, MODEL, CUTOFF, LEETCODE_MARGIN_DAYS, HOLDOUT_FRAC, SEED)
    prev = {}
    previous_provenance = {}
    latest = LOGS / "latest.json"
    if latest.exists():
        previous = json.loads(latest.read_text(encoding="utf-8"))
        prev = previous.get("results", {})
        previous_provenance = previous.get("provenance", {})
    results = {k: v for k, v in prev.items()}
    for ds in a.datasets.split(","):
        results[ds] = process(ds)

    items = {ds: r.pop("eval_items") for ds, r in results.items() if "eval_items" in r}
    set_path = SETS / f"post_cutoff_{VERSION}.json"
    frozen = json.loads(set_path.read_text(encoding="utf-8")) if set_path.exists() else {"items": {}, "buckets": {}}
    for ds, its in items.items():
        frozen["items"][ds] = [{**it, "id": it["uid"]} for it in its]
        frozen["buckets"][ds] = {"n": len(its), "max_tokens": graders.MAX_TOKENS[ds]}
    frozen.update(schema=f"post_cutoff_{VERSION}", model=MODEL, cutoff=CUTOFF, created=stamp[:8], seed=SEED)
    set_path.write_bytes(json.dumps(frozen, ensure_ascii=False, indent=1).encode("utf-8"))
    log.info("eval set -> %s (%s)", set_path, {k: v["n"] for k, v in frozen["buckets"].items()})

    # Baseline = BASELINE per bucket, drawn from the eval split (never the holdout), lowest hash(uid) first.
    base = {**frozen, "schema": f"post_cutoff_{VERSION}_5pb", "items": {}, "buckets": {}}
    for ds, its in frozen["items"].items():
        pick = sorted(its, key=lambda it: hfrac(it["uid"]))[:BASELINE.get(ds, 0)]
        base["items"][ds] = pick
        base["buckets"][ds] = {**frozen["buckets"][ds], "n": len(pick)}
    base_path = SETS / f"post_cutoff_{VERSION}_5pb.json"
    base_path.write_bytes(json.dumps(base, ensure_ascii=False, indent=1).encode("utf-8"))
    log.info("baseline set -> %s (%s)", base_path, {k: v["n"] for k, v in base["buckets"].items()})

    summary = {"stamp": stamp, "model": MODEL, "cutoff": CUTOFF, "version": VERSION,
               "provenance": merge_provenance(previous_provenance, PROV),
               "eval_set": str(set_path.relative_to(REPO)), "eval_set_sha256": sha256_file(set_path),
               "results": results}
    for p in (LOGS / f"prepare-{stamp}.json", latest):
        p.write_bytes(json.dumps(summary, indent=1, sort_keys=True).encode("utf-8"))
    write_rules_table(results, stamp)
    log.info("done -> RULES.md table updated, summary logs/prepare-%s.json", stamp)


if __name__ == "__main__":
    main()
