"""Salvage capped failures: cut the thinking at a conclusion point, insert only
</think>, let the SAME model finish, keep it only if its own answer grades correct.

  .venv\\Scripts\\python.exe sft\\salvage.py --name r3-pod-mhqa-h34 --base http://127.0.0.1:8001

Reads sft/data/<name>.fail.jsonl (raw ids of every failed attempt) and
<name>.stats.json (the collection's args, to rebuild each task and to know which
model wrote the traces). Writes sft/data/<name>.salvaged.jsonl - never the
natural-wins file - with each record tagged "salvaged": {from, cut, inserted_ids}.

Rules (canon): cuts sit on a fixed grid (every CUT_STEP tokens after CUT_START,
up to MAX_CUTS), chosen without the gold answer or any phrase list. At each cut
the model answers twice (n=2); a cut counts only if BOTH own answers grade
correct, so a lucky guess before she found the answer (~8% per try) almost never
passes (~0.7%). The shortest passing completion is kept. The only token we write is </think>; everything after it is
sampled from the same served model that produced the trace. Only capped
(truncated) traces are salvaged: a finished-but-wrong attempt has no answer to
rescue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import requests
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect  # noqa: E402  (task rebuild + leak gate, same code as collection)
from collect import graders  # noqa: E402
from sft_schema import MAX_TOKENS, template_sha, validate  # noqa: E402

DATA = HERE / "data"
CUT_START = 200   # first cut, in thinking tokens
CUT_STEP = 150    # then every N tokens
MAX_CUTS = 10
N_CHECK = 2       # own answers per cut; all must pass


def cut_points(ids: list[int]) -> list[int]:
    return list(range(CUT_START, len(ids), CUT_STEP))[:MAX_CUTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="collection name (reads <name>.fail.jsonl + .stats.json)")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    ap.add_argument("--served-model", help="must equal the model that wrote the traces (default: from stats)")
    ap.add_argument("--bucket", default="mhqa,code")
    ap.add_argument("--limit", type=int, help="salvage at most N traces (smoke test)")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()

    stats = json.loads((DATA / f"{a.name}.stats.json").read_text(encoding="utf-8"))
    cargs = SimpleNamespace(**stats["args"])
    wrote = cargs.served_model or cargs.model
    served = a.served_model or wrote
    if served != wrote:
        sys.exit(f"traces were written by '{wrote}'; continuing them with '{served}' would make "
                 "it someone else's ending. Refusing.")
    out = DATA / f"{a.name}.salvaged.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; salvage outputs are immutable")

    tok = AutoTokenizer.from_pretrained(cargs.model)
    eos, end_think = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</think>")
    tsha = template_sha(tok)
    tasks = {t["uid"]: t for t in collect.load_tasks(cargs)}
    bad = collect.forbidden_roots()
    buckets = set(a.bucket.split(","))
    fails = [json.loads(l) for l in open(DATA / f"{a.name}.fail.jsonl", encoding="utf-8") if l.strip()]
    fails = [f for f in fails if f["status"] == "truncated" and f["bucket"] in buckets
             and f["uid"] in tasks and collect.seal.root_of(f, "root_id") not in bad]
    if a.limit:
        fails = fails[: a.limit]
    H = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else None

    def try_cut(f: dict, k: int) -> dict:
        t, pids, cids = tasks[f["uid"]], f["prompt_ids"], f["completion_ids"]
        prefix = cids[:k] + [end_think]
        room = MAX_TOKENS - len(pids) - len(prefix)
        if room < 16:
            return {"f": f, "k": k, "ok": False, "conts": []}
        r = requests.post(f"{a.base}/v1/completions", headers=H, timeout=1800, json={
            "model": served, "prompt": pids + prefix, "max_tokens": room, "n": N_CHECK,
            "temperature": cargs.temperature, "top_p": cargs.top_p,
            "return_token_ids": True, "skip_special_tokens": False})
        r.raise_for_status()
        conts = []
        for ch in r.json()["choices"]:
            cont = ch["token_ids"]
            passed, detail = graders.grade(t, tok.decode(cont, skip_special_tokens=True))
            clean = ch["finish_reason"] == "stop" and bool(cont) and cont[-1] == eos
            conts.append({"ids": cont, "passed": passed and clean, "detail": detail})
        return {"f": f, "k": k, "ok": all(c["passed"] for c in conts), "conts": conts, "room": room}

    def record(f: dict, best: dict, tried: list) -> dict:
        t, pids = tasks[f["uid"]], f["prompt_ids"]
        cont = min((c for c in best["conts"] if c["passed"]), key=lambda c: len(c["ids"]))
        comp = f["completion_ids"][: best["k"]] + [end_think] + cont["ids"]
        rec = {
            "id": f"{f['id']}-salv", "root_id": f["root_id"], "bucket": f["bucket"],
            "source": t.get("source", ""), "split": "train",
            "messages": [{**m, "train": False} for m in graders.messages(t)],
            "prompt_ids": pids, "completion_ids": comp,
            "completion_text": tok.decode(comp, skip_special_tokens=False),
            "n_tokens": len(pids) + len(comp),
            "grade": {"pass": True, "detail": cont["detail"]},
            "model": f"{cargs.model}+{served}", "tokenizer_template_sha": tsha,
            "sampling": {"temperature": cargs.temperature, "top_p": cargs.top_p, "max_tokens": best["room"]},
            "salvaged": {"from": f["id"], "cut": best["k"], "inserted_ids": [end_think],
                         "orig_len": len(f["completion_ids"]), "n_check": N_CHECK,
                         "cuts": {str(x["k"]): [c["passed"] for c in x["conts"]] for x in tried}},
        }
        validate(rec, eos)
        return rec

    jobs = [(f, k) for f in fails for k in cut_points(f["completion_ids"])]
    print(f"[salvage] {len(fails)} capped traces from '{a.name}' (written by {wrote}), continuing with "
          f"{served}: {len(jobs)} cuts x {N_CHECK} answers", flush=True)
    by_trace: dict[str, list] = {f["id"]: [] for f in fails}
    with ThreadPoolExecutor(max_workers=min(a.workers, max(1, len(jobs)))) as ex:
        for i, fut in enumerate(as_completed([ex.submit(try_cut, f, k) for f, k in jobs]), 1):
            r = fut.result()
            by_trace[r["f"]["id"]].append(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"[salvage] {i}/{len(jobs)} cuts done", flush=True)
    res = []
    with out.open("w", encoding="utf-8") as fh:
        for f in fails:
            tried = sorted(by_trace[f["id"]], key=lambda x: x["k"])
            passing = [x for x in tried if x["ok"]]
            rec = None
            if passing:  # shortest passing completion overall
                best = min(passing, key=lambda x: x["k"] + min(len(c["ids"]) for c in x["conts"]))
                rec = record(f, best, tried)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            res.append({"f": f, "rec": rec, "tried": tried})

    summary = {}
    for b in sorted({r["f"]["bucket"] for r in res}):
        rs = [r for r in res if r["f"]["bucket"] == b]
        ok = [r for r in rs if r["rec"]]
        summary[b] = {
            "capped_traces": len(rs), "cuts_tried": sum(len(r["tried"]) for r in rs),
            "salvaged": len(ok),
            "median_cut_token": int(median(r["rec"]["salvaged"]["cut"] for r in ok)) if ok else None,
            "median_orig_len": int(median(len(r["f"]["completion_ids"]) for r in ok)) if ok else None,
            "median_salvaged_len": int(median(len(r["rec"]["completion_ids"]) for r in ok)) if ok else None,
            "cuts_passing_both": sum(1 for r in rs for t in r["tried"] if t["ok"]),
            "cuts_one_of_two": sum(1 for r in rs for t in r["tried"] if sum(c["passed"] for c in t["conts"]) == 1),
        }
    (DATA / f"{a.name}.salvaged.stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
