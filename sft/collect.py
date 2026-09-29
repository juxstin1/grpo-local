"""Collect rejection-sampled SFT wins from the served model.

  # multi-hop from fresh invented worlds (never the eval seed), easier world for collection
  .venv\\Scripts\\python.exe sft\\collect.py --mhqa-seeds 1000-1019 --hops 2 --world 12,6,6,6 --name mhqa-easy
  # any pool in the graders.py record shape (math / code / mhqa)
  .venv\\Scripts\\python.exe sft\\collect.py --pool evals\\data\\pools\\math.train.jsonl --limit 50 --name math-50

Per task: K attempts at temperature 0.7, graded by the SAME endpoint grader as
the eval (evals/graders.py). A win is kept only if it passed, ended on EOS (not
truncated), and prompt+completion <= 4096 tokens - so the data also nudges
toward concise, terminating answers. At most --keep-per-task wins per task.

Leak gate (fail closed): a task is refused if its root_id is in the eval set or
in any sealed holdout. Token fidelity: vLLM's prompt ids must equal the local
chat template's ids, or the run aborts - the stored ids are exactly what the
trainer and GRPO will see.

Also writes <name>.stats.json: per-task pass counts, which is what sorts a
bucket into format-failure (SFT) / sometimes-right (GRPO) / never-right (easier).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVALS = ROOT / "evals"
sys.path[:0] = [str(EVALS), str(EVALS / "holdout"), str(EVALS / "datasets"), str(HERE)]
import graders  # noqa: E402
import seal  # noqa: E402  (root_of canonicalization, shared with the leak gate)
import synth_multihop  # noqa: E402
from sft_schema import MAX_TOKENS, template_sha, validate  # noqa: E402

EVAL_SET = EVALS / "data" / "sets" / "post_cutoff_v1_5pb.json"
SEALED = EVALS / "holdout" / "sealed"
OUT_DIR = HERE / "data"
MIN_COMPLETION = 256  # a prompt that leaves less room than this can't produce a win


def forbidden_roots() -> set[str]:
    roots = set()
    s = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    for items in s["items"].values():
        roots |= {seal.root_of(it, "root_id") for it in items}
    for d in SEALED.iterdir() if SEALED.exists() else []:
        roots |= set(json.loads((d / "roots.json").read_bytes())["root_ids"])
    roots.discard(None)
    return roots


def mhqa_tasks(seeds: list[int], hops: str, per_hop: int, world: str | None) -> list[dict]:
    """Invented-world tasks from the vendored generator (not modified here). --world
    overrides its world size for easier SFT collection; ids carry the size tag."""
    if synth_multihop.SEED in seeds:
        sys.exit(f"seed {synth_multihop.SEED} is the eval world; never collect training data from it")
    orig = synth_multihop.build_world
    sizes = tuple(int(x) for x in world.split(",")) if world else None
    tag = f"{synth_multihop.WORLD}-{'x'.join(map(str, sizes))}" if sizes else None
    out = []
    try:
        if sizes:
            synth_multihop.build_world = lambda seed: orig(seed, *sizes)
        for s in seeds:
            for r in synth_multihop.records(s, {int(h): per_hop for h in hops.split(",")}):
                if tag:
                    for k in ("uid", "root_id", "source"):
                        r[k] = r[k].replace(synth_multihop.WORLD, tag, 1)
                out.append(r)
    finally:
        synth_multihop.build_world = orig
    return out


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def load_tasks(a) -> list[dict]:
    tasks = []
    if a.mhqa_seeds:
        tasks += mhqa_tasks(parse_seeds(a.mhqa_seeds), a.hops, a.per_hop, a.world)
    excluded = set()
    for f in a.exclude_roots_from:  # e.g. the SFT data: compare on tasks the adapter never saw
        excluded |= {json.loads(l)["root_id"] for l in open(f, encoding="utf-8") if l.strip()}
    tasks = [t for t in tasks if t["root_id"] not in excluded]
    pool = []
    for p in a.pool:
        pool += [t for t in graders.load_pool(p) if t.get("root_id") not in excluded]
    if a.limit and len(pool) > a.limit:
        pool = random.Random(a.seed).sample(pool, a.limit)
    return tasks + pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=[])
    ap.add_argument("--mhqa-seeds", help="invented-world seeds, e.g. 1000-1019 (never the eval seed)")
    ap.add_argument("--hops", default="2,3")
    ap.add_argument("--per-hop", type=int, default=4, help="mhqa tasks per hop per seed")
    ap.add_argument("--world", help="people,companies,cities,rivers override, e.g. 12,6,6,6")
    ap.add_argument("--limit", type=int, help="random sample of N pool tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--keep-per-task", type=int, default=2)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="base model: tokenizer + template")
    ap.add_argument("--served-model", help="name to request from vLLM, e.g. a loaded LoRA adapter")
    ap.add_argument("--exclude-roots-from", nargs="*", default=[], help="jsonl files whose root_ids to skip")
    ap.add_argument("--base", default="http://127.0.0.1:8000",
                    help="vLLM URL; a remote pod works, e.g. https://<pod>-8000.proxy.runpod.net")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                    help="bearer token for a vLLM started with --api-key (default: $VLLM_API_KEY)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    tsha = template_sha(tok)
    try:  # local snapshot commit of the tokenizer the ids were checked against
        from huggingface_hub import snapshot_download
        rev = Path(snapshot_download(a.model, local_files_only=True)).name[:12]
    except Exception:
        rev = "unknown"

    bad = forbidden_roots()
    tasks, refused = [], 0
    for t in load_tasks(a):
        if seal.root_of(t, "root_id") in bad or not t.get("root_id"):
            refused += 1
            continue
        tasks.append(t)
    print(f"[collect] {len(tasks)} tasks ({refused} refused by leak gate), "
          f"{a.attempts} attempts each", flush=True)

    # Render once per task; skip prompts that leave no room under the 4096 cap.
    jobs, prompts, too_long = [], {}, 0
    for t in tasks:
        msgs = graders.messages(t)
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=True)  # same mode as eval + GRPO
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        if MAX_TOKENS - len(ids) < MIN_COMPLETION:
            too_long += 1
            continue
        prompts[t["uid"]] = (msgs, ids)
        jobs += [(t, k) for k in range(1, a.attempts + 1)]


    def fail_rec(t, k, status, pids, cids, ch):
        # Failure trace for salvage (cut at a conclusion, insert </think>, let the model
        # finish). Only the task ref is stored; salvage re-derives the task from args.
        return {"id": f"{t['uid']}-a{k}", "uid": t["uid"], "root_id": t["root_id"],
                "bucket": t["dataset"], "status": status, "finish_reason": ch["finish_reason"],
                "prompt_ids": pids, "completion_ids": cids}

    def one(t: dict, k: int) -> dict:
        msgs, pids = prompts[t["uid"]]
        body = {"model": a.served_model or a.model, "messages": msgs, "max_tokens": MAX_TOKENS - len(pids),
                "temperature": a.temperature, "top_p": a.top_p, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": True}}
        for tries in range(3):
            try:
                r = requests.post(f"{a.base}/v1/chat/completions", json=body, timeout=3600,
                                  headers={"Authorization": f"Bearer {a.api_key}"} if a.api_key else None)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if tries == 2:
                    return {"t": t, "k": k, "status": f"request-error:{type(e).__name__}", "n": 0}
                time.sleep(10)
        if data["prompt_token_ids"] != pids:
            raise RuntimeError(f"{t['uid']}: vLLM prompt ids != local chat template - "
                               "stored ids would not match training; aborting")
        ch = data["choices"][0]
        cids = ch["token_ids"]
        passed, detail = graders.grade(t, ch["message"].get("content") or "")
        if ch["finish_reason"] != "stop" or not cids or cids[-1] != eos:
            return {"t": t, "k": k, "status": "truncated", "passed": False, "n": len(cids),
                    "fail": fail_rec(t, k, "truncated", pids, cids, ch)}
        if not passed:
            return {"t": t, "k": k, "status": "fail", "passed": False, "n": len(cids),
                    "fail": fail_rec(t, k, "fail", pids, cids, ch)}
        rec = {
            "id": f"{t['uid']}-a{k}", "root_id": t["root_id"], "bucket": t["dataset"],
            "source": t.get("source", ""), "split": "train",
            "messages": [{**m, "train": False} for m in msgs],
            "prompt_ids": pids, "completion_ids": cids,
            "completion_text": tok.decode(cids, skip_special_tokens=False),
            "n_tokens": len(pids) + len(cids),
            "grade": {"pass": True, "detail": detail},
            "model": f"{a.model}@{rev}" + (f"+{a.served_model}" if a.served_model else ""),
            "tokenizer_template_sha": tsha,
            "sampling": {"temperature": a.temperature, "top_p": a.top_p,
                         "max_tokens": body["max_tokens"]},
        }
        validate(rec, eos)
        return {"t": t, "k": k, "status": "win", "passed": True, "rec": rec, "n": len(cids)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; pick a new --name (collections are immutable)")
    per_task: dict[str, dict] = {}
    kept: dict[str, int] = {}
    n = 0
    with ThreadPoolExecutor(max_workers=min(a.workers, max(1, len(jobs)))) as ex, \
            out.open("w", encoding="utf-8") as fh, \
            (OUT_DIR / f"{a.name}.fail.jsonl").open("w", encoding="utf-8") as ffh:
        futs = [ex.submit(one, t, k) for t, k in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            uid = res["t"]["uid"]
            st = per_task.setdefault(uid, {"bucket": res["t"]["dataset"], "root_id": res["t"]["root_id"],
                                           "pass": 0, "truncated": 0, "fail": 0, "error": 0, "kept": 0,
                                           "win_tokens": [], "all_tokens": []})
            s = res["status"]
            st["pass" if s == "win" else s if s in ("truncated", "fail") else "error"] += 1
            st["all_tokens"].append(res["n"])
            if s == "win":
                st["win_tokens"].append(res["n"])
            if s == "win" and kept.get(uid, 0) < a.keep_per_task:
                kept[uid] = kept.get(uid, 0) + 1
                st["kept"] += 1
                fh.write(json.dumps(res["rec"], ensure_ascii=False) + "\n")
                fh.flush()
            if res.get("fail"):  # every failed attempt's raw ids, for salvage
                ffh.write(json.dumps(res["fail"]) + "\n")
                ffh.flush()
            n += 1
            if n % 10 == 0 or n == len(jobs):
                print(f"[collect] {n}/{len(jobs)} graded, {sum(kept.values())} kept", flush=True)

    by_bucket: dict[str, dict] = {}
    for st in per_task.values():
        b = by_bucket.setdefault(st["bucket"], {"tasks": 0, "never": 0, "sometimes": 0, "always": 0,
                                                "attempts_truncated": 0, "kept": 0})
        b["tasks"] += 1
        b["kept"] += st["kept"]
        b["attempts_truncated"] += st["truncated"]
        graded = st["pass"] + st["fail"] + st["truncated"]
        b["never" if st["pass"] == 0 else "always" if st["pass"] == graded else "sometimes"] += 1
    stats = {"name": a.name, "model": f"{a.model}@{rev}", "attempts": a.attempts,
             "args": {**vars(a), "api_key": None},  # never write the key to disk
             "refused_leak_gate": refused, "skipped_prompt_too_long": too_long,
             "by_bucket": by_bucket, "per_task": per_task}
    (OUT_DIR / f"{a.name}.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({"kept": sum(kept.values()), "refused": refused, "too_long": too_long,
                      "by_bucket": by_bucket}, indent=1))


if __name__ == "__main__":
    main()
