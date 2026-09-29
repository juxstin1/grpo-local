"""Eval runner for the sandbox gym.

Reads a set file (data/holdout/gym_v1.json or any repo_gen spec set), serves
each item pass@k through the opencode loop, grades mechanically, writes raw
per-attempt lines + a per-family summary. Resumable: done (item, attempt) pairs
are skipped; request-errors are retried.

Usage (Windows PowerShell):
  .\\cli.py eval --served-model r3 --family find --attempts 4

Sealed holdout: `--sealed data/holdout/gym_v1.json` verifies the canary + hash
and records pass/fail only (never transcripts).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agent
import seal
from config import Config
from runner import Runner
from schema import Task
from transformers import AutoTokenizer
from workspace import Workspace

MAX_WORKERS = 16


def make_task(item: dict) -> Task:
    """Holdout item dict -> Task with the family's mechanical reward."""
    task = Task(
        name=item["name"], family=item["family"], prompt=item["prompt"],
        tools=item["tools"], ground_truth=item["ground_truth"],
        reward=lambda turns, ws: 0.0,
        meta={"seed": item["seed"], "checks": item.get("checks", [])},
    )
    from graders import reward_for
    task.reward = reward_for(task)
    return task


def _raw_dir(results: Path, slug: str) -> Path:
    d = results / slug / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="data/holdout/gym_v1.json")
    ap.add_argument("--sealed", action="store_true",
                    help="verify seal, score aggregate-only (no transcripts)")
    ap.add_argument("--family", default=None, help="limit to one family")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--slug", required=True, help="results folder name")
    ap.add_argument("--served-model", default="", help="LoRA adapter name on vLLM")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="tokenizer source")
    ap.add_argument("--runner", default="docker")
    a = ap.parse_args()

    set_path = Path(a.set)
    if a.sealed:
        ok, m = seal.verify(set_path)
        if not ok:
            sys.exit(f"seal verify FAILED for {set_path} -- holdout not provably clean")
        items = m["items"]
        print(f"[seal] verified {set_path} ({len(items)} items)")
    else:
        items = json.loads(set_path.read_text(encoding="utf-8"))["items"]
    if a.family:
        items = [it for it in items if it["family"] == a.family]

    cfg = Config()
    cfg.serve.base = a.base
    cfg.serve.served_model = a.served_model
    cfg.episode.runner = a.runner
    tok = AutoTokenizer.from_pretrained(a.model)

    def prompt_len(msgs) -> int:
        from encoding import render_chat
        return len(render_chat(tok, msgs, cfg.episode.thinking))

    raw = _raw_dir(cfg.episode.data_root / "results", a.slug)
    results = cfg.episode.data_root / "results" / a.slug
    done: set[tuple[str, int]] = set()
    for f in raw.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("passed") is not None:
                    done.add((rec["name"], rec["attempt"]))
            except Exception:
                pass

    jobs = [(it, k) for it in items for k in range(1, a.attempts + 1)
            if (it["name"], k) not in done]
    print(f"[eval] {len(jobs)} attempts across {len(items)} items "
          f"({a.family or 'all families'})", flush=True)

    def one(pair):
        it, k = pair
        task = make_task(it)
        st = time.time()
        ws = Workspace(cfg.episode.data_root / "workspaces" / f"{it['name']}-a{k}",
                       Runner(a.runner))
        ws.reset()
        ws.builds_left = cfg.episode.max_builds
        from repo_gen import find_spec, write_files
        write_files(find_spec(it["seed"], it["family"], it["name"])["files"], ws.root)
        ep = agent.run_episode(cfg, task, ws, prompt_len_fn=prompt_len)
        if ep.error:
            return it, k, {"name": it["name"], "attempt": k, "passed": None,
                           "error": ep.error}
        passed = bool(task.reward(ep.turns, ws) >= 0.5)
        rec = {"name": it["name"], "family": it["family"], "attempt": k,
               "passed": passed,
               "latency_ms": int((time.time() - st) * 1000),
               "n_turns": len(ep.turns), "truncated": ep.truncated,
               "tool_uses": sum(1 for t in ep.turns if t.action is not None)}
        if not a.sealed:
            rec.update(answer=ep.answer, head=(ep.turns[-1].content[:200]
                                               if ep.turns else ""))
        return it, k, rec

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(jobs)))) as ex:
        futs = {ex.submit(one, pair): pair for pair in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            it, k, rec = fut.result()
            fam = rec["family"]
            with (raw / f"{fam}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(jobs):
                print(f"[eval] {n}/{len(jobs)} graded", flush=True)

    # summarize per family
    for fam_file in raw.glob("*.jsonl"):
        fam = fam_file.stem
        fam_items = [it for it in items if it["family"] == fam]
        item_pass, att_n, att_pass, att_trunc = {}, 0, 0, 0
        for line in fam_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("passed") is None:
                continue
            att_n += 1
            att_pass += bool(rec["passed"])
            att_trunc += bool(rec.get("truncated"))
            item_pass.setdefault(rec["name"], 0)
            item_pass[rec["name"]] = max(item_pass[rec["name"]], int(rec["passed"]))
        solved = sum(item_pass.values())
        total = len(fam_items)
        acc = solved / total if total else 0.0
        print(f"[{fam}] pass@{a.attempts} {solved}/{total} ({acc:.1%}) | "
              f"per-attempt {att_pass}/{att_n} | truncated {att_trunc}/{att_n}")
        (results / f"{fam}.summary.json").write_text(json.dumps({
            "family": fam, "n": total, "pass_k": a.attempts, "accuracy": round(acc, 4),
            "per_attempt_acc": round(att_pass / att_n, 4) if att_n else 0.0,
            "truncated": att_trunc, "sealed": a.sealed,
            **({} if a.sealed else {"per_item": dict(sorted(item_pass.items()))}),
        }, indent=1), encoding="utf-8")
    print("DONE", a.slug)


if __name__ == "__main__":
    main()