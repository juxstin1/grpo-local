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
import gym_budget as GB
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
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override the episode turn cap (default: config, 10)")
    ap.add_argument("--turn-counter", action=argparse.BooleanOptionalAction, default=None,
                    help="override config.episode.turn_counter (default: on)")
    ap.add_argument("--transcripts", action="store_true",
                    help="save full per-attempt transcripts (refused with --sealed)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help="concurrent episodes (lower it when the desktop shares the GPU)")
    ap.add_argument("--budget", type=int, default=None,
                    help="override the token budget (default 8192, or env GYM_BUDGET) -- "
                         "recorded in every record; never compare across budgets")
    ap.add_argument("--compact", action="store_true",
                    help="in-episode compaction (75%% trigger + checkpoint tool); default off")
    ap.add_argument("--compact-note-mode", choices=["off", "force"], default="off")
    ap.add_argument("--max-resets", type=int, default=None)
    ap.add_argument("--ws-root", default=None,
                    help="workspace dir (default data/workspaces). Give each concurrently "
                         "running model its own, or ws.reset() wipes the other's episodes")
    a = ap.parse_args()

    if a.transcripts and a.sealed:
        sys.exit("--transcripts is refused on a sealed holdout")
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
    # drift guard: the generator rebuilds each workspace at eval time, so a set
    # sealed under an older generator would silently score a different task
    from repo_gen import find_spec
    for it in items:
        sp = find_spec(it["seed"], it["family"], it["name"])
        if (sp["prompt"], sp["ground_truth"]) != (it["prompt"], it["ground_truth"]):
            sys.exit(f"generator drift on {it['name']}: {set_path} was built by an older "
                     "generator -- reseal from fresh output, don't score it")

    cfg = Config()
    cfg.serve.base = a.base
    cfg.serve.served_model = a.served_model
    cfg.episode.runner = a.runner
    if a.turn_counter is not None:
        cfg.episode.turn_counter = a.turn_counter
    if a.max_turns:
        cfg.episode.max_turns = a.max_turns
    if a.budget:
        cfg.episode.budget = a.budget      # same value gym_budget read from argv
    GB.check_server(cfg.serve.base, cfg.episode.budget, api_key=cfg.serve.api_key)
    cfg.episode.compact = a.compact
    cfg.episode.compact_note_mode, cfg.episode.max_resets = a.compact_note_mode, a.max_resets
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
                    GB.require_same([rec.get("budget"), cfg.episode.budget],
                                    f"existing raw records in {raw} vs this run")
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
        ws_root = Path(a.ws_root) if a.ws_root else cfg.episode.data_root / "workspaces"
        ws = Workspace(ws_root / f"{it['name']}-a{k}",
                       Runner(a.runner))
        ws.reset()
        ws.builds_left = cfg.episode.max_builds
        from repo_gen import find_spec, write_files
        write_files(find_spec(it["seed"], it["family"], it["name"])["files"], ws.root)
        ep = agent.run_episode(cfg, task, ws, prompt_len_fn=prompt_len,
                               tokenizer=tok if cfg.episode.compact else None)
        if ep.error:
            return it, k, {"name": it["name"], "family": it["family"], "attempt": k, "passed": None,
                           "error": ep.error}
        passed = bool(task.reward(ep.turns, ws) >= 0.5)
        from graders import files_ok_ignoring_answer
        no_answer_ok = (not passed and ep.answer is None
                        and files_ok_ignoring_answer(task, ep.turns, ws))
        rec = {"name": it["name"], "family": it["family"], "attempt": k,
               "model": cfg.serve.served_model or cfg.serve.model_name,
               "passed": passed, "budget": cfg.episode.budget,
               "latency_ms": int((time.time() - st) * 1000),
               "n_turns": len(ep.turns), "truncated": ep.truncated,
               "cap_hit": ep.cap_hit, "stop": ep.stop, "files_ok_no_answer": no_answer_ok,
               "tool_uses": sum(1 for t in ep.turns if t.action is not None)}
        if cfg.episode.compact:
            import compact
            rec["compact"] = compact.summary(ep)
            rec["compact"]["note_mode"] = cfg.episode.compact_note_mode
            if not a.sealed:
                rec["compact"]["notes"] = [r.get("note_text") for r in ep.resets]
        if not a.sealed:
            rec.update(answer=ep.answer, head=(ep.turns[-1].content[:200]
                                               if ep.turns else ""))
        if a.transcripts:
            tdir = results / "transcripts"
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"{it['name']}-a{k}.json").write_text(json.dumps({
                "name": it["name"], "passed": passed, "stop": ep.stop,
                "ground_truth": it["ground_truth"],
                "turns": [{"content": t.content, "reasoning": t.reasoning[-1500:],
                           "action": ({"name": t.action.name, "arguments": t.action.arguments}
                                      if t.action else None),
                           "obs": t.obs[:3000]} for t in ep.turns]},
                indent=1, ensure_ascii=False), encoding="utf-8")
        return it, k, rec

    with ThreadPoolExecutor(max_workers=min(a.workers, max(1, len(jobs)))) as ex:
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
        item_pass, att_n, att_pass, att_trunc, att_cap, att_noans, att_empty = {}, 0, 0, 0, 0, 0, 0
        comp = {"episodes": 0, "resets": 0, "harness": 0, "self": 0, "notes_truncated": 0,
                "views_trimmed": 0, "with_truncated_note": [0, 0], "no_truncated_note": [0, 0]}
        for line in fam_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("passed") is None:
                continue
            att_n += 1
            att_pass += bool(rec["passed"])
            att_trunc += bool(rec.get("truncated"))
            att_noans += bool(rec.get("files_ok_no_answer"))
            att_empty += rec.get("stop") == "empty"
            cp = rec.get("compact")
            if cp is not None:
                comp["episodes"] += 1
                comp["resets"] += cp["resets"]
                comp["harness"] += cp["resets_harness"]
                comp["self"] += cp["resets_self"]
                comp["notes_truncated"] += cp["notes_truncated"]
                comp["views_trimmed"] += cp["views_trimmed"]
                if cp["compact_fail"]:
                    key = "fail_" + cp["compact_fail"]
                    comp[key] = comp.get(key, 0) + 1
                grp = "with_truncated_note" if cp["notes_truncated"] else "no_truncated_note"
                comp[grp][0] += bool(rec["passed"])
                comp[grp][1] += 1
            # older records lack cap_hit: no answer after max_turns = a cap hit
            att_cap += bool(rec.get("cap_hit", rec.get("n_turns", 0) >= Config().episode.max_turns
                                    and not rec.get("answer") and not rec.get("passed")))
            item_pass.setdefault(rec["name"], 0)
            item_pass[rec["name"]] = max(item_pass[rec["name"]], int(rec["passed"]))
        solved = sum(item_pass.values())
        total = len(fam_items)
        acc = solved / total if total else 0.0
        print(f"[{fam}] pass@{a.attempts} {solved}/{total} ({acc:.1%}) | "
              f"per-attempt {att_pass}/{att_n} | truncated {att_trunc}/{att_n} | "
              f"turn-cap {att_cap}/{att_n} | files-ok-no-answer {att_noans}/{att_n} | "
              f"empty(unclosed think) {att_empty}/{att_n}")
        if comp["episodes"]:
            w, n = comp["with_truncated_note"], comp["no_truncated_note"]
            fails = {k: v for k, v in comp.items() if k.startswith("fail_")}
            print(f"[{fam}] compaction: resets {comp['resets']} (harness {comp['harness']}, "
                  f"self {comp['self']}) | notes truncated {comp['notes_truncated']} | "
                  f"views trimmed {comp['views_trimmed']} | fails {fails} | pass "
                  f"with truncated note {w[0]}/{w[1]}, without {n[0]}/{n[1]}")
        (results / f"{fam}.summary.json").write_text(json.dumps({
            "family": fam, "budget": cfg.episode.budget, "n": total, "pass_k": a.attempts, "accuracy": round(acc, 4),
            "per_attempt_acc": round(att_pass / att_n, 4) if att_n else 0.0,
            "truncated": att_trunc, "turn_cap": att_cap,
            "files_ok_no_answer": att_noans, "empty_unclosed_think": att_empty, "sealed": a.sealed,
            **({"compaction": comp} if comp["episodes"] else {}),
            **({} if a.sealed else {"per_item": dict(sorted(item_pass.items()))}),
        }, indent=1), encoding="utf-8")
    print("DONE", a.slug)


if __name__ == "__main__":
    main()