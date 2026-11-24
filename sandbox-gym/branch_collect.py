"""Early-branch collector: her own fast paths, found by branching her real runs.

Per task (training seeds only):
  1. a ROOT episode of hers (served, token-exact ids).
  2. at each branch point t in --at (default 1,2,3): rebuild the workspace to its
     state after turn t (seed files + replay of her edits), then sample --branches
     continuations from that exact history (agent.run_episode prefix=root).
  3. keep a continuation only if the WHOLE episode (her t prefix turns + the
     branch) finishes inside the normal turn cap (config, 10) with:
       - a run_tests result that says PASS, then an ANSWER line,
       - the family grader passing (ANSWER required),
       - no edit to tests/, not truncated, token fidelity intact.
No cap or budget changes: kept wins have the pacing she must have at eval.
Every token is hers (prefix from the root run, the rest from the branch).

  ..\\.venv\\Scripts\\python.exe branch_collect.py --family editv_h --seeds 1000-1019 \\
      --name editv-branch-s1 --served-model r3

Output: data/collect/<name>.jsonl (same SFT record v2 shape as collect.py, plus
"branch": {"at": t, "root_turns": n}) and <name>.stats.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import agent
import gym_budget as GB
import encoding
import repo_gen
from collect import _segments, fidelity_error, touched_graded
from config import Config
from eval import make_task
from runner import Runner
from transformers import AutoTokenizer
from workspace import Workspace

TEST_ACTIONS = ("run_tests", "build")


def holdout_seeds() -> set[int]:
    """Every seed named by any sealed holdout manifest: never collected from."""
    out: set[int] = set()
    for p in Path("data/holdout").glob("*.json"):
        try:
            out |= {int(s) for s in json.loads(p.read_text(encoding="utf-8")).get("seeds", [])}
        except (ValueError, OSError):
            pass
    return out


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def tests_passed_then_answered(ep: agent.Episode) -> bool:
    saw_pass = False
    for t in ep.turns:
        if t.action and t.action.name == "run_tests" and t.obs.split("\n")[0].find("PASS") >= 0:
            saw_pass = True
    return saw_pass and ep.stop == "answer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="editv_h")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--at", default="1,2,3", help="branch after these turn counts")
    ap.add_argument("--branches", type=int, default=4, help="continuations per branch point")
    ap.add_argument("--roots", type=int, default=1, help="root episodes per task")
    ap.add_argument("--keep-per-task", type=int, default=2)
    ap.add_argument("--served-model", default="")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runner", default="docker")
    ap.add_argument("--hint", default="")
    ap.add_argument("--turn-counter", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="episode token budget (default 8192 / env GYM_BUDGET); read by gym_budget at import")
    a = ap.parse_args()

    seeds = parse_seeds(a.seeds)
    bad = set(seeds) & holdout_seeds()
    if bad:
        sys.exit(f"refusing holdout seeds {sorted(bad)}")
    out_dir = Path("data/collect")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists -- collections are immutable, pick a new --name")

    cfg = Config()
    cfg.serve.base, cfg.serve.served_model, cfg.episode.runner = a.base, a.served_model, a.runner
    GB.check_server(cfg.serve.base, cfg.episode.budget, api_key=cfg.serve.api_key)
    if a.turn_counter is not None:
        cfg.episode.turn_counter = a.turn_counter
    cfg.episode.hint = a.hint
    tok = AutoTokenizer.from_pretrained(a.model)
    at = [int(x) for x in a.at.split(",")]

    def plen(msgs):
        return len(encoding.render_chat(tok, msgs, cfg.episode.thinking))

    def fresh_ws(task, tag: str) -> Workspace:
        ws = Workspace(cfg.episode.data_root / "workspaces" / f"br-{task.name}-{tag}",
                       Runner(a.runner))
        ws.reset()
        ws.builds_left = cfg.episode.max_builds
        repo_gen.write_files(repo_gen.find_spec(task.meta["seed"], task.family,
                                                task.name)["files"], ws.root)
        return ws

    def keep(task, ep) -> str | None:
        """None = keep; else the reason it's dropped."""
        if ep.error:
            return "error"
        if ep.truncated or ep.cap_hit or len(ep.turns) > cfg.episode.max_turns:
            return "cap/truncated"
        if not tests_passed_then_answered(ep):
            return "no-pass-then-answer"
        if task.reward(ep.turns, ep.ws) < 0.5:
            return "grade-fail"
        if touched_graded(ep):
            return "touched-tests"
        if fidelity_error(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg),
                          cfg.episode.hint):
            return "fidelity"
        return None

    def run_task(spec: dict) -> tuple[str, list[dict], dict]:
        task = make_task({**spec, "checks": spec.get("checks", [])})
        st = {"roots": 0, "root_wins": 0, "branches": 0, "kept": 0, "drop": {}}
        recs: list[dict] = []
        for r in range(a.roots):
            root = agent.run_episode(cfg, task, fresh_ws(task, f"r{r}"),
                                     prompt_len_fn=plen, return_ids=True)
            st["roots"] += 1
            if root.error:
                st["drop"]["root-error"] = st["drop"].get("root-error", 0) + 1
                continue
            if keep(task, root) is None:
                st["root_wins"] += 1          # natural fast win: already collectable
            for t in at:
                if t >= len(root.turns) or root.turns[t - 1].action is None:
                    continue                  # root ended before this point
                for b in range(a.branches):
                    if st["kept"] >= a.keep_per_task:
                        break
                    ws = fresh_ws(task, f"r{r}t{t}b{b}")
                    agent.replay_edits(ws, root.turns[:t])
                    ws.builds_left -= sum(1 for x in root.turns[:t]
                                          if x.action and x.action.name in TEST_ACTIONS)
                    ep = agent.run_episode(cfg, task, ws, prompt_len_fn=plen,
                                           return_ids=True, prefix=root, prefix_len=t)
                    st["branches"] += 1
                    why = keep(task, ep)
                    if why:
                        st["drop"][why] = st["drop"].get(why, 0) + 1
                        continue
                    segs = _segments(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg))
                    recs.append({
                        "id": f"{task.name}-r{r}-t{t}-b{b}", "root_id": task.meta["seed"],
                        "bucket": task.family, "split": "train",
                        "branch": {"at": t, "root_turns": len(root.turns),
                                   "turns": len(ep.turns)},
                        "messages": [
                            {"role": "system", "content": agent.system_prompt(task.tools, agent.counter_turns(cfg),
                                                                   cfg.episode.hint),
                             "train": False},
                            {"role": "user", "content": task.prompt, "train": False}],
                        "prompt_ids": ep.prompt_ids_per_turn[0],
                        "segments": segs,
                        "completion_ids": [i for s in segs if s["train"] for i in s["ids"]],
                        "n_tokens": ep.n_tokens,
                        "grade": {"pass": True, "detail": "branch-win"},
                        "model": a.model + (f"+{a.served_model}" if a.served_model else ""),
                        "sampling": {"temperature": cfg.serve.temperature,
                                     "top_p": cfg.serve.top_p,
                                     "budget": cfg.episode.budget,
                                     "max_turns": cfg.episode.max_turns,
                                     "turn_counter": cfg.episode.turn_counter},
                    })
                    st["kept"] += 1
        return task.name, recs, st

    specs = [sp for s in seeds for sp in repo_gen.make_specs(s, a.family)]
    print(f"[branch] {len(specs)} tasks, branch at {at} x{a.branches}, "
          f"cap {cfg.episode.max_turns} turns", flush=True)
    stats, kept = {}, 0
    with ThreadPoolExecutor(a.workers) as ex, out.open("w", encoding="utf-8") as fh:
        for name, recs, st in ex.map(run_task, specs):
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            kept += len(recs)
            stats[name] = st
            print(f"  {name}: kept {st['kept']}/{st['branches']} branches "
                  f"(root win: {st['root_wins']}) drops {st['drop']}", flush=True)
    tot = {"tasks": len(specs), "kept": kept,
           "branches": sum(s["branches"] for s in stats.values()),
           "tasks_with_win": sum(1 for s in stats.values() if s["kept"]),
           "root_wins": sum(s["root_wins"] for s in stats.values())}
    drops: dict = {}
    for s in stats.values():
        for k, v in s["drop"].items():
            drops[k] = drops.get(k, 0) + v
    tot["drops"] = drops
    (out_dir / f"{a.name}.stats.json").write_text(
        json.dumps({"total": tot, "per_task": stats, "family": a.family, "at": at,
                    "branches": a.branches, "served_model": a.served_model}, indent=1),
        encoding="utf-8")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
