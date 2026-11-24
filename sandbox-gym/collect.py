"""Rejection-sampled SFT collector from the served model (multi-turn, token-exact).

Per task: K attempts at temperature 0.7 through the opencode loop, each in a
FRESH workspace, graded by the family's mechanical grader (the SAME one eval and
GRPO use). A record is kept only if it graded a win and finished under budget.
At most --keep-per-task wins per task.

Token fidelity (fail closed): every turn's served prompt ids must equal the
local chat-template render (encoding.render_chat) of the same messages, or the
run aborts -- stored ids are exactly what the trainer and GRPO will see.

Record shape (SFT record v2, per-segment trainable mask):
  {"id", "root_id": seed, "bucket": family, "split": "train",
   "messages": [...transcript...],
   "prompt_ids": [...initial prompt...],
   "segments": [{"kind": "prompt"|"assistant"|"obs", "ids": [...], "train": bool}, ...],
   "completion_ids": [...all assistant ids concatenated...],
   "n_tokens", "grade", "model", "sampling", "budget"}

Leak gate: seeds in the sealed holdout are refused. Split is by seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agent
import compact
import encoding
import repo_gen
import seal
from config import Config
import gym_budget as GB
from eval import make_task
from runner import Runner
from schema import Task
from transformers import AutoTokenizer
from workspace import Workspace

HOLDOUT = Path("data/holdout/gym_v1.json")


def forbidden_seeds() -> set[int]:
    if not HOLDOUT.exists():
        return set()
    ok, m = seal.verify(HOLDOUT)
    return set(m.get("seeds", [])) if ok else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="e.g. 2000-2019")
    ap.add_argument("--family", default="find")
    ap.add_argument("--name", required=True, help="collection name (immutable)")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--keep-per-task", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--served-model", default="", help="LoRA adapter name")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="tokenizer source")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--runner", default="docker")
    ap.add_argument("--hint", default="",
                    help="collection-only system line (context distillation); "
                         "strip with sft/from_gym.py --strip-hint")
    ap.add_argument("--compact", action="store_true",
                    help="in-episode compaction (75%% trigger + checkpoint tool); default off")
    ap.add_argument("--compact-note-mode", choices=["off", "force"], default="off")
    ap.add_argument("--max-resets", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None,
                    help="override sampling temperature (match GRPO's for prescreens)")
    ap.add_argument("--budget", type=int, default=None,
                    help="episode token budget (default 8192 / env GYM_BUDGET); read by gym_budget at import")
    a = ap.parse_args()

    cfg = Config()
    if a.temperature is not None:
        cfg.serve.temperature = a.temperature
    cfg.serve.base, cfg.serve.served_model = a.base, a.served_model
    cfg.serve.api_key = a.api_key
    cfg.episode.family = a.family
    cfg.episode.runner = a.runner
    cfg.episode.hint = a.hint
    GB.check_server(cfg.serve.base, cfg.episode.budget, cfg.serve.api_key)
    cfg.episode.compact = a.compact
    cfg.episode.compact_note_mode, cfg.episode.max_resets = a.compact_note_mode, a.max_resets
    tok = AutoTokenizer.from_pretrained(a.model)

    seeds = []
    for part in a.seeds.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    bad = forbidden_seeds()
    refused = [s for s in seeds if s in bad]
    seeds = [s for s in seeds if s not in bad]
    if refused:
        print(f"[collect] refused holdout seeds: {refused}", flush=True)

    tasks = [make_task(sp) for s in seeds for sp in repo_gen.make_specs(s, a.family)]
    if not tasks:
        sys.exit("no tasks; pick seeds/family that generate tasks")
    print(f"[collect] {len(tasks)} tasks, {a.attempts} attempts each", flush=True)

    out_dir = Path("data/collect")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; pick a new --name (collections are immutable)")

    def prompt_len(msgs) -> int:
        return len(encoding.render_chat(tok, msgs, cfg.episode.thinking))

    def one(task, k) -> tuple[Task, int, agent.Episode]:
        ws = Workspace(cfg.episode.data_root / "workspaces" /
                       f"collect-{task.name}-a{k}", Runner(a.runner))
        ws.reset()
        ws.builds_left = cfg.episode.max_builds
        repo_gen.write_files(repo_gen.find_spec(task.meta["seed"],
                                                task.family, task.name)["files"],
                             ws.root)
        ep = agent.run_episode(cfg, task, ws, prompt_len_fn=prompt_len,
                               return_ids=True, tokenizer=tok)
        return task, k, ep

    kept, n = 0, 0
    per_task: dict[str, dict] = {}
    jobs = [(t, k) for t in tasks for k in range(1, a.attempts + 1)]
    with ThreadPoolExecutor(max_workers=min(a.workers, max(1, len(jobs)))) as ex, \
            out.open("w", encoding="utf-8") as fh, \
            (out_dir / f"{a.name}.fail.jsonl").open("w", encoding="utf-8") as ffh:
        futs = {ex.submit(one, t, k): (t, k) for t, k in jobs}
        for fut in as_completed(futs):
            task, k, ep = fut.result()
            n += 1
            st = per_task.setdefault(task.name, {
                "family": task.family, "root_id": task.meta["seed"],
                "pass": 0, "truncated": 0, "fail": 0, "error": 0,
                "kept": 0, "touched_tests": 0, "cap_hit": 0, "win_tokens": []})
            if ep.error:
                st["error"] += 1
                ffh.write(json.dumps({"id": f"{task.name}-a{k}", "seed": task.meta["seed"],
                                      "family": task.family, "status": "error",
                                      "error": ep.error}, ensure_ascii=False) + "\n")
            else:
                ws = ep.ws
                assert ws is not None
                passed = bool(task.reward(ep.turns, ws) >= 0.5)
                if ep.truncated:
                    st["truncated"] += 1
                elif not passed:
                    st["fail"] += 1
                else:
                    st["pass"] += 1
                if ep.cap_hit:
                    st["cap_hit"] += 1
                    passed = False            # never finished: not a natural win
                if ep.compact_fail:
                    st["compact_fail"] = st.get("compact_fail", 0) + 1
                    passed = False
                if passed and not ep.truncated and touched_graded(ep):
                    st["touched_tests"] += 1
                    passed = False            # logged as fail below, never kept
                if passed and not ep.truncated and st["kept"] < a.keep_per_task and cfg.episode.compact:
                    f = compact.fidelity_error(ep, tok, cfg.episode.thinking,
                                               agent.counter_turns(cfg))
                    if f:
                        raise RuntimeError(f"[collect] {task.name}: {f} -- aborting")
                    seqs = compact.sequences(ep, tok, cfg.episode.thinking,
                                             agent.counter_turns(cfg))
                    summ = compact.summary(ep)
                    for j, sq in enumerate(seqs):   # one SFT record per window / note call
                        fh.write(json.dumps({
                            "id": f"{task.name}-a{k}-s{j}", "root_id": task.meta["seed"],
                            "bucket": task.family, "split": "train",
                            "compact": {"seq": j, "kind": sq["kind"], "window": sq["window"],
                                        "of": len(seqs), "resets": summ["resets"]},
                            "messages": [],
                            "prompt_ids": sq["segments"][0]["ids"],
                            "segments": sq["segments"],
                            "completion_ids": [i for s in sq["segments"] if s["train"]
                                               for i in s["ids"]],
                            "grade": {"pass": True, "detail": ""},
                            "model": a.model + (f"+{a.served_model}" if a.served_model else ""),
                            "sampling": {"temperature": cfg.serve.temperature,
                                         "top_p": cfg.serve.top_p,
                                         "budget": cfg.episode.budget},
                        }, ensure_ascii=False) + "\n")
                    fh.flush()
                    st["kept"] += 1
                    kept += 1
                    st["win_tokens"].append(ep.n_tokens)
                elif passed and not ep.truncated and st["kept"] < a.keep_per_task:
                    f = fidelity_error(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg),
                                       cfg.episode.hint)
                    if f:
                        raise RuntimeError(f"[collect] {task.name}: {f} -- aborting")
                    segs = _segments(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg))
                    rec = {
                        "id": f"{task.name}-a{k}", "root_id": task.meta["seed"],
                        "bucket": task.family, "split": "train",
                        "messages": [
                            {"role": "system", "content": agent.system_prompt(task.tools, agent.counter_turns(cfg),
                                                                   cfg.episode.hint),
                             "train": False},
                            {"role": "user", "content": task.prompt, "train": False},
                        ],
                        "prompt_ids": ep.prompt_ids_per_turn[0] if ep.prompt_ids_per_turn else [],
                        "segments": segs,
                        "completion_ids": [i for s in segs if s["train"] for i in s["ids"]],
                        "n_tokens": ep.n_tokens,
                        "grade": {"pass": True, "detail": ""},
                        "model": a.model + (f"+{a.served_model}" if a.served_model else ""),
                        "sampling": {"temperature": cfg.serve.temperature,
                                     "top_p": cfg.serve.top_p,
                                     "budget": cfg.episode.budget},
                    }
                    st["kept"] += 1
                    kept += 1
                    st["win_tokens"].append(ep.n_tokens)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                if not passed or ep.truncated:
                    ffh.write(json.dumps({"id": f"{task.name}-a{k}", "seed": task.meta["seed"],
                                          "family": task.family,
                                          "status": "truncated" if ep.truncated else "fail",
                                          "answer": ep.answer}, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(jobs):
                print(f"[collect] {n}/{len(jobs)} graded, {kept} kept", flush=True)

    (out_dir / f"{a.name}.stats.json").write_text(json.dumps({
        "name": a.name, "family": a.family, "attempts": a.attempts,
        "refused_seeds": refused, "per_task": per_task,
    }, indent=1), encoding="utf-8")
    print(json.dumps({"kept": kept, "refused_seeds": refused}, indent=1))


def fidelity_error(ep: agent.Episode, tok, thinking: bool,
                   max_turns: int | None = None, hint: str = "") -> str | None:
    """Every turn's served prompt ids must equal the local template render."""
    msgs = [{"role": "system", "content": agent.system_prompt(ep.task.tools, max_turns, hint)},
            {"role": "user", "content": ep.task.prompt}]
    for i, turn in enumerate(ep.turns):
        served = ep.prompt_ids_per_turn[i] if i < len(ep.prompt_ids_per_turn) else []
        local = encoding.render_chat(tok, msgs, thinking)
        if served != local:
            return (f"turn {i}: served prompt ids != local template "
                    f"(len {len(served)} vs {len(local)})")
        msgs.append(agent.assistant_message(turn.content, turn.reasoning))
        if turn.obs:
            msgs.append({"role": "user", "content": agent.observation_text(
                turn.action.name if turn.action else "",
                turn.action.arguments if turn.action else {}, turn.obs,
                i + 1, max_turns)})
    return None


def touched_graded(ep: agent.Episode) -> str | None:
    """A win that wrote to tests/ may be a cheat; SFT would teach it. Drop it."""
    for t in ep.turns:
        if t.action and t.action.name in ("edit_file", "write_file"):
            path = str(t.action.arguments.get("path", "")).replace("\\", "/").lstrip("./")
            if path.startswith("tests/") or "/tests/" in path:
                return path
    return None


def _segments(ep: agent.Episode, tok, thinking: bool,
              max_turns: int | None = None) -> list[dict]:
    """prompt + assistant turns (trainable) + observations (not)."""
    segs: list[dict] = []
    if ep.prompt_ids_per_turn:
        segs.append({"kind": "prompt", "ids": list(ep.prompt_ids_per_turn[0]),
                     "train": False})
    for i, turn in enumerate(ep.turns):
        ids = ep.token_ids_per_turn[i] if i < len(ep.token_ids_per_turn) else []
        segs.append({"kind": "assistant", "ids": list(ids), "train": True})
        if turn.obs:
            segs.append({"kind": "obs",
                         "ids": encoding.encode_observation(
                             tok, encoding.observation_body(turn.action.name,
                                                            turn.action.arguments, turn.obs,
                                                            i + 1, max_turns),
                             thinking),
                         "train": False})
    return segs


if __name__ == "__main__":
    main()