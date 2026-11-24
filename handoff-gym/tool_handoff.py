"""Compaction on TOOL episodes: freeze a real sandbox-gym episode mid-run, write a
note, resume a fresh instance in the REAL agent loop (agent.run_episode) with the
same tools and the workspace as it was at the cut. Graded by the family's own reward.

Pipeline per frozen state (one state = one episode cut after k completed turns):
  1. freeze  task + her first k turns (calls, results, reasoning) + the workspace at
             the cut: seed files + her edits from turns 1..k re-applied
             (agent.replay_edits), so changes made before the cut persist.
  2. control blank-note resumes. States are SELECTED on the first --control-n
             (all must fail); the held-out blank baseline is measured on the rest
             (--baseline-n total), never on the selection attempts. Baseline >= 15%
             -> stop before notes (tasks too easy to need a note).
  3. notes   modes: off / force (common.write_note: the one note call, notes.py caps),
             oracle (ground truth at the cut: what's verified + the remaining steps,
             never the answer; mhqa_t and findh-effective only), blank (= step 2).
  4. resume  per note, --reads fresh episodes: system prompt + notes.resume_prompt
             (task + note, check_against "the files"), same tools, snapshot workspace.
             Turn budget rule: the reader gets R = cap - k turns (what the original
             episode had left) and her counter shows "[turn t of R]".
             Token budget: the usual budget (gym_budget.py, 8192) per call (fresh context is the point).
             A note passes only if every read passes. Request errors = missing.

Sources (training seeds only; any (family, seed) in sandbox-gym/data/holdout is refused):
  --source live      run her full episodes now (writer model), then cut them
  --source fixture   fixtures/rollouts_find_h.jsonl (pod GRPO rollouts; tests)
  --source file      --episodes PATH: any tool_episodes JSONL (other runs, tests)

  # pilot (home GPU, r3 writer+reader): mhqa_t + find_h, 5 states, k=2
  ..\\.venv\\Scripts\\python.exe -u tool_handoff.py --source live --families mhqa_t,find_h ^
      --only mhqat-h5,mhqat-h7,findh-effective --seeds 3100-3119 --states 5 --k 2 ^
      --modes blank,oracle,off,force --tag pilot-tool

Outputs (handoff-gym/data/): tool_episodes-<tag>.jsonl (cache), tool_snapshots-<tag>.jsonl,
tool_controls-<tag>.json (cache), tool_rows-<tag>.partial.jsonl, tool_stats-<tag>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SANDBOX = ROOT / "sandbox-gym"
sys.path.insert(0, str(SANDBOX))

import agent  # noqa: E402  (sandbox-gym: the real loop)
import encoding  # noqa: E402
import gyms  # noqa: E402
import repo_gen  # noqa: E402
from config import Config  # noqa: E402
from runner import Runner  # noqa: E402
from schema import Action, Turn  # noqa: E402
from workspace import Workspace  # noqa: E402


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_common():
    """handoff-gym/common.py imports evals/graders.py as `graders` and prepends the
    grpo-local root to sys.path; sandbox-gym has its own graders/config/schema.
    Load it with the sandbox modules set aside, then put everything back."""
    saved_path, saved = list(sys.path), {}
    for name in ("graders", "config", "schema"):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    try:
        c = _load_by_path("common", HERE / "common.py")
        b = _load_by_path("hg_branch", HERE / "branch.py")
    finally:
        for name in ("graders", "config", "schema"):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
    return c, b


C, B = _load_common()
notes = C.notes
DATA = HERE / "data"
SNAP_SCHEMA = "tool_snapshot_v1"
READER_SEES = "the task, the workspace files and the tools"


# --- holdout guard -------------------------------------------------------------

def holdout_keys() -> set[tuple[str, int]]:
    out = set()
    for f in (SANDBOX / "data" / "holdout").glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for it in d.get("items", []) if isinstance(d, dict) else []:
            out.add((it.get("family"), int(it.get("seed", -1))))
    return out


def parse_seeds(spec: str) -> list[int]:
    return B.parse_seeds(spec)


# --- turns <-> json --------------------------------------------------------------

def turn_to_dict(t: Turn) -> dict:
    return {"content": t.content, "reasoning": t.reasoning, "obs": t.obs,
            "truncated": t.truncated,
            "action": ({"name": t.action.name, "arguments": t.action.arguments}
                       if t.action else None)}


def turn_from_dict(d: dict) -> Turn:
    a = d.get("action")
    return Turn(content=d.get("content", ""), resp_text=d.get("content", ""),
                reasoning=d.get("reasoning", ""), obs=d.get("obs", ""),
                truncated=bool(d.get("truncated")),
                action=Action(a["name"], a.get("arguments", {})) if a else None)


def make_task(family: str, seed: int, name: str):
    spec = repo_gen.find_spec(seed, family, name)
    return spec, gyms.spec_to_task(spec)


def fresh_ws(cfg: Config, spec: dict, turns: list[Turn], label: str) -> Workspace:
    """Workspace at the cut: seed files, then her edits from the frozen turns."""
    root = cfg.episode.data_root / "workspaces-toolhandoff" / re.sub(r"[^\w.@#-]+", "_", label)
    ws = Workspace(root, Runner(cfg.episode.runner))
    ws.reset()
    ws.builds_left = cfg.episode.max_builds
    repo_gen.write_files(spec["files"], ws.root)
    agent.replay_edits(ws, turns)
    return ws


# --- oracle notes ------------------------------------------------------------------

def _read_paths(turns: list[Turn]) -> list[str]:
    return [str(t.action.arguments.get("path")) for t in turns
            if t.action and t.action.name == "read_file" and t.action.arguments.get("path")]


def _note(goal, done, state, hyp, nxt, dont) -> str:
    return "\n".join([f"Goal: {goal}", f"Done: {done}", f"Current state: {state}",
                      f"Hypothesis: {hyp}", f"Next step: {nxt}", f"Don't repeat: {dont}"])


def oracle_mhqa_t(spec: dict, turns: list[Turn]) -> str:
    import mhqa_tool as M
    chain = spec["ground_truth"].split(M.SEP)
    kinds = M._kinds(spec["hops"])
    seen = "\n".join((t.obs or "") for t in turns).lower()
    i = 1                                    # consecutive hops her results already showed
    while i < len(chain) and chain[i].lower() in seen:
        i += 1
    done = " -> ".join(f"{chain[j]} ({kinds[j]})" for j in range(i))
    last, lk = chain[i - 1], kinds[i - 1]
    rest = [f"{M._FIELD[(a, b)]} ({b})" for a, b in zip(kinds[i - 1:], kinds[i:])]
    if i < len(chain):
        nxt = (f"open kb/{M._KIND_DIR[lk]}/{M.slug(last)}.md and read its "
               f"'{M._FIELD[(lk, kinds[i])]}' line, then follow each next file the same way")
    else:
        nxt = "write the ANSWER line with the full chain"
    return _note(
        f"the full chain {' -> '.join(kinds)} from {chain[0]}, as an ANSWER line",
        f"verified in the kb files: {done}.",
        f"{i - 1} of {len(chain) - 1} hops found; last confirmed {last} ({lk}). "
        f"Remaining {len(rest)} lookups, in order: {', then '.join(rest) or 'none'}.",
        "none; everything under Done was read from the files.",
        nxt + ".", "re-reading the files for the links under Done.")


def oracle_findh_effective(spec: dict, turns: list[Turn]) -> str:
    import repo_gen_hard as H
    m = H.generate_hard(spec["seed"]).meta
    P, S = m["pkg"], m["settings"]
    oc = re.search(r"What value does (\w+) have", spec["prompt"]).group(1)
    read = sorted(set(_read_paths(turns)))
    return _note(
        f"the runtime value of {oc} as imported from {P}.{S}; answer with the number only.",
        f"files read so far: {', '.join(read) or 'none'}.",
        f"{P}/{S}/__init__.py starts from {P}/{S}/defaults.py and then applies the ACTIVE "
        f"profile '{m['active']}' ({P}/{S}/{m['active']}.py); {m['inactive']}.py is not "
        "loaded.",
        "none beyond the loading order above (checked against the files).",
        f"read {P}/{S}/defaults.py and {P}/{S}/{m['active']}.py; if {m['active']}.py sets "
        f"{oc}, that value wins, else the default. Then ANSWER.",
        f"{m['inactive']}.py, legacy/ and modules that only use {oc}.")


ORACLES = {"mhqa_t": oracle_mhqa_t, "findh-effective": oracle_findh_effective}


def oracle_for(family: str, name: str):
    return ORACLES.get(family) or next(
        (f for k, f in ORACLES.items() if name.startswith(k + "-")), None)


# --- episodes -> snapshots -----------------------------------------------------------

def make_cfg(a, model: str, max_turns: int | None = None) -> Config:
    cfg = Config()
    cfg.serve.base = a.base
    cfg.serve.served_model = model
    cfg.episode.runner = a.runner
    cfg.episode.data_root = SANDBOX / "data"
    if max_turns:
        cfg.episode.max_turns = max_turns
    return cfg


def prompt_len_fn(cfg: Config):
    tok = C.tokenizer()
    return lambda msgs: len(encoding.render_chat(tok, msgs, cfg.episode.thinking))


def live_episodes(a, families: list[str], seeds: list[int], only: list[str]) -> list[dict]:
    """Her full episodes on training seeds (cached, resumable)."""
    cache = DATA / f"tool_episodes-{a.tag}.jsonl"
    have = {json.loads(ln)["key"]: json.loads(ln) for ln in
            (cache.read_text(encoding="utf-8").splitlines() if cache.exists() else [])}
    bad, jobs = holdout_keys(), []
    for fam in families:
        for s in seeds:
            if (fam, s) in bad:
                raise SystemExit(f"{fam} seed {s} is in a sealed holdout")
            for sp in repo_gen.make_specs(s, fam):
                if only and not any(sp["name"].startswith(o) for o in only):
                    continue
                for r in range(a.episodes_per_task):
                    key = f"{sp['name']}#e{r}"
                    if key not in have:
                        jobs.append((fam, s, sp["name"], key))
    cfg = make_cfg(a, a.writer)
    lock = threading.Lock()

    def one(job):
        fam, s, name, key = job
        spec, task = make_task(fam, s, name)
        ws = fresh_ws(cfg, spec, [], f"ep-{key}")
        ep = agent.run_episode(cfg, task, ws, prompt_len_fn=prompt_len_fn(cfg))
        if ep.error:
            return None
        return {"key": key, "family": fam, "seed": s, "name": name, "cap": cfg.episode.max_turns,
                "passed": bool(task.reward(ep.turns, ws) >= 0.5), "stop": ep.stop,
                "turns": [turn_to_dict(t) for t in ep.turns]}

    print(f"[episodes] {len(have)} cached, {len(jobs)} to run (writer {a.writer})", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, j) for j in jobs]):
            rec = fut.result()
            if rec is None:
                continue
            with lock, cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            have[rec["key"]] = rec
            print(f"  episode {rec['key']}: {len(rec['turns'])} turns, stop {rec['stop']}, "
                  f"pass {rec['passed']}", flush=True)
    return list(have.values())


def fixture_episodes() -> list[dict]:
    rows = C.jsonl_read(HERE / "fixtures" / "rollouts_find_h.jsonl")
    out = []
    for r in rows:
        out.append({"key": f"{r['task']}#s{r['step']}a{r['attempt']}", "family": r["family"],
                    "seed": r["seed"], "name": r["task"], "cap": Config().episode.max_turns,
                    "passed": r["reward"] >= 0.5, "stop": r["stop"],
                    "turns": [{"content": t["content"], "reasoning": t.get("reasoning", ""),
                               "obs": t.get("obs", ""), "truncated": t.get("truncated", False),
                               "action": t.get("action")} for t in r["messages"]]})
    return out


def cut_points(n: int, cap: int, fracs: list[float]) -> list[int]:
    """Turn boundaries strictly inside the episode, leaving the reader >= 2 turns."""
    ks = sorted({max(1, min(n - 1, round(f * n))) for f in fracs})
    return [k for k in ks if 1 <= k <= n - 1 and cap - k >= 2]


def freeze(ep: dict, k: int) -> dict:
    turns = [turn_from_dict(d) for d in ep["turns"]]
    spec, task = make_task(ep["family"], ep["seed"], ep["name"])
    trace = B.transcript_trace([turn_to_dict(t) for t in turns[:k]])
    snap = {"schema": SNAP_SCHEMA, "snap_id": f"{ep['key']}@t{k}", "family": ep["family"],
            "seed": ep["seed"], "name": ep["name"], "cut_turn": k, "cap": ep["cap"],
            "reader_turns": ep["cap"] - k, "episode_turns": len(turns),
            "episode_passed": ep["passed"], "episode_stop": ep["stop"],
            "depth_frac": round(k / len(turns), 3),
            "turns": [turn_to_dict(t) for t in turns[:k]], "trace": trace,
            "gt_in_trace": spec["ground_truth"].lower() in trace.lower()}
    if ep["family"] == "mhqa_t":
        import mhqa_tool as M
        hops = spec["ground_truth"].split(M.SEP)[1:]
        if all(h.lower() in trace.lower() for h in hops):
            return {"skip": "chain-in-trace", "snap_id": snap["snap_id"]}
    hp = notes.handoff_prompt(task.prompt, trace, READER_SEES)
    if C.room(hp, True) < notes.NOTE_CALL_MAX:
        return {"skip": "no-room-for-note-call", "snap_id": snap["snap_id"]}
    return snap


# --- notes + resume ---------------------------------------------------------------------

def write(mode: str, snap: dict, writer: str) -> dict:
    spec, task = make_task(snap["family"], snap["seed"], snap["name"])
    turns = [turn_from_dict(d) for d in snap["turns"]]
    if mode == "blank":
        return {"text": "", "note_tokens": 0, "over_cap": False, "forced": False}
    if mode == "oracle":
        fn = oracle_for(snap["family"], snap["name"])
        if fn is None:
            return {"error": "oracle-unsupported"}
        text = fn(spec, turns)
        n = len(C.tokenizer()(text, add_special_tokens=False)["input_ids"])
        return {"text": text, "note_tokens": n, "over_cap": n > notes.NOTE_CAP, "forced": False}
    hp = notes.handoff_prompt(task.prompt, snap["trace"], READER_SEES)
    w = C.write_note(writer, hp, mode)
    if w.get("error"):
        return w
    want = encoding.render_chat(C.tokenizer(), [{"role": "user", "content": hp}], mode != "off")
    if w.get("prompt_ids") and w["prompt_ids"] != want:
        return {"error": "fidelity: note-call prompt ids != render_chat"}
    return w


def resume(a, snap: dict, note: str, label: str) -> bool | None:
    """One fresh reader episode in the real loop. None = request error (missing)."""
    spec, task = make_task(snap["family"], snap["seed"], snap["name"])
    rtask = dataclasses.replace(task, prompt=notes.resume_prompt(task.prompt, note, "the files"))
    cfg = make_cfg(a, a.reader, max_turns=snap["reader_turns"])
    turns = [turn_from_dict(d) for d in snap["turns"]]
    for wait in (0, 30, 120):
        if wait:
            threading.Event().wait(wait)
        ws = fresh_ws(cfg, spec, turns, label)
        ep = agent.run_episode(cfg, rtask, ws, prompt_len_fn=prompt_len_fn(cfg))
        if not ep.error:
            return bool(task.reward(ep.turns, ws) >= 0.5)
    return None


# --- main -------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict | None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["live", "fixture", "file"], default="live")
    ap.add_argument("--episodes", default="", help="--source file: episodes JSONL "
                    "(tool_episodes schema: key, family, seed, name, cap, passed, stop, turns)")
    ap.add_argument("--families", default="mhqa_t,find_h")
    ap.add_argument("--only", default="", help="spec-name prefixes, e.g. mhqat-h5,findh-effective")
    ap.add_argument("--seeds", default="3100-3119")
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--cut-fracs", default="0.34,0.5,0.67")
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--k", type=int, default=4, help="notes per mode per state")
    ap.add_argument("--reads", type=int, default=2, help="reader episodes per note")
    ap.add_argument("--modes", default="blank,oracle,off,force")
    ap.add_argument("--control-n", type=int, default=2, help="blank resumes used to SELECT")
    ap.add_argument("--baseline-n", type=int, default=4,
                    help="total blank resumes per state; those past --control-n = baseline")
    ap.add_argument("--max-control-rate", type=float, default=0.15)
    ap.add_argument("--oracle-margin", type=float, default=0.2,
                    help="oracle read rate must beat blank by this much, else 'cut point wrong'")
    ap.add_argument("--allow-easy", action="store_true")
    ap.add_argument("--writer", default="r3")
    ap.add_argument("--reader", default="r3", help="frozen reader; never the trained adapter")
    ap.add_argument("--base", default=C.BASE)
    ap.add_argument("--runner", default="docker")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot-tool")
    ap.add_argument("--budget", type=int, default=None,
                    help="episode token budget (default 8192 / env GYM_BUDGET); read by gym_budget at import")
    a = ap.parse_args(argv)
    os.chdir(SANDBOX)
    modes = [m for m in a.modes.split(",") if m]
    if a.source == "live":
        C.check_server()   # budget > server max_model_len = fail fast

    fams = a.families.split(",")
    only = [o for o in a.only.split(",") if o]
    if a.source == "fixture":
        eps = fixture_episodes()
    elif a.source == "file":
        eps = C.jsonl_read(Path(a.episodes))
        bad = holdout_keys()
        if any((e["family"], e["seed"]) in bad for e in eps):
            raise SystemExit("episodes include a sealed-holdout (family, seed)")
    else:
        eps = live_episodes(a, fams, parse_seeds(a.seeds), only)
    eps = [e for e in eps if e["family"] in fams and (not only or any(
        e["name"].startswith(o) for o in only))]
    fracs = [float(f) for f in a.cut_fracs.split(",")]
    snaps, skips = [], {}
    for e in eps:
        for k in cut_points(len(e["turns"]), e["cap"], fracs):
            s = freeze(e, k)
            if "skip" in s:
                skips[s["skip"]] = skips.get(s["skip"], 0) + 1
            else:
                snaps.append(s)
    C.jsonl_write(DATA / f"tool_snapshots-{a.tag}.jsonl", snaps)
    print(f"[freeze] {len(eps)} episodes -> {len(snaps)} states; skipped {skips}", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(snaps)

    # --- blank controls (cached): select on [:control_n], baseline on the rest -----------
    cpath = DATA / f"tool_controls-{a.tag}.json"
    controls = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    need = max(a.control_n, a.baseline_n)
    todo = [(s, j) for s in snaps for j in range(need)
            if len(controls.get(s["snap_id"], [])) <= j]
    lock = threading.Lock()
    print(f"[control] {len(snaps)} states, {len(todo)} blank resumes to run", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(resume, a, s, "", f"{s['snap_id']}-blank{j}"): s for s, j in todo}
        for fut in as_completed(futs):
            ok = fut.result()
            if ok is None:
                continue
            s = futs[fut]
            with lock:
                controls.setdefault(s["snap_id"], []).append(ok)
                cpath.write_text(json.dumps(controls, indent=1), encoding="utf-8")
    alive = [s for s in snaps if len(controls.get(s["snap_id"], [])) >= need
             and not any(controls[s["snap_id"]][: a.control_n])]
    picked = alive[: a.states]
    held = [c for s in picked for c in controls[s["snap_id"]][a.control_n:need]]
    base = sum(held) / len(held) if held else None
    print(f"[baseline] {len(alive)} states survive selection; scoring {len(picked)}. Held-out "
          f"blank resume pass {sum(held)}/{len(held)}" + (f" = {base:.1%}" if held else ""),
          flush=True)
    if base is not None and base >= a.max_control_rate and not a.allow_easy:
        print(f"[baseline] >= {a.max_control_rate:.0%}: too easy, not scoring notes", flush=True)
        return None

    # --- notes per mode --------------------------------------------------------------------
    part = DATA / f"tool_rows-{a.tag}.partial.jsonl"
    part.write_text("", encoding="utf-8")
    note_modes = [m for m in modes if m != "blank"]
    tally = {m: {"notes": 0, "written": 0, "over": 0, "err": 0, "reads": 0, "read_pass": 0,
                 "note_pass": 0} for m in note_modes}

    def score(s):
        row = {"snap_id": s["snap_id"], "family": s["family"], "cut_turn": s["cut_turn"],
               "reader_turns": s["reader_turns"], "depth_frac": s["depth_frac"],
               "control": controls[s["snap_id"]][:need], "modes": {}}
        for m in note_modes:
            out = []
            for j in range(a.k):
                w = write(m, s, a.writer)
                if w.get("error"):
                    out.append({"error": w["error"]})
                    print(f"    {s['snap_id']} {m} note {j + 1}/{a.k}: ERROR {w['error']}",
                          flush=True)
                    if w["error"] == "oracle-unsupported":
                        break
                    continue
                over = w["over_cap"]
                reads = [] if over else [resume(a, s, w["text"], f"{s['snap_id']}-{m}{j}r{r}")
                                         for r in range(a.reads)]
                print(f"    {s['snap_id']} {m} note {j + 1}/{a.k}: {w['note_tokens']} tok"
                      f"{' FORCED' if w.get('forced') else ''}{' OVER-CAP' if over else ''} "
                      f"reads {reads}", flush=True)
                out.append({"text": w["text"], "n_tokens": w["note_tokens"], "over_cap": over,
                            "forced": w.get("forced", False), "reads": reads,
                            "incomplete": None in reads,
                            "pass": (not over) and bool(reads) and None not in reads
                            and all(reads)})
            row["modes"][m] = out
        return row

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for n, fut in enumerate(as_completed([ex.submit(score, s) for s in picked]), 1):
            row = fut.result()
            rows.append(row)
            with part.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            line = []
            for m, ns in row["modes"].items():
                t = tally[m]
                good = [x for x in ns if "error" not in x]
                t["notes"] += len(ns)
                t["err"] += len(ns) - len(good)
                t["over"] += sum(x["over_cap"] for x in good)
                t["written"] += sum(not x["over_cap"] and x["n_tokens"] > 0 for x in good)
                rd = [r for x in good for r in x["reads"] if r is not None]
                t["reads"] += len(rd)
                t["read_pass"] += sum(rd)
                t["note_pass"] += sum(x["pass"] for x in good)
                line.append(f"{m} written {t['written']}/{t['notes']} reads "
                            f"{t['read_pass']}/{t['reads']} notes {t['note_pass']}/{t['notes']}")
            print(f"  [{n}/{len(picked)}] {row['snap_id']} | running: " + " | ".join(line),
                  flush=True)

    # --- side by side ------------------------------------------------------------------------
    def rate(p, n):
        return round(p / n, 3) if n else None
    stats = {"tag": a.tag, "writer": a.writer, "reader": a.reader, "states": len(picked),
             "k": a.k, "reads_per_note": a.reads, "budget": C.BUDGET,
             "turn_rule": "reader turns = cap - cut_turn",
             "blank_heldout": {"pass": sum(held), "n": len(held), "read_rate": rate(sum(held),
                                                                                   len(held))}}
    for m, t in tally.items():
        stats[m] = {**t, "read_rate": rate(t["read_pass"], t["reads"]),
                    "note_rate": rate(t["note_pass"], t["notes"] - t["err"])}
    o = stats.get("oracle", {}).get("read_rate")
    b = stats["blank_heldout"]["read_rate"]
    if o is not None and b is not None:
        stats["verdict"] = ("ok: oracle clearly beats blank" if o >= b + a.oracle_margin
                            else "cut point wrong: oracle does not clearly beat blank")
    (DATA / f"tool_stats-{a.tag}.json").write_text(json.dumps(stats, indent=1),
                                                   encoding="utf-8")
    print(f"\n{'mode':8} {'written':>9} {'over':>5} {'read pass':>11} {'note pass':>11}")
    print(f"{'blank':8} {'-':>9} {'-':>5} {sum(held):>4}/{len(held):<4} "
          f"{(b if b is not None else 0):>4.0%}  {'-':>11}")
    for m in note_modes:
        t = stats[m]
        print(f"{m:8} {t['written']:>4}/{t['notes']:<4} {t['over']:>5} "
              f"{t['read_pass']:>4}/{t['reads']:<4} {(t['read_rate'] or 0):>4.0%}  "
              f"{t['note_pass']:>4}/{t['notes'] - t['err']:<4}")
    if "verdict" in stats:
        print(f"[verdict] {stats['verdict']} (oracle {o:.0%} vs blank {b:.0%})")
    return stats


if __name__ == "__main__":
    main()
