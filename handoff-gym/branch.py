"""Branched handoff groups -> GRPO on a separate NOTE adapter (outcome-only reward).

One group per frozen point:
  1. run an episode until its context hits the compaction budget (--compact-at,
     prompt + generated tokens) and FREEZE it: a serializable snapshot holding the
     messages, the token-exact context ids and the work trace.
  2. RESUME the snapshot (fidelity: messages re-render to the frozen ids) and ask
     the writer for K notes (the one note call: grpo-local/notes.py via
     common.write_note). The note turn's prompt ids must equal
     sandbox-gym/encoding.render_chat of the handoff prompt, or the group is dropped.
  3. per note, a fresh reader gets ONLY task + note (common.resume_prompt), runs to
     an answer at the normal 8k budget, graded by the multi-hop exact-match grader
     (evals/graders.grade_mhqa, via common.grade). reward = 1 if every read passes.
     Over-cap notes = 0, never truncated, no read spent.

Reuses common.py (tasks prompts, served client, write_note, grade) and
synth_multihop.records (the h3/h4 seeds collect.py uses). Nothing copied.

  # dry run (not yet run): 10 existing multi-hop seeds, r3 writer+reader, pod tunnel
  ..\\.venv\\Scripts\\python.exe branch.py --seeds 6000-6009 --hops 3,4 --per-hop 2 \\
      --writer r3 --reader r3 --base http://127.0.0.1:8001 --k 4 --tag dryrun
  # same pipeline, deterministic stub model, no server
  ..\\.venv\\Scripts\\python.exe branch.py --stub --seeds 6000-6001 --per-hop 1 --tag stub

Outputs data/branch_snapshots-<tag>.jsonl, data/branch_groups-<tag>.jsonl,
data/branch_stats-<tag>.json. Group schema: see make_group().
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as C

SNAP_SCHEMA = "handoff_snapshot_v1"
GROUP_SCHEMA = "handoff_group_v1"
TOKENIZER = "Qwen/Qwen3.5-4B"
SANDBOX = C.ROOT / "sandbox-gym"

# sandbox-gym/encoding.py by path: its siblings (config, schema) clash with the
# grpo-local root modules of the same name, and encoding.py has no imports.
_spec = importlib.util.spec_from_file_location("sg_encoding", SANDBOX / "encoding.py")
encoding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encoding)


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def mhqa_tasks(seeds: list[int], hops: tuple[int, ...], per_hop: int) -> list[dict]:
    """The standard w3 multi-hop records (same as sft/collect.py --mhqa-seeds)."""
    out = []
    for s in seeds:
        if s == C.mh.SEED or s in C.HOLDOUT_SEEDS:
            raise SystemExit(f"seed {s} is an eval/holdout world")
        for r in C.mh.records(s, {h: per_hop for h in hops}):
            out.append({**r, "id": r["uid"], "seed": s})
    return out


# --- freeze / serialize / resume ---------------------------------------------

def user_msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def freeze_mhqa(task: dict, model: str, compact_at: int) -> dict:
    """Run her episode with max_tokens = compact_at - prompt, so the call stops
    exactly at the compaction point. Returns a snapshot, or {"skip": reason}."""
    tok = C.tokenizer()
    msgs = user_msgs(task["prompt"])
    local = encoding.render_chat(tok, msgs, True)
    room = compact_at - len(local)
    if room <= 0:
        return {"skip": "prompt-over-compaction-budget"}
    res = C.chat(model, task["prompt"], room, thinking=True)
    if res.get("error"):
        return {"skip": f"request-error: {res['error']}"}
    if res["prompt_ids"] != local:
        return {"skip": "fidelity: served prompt ids != render_chat"}
    if res["finish"] != "length":           # she answered before the compaction point
        return {"skip": "finished-before-compaction",
                "pass": C.grade(task, res["content"])[0]}
    ids = list(res["ids"])
    trace = tok.decode(ids).replace("<think>", "").strip()
    if C.chain_in_trace(task, trace):       # a note would just paste the answer
        return {"skip": "chain-in-trace"}
    return {"schema": SNAP_SCHEMA, "snap_id": f"{task['id']}@{compact_at}",
            "source": "mhqa", "task": task, "seed": task["seed"],
            "freeze_turn": 1,                # mid-way through her 1st generation
            "freeze_tokens": len(local) + len(ids), "budget": compact_at,
            "episode_budget": C.BUDGET, "thinking": True, "tokenizer": TOKENIZER,
            "messages": msgs, "prompt_len": len(local), "context_ids": local + ids,
            "trace": trace, "links_found": C.links_in_trace(task, trace),
            "links_total": len(task["answer"]["chain"]) - 1, "model": model}


def transcript_messages(rows: list[dict]) -> list[list[dict]]:
    """Rollout rows (train_grpo.rollout_record) -> the exact served message lists,
    built by sandbox-gym's own agent/gyms code in a subprocess (import clash)."""
    code = r"""
import json, sys
import agent, repo_gen_hard as H
from config import Config
from gyms import system_prompt
TOOLS = {"find_h": H.FIND_TOOLS, "edit_h": H.EDIT_TOOLS, "fix_h": H.FIX_TOOLS}
N = agent.counter_turns(Config())
out = []
for r in json.loads(sys.stdin.read()):
    m = [{"role": "system", "content": system_prompt(TOOLS[r["family"]], N)},
         {"role": "user", "content": r["prompt"]}]
    for i, t in enumerate(r["messages"]):
        m.append(agent.assistant_message(t["content"], t["reasoning"]))
        if t.get("obs"):
            a = t.get("action") or {}
            m.append({"role": "user", "content": agent.observation_text(
                a.get("name", ""), a.get("arguments", {}), t["obs"], i + 1, N)})
    out.append(m)
sys.stdout.write(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", code], input=json.dumps(rows), cwd=SANDBOX,
                       capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(p.stdout)


def transcript_trace(turns: list[dict]) -> str:
    out = []
    for i, t in enumerate(turns, 1):
        a = t.get("action") or {}
        out.append(f"[turn {i}]\n{(t.get('reasoning') or '').strip()}\n"
                   f"call: {a.get('name', '')}({json.dumps(a.get('arguments', {}))})\n"
                   f"result:\n{t.get('obs') or ''}")
    return "\n\n".join(out)


def freeze_transcript(row: dict, msgs: list[dict], compact_at: int) -> dict:
    """Freeze a real multi-turn transcript at the first turn boundary whose
    rendered context reaches compact_at (and before her final turn)."""
    tok = C.tokenizer()
    n_turns = len(row["messages"])
    for k in range(1, n_turns):              # k completed turns; never after the last
        cut = 2 + sum(1 + bool(t.get("obs")) for t in row["messages"][:k])
        ids = encoding.render_chat(tok, msgs[:cut], True)
        if len(ids) >= compact_at:
            prompt_len = len(encoding.render_chat(tok, msgs[:2], True))
            return {"schema": SNAP_SCHEMA,
                    "snap_id": f"{row['task']}#s{row['step']}a{row['attempt']}@{compact_at}",
                    "source": "rollout", "task": {"id": row["task"], "prompt": row["prompt"],
                                                  "family": row["family"]},
                    "seed": row["seed"], "freeze_turn": k, "freeze_tokens": len(ids),
                    "budget": compact_at, "episode_budget": C.BUDGET, "thinking": True,
                    "tokenizer": TOKENIZER, "messages": msgs[:cut], "prompt_len": prompt_len,
                    "context_ids": ids, "trace": transcript_trace(row["messages"][:k]),
                    "episode": {"stop": row["stop"], "reward": row["reward"],
                                "turns": n_turns}}
    return {"skip": "never-reached-compaction"}


def dumps(snap: dict) -> str:
    return json.dumps(snap, ensure_ascii=False)


def loads(line: str) -> dict:
    snap = json.loads(line)
    if snap.get("schema") != SNAP_SCHEMA:
        raise ValueError(f"not a {SNAP_SCHEMA} snapshot")
    return snap


def resume(snap: dict) -> str:
    """Re-render the frozen context from its messages; token-exact or raise.
    Returns the handoff prompt the writer gets at this point."""
    tok = C.tokenizer()
    ids = snap["context_ids"]
    head = encoding.render_chat(tok, snap["messages"], snap["thinking"])
    if snap["source"] == "rollout":
        if head != ids:
            raise ValueError(f"{snap['snap_id']}: messages re-render != frozen context")
        return C.notes.handoff_prompt(snap["task"]["prompt"], snap["trace"],
                                      "the task, the workspace files")
    if head != ids[: snap["prompt_len"]] or len(head) != snap["prompt_len"]:
        raise ValueError(f"{snap['snap_id']}: prompt re-render != frozen context")
    if tok.decode(ids[snap["prompt_len"]:]).replace("<think>", "").strip() != snap["trace"]:
        raise ValueError(f"{snap['snap_id']}: frozen ids != trace")
    return C.handoff_prompt(snap["task"], snap["trace"])


# --- notes + continuations -> group -------------------------------------------

def read_note(task: dict, note: str, reader: str) -> dict:
    p = C.resume_prompt(task, note)
    r = C.chat(reader, p, C.room(p, True), thinking=True)
    if r.get("error"):
        return {"pass": None, "why": r["error"]}
    ok, why = C.grade(task, r["content"])
    return {"pass": ok, "why": why, "finish": r["finish"], "n_tokens": len(r["ids"])}


def make_group(snap: dict, writer: str, reader: str, k: int, reads: int,
               note_mode: str) -> dict:
    tok = C.tokenizer()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    hp = resume(snap)
    want = encoding.render_chat(tok, user_msgs(hp), C.notes.NOTE_THINKING)
    if len(want) + C.notes.NOTE_CALL_MAX > C.BUDGET:
        return {"skip": "no-room-for-note-call"}
    comps = []
    for _ in range(k):
        w = C.write_note(writer, hp, note_mode)
        if w.get("error"):
            return {"skip": f"note request-error: {w['error']}"}
        if w["prompt_ids"] != want:
            return {"skip": "fidelity: note prompt ids != render_chat"}
        ids, over = list(w["ids"]), w["over_cap"]
        if not over and (not ids or ids[-1] != eos):
            ids.append(eos)                   # the EOS she emitted (vLLM strips it)
        conts = [] if over else [read_note(snap["task"], w["text"], reader)
                                 for _ in range(reads)]
        if any(c["pass"] is None for c in conts):
            return {"skip": "reader request-error"}
        passed = (not over) and all(c["pass"] for c in conts)
        comps.append({"text": w["text"], "completion_ids": ids, "reward": float(passed),
                      "over_cap": over, "forced": w["forced"], "n_tokens": w["note_tokens"],
                      "continuations": conts})
    rewards = [c["reward"] for c in comps]
    t = snap["task"]
    return {
        "schema": GROUP_SCHEMA, "group_id": snap["snap_id"], "adapter": "note",
        "prompt_ids": want, "prompt_text": hp, "completions": comps, "rewards": rewards,
        "mixed": 0 < sum(rewards) < len(rewards),
        "meta": {"seed": snap["seed"], "task_id": t["id"], "root_id": t.get("root_id"),
                 "hops": t.get("hops"), "source": snap["source"],
                 "freeze_turn": snap["freeze_turn"], "freeze_tokens": snap["freeze_tokens"],
                 "budget": snap["budget"], "episode_budget": snap["episode_budget"],
                 "links_found": snap.get("links_found"), "links_total": snap.get("links_total"),
                 "writer": writer, "reader": reader, "k": k, "reads": reads,
                 "note_mode": note_mode, "note_cap": C.notes.NOTE_CAP,
                 "note_call_max": C.notes.NOTE_CALL_MAX, "fidelity": "render_chat-ok",
                 "tokenizer": TOKENIZER},
    }


def run(tasks: list[dict], writer: str, reader: str, compact_at: int, k: int, reads: int,
        note_mode: str, workers: int, tag: str) -> dict:
    snaps, skips = [], {}
    with ThreadPoolExecutor(workers) as ex:
        for s in ex.map(lambda t: freeze_mhqa(t, writer, compact_at), tasks):
            if "skip" in s:
                skips[s["skip"]] = skips.get(s["skip"], 0) + 1
            else:
                snaps.append(s)
    sp = C.DATA / f"branch_snapshots-{tag}.jsonl"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text("".join(dumps(s) + "\n" for s in snaps), encoding="utf-8")
    loaded = [loads(ln) for ln in sp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    groups = []
    with ThreadPoolExecutor(workers) as ex:
        for g in ex.map(lambda s: make_group(s, writer, reader, k, reads, note_mode), loaded):
            if "skip" in g:
                skips[g["skip"]] = skips.get(g["skip"], 0) + 1
            else:
                groups.append(g)
    C.jsonl_write(C.DATA / f"branch_groups-{tag}.jsonl", groups)
    notes_ = [c for g in groups for c in g["completions"]]
    stats = {"tasks": len(tasks), "snapshots": len(snaps), "groups": len(groups),
             "mixed_groups": sum(g["mixed"] for g in groups),
             "note_pass": f"{sum(c['reward'] for c in notes_):.0f}/{len(notes_)}",
             "over_cap": sum(c["over_cap"] for c in notes_), "skips": skips,
             "writer": writer, "reader": reader, "compact_at": compact_at, "k": k,
             "reads": reads, "note_mode": note_mode}
    (C.DATA / f"branch_stats-{tag}.json").write_text(json.dumps(stats, indent=1),
                                                    encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="6000-6009")
    ap.add_argument("--hops", default="3,4")
    ap.add_argument("--per-hop", type=int, default=2)
    ap.add_argument("--writer", default="r3")
    ap.add_argument("--reader", default="r3", help="frozen reader; never the trained adapter")
    ap.add_argument("--base", default=C.BASE, help="OpenAI-compatible vLLM URL")
    ap.add_argument("--compact-at", type=int, default=3072,
                    help="context tokens (prompt + generated) at which the episode freezes")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reads", type=int, default=1, help="reader continuations per note")
    ap.add_argument("--note-mode", choices=["off", "think", "force"], default="think",
                    help="think = every completion token sampled (clean for GRPO)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="dryrun")
    ap.add_argument("--stub", action="store_true", help="deterministic stub model, no server")
    ap.add_argument("--budget", type=int, default=None,
                    help="episode token budget (default 8192 / env GYM_BUDGET); read by gym_budget at import")
    a = ap.parse_args()
    C.BASE = a.base
    if not a.stub:
        C.check_server()
    tasks = mhqa_tasks(parse_seeds(a.seeds), tuple(int(h) for h in a.hops.split(",")),
                       a.per_hop)
    if a.stub:
        from stub_model import StubModel
        with StubModel(tasks).installed():
            stats = run(tasks, a.writer, a.reader, a.compact_at, a.k, a.reads, a.note_mode,
                        a.workers, a.tag)
    else:
        stats = run(tasks, a.writer, a.reader, a.compact_at, a.k, a.reads, a.note_mode,
                    a.workers, a.tag)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
