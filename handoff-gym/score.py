"""Score frozen states: empty-note control, K notes x 2 fresh continuations.

Per candidate state (depth-stratified order, so early/mid/late cuts all show up):
  1. control: 2 continuations from an EMPTY note. Either passes -> drop the
     state (the note didn't matter).
  2. K notes from the writer at the frozen state, via the shared note call
     (grpo-local/notes.py: thinking on, <=600 think + <=400 note; over-cap =
     fail, never truncated).
  3. each note: 2 continuations by the frozen reader; note passes only if both do.

The reader is frozen (default r3) and is never the model being trained, so the
reward can't drift as the writer learns.

Outputs (data/):
  handoff_wins.jsonl    SFT records: handoff prompt ids (train=False) + note ids
                        (train=True, EOS included), from control-surviving states
  handoff_groups.jsonl  per state, all K notes with pass/fail (mixed = GRPO-usable)
  stats.json            kept/dropped, empty-note pass rate, note pass rate,
                        median note length, all split by depth bin

  ..\\.venv\\Scripts\\python.exe score.py --states 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer

import common as C

BINS = (("early", 0.0, 0.34), ("mid", 0.34, 0.67), ("late", 0.67, 1.01))


def depth_bin(frac: float) -> str:
    return next(n for n, lo, hi in BINS if lo <= frac < hi)


def stratified(states: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by = {n: [s for s in states if depth_bin(s["depth_frac"]) == n] for n, _, _ in BINS}
    for v in by.values():
        rng.shuffle(v)
    out = []
    while any(by.values()):
        for n, _, _ in BINS:
            if by[n]:
                out.append(by[n].pop())
    return out


# Hop relations of a _deep_hop chain, as the question words them.
REL = {"mentor": "mentor of {x}", "company": "company {x} works for",
       "founder": "founder of {x}", "city": "birth city of {x}",
       "river": "river that flows through {x}"}
REST = {"mentor": "their mentor", "company": "the company that person works for",
        "founder": "that company's founder", "city": "the founder's birth city",
        "river": "the river through that city"}


def oracle_note(task: dict, trace: str, tok) -> dict:
    """Ceiling control: a note built from ground truth at the frozen cut. Done = the
    verified chain prefix her trace already reached (consecutive links only); the
    remaining hops are named by relation, never by answer. Same sections and cap
    as her notes, so the reader sees the best note she could have written."""
    chain, t = task["answer"]["chain"], trace.lower()
    d = len(chain) - 5
    kinds = ["person"] + ["mentor"] * d + ["company", "founder", "city", "river"]
    i = 1
    while i < len(chain) and chain[i].lower() in t:
        i += 1
    done = " -> ".join(f"{chain[j]} ({kinds[j]})" for j in range(i))
    rest = [REST[kinds[j]] for j in range(i, len(chain))]
    nxt = REL[kinds[i]].format(x=chain[i - 1]) if i < len(chain) else "write the ANSWER line"
    text = "\n".join([
        f"Goal: the full chain {' -> '.join(kinds)} starting from {chain[0]}; "
        "the last line must be ANSWER: with every hop.",
        f"Done: verified in the facts: {done}.",
        f"Current state: {i - 1} of {len(chain) - 1} hops found; last confirmed is "
        f"{chain[i - 1]}. Remaining {len(rest)} hops, in order: "
        + (", then ".join(rest) or "none") + ".",
        "Hypothesis: none; everything under Done is checked against the facts.",
        f"Next step: find the {nxt} in the facts, then continue hop by hop.",
        "Don't repeat: don't re-derive the links under Done."])
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    return {"text": text, "note_tokens": n, "ids": [], "prompt_ids": [],
            "over_cap": n > C.notes.NOTE_CAP, "forced": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=20, help="survivors to score fully")
    ap.add_argument("--max-candidates", type=int, default=1000)
    ap.add_argument("--writer", default="r3")
    ap.add_argument("--reader", default="r3")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)   # desktop shares this GPU
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-n", type=int, default=2,
                    help="empty-note continuations per task")
    ap.add_argument("--max-control-rate", type=float, default=0.15)
    ap.add_argument("--note-mode", choices=["off", "think", "force", "oracle"], default="force",
                    help="how the note is written (common.write_note)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. force")
    ap.add_argument("--hops", default="", help="only states of these hop counts, e.g. 8")
    ap.add_argument("--baseline-n", type=int, default=0,
                    help="ensure this many cached empty-note continuations per task; the ones "
                         "past --control-n are the held-out blank-note baseline")
    ap.add_argument("--allow-easy", action="store_true",
                    help="score notes even if the held-out blank-note baseline is >= max-control-rate")
    ap.add_argument("--control-only", action="store_true",
                    help="run phase 1 only: measure from-scratch success, no notes")
    a = ap.parse_args()
    C.check_server()     # budget > server max_model_len = fail fast, never silent truncation

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    cands = stratified(C.jsonl_read(C.DATA / "states.jsonl"), a.seed)[: a.max_candidates]
    # states frozen under an older note spec may not leave room for the note call
    fit = [st for st in cands if C.room(C.handoff_prompt(st["task"], st["trace"]),
                                        C.notes.NOTE_THINKING) >= C.notes.NOTE_CALL_MAX]
    if len(fit) < len(cands):
        print(f"[notes] {len(cands) - len(fit)} states dropped: no room for the "
              f"{C.notes.NOTE_CALL_MAX}-token note call", flush=True)
    cands = fit
    if a.hops:
        keep = {int(h) for h in a.hops.split(",")}
        cands = [st for st in cands if st["task"]["hops"] in keep]
        print(f"[notes] --hops {a.hops}: {len(cands)} candidate states", flush=True)

    def cont(task, note, sink=None):
        """True/False = the reader's graded outcome. None = the request itself
        failed (server down/restarting) -- never scored as a failed read.
        sink: list that gets one record per read -- truncated vs answered-wrong."""
        p = C.resume_prompt(task, note)       # normal 8k budget, never squeezed
        room = C.room(p, True)
        for wait in (0, 30, 120):
            time.sleep(wait)
            r = C.chat(a.reader, p, room, thinking=True)
            if not r.get("error"):
                ok = C.grade(task, r["content"])[0]
                if sink is not None:
                    sink.append({"pass": ok, "finish": r["finish"], "room": room,
                                 "gen_tokens": len(r["ids"]),
                                 "answer_line": "ANSWER:" in r["content"],
                                 "tail": r["content"][-240:]})
                return ok
        return None

    # --- phase 1: empty-note controls, per TASK, one batch -------------------
    # The empty-note prompt depends only on the task, so every cut of a task
    # shares one control. Cached in data/controls.json (resumable).
    ctrl_path = C.DATA / "controls.json"
    controls = json.loads(ctrl_path.read_text(encoding="utf-8")) if ctrl_path.exists() else {}
    tasks = {s["task"]["id"]: s["task"] for s in cands}
    todo = [(tid, j) for tid in tasks for j in range(max(a.control_n, a.baseline_n))
            if len(controls.get(tid, [])) <= j]
    print(f"[control] {len(tasks)} tasks, {len(todo)} empty-note continuations to run",
          flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for (tid, _), ok in zip(todo, ex.map(lambda tj: cont(tasks[tj[0]], ""), todo)):
            if ok is None:
                continue                      # request failed: re-run next time, never a fail
            controls.setdefault(tid, []).append(bool(ok))
            ctrl_path.write_text(json.dumps(controls, indent=1), encoding="utf-8")
    flat = [c for tid in tasks for c in controls[tid][: a.control_n]]
    rate = sum(flat) / len(flat) if flat else 0.0
    by_hops: dict = {}
    for tid, t in tasks.items():
        by_hops.setdefault(t["hops"], []).extend(controls[tid][: a.control_n])
    print(f"[control] from-scratch pass rate {rate:.1%} ({sum(flat)}/{len(flat)}); by hops: "
          + ", ".join(f"h{h} {sum(v)}/{len(v)}" for h, v in sorted(by_hops.items())), flush=True)
    if rate > a.max_control_rate:
        print(f"[control] above {a.max_control_rate:.0%}: tasks too easy to need a note -- "
              "push hops/world before spending GPU on notes")
    if a.control_only:
        return

    def score(state):
        task = state["task"]
        ctrl = controls[task["id"]][: a.control_n]
        row = {"state_id": state["state_id"], "depth": state["depth"],
               "depth_frac": state["depth_frac"], "bin": depth_bin(state["depth_frac"]),
               "traj_pass": state["traj_pass"], "control": ctrl, "notes": [],
               "links_found": C.links_in_trace(task, state["trace"]),
               "links_total": len(task["answer"]["chain"]) - 1}
        if any(ctrl):
            row["dropped"] = "control-passed"
            return row
        hp = C.handoff_prompt(task, state["trace"])
        for _ in range(a.k):
            w = (oracle_note(task, state["trace"], tok) if a.note_mode == "oracle"
                 else C.write_note(a.writer, hp, a.note_mode))
            if w.get("error"):
                row["notes"].append({"error": w["error"]})
                continue
            ids, over = list(w["ids"]), w["over_cap"]
            if not over and (not ids or ids[-1] != eos):
                ids.append(eos)
            meta = []
            conts = [] if over else [cont(task, w["text"], meta) for _ in range(2)]
            print(f"    {state['state_id']} note {len(row['notes']) + 1}/{a.k}: "
                  f"{w['note_tokens']} tok{' FORCED' if w['forced'] else ''}"
                  f"{' OVER-CAP' if over else ''} reads {conts} "
                  f"finish {[(m['finish'], m['gen_tokens'], m['answer_line']) for m in meta]}",
                  flush=True)
            row["notes"].append({
                "text": w["text"], "n_tokens": w["note_tokens"], "gen_tokens": len(w["ids"]),
                "forced": w["forced"], "over_cap": over,
                "conts": conts, "read_meta": meta, "incomplete": None in conts,
                "pass": (not over) and None not in conts and all(conts),
                "prompt_ids": w["prompt_ids"], "ids": ids})
        return row

    # --- phase 2: notes, only on control-surviving states ----------------------
    rows = [score(st) for st in cands if any(controls[st["task"]["id"]][: a.control_n])]
    alive = [st for st in cands if not any(controls[st["task"]["id"]][: a.control_n])]
    print(f"[notes] {len(alive)} states survive the control; scoring {min(a.states, len(alive))}",
          flush=True)
    # Held-out blank-note baseline: states were selected on controls[:control_n], so
    # those read ~0% by construction. Controls [control_n:] were not used to select.
    picked = alive[: a.states]
    held = [c for tid in {s["task"]["id"] for s in picked}
            for c in controls[tid][a.control_n:]]
    base = sum(held) / len(held) if held else None
    print(f"[baseline] held-out blank-note pass rate on the {len({s['task']['id'] for s in picked})}"
          f" picked tasks: {sum(held)}/{len(held)}" + (f" = {base:.1%}" if held else " (none cached)"),
          flush=True)
    if base is not None and base >= a.max_control_rate and not a.allow_easy:
        print(f"[baseline] >= {a.max_control_rate:.0%}: too easy, not scoring notes "
              "(push hops/world, or --allow-easy)", flush=True)
        return
    sfx = f"-{a.tag}" if a.tag else ""
    part = C.DATA / f"handoff_rows{sfx}.partial.jsonl"
    part.write_text("", encoding="utf-8")
    tally = {"states": 0, "notes": 0, "written": 0, "over": 0, "err": 0, "reads": 0, "read_pass": 0,
             "note_pass": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(score, st) for st in picked]):
            row = fut.result()
            rows.append(row)
            with part.open("a", encoding="utf-8") as f:
                f.write(json.dumps({k: v for k, v in row.items() if k != "notes"}
                                   | {"notes": [{k: v for k, v in n.items()
                                                 if k not in ("ids", "prompt_ids")}
                                                for n in row["notes"]]}) + "\n")
            ns = row["notes"]
            tally["states"] += 1
            tally["notes"] += len(ns)
            tally["err"] += sum("error" in n for n in ns)
            tally["over"] += sum(n.get("over_cap", False) for n in ns)
            tally["written"] += sum("error" not in n and not n["over_cap"] and n["n_tokens"] > 0
                                    for n in ns)
            rd = [c for n in ns for c in n.get("conts", []) if c is not None]
            tally["reads"] += len(rd)
            tally["read_pass"] += sum(rd)
            tally["note_pass"] += sum(n.get("pass", False) for n in ns)
            p = sum(n.get("pass", False) for n in ns)
            print(f"  [{tally['states']}/{len(picked)}] {row['state_id']} [{row['bin']} "
                  f"{row['depth_frac']}] notes pass {p}/{len(ns)} | running: written "
                  f"{tally['written']}/{tally['notes']}, over-cap {tally['over']}, err {tally['err']}, "
                  f"reads {tally['read_pass']}/{tally['reads']}, note pass "
                  f"{tally['note_pass']}/{tally['notes']}", flush=True)

    # --- outputs --------------------------------------------------------------
    by_id = {s["state_id"]: s for s in cands}
    wins, groups = [], []
    for row in rows:
        if "dropped" in row:
            continue
        good = [n for n in row["notes"] if "error" not in n and not n.get("incomplete")]
        npass = sum(n["pass"] for n in good)
        mixed = 0 < npass < len(good)
        groups.append({"state_id": row["state_id"], "bin": row["bin"], "depth": row["depth"],
                       "mixed": mixed,
                       "notes": [{"text": n["text"], "pass": n["pass"], "over_cap": n["over_cap"],
                                  "n_tokens": n["n_tokens"], "ids": n["ids"],
                                  "prompt_ids": n["prompt_ids"]} for n in good]})
        for j, n in enumerate(good):
            if n["pass"] and a.note_mode != "oracle":   # oracle text is not her tokens
                wins.append({
                    "id": f"{row['state_id']}#n{j}", "root_id": by_id[row["state_id"]]["task"]["root_id"],
                    "bucket": "handoff_mhqa", "split": "train", "bin": row["bin"],
                    "depth": row["depth"], "writer": a.writer, "reader": a.reader,
                    "segments": [{"kind": "prompt", "ids": n["prompt_ids"], "train": False},
                                 {"kind": "assistant", "ids": n["ids"], "train": True}],
                    "note": n["text"]})
    C.jsonl_write(C.DATA / f"handoff_wins{sfx}.jsonl", wins)
    C.jsonl_write(C.DATA / f"handoff_groups{sfx}.jsonl", groups)

    def summarize(rs):
        kept = [r for r in rs if "dropped" not in r]
        notes = [n for r in kept for n in r["notes"]
                 if "error" not in n and not n.get("incomplete")]
        ctrl = [c for r in rs for c in r["control"]]
        return {
            "candidates": len(rs), "kept": len(kept), "dropped_control": len(rs) - len(kept),
            "empty_note_pass_rate": round(sum(ctrl) / len(ctrl), 3) if ctrl else None,
            "note_pass_rate": round(sum(n["pass"] for n in notes) / len(notes), 3) if notes else None,
            # per-read rate: same unit as the blank-note baseline (one continuation)
            "read_pass_rate": (round(sum(c for n in notes for c in n["conts"])
                                     / max(1, sum(len(n["conts"]) for n in notes)), 3)
                               if notes else None),
            "notes_written": sum(not n["over_cap"] and n["n_tokens"] > 0 for n in notes),
            "over_cap_rate": round(sum(n["over_cap"] for n in notes) / len(notes), 3) if notes else None,
            "median_note_tokens": statistics.median([n["n_tokens"] for n in notes]) if notes else None,
            "mixed_groups": sum(1 for r in kept if 0 < sum(n.get("pass", False) for n in r["notes"])
                                < len(r["notes"])),
        }

    by_links = {}
    for r in rows:
        if "dropped" in r:
            continue
        k = f"{r['links_found']}/{r['links_total']}"
        ns = [n for n in r["notes"] if "error" not in n and not n.get("incomplete")]
        d = by_links.setdefault(k, [0, 0])
        d[0] += sum(n["pass"] for n in ns)
        d[1] += len(ns)
    stats = {"note_pass_by_links_found": {k: f"{p}/{n}" for k, (p, n) in sorted(by_links.items())},
             "all": summarize(rows), "writer": a.writer, "reader": a.reader, "k": a.k,
             "budget": C.BUDGET, "note_cap": C.notes.NOTE_CAP,
             "note_think_cap": C.notes.NOTE_THINK_CAP, "note_thinking": C.notes.NOTE_THINKING,
             **{b: summarize([r for r in rows if r["bin"] == b]) for b, _, _ in BINS}}
    stats["note_mode"] = a.note_mode
    stats["heldout_blank_baseline"] = {"pass": sum(held), "n": len(held),
                                       "rate": round(base, 3) if base is not None else None}
    (C.DATA / f"stats{sfx}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))

    # three notes to read: one pass, one fail, one over-cap
    notes = [(r, n) for r in rows for n in r["notes"] if "error" not in n]
    for label, pick in (("PASS", lambda n: n["pass"]),
                        ("FAIL", lambda n: not n["pass"] and not n["over_cap"]),
                        ("OVER-CAP", lambda n: n["over_cap"])):
        hit = next(((r, n) for r, n in notes if pick(n)), None)
        print(f"\n===== {label} " + (f"{hit[0]['state_id']} [{hit[0]['bin']}] "
                                     f"{hit[1]['n_tokens']} tok =====\n{hit[1]['text']}"
                                     if hit else "(none) ====="))


if __name__ == "__main__":
    main()
