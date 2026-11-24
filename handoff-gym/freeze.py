"""Freeze real r3 multi-hop trajectories at several depths -> data/states.jsonl.

A trajectory is one served episode (thinking on, 8k budget). Cuts are taken at
fixed token depths inside her reasoning (every --every tokens after the first
--every), snapped forward to the next newline -- never chosen from the gold
chain. Every depth is kept: a state is (task, her trace up to the cut), tagged
with depth in tokens and as a fraction of that trajectory's reasoning.

  ..\\.venv\\Scripts\\python.exe freeze.py --seeds 4242,4243 --model r3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

import common as C


def cuts(ids: list[int], tok, every: int, think_end: int) -> list[int]:
    out, pos = [], every
    while pos < think_end:
        snap = pos
        for j in range(pos, min(pos + 40, think_end)):
            if "\n" in tok.decode([ids[j]]):
                snap = j + 1
                break
        out.append(snap)
        pos += every
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="4242,4243")
    ap.add_argument("--model", default="r3")
    ap.add_argument("--hops", default="7,8")
    ap.add_argument("--per-hop", type=int, default=6)
    ap.add_argument("--every", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    think_close = tok.convert_tokens_to_ids("</think>")
    hops = tuple(int(h) for h in a.hops.split(","))
    tasks = [t for s in a.seeds.split(",") for t in C.make_tasks(int(s), hops, a.per_hop)]
    print(f"[freeze] {len(tasks)} tasks, model={a.model}", flush=True)

    def run(task):
        return task, C.chat(a.model, task["prompt"], C.room(task["prompt"], True),
                            thinking=True)

    trajs, states, n_solved, n_noroom = [], [], 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for task, res in ex.map(run, tasks):
            if res.get("error"):
                print(f"  error {task['id']}: {res['error']}")
                continue
            ids = res["ids"]
            think_end = ids.index(think_close) if think_close in ids else len(ids)
            ok, why = C.grade(task, res["content"])
            trajs.append({"task_id": task["id"], "pass": ok, "why": why,
                          "finish": res["finish"], "n_ids": len(ids),
                          "think_tokens": think_end, "ids": ids,
                          "prompt_ids": res["prompt_ids"]})
            for i, cut in enumerate(cuts(ids, tok, a.every, think_end)):
                trace = tok.decode(ids[:cut]).replace("<think>", "").strip()
                if C.chain_in_trace(task, trace):
                    n_solved += 1              # chain already found: note = paste
                    continue
                if C.room(C.handoff_prompt(task, trace),
                          C.notes.NOTE_THINKING) < C.notes.NOTE_CALL_MAX:
                    n_noroom += 1              # note can't fit inside the 8k budget
                    continue
                states.append({
                    "state_id": f"{task['id']}@{cut}", "task": task,
                    "depth": cut, "depth_frac": round(cut / max(think_end, 1), 3),
                    "cut_index": i, "traj_pass": ok, "trace": trace,
                })
            print(f"  {task['id']} pass={ok} think={think_end} finish={res['finish']}",
                  flush=True)

    C.jsonl_write(C.DATA / "trajectories.jsonl", trajs)
    C.jsonl_write(C.DATA / "states.jsonl", states)
    n_pass = sum(t["pass"] for t in trajs)
    print(f"[freeze] {len(trajs)} trajectories ({n_pass} pass), {len(states)} states "
          f"-> {C.DATA / 'states.jsonl'}")


if __name__ == "__main__":
    main()
