"""Stub + fixture tests for branch.py. No server, no GPU.

  ..\\.venv\\Scripts\\python.exe test_branch.py

1. stub pipeline on multi-hop seeds 6000-6001 (h3/h4): freeze -> serialize ->
   resume -> K=4 notes -> continuation -> grade_mhqa -> group; every group has
   >=1 pass and >=1 fail, token-exact note prompt, EOS-terminated completions.
2. fidelity guard: tampered served prompt ids -> no group.
3. real find_h transcripts (fixtures/rollouts_find_h.jsonl, from the pod):
   freeze at a compaction point -> serialize -> resume re-renders the frozen ids.
"""

from __future__ import annotations

import json
import sys

import branch as B
import common as C
from stub_model import StubModel

FIX = C.HERE / "fixtures" / "rollouts_find_h.jsonl"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_stub_pipeline() -> str:
    tasks = B.mhqa_tasks([6000, 6001], (3, 4), 1)
    with StubModel(tasks).installed() as stub:
        stats = B.run(tasks, "stub", "stub", 2048, 4, 1, "think", 2, "stubtest")
    groups = C.jsonl_read(C.DATA / "branch_groups-stubtest.jsonl")
    snaps = C.DATA / "branch_snapshots-stubtest.jsonl"
    check(len(groups) == len(tasks) == stats["groups"], f"groups {stats}")
    tok = C.tokenizer()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for g, line in zip(groups, snaps.read_text(encoding="utf-8").splitlines()):
        snap = B.loads(line)
        check(B.dumps(B.loads(B.dumps(snap))) == B.dumps(snap), "snapshot round-trip")
        check(snap["freeze_tokens"] == 2048 == len(snap["context_ids"]), "compaction point")
        check(g["prompt_ids"] == B.encoding.render_chat(
            tok, B.user_msgs(B.resume(snap)), True), "note prompt not token-exact")
        check(len(g["completions"]) == 4 and g["rewards"] == [1.0, 0.0, 1.0, 0.0],
              f"{g['group_id']} rewards {g['rewards']}")
        check(g["mixed"], "group not mixed")
        check(all(c["completion_ids"][-1] == eos for c in g["completions"]), "EOS")
        check(all(tok.decode(c["completion_ids"][:-1]).split("</think>")[-1].strip()
                  == c["text"].strip() for c in g["completions"]), "ids != text")
        m = g["meta"]
        check(all(m[k] is not None for k in ("seed", "freeze_turn", "budget", "task_id")),
              "meta")
        check(all(c["continuations"][0]["pass"] == bool(c["reward"])
                  for c in g["completions"]), "outcome != reward")
    return (f"stub: {stats['groups']} groups x4 notes, rewards {groups[0]['rewards']}, "
            f"note_pass {stats['note_pass']}, mixed {stats['mixed_groups']}/{stats['groups']}, "
            f"stub notes {stub.log.count('note-pass')}p/{stub.log.count('note-fail')}f")


def test_fidelity_guard() -> str:
    tasks = B.mhqa_tasks([6000], (3,), 1)
    with StubModel(tasks, tamper_prompt_ids=True).installed():
        s = B.freeze_mhqa(tasks[0], "stub", 2048)
    check(s.get("skip", "").startswith("fidelity"), f"tamper not caught: {s.get('skip')}")
    return "fidelity guard: tampered prompt ids rejected"


def test_fixture_transcripts() -> str:
    rows = [json.loads(ln) for ln in FIX.read_text(encoding="utf-8").splitlines() if ln.strip()]
    msgs = B.transcript_messages(rows)
    out = []
    for row, m in zip(rows, msgs):
        snap = B.freeze_transcript(row, m, 1536)
        if "skip" in snap:
            out.append(f"{row['task']}:{snap['skip']}")
            continue
        back = B.loads(B.dumps(snap))
        hp = B.resume(back)                      # raises unless ids re-render exactly
        check(back["context_ids"] == snap["context_ids"], "ids round-trip")
        check(1 <= back["freeze_turn"] < len(row["messages"]), "freeze turn")
        check(C.room(hp, True) >= C.notes.NOTE_CALL_MAX, "no room for note call")
        out.append(f"{row['task']}:t{back['freeze_turn']}/{len(row['messages'])}"
                   f"@{back['freeze_tokens']}")
    check(sum(":t" in o for o in out) == len(rows), f"too few frozen: {out}")
    return "fixture: " + ", ".join(out)


if __name__ == "__main__":
    for t in (test_stub_pipeline, test_fidelity_guard, test_fixture_transcripts):
        print(f"PASS {t.__name__}: {t()}", flush=True)
    print("ALL PASS")
    sys.exit(0)
