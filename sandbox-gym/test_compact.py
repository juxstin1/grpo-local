"""In-episode compaction (compact.py + agent.run_episode), stub model, no server.

Ids are characters (compact_stub.render), so the 8192 budget is 8192 characters:
the 75% trigger fires at 6144, a rebuilt context must fit in 3277.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

import agent
import compact
import compact_stub as S
from config import Config

NOTES = compact.notes
plen = lambda m: len(S.render(m))  # noqa: E731
TOK = S.FakeTok()


@pytest.fixture
def patch(monkeypatch):
    def install(stub):
        monkeypatch.setattr(agent, "chat_once", stub.chat_once)
        monkeypatch.setattr(agent, "wait_healthy", lambda base, max_wait=900: True)
        monkeypatch.setattr(compact, "complete_ids", stub.complete_ids)
        return stub
    return install


def long_read(n_files=6, size=1500):
    """Six files of ~1500 chars, one code each; the answer needs all six."""
    tmp = Path(tempfile.mkdtemp())
    codes = [f"C{i}{i * 7 % 10}" for i in range(n_files)]
    files = {f"doc/part{i}.txt": (f"code {codes[i]}\n" + "filler text. " * (size // 13))
             for i in range(n_files)}
    ws = S.make_ws(tmp / "ws", files)
    answer = "-".join(codes)
    task = S.make_task("Read every file in doc/ and answer with all codes joined by '-'.",
                       answer)
    return ws, task, answer


def cfg_on(**kw):
    c = Config()
    c.episode = replace(c.episode, compact=True, **kw)
    return c


def run(cfg, task, ws, return_ids=True):
    return agent.run_episode(cfg, task, ws, prompt_len_fn=plen, return_ids=return_ids,
                             tokenizer=TOK)


def ckpt_note(extra=""):
    return S.NOTE + extra


# --- compact OFF is today's behaviour ---------------------------------------------

def test_compact_off_identical_to_pre_change_golden(patch):
    tmp = Path(tempfile.mkdtemp())
    ws = S.make_ws(tmp / "ws", {"a.txt": "alpha\n", "dir/b.txt": "the code is 42\n"})
    task = S.make_task("Find the code in the workspace.", "42")
    script = patch(S.Script([
        {"content": S.call("list_dir", path="."), "reasoning": "look around"},
        {"content": S.call("read_file", path="dir/b.txt"), "reasoning": "open b"},
        {"content": S.call("checkpoint", note="Goal: x"), "reasoning": "try ckpt"},
        {"content": "ANSWER: 42", "reasoning": "done"},
    ]))
    ep = agent.run_episode(Config(), task, ws, prompt_len_fn=plen, return_ids=True)
    now = json.loads(json.dumps({
        "calls": script.calls, "prompt_ids": ep.prompt_ids_per_turn,
        "gen_ids": ep.token_ids_per_turn, "stop": ep.stop, "answer": ep.answer,
        "obs": [t.obs for t in ep.turns]}))
    gold = json.loads((Path(__file__).parent / "tests_fixtures" /
                       "compact_off_golden.json").read_text(encoding="utf-8"))
    assert now == gold                     # recorded with agent.py before compaction existed
    assert ep.resets == [] and ep.windows == []


def test_checkpoint_tool_only_with_compact_on():
    task = S.make_task("x", "y")
    assert "checkpoint(" not in agent.system_prompt(task.tools, 10)
    assert "checkpoint(note, plan?)" in compact.system_text(task, 10)
    assert "75%" in compact.system_text(task, 10)


# --- the main path: harness reset + self checkpoint ---------------------------------

def work_script(answer):
    return [S.call("read_file", path=f"doc/part{i}.txt") for i in range(4)] + [
        S.call("read_file", path="doc/part4.txt"),
        S.call("checkpoint", note=ckpt_note("\nfound C0..C4"), plan="read part5 then answer"),
        S.call("read_file", path="doc/part5.txt"),
        f"ANSWER: {answer}",
    ]


def test_long_read_survives_two_resets(patch):
    ws, task, answer = long_read()
    stub = patch(S.Adaptive(work_script(answer)))
    cfg = cfg_on()
    # without compaction the same reads overflow the budget
    assert sum(len(p.read_text()) for p in (ws.root / "doc").iterdir()) > cfg.episode.budget
    ep = run(cfg, task, ws)
    assert ep.stop == "answer" and ep.answer == answer
    assert task.reward(ep.turns, ws) == 1.0
    ok = [r for r in ep.resets if r["outcome"] == "ok"]
    assert [r["trigger"] for r in ok] == ["harness", "self"]
    h, s = ok
    assert h["tokens_before"] >= NOTES.compact_trigger_tokens(8192) == 6144
    assert h["plan_set_by"] == "first-note" and "Next step" in h["plan"]
    assert s["plan_set_by"] == "checkpoint" and s["plan"] == "read part5 then answer"
    for r in ok:
        assert r["context_tokens"] <= NOTES.compact_context_max(8192) == 3277
    # every work call stayed under the trigger after a reset; nothing exceeded the budget
    for c in stub.calls:
        assert plen(c["messages"]) + c["max_tokens"] <= 8192 + 64
    # rebuilt context: system + one user message with task, plan, note, workspace view
    m = ep.windows[-1]["messages"]
    assert [x["role"] for x in m] == ["system", "user"]
    u = m[1]["content"]
    for part in (task.prompt, "Plan (pinned):\nread part5 then answer", "<handoff>",
                 "Workspace files (they persist", "doc/part5.txt", "turns used]"):
        assert part in u
    # global turn counter continues across resets
    last = stub.calls[-1]["messages"]           # the ANSWER call, in the 3rd window
    assert "<handoff>" in last[1]["content"] and "[6 of 10 turns used]" in last[1]["content"]
    assert "[turn 7 of 10]" in last[-1]["content"]
    assert len(ep.turns) == 8


def test_note_call_fits_remaining_budget(patch):
    ws, task, answer = long_read()
    stub = patch(S.Adaptive(work_script(answer)))
    ep = run(cfg_on(), task, ws)
    note_calls = [c for c in stub.calls if len(c["messages"]) == 1]
    assert note_calls and all(not c["thinking"] for c in note_calls)       # off mode
    for c in note_calls:
        assert plen(c["messages"]) + c["max_tokens"] <= 8192
        assert c["max_tokens"] == NOTES.NOTE_CAP
    assert ep.resets[0]["note_room"] >= NOTES.NOTE_CAP


# --- training fidelity ------------------------------------------------------------

def test_sequences_are_token_exact(patch):
    ws, task, answer = long_read()
    stub = patch(S.Adaptive(work_script(answer)))
    ep = run(cfg_on(), task, ws)
    assert compact.fidelity_error(ep, TOK, True, 10) is None
    seqs = compact.sequences(ep, TOK, True, 10)
    assert [q["kind"] for q in seqs] == ["window", "note", "window", "window"]
    note = seqs[1]["segments"]
    assert note[0]["train"] is False and note[1]["train"] is True
    assert note[0]["ids"] == S.render(ep.resets[0]["note_messages"], False)
    # every token she generated (work turns + the harness note) is trainable exactly once
    trained = [i for q in seqs for s in q["segments"] if s["train"] for i in s["ids"]]
    hers = [i for ids in ep.token_ids_per_turn for i in ids] + note[1]["ids"]
    assert sorted(trained) == sorted(hers) and len(trained) == len(hers)
    # each window starts with the prompt the server actually saw
    starts = [0] + [w["start_turn"] for w in ep.windows]
    win = [q for q in seqs if q["kind"] == "window"]
    for q, st in zip(win, starts):
        assert q["segments"][0]["ids"] == ep.prompt_ids_per_turn[st]


def test_tampered_prompt_is_a_fidelity_error(patch):
    ws, task, answer = long_read()
    patch(S.Adaptive(work_script(answer)))
    ep = run(cfg_on(), task, ws)
    ep.resets[0]["segments"][0]["ids"] = ep.resets[0]["segments"][0]["ids"][:-1]
    assert "note prompt" in compact.fidelity_error(ep, TOK, True, 10)


def test_rl_counts_tokens_across_windows():
    import rl
    from schema import Trajectory
    t = Trajectory(task=None, turns=[], tok_ids=[1, 2], trainable=[False, True],
                   windows=[([1, 2, 3], [False, True, True]), ([4, 5], [False, True])])
    assert rl._n_policy_tokens(t) == 3
    plain = Trajectory(task=None, turns=[], tok_ids=[1, 2, 3], trainable=[False, True, True])
    assert rl._n_policy_tokens(plain) == 2


# --- checkpoint validation, truncation, caps, guards --------------------------------

def test_checkpoint_missing_sections_is_error_no_reset(patch):
    ws, task, answer = long_read(n_files=1)
    patch(S.Adaptive([S.call("checkpoint", note="Goal: x\nDone: y"),
                      S.call("read_file", path="doc/part0.txt"), f"ANSWER: {answer}"]))
    ep = run(cfg_on(), task, ws)
    assert ep.resets == [] and ep.stop == "answer"
    assert "missing: Current state, Hypothesis, Next step, Don't repeat" in ep.turns[0].obs
    assert "No reset happened" in ep.turns[0].obs


def test_truncated_note_continues_and_is_flagged(patch):
    ws, task, answer = long_read()
    long_note = S.NOTE + "\n" + "detail " * 200          # > 400 tokens: max_tokens cuts it
    stub = patch(S.Adaptive(work_script(answer), note=lambda i: long_note))
    ep = run(cfg_on(), task, ws)
    h = ep.resets[0]
    assert h["outcome"] == "ok" and h["truncated"] is True
    assert h["note_tokens_used"] <= NOTES.NOTE_CAP
    assert ep.stop == "answer" and ep.answer == answer           # not a fail
    assert compact.summary(ep)["notes_truncated"] == 1
    assert stub.notes == 1


def test_self_checkpoint_note_over_cap_is_cut(patch):
    ws, task, answer = long_read(n_files=2)
    big = S.NOTE + "\n" + "x" * 900
    patch(S.Adaptive([S.call("read_file", path="doc/part0.txt"),
                      S.call("checkpoint", note=big),
                      S.call("read_file", path="doc/part1.txt"), f"ANSWER: {answer}"]))
    ep = run(cfg_on(), task, ws)
    r = ep.resets[0]
    assert r["trigger"] == "self" and r["truncated"] and r["cut_at"] == NOTES.NOTE_CAP
    assert len(r["note_text"]) == NOTES.NOTE_CAP and ep.stop == "answer"


def test_turn_cap_binds_across_resets(patch):
    ws, task, answer = long_read()
    patch(S.Adaptive(work_script(answer)))
    cfg = cfg_on(max_turns=6)
    ep = run(cfg, task, ws)
    assert ep.stop == "cap" and ep.cap_hit and len(ep.turns) == 6
    assert any(r["outcome"] == "ok" for r in ep.resets)


def test_thrash_guard(patch):
    ws, task, answer = long_read(n_files=1)
    patch(S.Adaptive([S.call("read_file", path="doc/part0.txt"),
                      S.call("checkpoint", note=S.NOTE),
                      S.call("checkpoint", note=S.NOTE),
                      f"ANSWER: {answer}"]))
    ep = run(cfg_on(), task, ws)
    assert ep.stop == "compact-fail" and ep.compact_fail == "thrash"
    assert [r["outcome"] for r in ep.resets] == ["ok", "thrash"]
    assert compact.summary(ep)["compact_fail"] == "thrash"


def test_view_trimmed_to_fit(patch):
    tmp = Path(tempfile.mkdtemp())
    files = {f"d{i:02d}/file_with_a_long_descriptive_name_{j:02d}.txt": "x"
             for i in range(12) for j in range(8)}
    ws = S.make_ws(tmp / "ws", files)
    task = S.make_task("Answer 7.", "7")
    patch(S.Adaptive([S.call("list_dir", path="."), S.call("checkpoint", note=S.NOTE),
                      "ANSWER: 7"]))
    ep = run(cfg_on(), task, ws)
    r = ep.resets[0]
    assert r["outcome"] == "ok" and r["view_trimmed"] and r["view_trimmed_tokens"] > 0
    assert r["context_tokens"] <= 3277 < r["context_tokens_untrimmed"]
    assert ep.stop == "answer"


def test_context_too_big_ends_episode(patch):
    ws, _, _ = long_read(n_files=1)
    task = S.make_task("Long task statement. " * 160, "z")     # ~3.4k chars alone
    patch(S.Adaptive([S.call("read_file", path="doc/part0.txt"),
                      S.call("checkpoint", note=S.NOTE), "ANSWER: z"]))
    ep = run(cfg_on(), task, ws)
    assert ep.stop == "compact-fail" and ep.compact_fail == "context"
    r = ep.resets[-1]
    assert r["outcome"] == "context-too-big" and r["view_depth"] == 0


def test_force_mode_caps_thinking_and_forces_close(patch):
    ws, task, answer = long_read()
    stub = patch(S.Adaptive(work_script(answer), note_reasoning="ponder " * 400))
    ep = run(cfg_on(compact_note_mode="force"), task, ws)
    h = ep.resets[0]
    assert h["mode"] == "force" and h["forced"] is True
    kinds = [s["kind"] for s in h["segments"]]
    assert kinds == ["prompt", "note", "forced", "note"]
    assert [s["train"] for s in h["segments"]] == [False, True, False, True]
    first = next(c for c in stub.calls if len(c.get("messages", [])) == 1)
    assert first["thinking"] and first["max_tokens"] == h["think_cap"]
    total = len(h["segments"][0]["ids"]) + sum(len(s["ids"]) for s in h["segments"][1:])
    assert total <= 8192
    assert ep.stop == "answer"


def test_max_resets_stops_triggering(patch):
    ws, task, answer = long_read()
    patch(S.Adaptive(work_script(answer)))
    ep = run(cfg_on(max_resets=0), task, ws)
    assert ep.resets == []


def test_compact_requires_tokenizer():
    ws, task, _ = long_read(n_files=1)
    with pytest.raises(ValueError):
        agent.run_episode(cfg_on(), task, ws, prompt_len_fn=plen)
