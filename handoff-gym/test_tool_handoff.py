"""Stub tests for tool_handoff.py (compaction on TOOL episodes). No server, no GPU.

  ..\\.venv\\Scripts\\python.exe test_tool_handoff.py

1. mhqa_t end to end through the REAL agent loop (agent.run_episode, real workspace
   tools, real reward): a scripted episode (real tool results) is frozen mid-chain;
   blank resumes fail (gate passes), oracle notes resume by reading the next kb
   files and pass, her (stub) notes alternate good/bad -> mixed. Partial rows + stats.
2. oracle notes never contain an unfound hop / the find_h answer.
3. edits before the cut persist in the reader's workspace (edit family).
4. note-call fidelity: tampered prompt ids -> error (missing), not a note.
5. real find_h transcripts (fixtures) freeze at turn boundaries with room for the note.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import tool_handoff as TH

agent, C, notes = TH.agent, TH.C, TH.notes
import mhqa_tool as M  # noqa: E402  (sandbox-gym, on sys.path via tool_handoff)
from schema import Action, Turn  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def obs_of(spec: dict, action: Action, label: str) -> str:
    cfg = TH.make_cfg(_args(), "stub")
    ws = TH.fresh_ws(cfg, spec, [], label)
    return agent.run_action(ws, action, spec["tools"])


def _args():
    class A:
        base, runner = "http://127.0.0.1:1", "host"
    return A()


def mhqa_episode(seed: int, hops: int) -> dict:
    """Her-shaped episode with REAL tool results: list kb, read one file per hop, answer."""
    spec = next(s for s in M.specs(seed) if s["hops"] == hops)
    chain, kinds = spec["ground_truth"].split(M.SEP), M._kinds(hops)
    acts = [Action("list_dir", {"path": "kb"})]
    acts += [Action("read_file", {"path": f"kb/{M._KIND_DIR[k]}/{M.slug(c)}.md"})
             for c, k in zip(chain[:-1], kinds[:-1])]
    turns = [{"content": json.dumps({"name": x.name, "arguments": x.arguments}),
              "reasoning": "next lookup", "truncated": False,
              "action": {"name": x.name, "arguments": x.arguments},
              "obs": obs_of(spec, x, f"t-{spec['name']}")} for x in acts]
    turns.append({"content": f"ANSWER: {spec['ground_truth']}", "reasoning": "", "obs": "",
                  "truncated": False, "action": None})
    return {"key": f"{spec['name']}#e0", "family": "mhqa_t", "seed": seed, "name": spec["name"],
            "cap": 10, "passed": True, "stop": "answer", "turns": turns}


# --- stub reader (agent.chat_once) and stub writer (common.write_note) --------------------

def stub_reader(cfg, messages, max_tokens, return_ids=False, timeout=600):
    """Follows the note: starts at its 'last confirmed' entity and reads one kb file per
    remaining lookup with read_file, parsing each result it gets back. No note -> gives up."""
    def say(text):
        return {"content": text, "reasoning": "", "finish_reason": "stop",
                "token_ids": [], "prompt_token_ids": []}
    note = re.search(r"<handoff>\n(.*?)\n</handoff>", messages[1]["content"], re.S).group(1)
    m = re.search(r"last confirmed (.+?) \((\w+)\)\. Remaining \d+ lookups, in order: (.+?)\.\n",
                  note)
    if not m:
        return say("ANSWER: nobody")
    done = re.search(r"verified in the kb files: (.+?)\.\n", note).group(1)
    names = [re.sub(r" \(\w+\)$", "", x) for x in done.split(" -> ")]
    steps = [] if m.group(3) == "none" else [
        re.match(r"(.+) \((\w+)\)", x).groups() for x in m.group(3).split(", then ")]
    cur, kind = m.group(1), m.group(2)
    results = [x["content"] for x in messages[2:] if x["role"] == "user"]
    for (field, nk), res in zip(steps, results):
        v = re.search(rf"^\s*(?:\d+: )?- {re.escape(field)}: (.+)$", res, re.M)
        if not v:
            return say("ANSWER: lost")
        names.append(v.group(1).strip())
        cur, kind = names[-1], nk
    if len(results) < len(steps):
        path = f"kb/{M._KIND_DIR[kind]}/{M.slug(cur)}.md"
        return say(json.dumps({"name": "read_file", "arguments": {"path": path}}))
    return say("ANSWER: " + M.SEP.join(names))


class StubWriter:
    """Even calls: the oracle note (good); odd calls: same note with a decoy as the last
    confirmed entity (bad). prompt_ids = render_chat of the handoff prompt (fidelity ok)."""
    def __init__(self, tamper=False):
        self.n, self.tamper = 0, tamper

    def __call__(self, model, hp, mode):
        # the frozen state this handoff prompt was built from (its trace is in hp)
        snap = next(s for s in self.snaps if s["trace"] in hp)
        spec, _ = TH.make_task(snap["family"], snap["seed"], snap["name"])
        text = TH.oracle_mhqa_t(spec, [TH.turn_from_dict(d) for d in snap["turns"]])
        if self.n % 2:
            last = re.search(r"last confirmed (.+?) \(", text).group(1)
            text = text.replace(f"last confirmed {last}", "last confirmed Nobody Atall")
        self.n += 1
        ids = TH.encoding.render_chat(C.tokenizer(), [{"role": "user", "content": hp}],
                                      mode != "off")
        return {"text": text, "note_tokens": 120, "ids": [], "over_cap": False,
                "forced": mode == "force", "prompt_ids": ids[:-1] if self.tamper else ids}


def installed(writer):
    old = agent.chat_once, agent.wait_healthy, C.write_note
    agent.chat_once, agent.wait_healthy, C.write_note = stub_reader, (lambda *a, **k: True), writer
    return old


def restore(old):
    agent.chat_once, agent.wait_healthy, C.write_note = old


# --- tests ----------------------------------------------------------------------------------

def test_mhqa_pipeline() -> str:
    eps = [mhqa_episode(s, h) for s in (3001, 3002) for h in (5, 7)]
    tmp = Path(tempfile.mkdtemp()) / "eps.jsonl"
    C.jsonl_write(tmp, eps)
    w = StubWriter()
    w.snaps = [TH.freeze(e, k) for e in eps
               for k in TH.cut_points(len(e["turns"]), e["cap"], [0.34, 0.5, 0.67])]
    w.snaps = [x for x in w.snaps if "skip" not in x]
    old = installed(w)
    try:
        stats = TH.main(["--source", "file", "--episodes", str(tmp), "--families", "mhqa_t",
                         "--runner", "host", "--states", "5", "--k", "2", "--reads", "1",
                         "--modes", "blank,oracle,off,force", "--workers", "2",
                         "--tag", "stubtest-tool"])
    finally:
        restore(old)
    check(stats is not None, "gate stopped a stub run whose blank always fails")
    check(stats["blank_heldout"]["read_rate"] == 0.0, f"blank {stats['blank_heldout']}")
    check(stats["oracle"]["read_rate"] == 1.0, f"oracle {stats['oracle']}")
    for m in ("off", "force"):
        check(0 < stats[m]["note_pass"] < stats[m]["notes"], f"{m} not mixed: {stats[m]}")
    check(stats["verdict"].startswith("ok"), stats["verdict"])
    rows = C.jsonl_read(TH.DATA / "tool_rows-stubtest-tool.partial.jsonl")
    check(len(rows) == stats["states"] == 5, f"partial rows {len(rows)}")
    snaps = C.jsonl_read(TH.DATA / "tool_snapshots-stubtest-tool.jsonl")
    check(all(s["reader_turns"] == s["cap"] - s["cut_turn"] for s in snaps), "turn rule")
    return (f"mhqa_t: {len(snaps)} states, scored {stats['states']}; blank "
            f"{stats['blank_heldout']['pass']}/{stats['blank_heldout']['n']}, oracle reads "
            f"{stats['oracle']['read_pass']}/{stats['oracle']['reads']}, off notes "
            f"{stats['off']['note_pass']}/{stats['off']['notes']}, force notes "
            f"{stats['force']['note_pass']}/{stats['force']['notes']}; {stats['verdict']}")


def test_oracle_no_answers() -> str:
    e = mhqa_episode(3003, 7)
    spec, _ = TH.make_task("mhqa_t", 3003, e["name"])
    chain = spec["ground_truth"].split(M.SEP)
    for k in (2, 4):
        s = TH.freeze(e, k)
        check("skip" not in s, f"freeze skipped: {s}")
        text = TH.oracle_mhqa_t(spec, [TH.turn_from_dict(d) for d in s["turns"]])
        seen = "\n".join(d["obs"] for d in s["turns"]).lower()
        unfound = [c for c in chain[1:] if c.lower() not in seen]
        check(unfound and not any(c in text for c in unfound), f"oracle leaks {unfound}")
    rows = C.jsonl_read(TH.HERE / "fixtures" / "rollouts_find_h.jsonl")
    n = 0
    for r in rows:
        if r["task"].startswith("findh-effective"):
            sp, _ = TH.make_task("find_h", r["seed"], r["task"])
            text = TH.oracle_findh_effective(sp, [])
            check(not re.search(rf"\b{re.escape(sp['ground_truth'])}\b", text),
                  "find_h oracle names the answer")
            n += 1
    return f"oracle: mhqa_t h7 cuts t2/t4 leak no unfound hop; findh-effective {n} checked"


def test_edits_persist() -> str:
    spec = next(s for s in TH.repo_gen.make_specs(3001, "edit") if s["name"] == "edit-tax-3001")
    line = next(ln for ln in spec["files"]["config.py"].splitlines() if ln.startswith("TAX_RATE"))
    edit = Action("edit_file", {"path": "config.py", "old_string": line,
                                "new_string": "TAX_RATE = 0.06"})
    turns = [Turn(content="edit", action=edit, obs="ok"),
             Turn(content="ANSWER: done")]
    cfg = TH.make_cfg(_args(), "stub")
    ws = TH.fresh_ws(cfg, spec, turns[:1], "persist-test")
    check("TAX_RATE = 0.06" in (ws.root / "config.py").read_text(encoding="utf-8"),
          "edit before the cut not in the reader's workspace")
    _, task = TH.make_task("edit", 3001, "edit-tax-3001")
    check(task.reward([turns[1]], ws) >= 0.5, "reward on persisted edit + answer")
    ws0 = TH.fresh_ws(cfg, spec, [], "persist-test0")
    check(task.reward([turns[1]], ws0) < 0.5, "unedited workspace should fail")
    return "edits: change from turn 1 survives the cut; reward 1 with it, 0 without"


def test_note_fidelity() -> str:
    e = mhqa_episode(3001, 5)
    s = TH.freeze(e, 3)
    w = StubWriter(tamper=True)
    w.snaps = [s]
    old = installed(w)
    try:
        r = TH.write("force", s, "stub")
    finally:
        restore(old)
    check(r.get("error", "").startswith("fidelity"), f"tamper not caught: {r}")
    return "note call: tampered prompt ids -> error (missing), not scored"


def test_fixture_freeze() -> str:
    eps = TH.fixture_episodes()
    out = []
    for e in eps:
        ks = TH.cut_points(len(e["turns"]), e["cap"], [0.34, 0.5, 0.67])
        for k in ks:
            s = TH.freeze(e, k)
            if "skip" in s:
                out.append(f"{e['name']}@t{k}:{s['skip']}")
                continue
            hp = notes.handoff_prompt(TH.make_task(e["family"], e["seed"], e["name"])[1].prompt,
                                      s["trace"], TH.READER_SEES)
            check(C.room(hp, True) >= notes.NOTE_CALL_MAX, "room")
            check(json.loads(json.dumps(s)) == s, "snapshot round-trip")
            out.append(f"{e['name']}@t{k}/{len(e['turns'])}")
    check(any("/" in o for o in out), f"no fixture state frozen: {out}")
    return "fixture: " + ", ".join(out)


if __name__ == "__main__":
    for t in (test_oracle_no_answers, test_edits_persist, test_note_fidelity,
              test_fixture_freeze, test_mhqa_pipeline):
        print(f"PASS {t.__name__}: {t()}", flush=True)
    print("ALL PASS")
    sys.exit(0)
