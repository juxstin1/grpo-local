"""In-episode compaction for agent.run_episode (switch: config Episode.compact).

With compact ON, an episode can outlive one context window:

  harness trigger  before a model call, if the rendered prompt reaches
                   notes.COMPACT_TRIGGER of the budget (8192 -> 6144 tokens), she
                   is asked for a handoff note (notes.handoff_prompt, fixed
                   skeleton notes.SECTIONS) and the context is rebuilt.
  checkpoint tool  she can do the same herself at any time:
                   checkpoint(note, plan?) -- skeleton-validated; a note missing
                   sections is an error observation and no reset.

Rebuilt context = system prompt (+ compaction rules) and ONE user message:
pinned task, pinned plan, her latest note (notes.resume_prompt), a capped view of
the workspace files (they persist: same workspace, nothing replayed), and the
turns used so far. The turn counter is global; max_turns binds the whole episode.

Plan rule: a checkpoint `plan` argument pins (and replaces) the plan. If she has
never set one, the first note's "Goal" + "Next step" become the pinned plan.

Note call (harness trigger) always fits the remaining budget: the room is
computed from the note prompt's actual rendered length, and the oldest trace
turns are dropped until NOTE_CAP (+ COMPACT_MIN_THINK in force mode) fits.
  off    thinking off, max_tokens = NOTE_CAP.
  force  thinking on, capped at min(NOTE_THINK_CAP, room - NOTE_CAP); if she is
         still thinking at the cap, "\\n</think>\\n\\n" is appended (not trainable)
         and the note continues for what is left of NOTE_CAP.
A note longer than NOTE_CAP (or cut by max_tokens) is NOT a fail: she continues
with her own tokens cut at NOTE_CAP tokens of note text; the reset is flagged
truncated=True. An empty note continues as "(empty)" and is flagged.

Only two compaction conditions end an episode early (stop="compact-fail"):
  thrash   two resets in a row with no (non-checkpoint) tool call between them
  context  the rebuilt context is still > COMPACT_CONTEXT_MAX of the budget with
           the workspace view trimmed to nothing
Everything is logged per reset in Episode.resets.

Training: sequences(ep) splits a compacted episode into token-exact sequences --
one per context window (prompt, her generations, observations) and one per
harness note call (prompt, her note ids; a forced </think> is not trainable).
Eval, collection and GRPO all run agent.run_episode, so all three see the same
tokens. rollout.py (legacy local HF path) refuses compact.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

import encoding
import tools as T
from gyms import system_prompt
from schema import Turn



def _load_notes():
    """grpo-local/notes.py (the ONE note-call spec) by explicit path: putting
    grpo-local on sys.path would shadow sandbox-gym's config/schema/rl modules."""
    import importlib.util
    import sys
    if "notes" in sys.modules:
        return sys.modules["notes"]
    spec = importlib.util.spec_from_file_location(
        "notes", Path(__file__).resolve().parent.parent / "notes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["notes"] = mod
    spec.loader.exec_module(mod)
    return mod


notes = _load_notes()
READER_SEES = "the task, the plan, the workspace files and the tools"
CHECK_AGAINST = "the files"
COMPACT_RULES = (
    "Context limit: when your context reaches {pct}% of the budget, the harness stops "
    "you, asks for a handoff note and restarts you with only the task, the pinned plan, "
    "your note and the workspace files. You can do this yourself at any time with "
    "checkpoint(note, plan?); the note must use the sections: {sections}.\n"
)
# workspace view trim ladder: (depth, max entries); (0, 0) = view omitted
VIEW_STEPS = ((2, 60), (2, 30), (1, 30), (1, 15), (1, 5), (0, 0))
VIEW_OMITTED = "(workspace view omitted to fit the context limit; use list_dir)"


def system_text(task, max_turns: int | None, hint: str = "") -> str:
    """System prompt with compaction ON: checkpoint enabled + the rules line."""
    s = system_prompt(enabled_tools(task), max_turns, hint)
    return s + COMPACT_RULES.format(pct=round(notes.COMPACT_TRIGGER * 100),
                                    sections=", ".join(notes.SECTIONS))


def enabled_tools(task) -> list[str]:
    return list(task.tools) + ["checkpoint"]


def complete_ids(cfg, prompt_ids: list[int], max_tokens: int, timeout: int = 600) -> dict:
    """Raw completion continuing an exact token stream (force-mode note)."""
    body = {"model": cfg.serve.served_model or cfg.serve.model_name, "prompt": prompt_ids,
            "max_tokens": max_tokens, "temperature": cfg.serve.temperature,
            "top_p": cfg.serve.top_p, "return_token_ids": True, "skip_special_tokens": False}
    try:
        r = requests.post(f"{cfg.serve.base.rstrip('/')}/v1/completions", json=body,
                          timeout=timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        return {"text": ch.get("text") or "", "finish": ch.get("finish_reason", ""),
                "ids": ch.get("token_ids") or []}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "text": "", "finish": "",
                "ids": []}


def workspace_view(root: Path, depth: int, max_entries: int) -> str:
    """Deterministic file tree (sorted, dirs end with '/'), at most max_entries lines."""
    if depth <= 0 or max_entries <= 0:
        return VIEW_OMITTED
    lines: list[str] = []
    total = 0

    def walk(d: Path, level: int) -> None:
        nonlocal total
        for name in sorted(os.listdir(d)):
            p = d / name
            rel = p.relative_to(root).as_posix()
            total += 1
            if len(lines) < max_entries:
                lines.append(rel + ("/" if p.is_dir() else ""))
            if p.is_dir() and level < depth:
                walk(p, level + 1)

    walk(root, 1)
    if total > len(lines):
        lines.append(f"... and {total - len(lines)} more entries (use list_dir)")
    return "\n".join(lines) or "(empty)"


def trace_text(turns: list[Turn], first_turn: int) -> str:
    """Her work in the current window as text for the note prompt (reasoning left out)."""
    out = []
    for i, t in enumerate(turns):
        out.append(f"[turn {first_turn + i + 1}] {(t.resp_text or t.content).strip()}")
        if t.obs:
            out.append(f"[result]\n{t.obs}")
    return "\n".join(out)


def _plan_from_note(note: str) -> str:
    s = notes.note_sections(note)
    return "\n".join(f"{k}: {s[k]}" for k in ("Goal", "Next step") if s.get(k))


class Compactor:
    """Per-episode compaction state. agent.run_episode owns the loop."""

    def __init__(self, cfg, task, ws, prompt_len_fn, tokenizer, max_turns: int | None):
        if prompt_len_fn is None or tokenizer is None:
            raise ValueError("compact needs prompt_len_fn and tokenizer (note sizing)")
        if cfg.episode.compact_note_mode not in ("off", "force"):
            raise ValueError(f"compact_note_mode {cfg.episode.compact_note_mode!r}")
        self.cfg, self.task, self.ws = cfg, task, ws
        self.plen, self.tok, self.N = prompt_len_fn, tokenizer, max_turns
        self.budget = cfg.episode.budget
        self.trigger = notes.compact_trigger_tokens(self.budget)
        self.ctx_max = notes.compact_context_max(self.budget)
        self.system = system_text(task, max_turns, cfg.episode.hint)
        self.plan: str | None = None
        self.window_start = 0           # index into ep.turns where this window began
        self.tools_since_reset = 0      # non-checkpoint tool calls since the last reset
        self.last_prompt_len = 0

    # --- trigger ----------------------------------------------------------------
    def should_reset(self, ep, prompt_len: int) -> bool:
        self.last_prompt_len = prompt_len
        mx = self.cfg.episode.max_resets
        return prompt_len >= self.trigger and (mx is None or len(ep.resets) < mx)

    def ntok(self, text: str) -> int:
        return len(encoding._ids(self.tok(text, add_special_tokens=False)))

    def cut(self, text: str) -> tuple[str, int, bool]:
        """Note text cut at NOTE_CAP tokens of note: (used text, tokens produced, cut?)."""
        ids = encoding._ids(self.tok(text, add_special_tokens=False))
        if len(ids) <= notes.NOTE_CAP:
            return text, len(ids), False
        return self.tok.decode(ids[: notes.NOTE_CAP]), len(ids), True

    # --- the reset ----------------------------------------------------------------
    def reset(self, ep, trigger: str, note: str | None = None,
              plan: str | None = None) -> list[dict] | None:
        """Write/take the note, rebuild the context. Returns the new messages, or None
        when the episode must end (ep.stop / ep.compact_fail / ep.error set)."""
        rec = {"n": len(ep.resets) + 1, "turn": len(ep.turns), "trigger": trigger,
               "tokens_before": self.last_prompt_len,
               "tools_since_last_reset": self.tools_since_reset}
        if ep.resets and self.tools_since_reset == 0:
            rec["outcome"] = "thrash"
            ep.resets.append(rec)
            ep.stop, ep.compact_fail = "compact-fail", "thrash"
            return None
        if trigger == "harness":
            note = self._write_note(ep, rec)
            if note is None:
                ep.resets.append(rec)
                return None
        else:
            rec.update(mode="checkpoint", segments=[])
        used, produced, was_cut = self.cut(note or "")
        rec.update(note_text=used, note_tokens=produced,
                   note_tokens_used=min(produced, notes.NOTE_CAP),
                   truncated=bool(rec.get("truncated")) or was_cut,
                   cut_at=notes.NOTE_CAP if was_cut else None,
                   empty=not used.strip(),
                   missing_sections=notes.missing_sections(used))
        if plan and plan.strip():
            self.plan, rec["plan_set_by"] = plan.strip(), "checkpoint"
        elif self.plan is None and _plan_from_note(used):
            self.plan, rec["plan_set_by"] = _plan_from_note(used), "first-note"
        rec["plan"] = self.plan
        msgs = self._context(used, len(ep.turns), rec)
        if msgs is None:
            rec["outcome"] = "context-too-big"
            ep.resets.append(rec)
            ep.stop, ep.compact_fail = "compact-fail", "context"
            return None
        rec["outcome"] = "ok"
        ep.resets.append(rec)
        ep.windows.append({"start_turn": len(ep.turns),        # snapshot: the loop
                           "messages": [dict(m) for m in msgs]})  # appends to msgs
        self.window_start = len(ep.turns)
        self.tools_since_reset = 0
        return msgs

    def _task_text(self) -> str:
        return self.task.prompt + (f"\n\nPlan (pinned):\n{self.plan}" if self.plan else "")

    def _context(self, note: str, turns_used: int, rec: dict) -> list[dict] | None:
        untrimmed = None
        for step, (depth, entries) in enumerate(VIEW_STEPS):
            view = workspace_view(self.ws.root, depth, entries)
            user = (notes.resume_prompt(self._task_text(), note, CHECK_AGAINST)
                    + f"\n\nWorkspace files (they persist across restarts):\n{view}")
            if self.N:
                user += f"\n\n[{turns_used} of {self.N} turns used]"
            msgs = [{"role": "system", "content": self.system},
                    {"role": "user", "content": user}]
            n = self.plen(msgs)
            untrimmed = n if untrimmed is None else untrimmed
            if n <= self.ctx_max:
                rec.update(context_tokens=n, context_tokens_untrimmed=untrimmed,
                           view_depth=depth, view_entries=entries, view_trimmed=step > 0,
                           view_trimmed_tokens=untrimmed - n, context_max=self.ctx_max)
                return msgs
        rec.update(context_tokens=n, context_tokens_untrimmed=untrimmed, view_depth=0,
                   view_entries=0, view_trimmed=True, view_trimmed_tokens=untrimmed - n,
                   context_max=self.ctx_max)
        return None

    def _write_note(self, ep, rec: dict) -> str | None:
        import agent as A  # late: agent imports this module
        mode = self.cfg.episode.compact_note_mode
        turns = ep.turns[self.window_start:]
        need = notes.NOTE_CAP + (notes.COMPACT_MIN_THINK if mode == "force" else 0)
        drop = 0
        while True:
            hp = notes.handoff_prompt(self._task_text(),
                                      trace_text(turns[drop:], self.window_start + drop),
                                      READER_SEES)
            msgs = [{"role": "user", "content": hp}]
            plen = self.plen(msgs)
            room = self.budget - plen - notes.COMPACT_NOTE_MARGIN
            if room >= need or drop >= len(turns):
                break
            drop += 1
        rec.update(mode=mode, note_prompt_tokens=plen, note_room=room,
                   trace_turns_dropped=drop, note_messages=msgs)
        if mode == "off":
            ch = A.chat_once(self.cfg, msgs, notes.NOTE_CAP, return_ids=True, thinking=False)
            if "error" in ch:
                ep.error = f"note: {ch['error']}"
                return None
            rec.update(finish=ch["finish_reason"], truncated=ch["finish_reason"] == "length",
                       segments=[{"kind": "prompt", "ids": list(ch["prompt_token_ids"]),
                                  "train": False},
                                 {"kind": "note", "ids": list(ch["token_ids"]), "train": True}],
                       thinking=False)
            return ch["content"]
        # force: thinking on, hard-capped so think + note fits the room
        think_cap = max(1, min(notes.NOTE_THINK_CAP, room - notes.NOTE_CAP))
        ch = A.chat_once(self.cfg, msgs, think_cap, return_ids=True, thinking=True)
        if "error" in ch:
            ep.error = f"note: {ch['error']}"
            return None
        segs = [{"kind": "prompt", "ids": list(ch["prompt_token_ids"]), "train": False},
                {"kind": "note", "ids": list(ch["token_ids"]), "train": True}]
        rec.update(thinking=True, think_cap=think_cap, forced=False)
        if ch["finish_reason"] != "length":
            rec.update(finish=ch["finish_reason"], truncated=False, segments=segs)
            return ch["content"]
        ids = list(ch["token_ids"])
        close = encoding._ids(self.tok("</think>", add_special_tokens=False))
        head = ""
        if _find(ids, close) < 0:
            forced = encoding._ids(self.tok("\n</think>\n\n", add_special_tokens=False))
            segs.append({"kind": "forced", "ids": forced, "train": False})
            ids += forced
            rec["forced"] = True
        else:
            k = _find(ids, close) + len(close)
            head = self.tok.decode(ids[k:])
        left = max(1, notes.NOTE_CAP - self.ntok(head))
        c = complete_ids(self.cfg, list(ch["prompt_token_ids"]) + ids, left)
        if "error" in c:
            ep.error = f"note: {c['error']}"
            return None
        segs.append({"kind": "note", "ids": list(c["ids"]), "train": True})
        rec.update(finish=c["finish"], truncated=c["finish"] == "length", segments=segs)
        return (head + c["text"]).replace("<|im_end|>", "").strip()


def _find(ids: list[int], sub: list[int]) -> int:
    for i in range(len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)] == sub:
            return i
    return -1


# --- training / fidelity ----------------------------------------------------------

def _windows(ep) -> list[tuple[int, int, list[dict]]]:
    """(first turn, end turn, starting messages) per context window."""
    starts = [(0, ep.initial_messages)] + [(w["start_turn"], w["messages"]) for w in ep.windows]
    out = []
    for i, (s, msgs) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else len(ep.turns)
        out.append((s, e, msgs))
    return out


def sequences(ep, tok, thinking: bool, max_turns: int | None) -> list[dict]:
    """Token-exact training sequences of a compact episode, in episode order:
    {"kind": "window"|"note", "segments": [...]} -- each segment {kind, ids, train}.
    A window = its first served prompt, then her ids and the observations between
    them (the window's last generation ends it). Needs return_ids=True."""
    out: list[dict] = []
    resets = [r for r in ep.resets if r.get("outcome") == "ok"]
    for w, (s, e, _) in enumerate(_windows(ep)):
        if s >= e:
            continue
        segs = [{"kind": "prompt", "ids": list(ep.prompt_ids_per_turn[s]), "train": False}]
        for i in range(s, e):
            t = ep.turns[i]
            segs.append({"kind": "assistant", "ids": list(ep.token_ids_per_turn[i]),
                         "train": True})
            if t.obs and i < e - 1:
                segs.append({"kind": "obs", "ids": encoding.encode_observation(
                    tok, encoding.observation_body(t.action.name, t.action.arguments, t.obs,
                                                   i + 1, max_turns), thinking),
                    "train": False})
        out.append({"kind": "window", "window": w, "segments": segs})
        if w < len(resets) and resets[w].get("segments"):
            out.append({"kind": "note", "window": w, "segments": resets[w]["segments"]})
    return out


def fidelity_error(ep, tok, thinking: bool, max_turns: int | None) -> str | None:
    """Every served prompt (work turns per window, harness note calls) must equal the
    local template render."""
    import agent as A
    for w, (s, e, msgs0) in enumerate(_windows(ep)):
        msgs = [dict(m) for m in msgs0]
        for i in range(s, e):
            served = ep.prompt_ids_per_turn[i] if i < len(ep.prompt_ids_per_turn) else []
            local = encoding.render_chat(tok, msgs, thinking)
            if served != local:
                return (f"window {w} turn {i}: served prompt ids != local template "
                        f"(len {len(served)} vs {len(local)})")
            t = ep.turns[i]
            msgs.append(A.assistant_message(t.content, t.reasoning))
            if t.obs:
                msgs.append({"role": "user", "content": A.observation_text(
                    t.action.name if t.action else "", t.action.arguments if t.action else {},
                    t.obs, i + 1, max_turns)})
    for r in ep.resets:
        if r.get("trigger") == "harness" and r.get("segments"):
            local = encoding.render_chat(tok, r["note_messages"], r.get("thinking", True))
            if r["segments"][0]["ids"] != local:
                return f"reset {r['n']}: note prompt ids != local template"
    return None


def summary(ep) -> dict:
    """Per-episode compaction stats for eval records (no note text)."""
    ok = [r for r in ep.resets if r.get("outcome") == "ok"]
    return {
        "resets": len(ok),
        "resets_harness": sum(r["trigger"] == "harness" for r in ok),
        "resets_self": sum(r["trigger"] == "self" for r in ok),
        "notes_truncated": sum(bool(r.get("truncated")) for r in ok),
        "notes_empty": sum(bool(r.get("empty")) for r in ok),
        "views_trimmed": sum(bool(r.get("view_trimmed")) for r in ok),
        "compact_fail": ep.compact_fail,
        "reset_log": [{k: r.get(k) for k in (
            "n", "turn", "trigger", "outcome", "tokens_before", "context_tokens",
            "context_tokens_untrimmed", "view_trimmed", "view_trimmed_tokens", "note_tokens",
            "note_tokens_used", "truncated", "empty", "missing_sections", "plan_set_by",
            "trace_turns_dropped", "note_room", "forced")} for r in ep.resets],
    }
