"""The ONE handoff-note call: prompt, skeleton, cap, thinking mode.

Every place that asks her for a note imports this -- the handoff gym
(handoff-gym/) and the workspace compaction harness (sandbox-gym). Train and
use must be the same call: same wording, same sections, same caps, thinking ON.
Change it here or nowhere.

Thinking on (2026-09-29): she sizes up the state ("link 4 of 8, what does the
next me need?") before compressing, and the K notes of a GRPO group vary more.
Budget: up to NOTE_THINK_CAP of thinking + NOTE_CAP of note in one call
(max_tokens = NOTE_CALL_MAX). Over-cap = the call hit max_tokens, or the note
section alone exceeds NOTE_CAP -- a fail, never truncated.
"""

from __future__ import annotations

NOTE_CAP = 400          # note tokens (after </think>); over = fail, never truncated
NOTE_THINK_CAP = 600    # thinking allowance before the note
NOTE_CALL_MAX = NOTE_THINK_CAP + NOTE_CAP   # max_tokens for the note call; also the
                                            # room a frozen state must leave in the budget
NOTE_THINKING = True    # the note is written with thinking ON
NOTE_TEMPERATURE = 0.8  # writer sampling for K-note groups
SECTIONS = ("Goal", "Done", "Current state", "Hypothesis", "Next step", "Don't repeat")

# In-episode compaction (sandbox-gym compact.py, switch: config Episode.compact).
COMPACT_TRIGGER = 0.75      # harness asks for a note once the rendered prompt reaches this
                            # share of the budget (8192 -> 6144; always computed: compact_trigger_tokens)
COMPACT_CONTEXT_MAX = 0.40  # a rebuilt context (system + task + plan + note + workspace
                            # view) must fit in this share (8192 -> 3277; compact_context_max); else trim the view
COMPACT_NOTE_MARGIN = 32    # tokens kept free for template differences (thinking-off header)
COMPACT_MIN_THINK = 64      # force mode: least thinking room before the trace gets trimmed


def compact_trigger_tokens(budget: int) -> int:
    return int(budget * COMPACT_TRIGGER)


def compact_context_max(budget: int) -> int:
    return round(budget * COMPACT_CONTEXT_MAX)


def note_sections(note: str) -> dict[str, str]:
    """Section name -> text of a skeleton note (case-insensitive 'Name:' line starts)."""
    out: dict[str, str] = {}
    cur = None
    for line in (note or "").splitlines():
        head = line.strip().lstrip("#*- ").strip()
        hit = next((s for s in SECTIONS if head.lower().startswith(s.lower() + ":")), None)
        if hit:
            cur = hit
            out[cur] = head[len(hit) + 1:].strip()
        elif cur:
            out[cur] = (out[cur] + "\n" + line.strip()).strip()
    return out


def missing_sections(note: str) -> list[str]:
    have = note_sections(note)
    return [s for s in SECTIONS if s not in have]


def handoff_prompt(task_text: str, trace: str, reader_sees: str) -> str:
    """Ask for a note mid-task. `reader_sees` names what the fresh instance will
    have besides the note, e.g. "the question, the facts" or "the task, the
    workspace files"."""
    skel = "\n".join(f"{s}:" for s in SECTIONS)
    return (
        f"{task_text}\n\n---\n"
        "You were working on this and got interrupted. Your work so far:\n"
        f"<work>\n{trace}\n</work>\n\n"
        "You're mid-task. Write a HANDOFF for a fresh instance that will only see "
        f"{reader_sees} and your note -- not your work above. Use exactly these "
        f"sections:\n{skel}\n"
        f"Hard limit {NOTE_CAP} tokens. Output only the handoff."
    )


def resume_prompt(task_text: str, note: str, check_against: str) -> str:
    """Fresh instance: task + note only. `check_against` is what it verifies the
    note's claims with, e.g. "the facts" or "the files"."""
    return (
        f"{task_text}\n\n---\n"
        "A previous instance worked on this and left you a handoff note:\n"
        f"<handoff>\n{note.strip() or '(empty)'}\n</handoff>\n"
        f"Continue from the note. Check what it claims against {check_against} "
        "instead of restarting from scratch."
    )


def over_cap(finish_reason: str, note_tokens: int) -> bool:
    """The one over-cap rule: ran out of the call budget, or the note is too long."""
    return finish_reason == "length" or note_tokens > NOTE_CAP
