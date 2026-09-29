"""Mechanical graders + GRPO reward adapter. No LLM judges, binary 1/0.

One module, two callers: eval.py (held-out eval) and rollout.py (GRPO reward).
Same check, so train reward == eval metric:

  find  normalized exact match on the final ANSWER line vs the generated ground truth
  edit  end-state content checks only (ring 2 has no run_tests; graded on the
        file after her edits)
  fix   the full test suite passes -- and, because the bug is only injected when
        it actually breaks a test (repo_gen guard), "tests pass" can't be gamed
        by changing nothing

A truncated final turn always earns 0.
"""

from __future__ import annotations

import re
import shutil
import uuid
from typing import Callable

from schema import Task, Turn
from workspace import Workspace

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Grade the response, not the path. Qwen3.5 thinking ends with </think>:
    keep only what follows the last one. An opened <think> with no close means
    the answer never came. vLLM with --reasoning-parser already returns the
    answer alone (no tags), which passes through unchanged."""
    text = text or ""
    if "</think>" in text:
        return text[text.rfind("</think>") + len("</think>"):]
    if "<think>" in text:
        return ""
    return text


def last_answer(turns: list[Turn]) -> str | None:
    for t in reversed(turns):
        m = ANSWER_RE.search(strip_reasoning(t.resp_text or t.content))
        if m:
            return m.group(1).strip().strip(".")
    return None


def truncated_final(turns: list[Turn]) -> bool:
    return bool(turns) and turns[-1].truncated


def normalize(s: str) -> str:
    return s.strip().lower().rstrip(".").replace(", ", ",")


# --- family graders ----------------------------------------------------------

def grade_find(turns: list[Turn], workspace: Workspace, gt: str) -> tuple[bool, str]:
    if truncated_final(turns):
        return False, "truncated"
    ans = last_answer(turns)
    if ans is None:
        return False, "no-answer-line"
    ok = normalize(ans) == normalize(gt)
    return ok, f"pred={ans[:60]}"


def grade_edit(turns: list[Turn], workspace: Workspace,
               checks: list[dict]) -> tuple[bool, str]:
    if truncated_final(turns):
        return False, "truncated"
    for c in checks:
        text = (workspace.root / c["file"]).read_text(encoding="utf-8")
        if c["op"] == "contains" and c["needle"] not in text:
            return False, f"missing {c['needle']!r} in {c['file']}"
        if c["op"] == "absent" and c["needle"] in text:
            return False, f"still present {c['needle']!r} in {c['file']}"
    return True, "end-state ok"


def grade_fix(turns: list[Turn], workspace: Workspace, pristine_tests: dict[str, str]) -> tuple[bool, str]:
    """Grade her code against tests she never touched.

    The tests in her workspace are visible and editable, so they are NOT the
    grade: copy the workspace, replace tests/ with the generator's pristine
    files, and require a per-grade random nonce in the runner's success line.
    Editing tests, deleting asserts, or exiting 0 early from core.py all fail.
    """
    if truncated_final(turns):
        return False, "truncated"
    nonce = uuid.uuid4().hex
    grade_root = workspace.root.parent / f"{workspace.root.name}__grade_{nonce[:8]}"
    try:
        shutil.copytree(workspace.root, grade_root)
        shutil.rmtree(grade_root / "tests", ignore_errors=True)
        for rel, text in pristine_tests.items():
            p = grade_root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text.replace("ALL TESTS PASSED", f"ALL TESTS PASSED {nonce}"),
                         encoding="utf-8", newline="\n")
        out = workspace.runner.run_tests(grade_root)
    finally:
        shutil.rmtree(grade_root, ignore_errors=True)
    ok = "PASS" in out.split("\n")[0] and nonce in out
    return ok, out.split("\n")[0][:80]


def pristine_tests(seed: int) -> dict[str, str]:
    """The generator's own test files for a seed (never the agent's copies)."""
    from repo_gen import generate_project
    return {k: v for k, v in generate_project(seed).files.items() if k.startswith("tests/")}


# --- reward adapter ----------------------------------------------------------

def reward_for(task: Task) -> Callable[[list[Turn], Workspace], float]:
    """Task.reward: 1.0 / 0.0. find uses the transcript; edit/fix use the end state."""
    if task.family == "find":
        def rw(turns, ws):
            return float(grade_find(turns, ws, task.ground_truth)[0])
        return rw
    if task.family == "edit":
        checks = task.meta.get("checks", [])
        def rw(turns, ws):
            return float(grade_edit(turns, ws, checks)[0])
        return rw
    if task.family == "fix":
        tests = pristine_tests(task.meta["seed"])
        def rw(turns, ws):
            return float(grade_fix(turns, ws, tests)[0])
        return rw
    raise ValueError(f"unknown family {task.family}")