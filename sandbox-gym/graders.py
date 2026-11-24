"""Mechanical graders + GRPO reward adapter. No LLM judges, binary 1/0.

One module, two callers: eval.py (held-out eval) and rollout.py (GRPO reward).
Same check, so train reward == eval metric:

  find  normalized exact match on the final ANSWER line vs the generated ground truth
  edit/fix  require a final ANSWER line (a task isn't done until she says so);
        a cap-hit with correct files scores 0 and is reported as a diagnostic
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
    s = s.strip().strip("`").strip().replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s.lower().rstrip(".").replace(", ", ",")


# --- family graders ----------------------------------------------------------

def grade_find(turns: list[Turn], workspace: Workspace, gt: str) -> tuple[bool, str]:
    if truncated_final(turns):
        return False, "truncated"
    ans = last_answer(turns)
    if ans is None:
        return False, "no-answer-line"
    ok = normalize(ans) == normalize(gt) or _same_number(ans, gt)
    return ok, f"pred={ans[:60]}"


def _same_number(a: str, b: str) -> bool:
    """42.5 == 42.50, 40.0 == 40: numbers compare by value, not spelling."""
    try:
        return abs(float(a.strip().rstrip(".")) - float(b.strip())) < 1e-9
    except ValueError:
        return False


def grade_edit(turns: list[Turn], workspace: Workspace,
               checks: list[dict], require_answer: bool = True) -> tuple[bool, str]:
    """Done means she said so: no ANSWER line = 0 even if the files are right
    (require_answer=False is the eval's files-ok-but-never-answered diagnostic)."""
    if truncated_final(turns):
        return False, "truncated"
    if require_answer and last_answer(turns) is None:
        return False, "no-answer-line"
    for c in checks:
        if c["op"] == "effective":
            from repo_gen_hard import effective_settings
            eff = effective_settings(workspace.root, c["pkg"], c["settings"])
            if eff is None:
                return False, "settings unparseable"
            for k, v in {**c["want"], **c["same"]}.items():
                if k not in eff or not _same_number(str(eff[k]), str(v)):
                    return False, f"{k}={eff.get(k)!r} want {v!r}"
            continue
        try:
            text = (workspace.root / c["file"]).read_text(encoding="utf-8")
        except OSError:
            return False, f"missing file {c['file']}"
        if c["op"] == "unchanged" and text != c["text"]:
            return False, f"modified {c['file']}"
        if c["op"] == "contains" and c["needle"] not in text:
            return False, f"missing {c['needle']!r} in {c['file']}"
        if c["op"] == "absent" and c["needle"] in text:
            return False, f"still present {c['needle']!r} in {c['file']}"
    return True, "end-state ok"


def grade_fix(turns: list[Turn], workspace: Workspace, pristine_tests: dict[str, str],
              require_answer: bool = True) -> tuple[bool, str]:
    """Grade her code against tests she never touched.

    The tests in her workspace are visible and editable, so they are NOT the
    grade: copy the workspace, replace tests/ with the generator's pristine
    files, and require a per-grade random nonce in the runner's success line.
    Editing tests, deleting asserts, or exiting 0 early from core.py all fail.
    No ANSWER line = 0 (same "done" as collect.py), unless require_answer=False.
    """
    if truncated_final(turns):
        return False, "truncated"
    if require_answer and last_answer(turns) is None:
        return False, "no-answer-line"
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


def pristine_tests(seed: int, family: str = "fix") -> dict[str, str]:
    """The generator's own test files for a seed (never the agent's copies)."""
    if family.endswith("_h"):
        import repo_gen_hard
        return repo_gen_hard.pristine_tests(seed)
    from repo_gen import generate_project
    return {k: v for k, v in generate_project(seed).files.items() if k.startswith("tests/")}


# --- reward adapter ----------------------------------------------------------

def reward_for(task: Task) -> Callable[[list[Turn], Workspace], float]:
    """Task.reward: 1.0 / 0.0. find uses the transcript; edit/fix use the end state."""
    fam = task.family.removesuffix("_h").removesuffix("_m")
    if fam == "mhqa_t":                    # full-chain grade_mhqa on the ANSWER line
        from mhqa_tool import grade
        def rw(turns, ws):
            return 0.0 if truncated_final(turns) else float(grade(last_answer(turns), task.ground_truth)[0])
        return rw
    if fam == "repo_q":                    # questions over the real plugin repo (repo_q.py)
        from repo_q import grade as grade_rq
        def rw(turns, ws):
            return 0.0 if truncated_final(turns) else float(grade_rq(last_answer(turns), task.ground_truth)[0])
        return rw
    if fam == "doc_make":                  # md/PDF report from notes + csv (doc_make.py)
        from doc_make import reward_fn
        return reward_fn(task)
    if fam == "gather_produce":            # lookups -> FINDINGS.md (gather_produce.py)
        from gather_produce import reward as reward_gp
        return reward_gp(task)
    if fam == "doc_digest":                # SUMMARY.md: all keys, no decoy, <= cap (doc_digest.py)
        from doc_digest import grade_workspace
        def rw(turns, ws):
            if truncated_final(turns) or last_answer(turns) is None:
                return 0.0
            return float(grade_workspace(ws.root, task.ground_truth, log_as=ws.root.name)[0])
        return rw
    if fam == "find":
        def rw(turns, ws):
            return float(grade_find(turns, ws, task.ground_truth)[0])
        return rw
    if fam in ("edit", "editv"):
        checks = task.meta.get("checks", [])
        def rw(turns, ws):
            return float(grade_edit(turns, ws, checks)[0])
        return rw
    if fam == "fix":
        tests = pristine_tests(task.meta["seed"], task.family)
        def rw(turns, ws):
            return float(grade_fix(turns, ws, tests)[0])
        return rw
    raise ValueError(f"unknown family {task.family}")


def files_ok_ignoring_answer(task: Task, turns: list[Turn], ws: Workspace) -> bool:
    """Diagnostic only, never a reward: would the end state pass without the
    ANSWER-line requirement? (edit skill present, finishing skill missing)"""
    fam = task.family.removesuffix("_h").removesuffix("_m")
    if fam in ("edit", "editv"):
        return grade_edit(turns, ws, task.meta.get("checks", []), require_answer=False)[0]
    if fam == "fix":
        return grade_fix(turns, ws, pristine_tests(task.meta["seed"], task.family),
                         require_answer=False)[0]
    if fam == "doc_digest":
        from doc_digest import grade_workspace
        return grade_workspace(ws.root, task.ground_truth)[0]
    return False
