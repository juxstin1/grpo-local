"""Historical Bionic task diagnostics, not qualified GRPO rewards.

make_task requires diagnostic=True. Substring/keyword checks cannot establish
semantic correctness; use the validated outcome contracts for new training work.

A task file (drafts from ops/fumble.ps1 land in data/bionic-real/drafts/; a draft becomes a task once it has
a "check"):

  {"name": "real-two-summaries", "prompt": "<what the user typed>",
   "files": {"docs/a.md": "...", ...},                  # start state of the folder (text files, relative paths)
   "check": {
     "ask": false,               # true: the right move is a question; final reply must ask, and NO file may change
     "answer": "docs/x.md",      # optional: last `ANSWER: <path>` must equal this
     "expect": {"docs/x.md": {"equals": "...", "contains": ["..."], "not_contains": ["..."], "exists": true}},
     "unchanged": "others",      # "others" (default): every file not in expect stays byte-identical; or a list
     "reply_contains": [], "reply_not_contains": [],      # case-insensitive, on the final reply
     "reply_contains_any": [],   # at least one of these (e.g. ways of saying "I can't check")
     "new_dir": "(?i)plugin",    # optional: some new folder whose relative path matches this regex must exist
     "allow_new": [], "allow_new_re": [],                 # extra files that may appear (else unexpected-file)
     "run": [{"cmd": ["python", "-m", "pytest", "-q"],    # spec 31: execution checks, run last in the folder;
              "stdout_contains": ["3 passed"]}]           # exit 0 required; "python" = this interpreter
   },
   "source": {"date": "...", "what_happened": "..."}}

The episode is the same Bionic native loop as the gym families (ops/grpo_q4.episode); only the start files, the
prompt and the grader come from here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from reward_contract import InvalidReward, UnqualifiedTask, final_answer, safe_rel

ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = ROOT / "data" / "bionic-real" / "tasks"
FAMILY = "bionic_real"
QUESTION_RE = re.compile(r"\?|\b(?:which|could you|can you|would you|please)\s+(?:one|file|tell|clarify|specify|"
                         r"confirm|choose|let me know)", re.I)


def load(task_dir: Path = TASK_DIR) -> list[dict]:
    out = []
    for p in sorted(task_dir.glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        if "check" not in t:
            raise SystemExit(f"{p}: no check (still a draft?)")
        out.append({"family": FAMILY, "seed": 0, "name": t["name"], "prompt": t["prompt"], "files": t["files"],
                    "check": t["check"], "tools": [], "ground_truth": json.dumps(t["check"], sort_keys=True)})
    return out


def write_files(files: dict[str, str], root: Path) -> None:
    for rel, text in files.items():
        safe_rel(rel)
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")


def _final_reply(turns) -> str | None:
    from graders import strip_reasoning
    if not turns or turns[-1].action is not None or turns[-1].truncated:
        return None
    return strip_reasoning(turns[-1].resp_text or turns[-1].content).strip()


def grade(root: Path, spec: dict, turns) -> tuple[bool, str]:
    root, chk, files = Path(root), spec["check"], spec["files"]
    reply = _final_reply(turns)
    if reply is None:
        return False, "no-final-reply"
    now, links = {}, []
    for p in root.rglob("*"):
        try:                                             # a link the model made (seen: `mcp -> /work`) can't be
            if p.is_symlink():                           # stat'ed on Windows; no task starts with one, so any
                links.append(p.relative_to(root).as_posix())  # link is an unasked-for change
            elif p.is_file():
                now[p.relative_to(root).as_posix()] = p
        except OSError:
            links.append(p.relative_to(root).as_posix())
    if links:
        return False, f"unexpected-link:{links[0]}"[:80]
    expect = chk.get("expect", {})
    if chk.get("ask"):
        if not QUESTION_RE.search(reply):
            return False, "missing-question"
        expect = {}
    keep = [p for p in files if p not in expect] if chk.get("unchanged", "others") == "others" else chk["unchanged"]
    for rel in keep:
        if rel not in now:
            return False, f"deleted:{rel}"[:80]
        if now[rel].read_text(encoding="utf-8", errors="replace") != files[rel]:
            return False, f"changed-unrelated-file:{rel}"[:80]
    if chk.get("ask") or chk.get("unchanged", "others") == "others":
        extra = sorted(set(now) - set(files) - set(expect) - set(chk.get("allow_new", [])))
        extra = [e for e in extra if not any(re.fullmatch(p, e) for p in chk.get("allow_new_re", []))]
        if extra:
            return False, f"unexpected-file:{extra[0]}"[:80]
    if "new_file" in chk:                                # some new file whose path matches re and text has contains[]
        nf = chk["new_file"]
        hits = [p for rel, p in now.items() if rel not in files and re.search(nf["re"], rel)
                and all(s.casefold() in p.read_text(encoding="utf-8", errors="replace").casefold()
                        for s in nf.get("contains", []))]
        if not hits:
            return False, "no-matching-new-file"
    if "new_dir" in chk:                                 # a folder the request asked to create (may be empty)
        old_dirs = {Path(f).parent.as_posix() for f in files} | {"."}
        new = [d.relative_to(root).as_posix() for d in root.rglob("*") if d.is_dir()]
        new = [d for d in new if d not in old_dirs and re.search(chk["new_dir"], d)]
        if not new:
            return False, "no-new-dir"
    if chk.get("answer"):
        ans = final_answer(turns)
        if ans is None or ans.strip().strip("` ").replace("\\", "/") != chk["answer"]:
            return False, "wrong-answer-path"
    for rel, rule in expect.items():
        exists = rel in now
        if rule.get("exists", True) != exists:
            return False, f"{'missing' if not exists else 'should-not-exist'}:{rel}"[:80]
        if not exists:
            continue
        text = now[rel].read_text(encoding="utf-8", errors="replace")
        if "equals" in rule and text.replace("\r\n", "\n") != rule["equals"].replace("\r\n", "\n"):
            return False, f"not-equal:{rel}"[:80]
        for s in rule.get("contains", []):
            if s not in text:
                return False, f"missing:{s}"[:80]
        for s in rule.get("not_contains", []):
            if s in text:
                return False, f"forbidden:{s}"[:80]
        any_of = rule.get("contains_any", [])
        if any_of and not any(s in text for s in any_of):
            return False, f"missing-any:{any_of[0]}"[:80]
    fold = lambda s: s.casefold().replace("’", "'")  # noqa: E731  models often write a curly apostrophe
    low = fold(reply)
    for s in chk.get("reply_contains", []):
        if fold(s) not in low:
            return False, f"reply-missing:{s}"[:80]
    for s in chk.get("reply_not_contains", []):
        if fold(s) in low:
            return False, f"reply-forbidden:{s}"[:80]
    any_of = chk.get("reply_contains_any", [])
    if any_of and not any(fold(s) in low for s in any_of):
        return False, f"reply-missing-any:{any_of[0]}"[:80]
    for r in chk.get("run", []):
        ok, why = run_check(root, r)
        if not ok:
            return False, why
    return True, "ok"


RUN_TIMEOUT_S = 60


def run_check(root: Path, r: dict) -> tuple[bool, str]:
    """One execution check: exit 0 and stdout+stderr contain every `stdout_contains` string.

    `hidden`: {relpath: text} grader files (e.g. tests the model never saw) written just for this run and
    removed after, so editing the visible tests can't make a task pass."""
    import os
    import shutil
    import subprocess
    import sys
    cmd = [sys.executable if c == "python" else c for c in r["cmd"]]
    if os.name == "nt" and cmd[0] != sys.executable:          # npm tools are .cmd shims on Windows
        cmd[0] = shutil.which(cmd[0] + ".cmd") or cmd[0]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    name = " ".join(r["cmd"])[:40]
    hidden = [(Path(root) / rel, text) for rel, text in r.get("hidden", {}).items()]
    for p, text in hidden:
        if p.exists():
            return False, f"hidden-file-collision:{p.name}"[:80]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
    try:
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=r.get("timeout", RUN_TIMEOUT_S), env=env)
    except subprocess.TimeoutExpired:
        return False, f"run-timeout:{name}"[:80]
    except OSError as exc:
        raise InvalidReward('Verifier executable unavailable; attempt is invalid, not reward zero.') from exc
    finally:
        for hp, _ in hidden:
            hp.unlink(missing_ok=True)
    if p.returncode != 0:
        return False, f"run-failed:{name}"[:80]
    out = p.stdout + p.stderr
    for s in r.get("stdout_contains", []):
        if s not in out:
            return False, f"run-missing:{s}"[:80]
    return True, "ok"


def make_task(spec: dict, native: bool = True, *, diagnostic: bool = False):
    """Historical keyword checks are available only with explicit diagnostic intent."""
    if not diagnostic:
        raise UnqualifiedTask('Legacy real-task graders are unqualified; use outcome contracts or explicit diagnostic=True.')
    from eval import make_task as gym_make_task
    item = {k: spec[k] for k in ("name", "prompt", "tools", "ground_truth", "seed")}
    task = gym_make_task({**item, "family": "file_work"}, native=native)   # any known family; reward replaced below
    task.family = FAMILY
    task.reward = lambda turns, ws: float(ws is not None and grade(ws.root, spec, turns)[0])
    return task
