"""Standalone endpoint graders for the math / code / mhqa buckets.

One module, two callers: eval_runner.py (eval + sealed holdout) and the GRPO
reward (make_task). Same prompt, same grader, so train reward == eval metric.

Rules (see evals/datasets/RULES.md):
  - Endpoint only. Math: last \\boxed{} vs gold via math-verify. Code: every
    test passes. MHQA: normalized exact match on the final ANSWER line vs gold
    or an alias. Reasoning is stripped and never inspected.
  - Binary: 1.0 / 0.0. No partial credit, no LLM judge, no format bonus.
  - A truncated final turn earns 0.

Record shape (one JSONL line in a pool / set / sealed holdout):
  {"uid", "dataset": math|code|mhqa, "root_id", "first_published", "prompt",
   "answer" (math: latex str; mhqa: {"answer", "aliases"}),
   "tests"  (code: {"kind": functional|stdin|assert, "entry", "cases"|"source"})}
"""

from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SYSTEM = {
    "math": (
        "Solve the problem. Put only the final answer, in simplest exact form, "
        "inside \\boxed{} at the end of your response."
    ),
    "code": (
        "You are an expert Python programmer. Complete the given starter code so it "
        "solves the problem. Output the complete solution in a single ```python fenced block."
    ),
    "mhqa": (
        "Answer the question using the paragraphs. The answer is a short span "
        "(an entity, date, or number). End with a final line of the form: ANSWER: <answer>, "
        "in the exact format the question asks for."
    ),
}
MAX_TOKENS = {"math": 8192, "code": 8192, "mhqa": 2048}

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Grade the response, not the path: drop inline <think> blocks (and an
    unclosed leading one, which means the answer never came)."""
    text = THINK_RE.sub("", text or "")
    if "<think>" in text:
        return ""
    return text


# --- math ------------------------------------------------------------------

def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def grade_math(record: dict, out: str) -> tuple[bool, str]:
    from math_verify import parse, verify  # lazy: heavy import

    pred = last_boxed(strip_reasoning(out))
    if pred is None:
        return False, "no-boxed"
    if len(pred) > 300:  # timeouts are disabled below; cap pathological input instead
        return False, "boxed-too-long"
    # timeout=None: math-verify's own timeout forks a process per call on Windows.
    gold = parse(f"${record['answer']}$", parsing_timeout=None)
    got = parse(f"$\\boxed{{{pred}}}$", parsing_timeout=None)
    if not gold:
        return False, "gold-unparseable"
    ok = bool(verify(gold, got, timeout_seconds=None))
    return ok, f"pred={pred[:60]}"


# --- mhqa ------------------------------------------------------------------

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.I | re.M)


def squad_norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


CHAIN_SEP = re.compile(r"\s*(?:->|→|=>)\s*")


def grade_mhqa(record: dict, out: str) -> tuple[bool, str]:
    hits = ANSWER_RE.findall(strip_reasoning(out))
    if not hits:
        return False, "no-answer-line"
    chain = record["answer"].get("chain")
    if chain:
        # Full-chain grading: every hop must be named, in order. The leading
        # subject (the person in the question) may be included or omitted.
        # Blocks "right endpoint via the wrong intermediate" and blind guessing.
        pred = [squad_norm(p) for p in CHAIN_SEP.split(hits[-1]) if p.strip()]
        gold = [squad_norm(c) for c in chain]
        if len(pred) == len(gold) - 1:
            gold = gold[1:]
        if pred != gold:
            wrong = next((i for i, (p, g) in enumerate(zip(pred, gold)) if p != g), min(len(pred), len(gold)))
            return False, f"chain-mismatch@hop{wrong} len={len(pred)}/{len(gold)}"
        return True, f"chain={'>'.join(pred)[:60]}"
    pred = squad_norm(hits[-1])
    gold = [record["answer"]["answer"], *record["answer"].get("aliases", [])]
    ok = bool(pred) and pred in {squad_norm(g) for g in gold}
    return ok, f"pred={pred[:60]}"


# --- code ------------------------------------------------------------------

PRELUDE = """\
import sys, os, re, math, json, heapq, bisect, string, random, itertools, functools, collections, operator
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
from math import *
sys.setrecursionlimit(1 << 20)
"""

# Runs inside the sandbox subprocess. Reads a JSON job from argv[1]; prints one
# JSON result line. Candidate code is exec'd; nothing is imported from here.
DRIVER = r'''
import json, sys, math
job = json.load(open(sys.argv[1], encoding="utf-8"))
def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try: return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-6)
        except Exception: return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b
g = {"__name__": "__sol__"}
exec(job["prelude"] + "\n" + job["code"], g)
cls_name, meth = job["entry"].split(".")  # "Solution.methodName"
fn = getattr(g[cls_name](), meth)
if job["kind"] == "assert":
    exec(job["source"], g)
    g["check"](fn)
    print(json.dumps({"passed": 1, "total": 1, "outputs": None}))
    sys.exit(0)
outs, passed = [], 0
for i, case in enumerate(job["cases"]):
    args = [json.loads(line) for line in case["input"].split("\n") if line.strip()]
    got = json.loads(json.dumps(fn(*args)))
    outs.append(got)
    if job["record"]:
        continue
    if not same(got, json.loads(case["output"])):
        print(json.dumps({"passed": passed, "total": len(job["cases"]), "fail_at": i}))
        sys.exit(0)
    passed += 1
print(json.dumps({"passed": passed, "total": len(job["cases"]), "outputs": outs if job["record"] else None}))
'''


def pick_code(text: str) -> str:
    fences = re.findall(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", text, re.S)
    if fences:
        return fences[-1]  # final answer = last fenced block, not the longest draft
    return ""


def _run(job: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="grade_") as td:
        jp = Path(td) / "job.json"
        dp = Path(td) / "driver.py"
        jp.write_text(json.dumps(job), encoding="utf-8")
        dp.write_text(DRIVER, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")}
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(dp), str(jp)], capture_output=True, text=True,
                timeout=timeout, cwd=td, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode != 0 or not lines:
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            return {"error": f"runtime:{err[:80]}"}
        return json.loads(lines[-1])


def run_functional(code: str, entry: str, cases: list[dict], record: bool = False,
                   kind: str = "functional", source: str = "") -> dict:
    job = {"prelude": PRELUDE, "code": code, "entry": entry, "cases": cases, "record": record,
           "kind": kind, "source": source}
    n = max(1, len(cases))
    return _run(job, timeout=min(10 + 3 * n, 120))


def run_stdin(code: str, cases: list[dict]) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="grade_") as td:
        sol = Path(td) / "solution.py"
        sol.write_text(code, encoding="utf-8")
        for i, tc in enumerate(cases):
            try:
                r = subprocess.run([sys.executable, "-I", str(sol)], input=tc["input"],
                                   capture_output=True, text=True, timeout=8, cwd=td)
            except subprocess.TimeoutExpired:
                return False, f"timeout@{i}"
            got = "\n".join(l.rstrip() for l in r.stdout.strip().splitlines())
            want = "\n".join(l.rstrip() for l in tc["output"].strip().splitlines())
            if got != want:
                return False, f"mismatch@{i}"
        return True, f"pass:{len(cases)}/{len(cases)}"


def grade_code(record: dict, out: str) -> tuple[bool, str]:
    code = pick_code(strip_reasoning(out))
    if not code.strip():
        return False, "no-code"
    try:
        compile(code, "<sol>", "exec")
    except SyntaxError as e:
        return False, f"syn:{(e.msg or '')[:40]}"
    t = record["tests"]
    if t["kind"] == "stdin":
        return run_stdin(code, t["cases"])
    res = run_functional(code, t["entry"], t.get("cases", []), kind=t["kind"], source=t.get("source", ""))
    if "error" in res:
        return False, res["error"]
    ok = res["passed"] == res["total"] and res["total"] > 0
    return ok, f"pass:{res['passed']}/{res['total']}"


GRADERS = {"math": grade_math, "code": grade_code, "mhqa": grade_mhqa}


def grade(record: dict, out: str) -> tuple[bool, str]:
    return GRADERS[record["dataset"]](record, out)


def messages(record: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM[record["dataset"]]},
            {"role": "user", "content": record["prompt"]}]


# --- GRPO side -------------------------------------------------------------

def make_task(record: dict):
    """Pool record -> schema.Task whose reward is the same endpoint grader.
    Single-turn: rollout treats a turn with no tool action as terminal."""
    from schema import Task  # grpo-local root must be on sys.path

    def reward(turns, done) -> float:
        if not turns or turns[-1].truncated:
            return 0.0
        text = turns[-1].resp_text or turns[-1].gen_text
        return 1.0 if grade(record, text)[0] else 0.0

    return Task(name=record["uid"], prompt=record["prompt"], reward_fn=reward,
                meta={"system": SYSTEM[record["dataset"]], "root_id": record["root_id"],
                      "dataset": record["dataset"]})


def load_pool(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
