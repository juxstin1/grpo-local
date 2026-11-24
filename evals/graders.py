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
import sys
from pathlib import Path
from typing import Any
import math

try:
    from .code_executor import execute, image_reference, InvalidGrading
except ImportError:
    from code_executor import execute, image_reference, InvalidGrading

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
                return text[j + 1 : k] if not text[k + 1:].strip() else None
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
        raise InvalidGrading("gold-unparseable")
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
    final = strip_reasoning(out).strip().splitlines()
    match = re.fullmatch(r"ANSWER[ \t]*:[ \t]*(.+)", final[-1], re.I) if final else None
    if not match or not match.group(1).strip():
        return False, "no-final-answer-line"
    hits = [match.group(1).strip()]
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

def pick_code(text: str) -> str:
    fences = re.findall(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", text, re.S)
    return fences[-1] if fences else ""


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if type(a) is int and type(b) is int:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
        except OverflowError:
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def run_functional(code: str, entry: str, cases: list[dict], record: bool = False,
                   kind: str = "functional", source: str = "") -> dict:
    if kind != "functional":
        raise InvalidGrading("legacy assertion suites require an independent verifier; disabled")
    if not isinstance(cases, list) or not cases:
        raise InvalidGrading("empty functional suite")
    try:
        inputs = [[json.loads(line) for line in c["input"].split("\n") if line.strip()] for c in cases]
        expected = None if record else [json.loads(c["output"]) for c in cases]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InvalidGrading("invalid functional test data") from exc
    # Expected outputs, assertion source, and verdict logic stay in this process.
    result = execute(PRELUDE + "\n" + code, kind="functional", inputs=inputs, entry=entry,
                     timeout=min(10 + 3 * len(cases), 120))
    if result["status"] != "ok":
        return {"error": result["reason"]}
    outputs = result["outputs"]
    if record:
        return {"passed": 0, "total": len(cases), "outputs": outputs}
    passed = sum(_same(got, want) for got, want in zip(outputs, expected))
    return {"passed": passed, "total": len(cases), "outputs": None}


def run_stdin(code: str, cases: list[dict]) -> tuple[bool, str]:
    if not isinstance(cases, list) or not cases:
        raise InvalidGrading("empty stdin suite")
    for i, tc in enumerate(cases):
        if not isinstance(tc, dict) or not isinstance(tc.get("input"), str) or not isinstance(tc.get("output"), str):
            raise InvalidGrading("invalid stdin test data")
        result = execute(code, kind="stdin", stdin=tc["input"], timeout=20)
        if result["status"] != "ok":
            return False, result["reason"]
        got = "\n".join(l.rstrip() for l in result["stdout"].strip().splitlines())
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


def preflight(records):
    """Refuse unsupported/unconfigured code tasks before spending model calls."""
    has_code = False
    for record in records:
        if record.get('dataset') != 'code':
            continue
        has_code = True
        tests = record.get('tests', {})
        if not isinstance(tests, dict) or tests.get('kind') not in ('functional', 'stdin'):
            raise InvalidGrading('unsupported code verifier; assertion suites remain disabled')
        cases = tests.get('cases')
        if not isinstance(cases, list) or not cases:
            raise InvalidGrading('empty code test suite')
        for case in cases:
            if not isinstance(case, dict) or not all(isinstance(case.get(k), str) for k in ('input', 'output')):
                raise InvalidGrading('invalid code test data')
            if tests['kind'] == 'functional':
                try:
                    [json.loads(line) for line in case['input'].split('\n') if line.strip()]
                    json.loads(case['output'])
                except (TypeError, ValueError) as exc:
                    raise InvalidGrading('invalid functional JSON test data') from exc
        if tests['kind'] == 'functional' and not re.fullmatch(r'[A-Za-z_]\w*\.[A-Za-z_]\w*', str(tests.get('entry', ''))):
            raise InvalidGrading('invalid code entry')
    if has_code:
        image_reference()


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
        rows = [json.loads(l) for l in f if l.strip()]
    ids = [r.get("uid", r.get("id")) for r in rows]
    if any(not isinstance(uid, str) or not uid for uid in ids) or len(ids) != len(set(ids)):
        raise InvalidGrading("pool requires nonempty unique item IDs")
    return rows
