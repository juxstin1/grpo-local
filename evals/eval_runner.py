"""Bake-off eval runner. Reads the frozen set, grades mechanically, writes raw.

Usage (Windows PowerShell):
  .venv\\Scripts\\python.exe evals\\eval_runner.py --model Qwen/Qwen3.5-2B ^
      --slug qwen3.5-2b --sampling '{"temperature":0.7,"top_p":0.8}'

Rules enforced here:
  - The set file is read-only input; the runner never edits it.
  - pass@4 per item (4 attempts, any pass counts).
  - Thinking stays ON (no enable_thinking passed; the model's template decides).
  - Holdout bucket is evaluated aggregate-only: no transcripts ever written.
  - Raw per-attempt lines are written per bucket; resume = skip done attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests

ROOT = Path(__file__).resolve().parent
SET_PATH = ROOT / "data" / "sets" / "baseline_v1.json"
HOLDOUT = ROOT / "holdout" / "holdout.py"

sys.path.insert(0, str(ROOT.parent))  # schema.py
sys.path.insert(0, str(HOLDOUT.parent))  # holdout.py
import holdout as HO  # noqa: E402  (sealed; imported by the runner only)
import graders  # noqa: E402  (math / code / mhqa endpoint graders, shared with GRPO reward)

SEAL_PY = HOLDOUT.parent / "seal.py"

MAX_WORKERS = 64  # cap on concurrent attempts; match vLLM --max-num-seqs


class _T:
    """Minimal Turn stand-in; reward fns only touch gen_text."""

    def __init__(self, text: str):
        self.gen_text = text


def norm(s: Any) -> str:
    return str(s).strip().lower().rstrip(".")


def parse_tool_call(text: str) -> Optional[dict]:
    """Tolerant extraction: qwen <tool_call> blocks, bare JSON objects, or
    {"name":..., "arguments":...} anyway they appear."""
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    i = text.find("{")
    while i != -1:
        depth, j = 0, i
        try:
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 0:
                obj = json.loads(text[i : j + 1])
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass
        i = text.find("{", i + 1)
    return None


def pick_code(text: str) -> str:
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if fences:
        return max(fences, key=len)
    body = re.sub(r"^.*?(?=def |import |from |class |print\(|#)", "", text, flags=re.S)
    return body.strip() or text.strip()


# --- graders ---------------------------------------------------------------

def grade_lcb(item: dict, out: str) -> tuple[bool, str]:
    code = pick_code(out)
    if not code:
        return False, "no-code"
    if item["tests"] and item["tests"][0].get("testtype") == "functional":
        # LeetCode-style LCB items call Solution().<method>; running them as a stdin
        # script (the old path) failed every attempt regardless of correctness.
        m = re.search(r"def (\w+)\(self", item.get("starter_code") or "")
        if not m:
            return False, "no-entry"
        res = graders.run_functional(code, f"Solution.{m.group(1)}", item["tests"])
        if "error" in res:
            return False, res["error"]
        return res["passed"] == res["total"], f"pass:{res['passed']}/{res['total']}"
    try:
        compile(code, "<sol>", "exec")
    except SyntaxError as e:
        return False, f"syn:{e.msg[:40]}"
    tmp = Path(os.environ.get("TMPDIR", tempfile_seed())) / "lcb_work"
    tmp.mkdir(parents=True, exist_ok=True)
    sol = tmp / "solution.py"
    sol.write_text(code, encoding="utf-8")
    passed = 0
    for tc in item["tests"]:
        inp = tc.get("input", "").strip("\n")
        exp = tc.get("output", "")
        try:
            r = subprocess.run(
                [sys.executable, str(sol)],
                input=inp,
                capture_output=True,
                text=True,
                timeout=8,
                cwd=str(tmp),
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout@{passed}"
        got = "\n".join(l.rstrip() for l in r.stdout.splitlines())
        want = "\n".join(l.rstrip() for l in exp.splitlines()).rstrip()
        if got.rstrip() != want.rstrip():
            return False, f"mismatch@{passed}"
        passed += 1
    return True, f"pass:{passed}/{len(item['tests'])}"


def _val_ok(got: Any, allowed: list) -> bool:
    if got is None:
        return False
    gv = str(got).strip().lower()
    for a in allowed:
        av = str(a).strip().lower()
        if gv == av:
            return True
        try:
            if float(gv) == float(av):
                return True
        except Exception:
            pass
    return False


def grade_bfcl(item: dict, out: str) -> tuple[bool, str]:
    gt = item.get("ground_truth") or {}
    answers = gt.get("ground_truth", []) if isinstance(gt, dict) else gt
    call = parse_tool_call(out)
    if not call:
        return False, "no-tool-call"
    name = str(call.get("name", "")).strip()
    args = call.get("arguments") or call.get("parameters") or {}
    if not isinstance(args, dict):
        return False, "bad-args"
    for cand in answers:
        if not isinstance(cand, dict):
            continue
        for cand_name, params in cand.items():
            if norm(cand_name) != norm(name):
                continue
            ok = True
            for param, allowed in params.items():
                if not isinstance(allowed, list):
                    allowed = [allowed]
                if param not in args:
                    ok = False
                    break
                if not _val_ok(args[param], allowed):
                    ok = False
                    break
            if ok:
                return True, f"{name}:match"
    return False, f"{name}:nomatch"


def _tool_names(tools: list) -> str:
    names = []
    for t in tools:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception:
                continue
        if isinstance(t, dict):
            names.append(str(t.get("function", {}).get("name") or t.get("name", "")))
    return ", ".join(names)


def grade_w2c(item: dict, out: str) -> tuple[bool, str]:
    needs_call = item["correct_answer"] != "cannot_answer"
    call = parse_tool_call(out)
    predicted = bool(call) if call else bool(
        re.search(r'"name"\s*:\s*"[^"]+"', out)
    )
    return (predicted == needs_call), f"pred={predicted} want={needs_call} tools=[{_tool_names(item['tools'])[:80]}]"


def grade_holdout(item: dict, out: str, turns: list[_T]) -> tuple[bool, str]:
    matches = [t for t in HO.make_holdout() if t.name.replace("hold-", "holdout-") == item["id"]]
    if not matches:
        return False, "no-holdout-task"
    r = matches[0].reward_fn(turns, True)
    return bool(r >= 0.5), f"reward={r}"


# --- request layer ---------------------------------------------------------

def _api_headers() -> dict:
    """Hosted endpoints (DeepInfra) need a bearer key: env var or evals/.env,
    never printed. Local vLLM needs none."""
    key = os.environ.get("DEEPINFRA_API_KEY", "")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPINFRA_API_KEY="):
                key = line.split("=", 1)[1].strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def wait_healthy(base: str, max_wait: int = 900) -> bool:
    """Block until the server answers /health (vLLM) or /v1/models. If vLLM is
    mid-restart (docker --restart unless-stopped), the runner waits instead of
    logging a pile of request-errors."""
    if "127.0.0.1" not in base and "localhost" not in base:
        return True  # hosted API: no /health to poll
    t0 = time.time()
    warned = False
    while time.time() - t0 < max_wait:
        for path in ("/health", "/v1/models"):
            try:
                if requests.get(f"{base}{path}", timeout=5).status_code == 200:
                    if warned:
                        print(f"[health] server back after {int(time.time() - t0)}s")
                    return True
            except Exception:
                pass
        if not warned:
            print(f"[health] waiting for {base} ...")
            warned = True
        time.sleep(5)
    print(f"[health] gave up after {max_wait}s")
    return False


def chat(
    base: str, model: str, messages: list[dict], max_tokens: int, sampling: dict, timeout: int
) -> tuple[Optional[str], float, int]:
    wait_healthy(base)
    t0 = time.time()
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        **sampling,
    }
    try:
        r = requests.post(
            # DeepInfra's OpenAI route is <base>/v1/openai/chat/completions
            f"{base}/chat/completions" if base.rstrip("/").endswith("/openai") else f"{base}/v1/chat/completions",
            json=body, timeout=timeout,
            headers=_api_headers(),
        )
        r.raise_for_status()
        data = r.json()
        out = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return (out or "", time.time() - t0, tokens)
    except Exception as e:
        return None, time.time() - t0, 0


def render(item: dict, bucket: str) -> list[dict]:
    if bucket in graders.SYSTEM:
        return graders.messages(item)
    if bucket == "lcb":
        sys_msg = (
            "You are a competitive programmer. Read the problem below and output a "
            "complete Python 3 solution reading from stdin and printing to stdout. "
            "Output only code inside a ```python fenced block."
        )
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": item["question"]},
        ]
    if bucket == "bfcl":
        funcs = json.dumps(item["functions"], indent=0)
        turns = item["question"]
        if turns and isinstance(turns[0], list):
            turns = turns[0]
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to these functions:\n"
                    f"{funcs}\nReply with a single JSON tool call "
                    '{"name": "...", "arguments": {...}} when a function applies; '
                    "otherwise answer directly in prose."
                ),
            }
        ]
        for t in turns:
            if t["role"] == "system":  # chat templates allow one leading system message: fold it in
                msgs[0]["content"] = t["content"] + "\n\n" + msgs[0]["content"]
                continue
            msgs.append({"role": t["role"], "content": t["content"]})
        return msgs
    if bucket == "when2call":
        funcs = json.dumps(item["tools"], indent=0)
        return [
            {
                "role": "system",
                "content": (
                    "Decide whether the user's request requires calling a function. "
                    "If it does, reply with exactly one JSON tool call "
                    f'{{"name": "...", "arguments": {{}}}}. The available functions:\n{funcs}\n'
                    'If no function applies, reply with the single word "no_call".'
                ),
            },
            {"role": "user", "content": item["question"]},
        ]
    return []


def run_holdout_episode(
    base: str, model: str, task, sampling: dict
) -> tuple[Optional[str], float, int, list[_T]]:
    messages = [{"role": "system", "content": HO.SYSTEM}] + [
        {"role": "user", "content": task.prompt}
    ]
    turns: list[_T] = []
    env = HO.HoldoutEnv()
    t0 = time.time()
    tokens = 0
    for _ in range(6):
        out, dt, tok = chat(
            base, model, messages, max_tokens=256, sampling=sampling, timeout=180
        )
        tokens += tok
        if out is None:
            return None, time.time() - t0, tokens, turns
        turns.append(_T(out))
        ans = HO.ANSWER_RE.search(out)
        if ans:
            return out, time.time() - t0, tokens, turns
        act = HO.parse_action(out)
        if not act:
            return out, time.time() - t0, tokens, turns
        tool, arg = act
        result = env.step(tool, arg)
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"Result of {tool}({arg}):\n{result}"})
    return turns[-1].gen_text, time.time() - t0, tokens, turns


def tempfile_seed() -> Path:
    return Path(os.environ.get("TMPDIR", ROOT / ".." / ".tmp"))


# --- main ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True, help="result folder name, e.g. qwen3.5-2b")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--sampling", default='{"temperature":0.7,"top_p":0.8}')
    ap.add_argument("--buckets", default=None, help="comma list; default = every bucket in the set")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--set", default=str(SET_PATH),
                    help="frozen set file (default baseline_v1.json; home: baseline_20.json; "
                         "post-cutoff: data/sets/post_cutoff_v1.json)")
    ap.add_argument("--holdout-seal", help="sealed holdout dir (holdout/sealed/<name>): verify the seal, "
                                           "then score aggregate-only")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer for --total-cap when --model is a served LoRA adapter name")
    ap.add_argument("--total-cap", type=int, default=None,
                    help="per-request max_tokens = cap - prompt tokens (e.g. 4096 to match SFT/GRPO)")
    args = ap.parse_args()

    prompt_len = None
    if args.total_cap:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

        def prompt_len(msgs):  # same default template (thinking on) the server applies
            ids = _tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)

    sealed = bool(args.holdout_seal)
    result_dir = ROOT / "results" / args.slug
    if sealed:
        sdir = Path(args.holdout_seal)
        v = subprocess.run([sys.executable, str(SEAL_PY), "verify", str(sdir)], capture_output=True, text=True)
        if v.returncode:
            sys.exit(f"seal verify FAILED for {sdir} -- holdout is not provably clean:\n{v.stdout}")
        rows = [json.loads(l) for l in (sdir / "holdout.jsonl").read_text(encoding="utf-8").splitlines() if l]
        frozen = {"items": {}, "buckets": {}}
        for r in rows:
            frozen["items"].setdefault(r["dataset"], []).append({**r, "id": r["uid"]})
        for b, its in frozen["items"].items():
            frozen["buckets"][b] = {"n": len(its), "max_tokens": graders.MAX_TOKENS[b]}
        result_dir = result_dir / f"sealed-{sdir.name}"
        args.buckets = ",".join(frozen["items"])
    else:
        frozen = json.loads(Path(args.set).read_text(encoding="utf-8"))
    wait_healthy(args.base)
    raw_dir = result_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampling = json.loads(args.sampling)
    wanted = [b for b in (args.buckets or ",".join(frozen["items"])).split(",") if b in frozen["items"]]

    def setup(bucket: str):
        """Per-bucket context: items, the attempt worker, raw file, attempts already graded."""
        items = frozen["items"][bucket]
        bucket_cfg = frozen["buckets"][bucket]
        max_tokens = bucket_cfg["max_tokens"]
        raw_file = raw_dir / f"{bucket}.jsonl"
        done: set[tuple[str, int]] = set()
        if raw_file.exists():
            for line in raw_file.open(encoding="utf-8", errors="ignore"):
                try:
                    rec = json.loads(line)
                    if rec.get("passed") is not None:  # request-errors are retried
                        done.add((rec["item_id"], rec["attempt"]))
                except Exception:
                    pass

        def one(item: dict, attempts: list[int]) -> dict:
            recs = []
            for a in attempts:
                if (item["id"], a) in done:
                    continue
                st = time.time()
                if bucket == "holdout":
                    matches = [
                        t
                        for t in HO.make_holdout()
                        if t.name.replace("hold-", "holdout-") == item["id"]
                    ]
                    if not matches:
                        recs.append({"item_id": item["id"], "attempt": a, "passed": None,
                                     "error": "no-task"})
                        continue
                    out, dt, tok, turns = run_holdout_episode(
                        args.base, args.model, matches[0], sampling
                    )
                    if out is None:
                        passed, note = None, "request-error"
                    else:
                        passed, note = grade_holdout(item, out, turns)
                    recs.append({
                        "item_id": item["id"], "attempt": a, "passed": passed,
                        "tokens": tok, "latency_ms": int(dt * 1000), "note": note,
                    })
                else:
                    msgs = render(item, bucket)
                    mt = max_tokens
                    if args.total_cap:  # prompt + completion <= cap, same budget as SFT/GRPO
                        mt = args.total_cap - prompt_len(msgs)
                    out, dt, tok = chat(
                        args.base, args.model, msgs, mt, sampling,
                        timeout=max(120, int(mt * 0.4)),
                    )
                    if out is None:
                        recs.append({"item_id": item["id"], "attempt": a, "passed": None,
                                     "tokens": tok, "latency_ms": int(dt * 1000),
                                     "note": "request-error"})
                        continue
                    try:
                        if bucket == "lcb":
                            passed, note = grade_lcb(item, out)
                        elif bucket in graders.GRADERS:
                            passed, note = graders.grade(item, out)
                        elif bucket == "bfcl":
                            passed, note = grade_bfcl(item, out)
                        else:
                            passed, note = grade_w2c(item, out)
                        passed = bool(passed)
                    except Exception as e:
                        passed, note = None, f"grade-error:{type(e).__name__}:{str(e)[:60]}"
                    rec = {"item_id": item["id"], "attempt": a, "passed": passed,
                           "tokens": tok, "max_tokens": mt, "hit_cap": tok >= mt,
                           "latency_ms": int(dt * 1000)}
                    if not sealed:  # sealed holdout: pass/fail only, never text
                        rec.update(note=note, len_chars=len(out), head=out[:200])
                    recs.append(rec)
            return {"item": item, "recs": recs}

        return items, one, raw_file, done

    def summarize(bucket: str, items: list, raw_file: Path) -> None:
        item_pass = {}
        att_n = att_pass = att_cap = 0
        with raw_file.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("passed") is None:
                    continue
                att_n += 1
                att_pass += bool(rec["passed"])
                att_cap += bool(rec.get("hit_cap"))
                item_pass.setdefault(rec["item_id"], 0)
                if rec["passed"]:
                    item_pass[rec["item_id"]] = max(item_pass[rec["item_id"]], 1)
        solved = sum(item_pass.values())
        total = len(items)
        acc = solved / total if total else 0.0
        per_att = att_pass / att_n if att_n else 0.0
        cap_rate = att_cap / att_n if att_n else 0.0
        floors = [it["guess_floor"]["follow"] for it in items if "guess_floor" in it]
        floor = sum(floors) / len(floors) if floors else None
        print(f"[{bucket}] per-attempt {att_pass}/{att_n} ({per_att:.1%})"
              + (f" vs guess floor {floor:.1%}" if floor is not None else "")
              + f" | cap-hit {att_cap}/{att_n} ({cap_rate:.1%}) | pass@{args.attempts} {solved}/{total}")
        (result_dir / f"{bucket}.summary.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "bucket": bucket,
                    "n": total,
                    "per_attempt_acc": round(per_att, 4),
                    "attempts": att_n,
                    "cap_hit_rate": round(cap_rate, 4),
                    "guess_floor_per_attempt": None if floor is None else round(floor, 4),
                    "total_cap": args.total_cap,
                    "pass4_items": solved,
                    "accuracy": round(acc, 4),
                    **({} if sealed else {"per_item": {k: v for k, v in sorted(item_pass.items())}}),
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    # Every pending (item, attempt) across every bucket goes out at once; the
    # server's scheduler does the batching. Each record is appended the moment
    # it is graded, so a crash loses only in-flight attempts (resume skips the rest).
    ctx = {b: setup(b) for b in wanted}
    jobs = [(b, it, a) for b in wanted for it in ctx[b][0]
            for a in range(1, args.attempts + 1) if (it["id"], a) not in ctx[b][3]]
    print(f"[run] {len(jobs)} attempts in flight across {len(wanted)} bucket(s)", flush=True)
    if jobs:
        with ThreadPoolExecutor(max_workers=min(len(jobs), MAX_WORKERS)) as ex:
            futs = {ex.submit(ctx[b][1], it, [a]): b for b, it, a in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                b = futs[fut]
                with ctx[b][2].open("a", encoding="utf-8") as fh:
                    for r in fut.result()["recs"]:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"[run] {n}/{len(jobs)} graded", flush=True)
    for b in wanted:
        summarize(b, ctx[b][0], ctx[b][2])

    print("DONE", args.slug)


if __name__ == "__main__":
    main()