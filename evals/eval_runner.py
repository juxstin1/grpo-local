"""Bake-off eval runner. Reads the frozen set, grades mechanically, writes raw.

Usage (Windows PowerShell):
  .venv\\Scripts\\python.exe evals\\eval_runner.py --model Qwen/Qwen3.5-2B ^
      --slug qwen3.5-2b --sampling '{"temperature":0.7,"top_p":0.8}'

Rules enforced here:
  - The set file is read-only input; the runner never edits it.
  - pass@k is reported only after all selected k attempts are valid.
  - Thinking stays ON (no enable_thinking passed; the model's template decides).
  - Private/holdout evaluation is refused by this development identity.
  - Resume requires exact dataset/prompt/grader/settings/serving identity.
  - The historical default set includes holdout IDs: explicitly select only
    public buckets with --buckets, or use a public-only set. No silent filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests

ROOT = Path(__file__).resolve().parent
SET_PATH = ROOT / "data" / "sets" / "baseline_v1.json"
sys.path.insert(0, str(ROOT.parent))  # schema.py
import gym_budget as GB  # noqa: E402  (THE episode budget; GYM_BUDGET / --budget)
from evals import graders, result_contract as RC, code_executor  # noqa: E402
from ops.run_policy import add_model_flag, require_model_opt_in  # noqa: E402

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
    if not isinstance(item.get('tests'), list) or not item['tests']:
        raise graders.InvalidGrading('LCB requires a nonempty test suite')
    code = pick_code(out)
    if not code:
        return False, "no-code"
    if item["tests"] and item["tests"][0].get("testtype") == "functional":
        # LeetCode-style LCB items call Solution().<method>; running them as a stdin
        # script (the old path) failed every attempt regardless of correctness.
        m = re.search(r"def (\w+)\(self", item.get("starter_code") or "")
        if not m:
            raise graders.InvalidGrading('LCB functional task has no declared entry point')
        res = graders.run_functional(code, f"Solution.{m.group(1)}", item["tests"])
        if "error" in res:
            return False, res["error"]
        return res["passed"] == res["total"], f"pass:{res['passed']}/{res['total']}"
    # The trusted endpoint executor owns isolation, return-code checks and
    # comparison. Never write a shared candidate solution.py in this process.
    return graders.run_stdin(code, item['tests'])


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
    raise RC.ResultIntegrityError('Private evaluation requires an independent evaluator')


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
    if not wait_healthy(base):
        return None, 0.0, 0
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
        choice = data['choices'][0]
        out = choice['message']['content']
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        if choice.get('finish_reason') != 'stop' or not isinstance(out, str):
            return None, time.time() - t0, tokens
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


def run_holdout_episode(*args, **kwargs):
    raise RC.ResultIntegrityError('Private evaluation requires an independent evaluator')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--slug', required=True)
    ap.add_argument('--base', default='http://127.0.0.1:8000')
    ap.add_argument('--sampling', default='{"temperature":0.7,"top_p":0.8}')
    ap.add_argument('--buckets', default=None)
    ap.add_argument('--attempts', type=int, default=4)
    ap.add_argument('--set', default=str(SET_PATH))
    ap.add_argument('--sealed', action='store_true')
    ap.add_argument('--holdout-seal')
    ap.add_argument('--tokenizer')
    ap.add_argument('--total-cap', type=int)
    ap.add_argument('--budget', type=int)
    ap.add_argument('--served-identity', type=Path,
                    help='Operator-verified JSON receipt: verified, served_model, base_url, artifacts_sha256, verification_method, verified_at.')
    add_model_flag(ap)
    args = ap.parse_args()
    require_model_opt_in(args.allow_model_run)
    if args.sealed or args.holdout_seal:
        raise RC.ResultIntegrityError('Private evaluation requires an independent evaluator; no private inputs were read.')
    RC.refuse_private_path(args.set)
    RC.refuse_private_path(Path(args.set).resolve())
    if args.attempts < 1:
        raise RC.ResultIntegrityError('attempts must be positive')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.slug):
        raise RC.ResultIntegrityError('slug must be a single safe folder name')
    if args.served_identity is None:
        raise RC.ResultIntegrityError('--served-identity is required; an alias alone does not identify weights')
    served = RC.validate_served_identity(RC.strict_json(args.served_identity.read_bytes()), args.model, args.base)
    sampling = RC.strict_json(args.sampling)
    if not isinstance(sampling, dict) or set(sampling) & {'model', 'messages', 'max_tokens', 'stream', 'tools'}:
        raise RC.ResultIntegrityError('sampling must be an object and cannot override request identity or inputs')
    budget = GB.resolve(['--budget', str(args.budget)] if args.budget is not None else [])
    if args.total_cap is None and budget != GB.DEFAULT_BUDGET:
        args.total_cap = budget
    if args.total_cap is not None and args.total_cap < 1:
        raise RC.ResultIntegrityError('total-cap must be positive')
    dataset_bytes = Path(args.set).read_bytes()
    frozen = RC.strict_json(dataset_bytes)
    if not isinstance(frozen, dict) or not isinstance(frozen.get('items'), dict) or not isinstance(frozen.get('buckets'), dict):
        raise RC.ResultIntegrityError('set needs items and buckets objects')
    wanted = (args.buckets or ','.join(frozen['items'])).split(',')
    supported = set(graders.GRADERS) | {'lcb', 'bfcl', 'when2call'}
    if not wanted or len(set(wanted)) != len(wanted) or any(b not in supported or b not in frozen['items'] for b in wanted):
        raise RC.ResultIntegrityError('unknown, duplicate, or private selected bucket')
    selected = {b: RC.selected_ids(frozen['items'][b]) for b in wanted}
    if any(item.get('dataset') != b for b in wanted if b in graders.GRADERS for item in frozen['items'][b]):
        raise RC.ResultIntegrityError('item dataset must match selected grading bucket')
    caps = {b: frozen['buckets'].get(b, {}).get('max_tokens') for b in wanted}
    if any(type(v) is not int or v < 1 for v in caps.values()):
        raise RC.ResultIntegrityError('every selected bucket needs positive integer max_tokens')
    rendered = {b: {item['id']: {'messages': render(item, b),
                                'tool_schemas': item.get('functions', item.get('tools', []))}
                     for item in frozen['items'][b]} for b in wanted}
    sources = {name: (ROOT / name).read_bytes() for name in ('eval_runner.py', 'result_contract.py', 'graders.py', 'code_executor.py')}
    identity = RC.make_identity(dataset_bytes, selected, rendered, sources,
                               dict(model=args.model, base=args.base, sampling=sampling, attempts=args.attempts,
                                    budget=budget, total_cap=args.total_cap, tokenizer=args.tokenizer,
                                    code_sandbox_image=os.environ.get('GRPO_CODE_SANDBOX_IMAGE'),
                                    python_version=sys.version, executor_profile=code_executor.EXECUTOR_PROFILE,
                                    max_tokens=caps, workers=MAX_WORKERS, score_scope='public-development'), served)
    result_dir = ROOT / 'results' / args.slug
    RC.validate_resume(result_dir, identity, selected, args.attempts)
    with RC.locked_run(result_dir, identity, selected, args.attempts) as cached:
        _run_validated(args, frozen, wanted, selected, caps, rendered, identity, result_dir, cached, sampling, budget)


def _run_validated(args, frozen, wanted, selected, caps, rendered, identity, result_dir, cached, sampling, budget):
    # All resume records and the immutable identity are checked before clients,
    # tokenizers, health probes, or output writes. Invalid attempts stay invalid;
    # a new invocation only fills missing slots, never replaces old observations.
    jobs = [(b, item, attempt) for b in wanted for item in frozen['items'][b]
            for attempt in range(1, args.attempts + 1) if (item['id'], attempt) not in cached[b]]
    if any(bucket == 'code' for bucket, _, _ in jobs):
        graders.preflight(frozen['items']['code'])
    if any(bucket == 'lcb' for bucket, _, _ in jobs):
        code_executor.image_reference()  # configuration only; never launches Docker
        for item in frozen['items']['lcb']:
            tests = item.get('tests')
            if not isinstance(tests, list) or not tests or any(not isinstance(case, dict) for case in tests):
                raise graders.InvalidGrading('LCB requires nonempty valid test cases')
            if tests[0].get('testtype') == 'functional' and not re.search(r'def (\w+)\(self', item.get('starter_code') or ''):
                raise graders.InvalidGrading('LCB functional task has no declared entry point')
    prompt_len = None
    if jobs and args.total_cap:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        def prompt_len(messages):
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return len(ids['input_ids'] if hasattr(ids, 'keys') else ids)
        GB.check_server(args.base, args.total_cap, model=args.model, api_key=os.environ.get('VLLM_API_KEY', ''))
    if jobs and not wait_healthy(args.base):
        raise RC.ResultIntegrityError('endpoint unavailable; no attempts or scores recorded')
    raw_dir = result_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)

    def one(bucket, item, attempt):
        messages = rendered[bucket][item['id']]['messages']
        cap = caps[bucket] if args.total_cap is None else args.total_cap - prompt_len(messages)
        base_row = dict(item_id=item['id'], attempt=attempt, bucket=bucket,
                        run_id=identity['run_id'], budget=budget, max_tokens=cap)
        if cap < 1:
            return {**base_row, 'passed': None, 'note': 'prompt-exceeds-total-cap'}
        out, elapsed, tokens = chat(args.base, args.model, messages, cap, sampling,
                                    timeout=max(120, int(cap * 0.4)))
        row = {**base_row, 'tokens': tokens, 'latency_ms': int(elapsed * 1000), 'hit_cap': tokens >= cap}
        if out is None:
            return {**row, 'passed': None, 'note': 'request-error-or-incomplete-completion'}
        try:
            if bucket == 'lcb':
                passed, note = grade_lcb(item, out)
            elif bucket in graders.GRADERS:
                passed, note = graders.grade(item, out)
            elif bucket == 'bfcl':
                passed, note = grade_bfcl(item, out)
            else:
                passed, note = grade_w2c(item, out)
            if type(passed) is not bool:
                raise RC.ResultIntegrityError('grader did not return a boolean verdict')
        except Exception as exc:
            passed, note = None, 'grade-error:' + type(exc).__name__
        return {**row, 'passed': passed, 'note': note, 'len_chars': len(out), 'head': out[:200]}

    if jobs:
        with ThreadPoolExecutor(max_workers=min(len(jobs), MAX_WORKERS)) as executor:
            futures = {executor.submit(one, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                bucket = futures[future]
                row = future.result()
                combined = [*cached[bucket].values(), row]
                indexed = RC.validate_rows(combined, selected[bucket], args.attempts, bucket, identity['run_id'])
                with (raw_dir / (bucket + '.jsonl')).open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                cached[bucket] = indexed
    for bucket in wanted:
        summary = RC.summarize(list(cached[bucket].values()), selected[bucket], args.attempts, bucket, identity['run_id'])
        summary.update(model=args.model, identity=identity, budget=budget, total_cap=args.total_cap,
                       score_kind='public-development', metric='call-presence' if bucket == 'when2call' else 'task-outcome')
        (result_dir / (bucket + '.summary.json')).write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(f"[{bucket}] {summary['status']} pass@{args.attempts}={summary['accuracy']} "
              f"valid={summary['valid_attempts']} invalid={summary['invalid_attempts']} missing={summary['missing_attempts']}")


if __name__ == '__main__':
    main()
