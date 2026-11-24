"""tau2-bench through OUR harness: our agent loop drives tau2's AgentGymEnv.

Agent side = the served loop of sandbox-gym (imported, not copied): vLLM chat
via agent.chat_once, thinking ON, reasoning sent back in history, Qwen-native
JSON tool calls parsed by tools.parse_action, tool results shown as
agent.observation_text (<tool_response> user turn). One token budget (8192 default, gym_budget.py) per
episode (prompt + completion); running out = fail (stop "cap" / "truncated").
Anything she writes without a tool call is sent to the user as her message;
she ends the conversation with the gym's done() tool.

User side = tau2's own LLM user simulator (LiteLLM). Sim model + temperature
are CLI flags, recorded in every result, and --compare refuses runs whose sim
settings, tau2 version, task ids or budget differ.

Grading = tau2's official reward (EvaluationType.ALL). airline/telecom are
mechanical (DB/COMMUNICATE/ENV_ASSERTION/ACTION); retail test tasks carry
NL_ASSERTION (an LLM judge) -> retail needs --judge-llm, recorded.

Request errors are never failures: a vLLM error after retries, or a tau2 run
that ended in user_error / infrastructure_error / unexpected_error, is retried
and then recorded as missing, excluded from pass^k.

EVAL-ONLY: tau2 tasks never enter an SFT or GRPO pool (see README.md).

  $env:DEEPSEEK_API_KEY = "..."
  .venv\\Scripts\\python.exe run_tau2.py --tag base4b --served-model Qwen/Qwen3.5-4B
  .venv\\Scripts\\python.exe run_tau2.py --tag r3 --served-model r3
  .venv\\Scripts\\python.exe run_tau2.py --compare results/base4b.summary.json results/r3.summary.json
"""

from __future__ import annotations

import os
import sys

if not sys.flags.utf8_mode:  # tau2 reads its data with the locale codec (cp1252 on Windows)
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

import argparse
import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sandbox-gym"))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import agent as A  # noqa: E402  (sandbox-gym served loop)
import tools as T  # noqa: E402  (sandbox-gym wire format)
from config import Config  # noqa: E402
import gym_budget as GB  # noqa: E402
from encoding import render_chat  # noqa: E402

import tau2  # noqa: E402
import tau2.evaluator.evaluator_nl_assertions as nl_eval  # noqa: E402
from tau2.data_model.message import (AssistantMessage, ToolMessage,  # noqa: E402
                                     UserMessage)
from tau2.gym.gym_agent import AgentGymEnv  # noqa: E402

TASKS_FILE = HERE / "tasks.json"
RESULTS = HERE / "results"
from gym_budget import BUDGET  # noqa: E402  one budget (8192 default; env GYM_BUDGET / --budget)
MAX_STEPS = 100            # tau2 gym default; our token budget binds long before it
SIM_DEFAULT = "deepseek/deepseek-flash"   # DeepSeek V4.1 Flash (api-docs.deepseek.com, 2026-10-02)
TASK_SEED = 20261002
PER_DOMAIN = 5
MISSING_REASONS = {"user_error", "infrastructure_error", "unexpected_error"}
DONE = '{"name": "done", "arguments": {}}'


def tau2_version() -> str:
    src = HERE / "src"
    try:
        sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    return f"{getattr(tau2, '__version__', '1.0.1')}@{sha[:12]}"


# --- task ids: fixed, seeded, same for every model ---------------------------
def pick_tasks(domains: list[str]) -> dict[str, list[str]]:
    """5 test-split ids per domain, seeded; written once to tasks.json, then frozen."""
    if TASKS_FILE.exists():
        fixed = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        return {d: fixed["tasks"][d] for d in domains}
    data = HERE / "src" / "data" / "tau2" / "domains"
    out = {}
    for d in ("airline", "retail", "telecom"):
        test = json.loads((data / d / "split_tasks.json").read_text(encoding="utf-8"))["test"]
        out[d] = random.Random(f"{TASK_SEED}:{d}").sample(sorted(test), PER_DOMAIN)
    TASKS_FILE.write_text(json.dumps({"seed": TASK_SEED, "split": "test", "per_domain": PER_DOMAIN,
                                      "eval_only": True, "tasks": out}, indent=1), encoding="utf-8")
    return {d: out[d] for d in domains}


# --- system prompt: tau2 policy + tools in sandbox-gym's tool-block style ----
def _type(s: dict, defs: dict) -> str:
    if "$ref" in s:
        name = s["$ref"].split("/")[-1]
        sub = defs.get(name, {})
        inner = ", ".join(f"{k}: {_type(v, defs)}" for k, v in sub.get("properties", {}).items())
        return f"{{{inner}}}"
    if "anyOf" in s:
        return " | ".join(_type(x, defs) for x in s["anyOf"] if x.get("type") != "null") or "any"
    if "enum" in s:
        return "|".join(json.dumps(e) for e in s["enum"])
    if s.get("type") == "array":
        return f"list[{_type(s.get('items', {}), defs)}]"
    return s.get("type", "any")


def describe_tau2_tools(tools) -> str:
    """Same shape as sandbox-gym tools.describe_tools: one line per tool,
    name(args, optional?) -- help; plus one indented line per argument (tau2
    arguments carry formats/ids the model must get exactly right)."""
    lines = ["You have access to these tools:"]
    for t in tools:
        sch = t.params.model_json_schema()
        req, props, defs = set(sch.get("required", [])), sch.get("properties", {}), sch.get("$defs", {})
        args = ", ".join(a if a in req else f"{a}?" for a in props)
        lines.append(f"  {t.name}({args}) -- {t.short_desc or t.long_desc}")
        for a, s in props.items():
            desc = s.get("description", "")
            lines.append(f"      {a}: {_type(s, defs)}" + (f" -- {desc}" if desc else ""))
    lines.append("  done() -- Call this when the conversation with the user is finished.")
    lines.append(
        "Call exactly one tool per turn as a JSON block:\n"
        '  {"name": "<tool>", "arguments": {"<arg>": value}}\n'
        "A result line follows each call. Anything you write without a tool call is "
        "sent to the user as your message, and the user's reply follows. When the "
        "user's request is fully handled, call done()."
    )
    return "\n".join(lines)


SYSTEM = """\
You are a customer service agent. Help the user according to the policy below.
Never invent information; get it from the user or from tools.
<policy>
{policy}
</policy>
{tools}
"""


def system_prompt(policy: str, tools) -> str:
    return SYSTEM.format(policy=policy.strip(), tools=describe_tau2_tools(tools))


# --- one trial -----------------------------------------------------------------
class Sim:
    def __init__(self, model: str, temperature: float):
        self.model, self.temperature = model, temperature

    def settings(self) -> dict:
        return {"sim_model": self.model, "sim_temperature": self.temperature}


def make_env(domain: str, task_id: str, sim: Sim) -> AgentGymEnv:
    return AgentGymEnv(domain=domain, task_id=task_id, max_steps=MAX_STEPS,
                       user_llm=sim.model, user_llm_args={"temperature": sim.temperature})


def _ingest(messages: list[dict], new: list, last_action, opening: bool = False) -> None:
    """tau2 messages since our last action -> our chat history (our own assistant
    message is already in it)."""
    for m in new:
        if isinstance(m, AssistantMessage):
            if m.content and opening:           # tau2's scripted opening greeting
                messages.append({"role": "assistant", "content": m.content})
            continue
        if isinstance(m, ToolMessage):
            name = last_action.name if last_action else "tool"
            args = last_action.arguments if last_action else {}
            messages.append({"role": "user",
                             "content": A.observation_text(name, args, m.content or "")})
        elif isinstance(m, UserMessage) and m.content:
            messages.append({"role": "user", "content": m.content})


def run_trial(domain: str, task_id: str, trial: int, cfg: Config, prompt_len, chat,
              env_factory, budget: int = BUDGET, transcripts: bool = False) -> dict:
    """One episode. Returns a record; record["missing"] = request error, not a result."""
    rec = {"domain": domain, "task_id": task_id, "trial": trial, "reward": 0.0,
           "missing": False, "error": "", "stop": "", "turns": 0, "prompt_tokens": 0}
    env = env_factory(domain, task_id)
    _, info = env.reset(seed=trial)
    messages = [{"role": "system", "content": system_prompt(info["policy"], info["tools"])}]
    seen = 0
    obs = env._agent.observation
    _ingest(messages, obs[seen:], None, opening=True)
    seen = len(obs)
    terminated, reward, last = False, 0.0, None
    while not terminated:
        n = prompt_len(messages)
        rec["prompt_tokens"] = n
        if n + 64 > budget:
            rec["stop"] = "cap"
            break
        ch = None
        for wait in (0, 30, 120):
            time.sleep(wait)
            ch = chat(cfg, messages, budget - n)
            if "error" not in ch:
                break
        if "error" in ch:
            rec.update(missing=True, error=f"agent request: {ch['error']}", stop="request-error")
            break
        rec["turns"] += 1
        content, reasoning = ch["content"], ch.get("reasoning", "")
        if ch["finish_reason"] == "length":
            rec["stop"] = "truncated"
            break
        if not content.strip():
            rec["stop"] = "empty"          # unclosed <think>: reply filed as reasoning
            break
        action = T.parse_action(content)
        messages.append(A.assistant_message(content, reasoning))
        out = T.tool_call_text(action) if action else content.strip()
        _, reward, terminated, _, step_info = env.step(out)
        last = action
        obs = env._agent.observation
        _ingest(messages, obs[seen + 1:], last)  # +1: skip our own assistant message
        seen = len(obs)
    if not terminated:                     # we stopped (cap/truncated/empty/error): close the sim
        try:
            env.step(DONE)
        except Exception:  # noqa: BLE001
            pass
        reward = 0.0
    run = env._simulation_run
    reason = run.termination_reason.value if run is not None else ""
    if terminated:
        rec["stop"] = reason or "terminated"
        if reason in MISSING_REASONS:
            rec.update(missing=True, error=f"tau2 {reason}")
        elif run is None:                  # orchestrator raised (e.g. sim API down): no run to grade
            rec.update(missing=True, error="tau2 orchestrator error: no simulation run",
                       stop="orchestrator-error")
    rec["reward"] = float(reward) if terminated and not rec["missing"] else 0.0
    rec["termination_reason"] = reason
    if transcripts:
        rec["messages"] = messages
    return rec


# --- pass^k ----------------------------------------------------------------------
def pass_hat_k(rows: list[dict], k_max: int) -> dict:
    """tau2's pass^k: mean over tasks of C(c,k)/C(n,k); n = non-missing trials.
    A task with fewer than k non-missing trials makes pass^k None."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r)
    out = {}
    for k in range(1, k_max + 1):
        vals = []
        for rs in by.values():
            ok = [r for r in rs if not r["missing"]]
            if len(ok) < k:
                vals = None
                break
            c = sum(r["reward"] >= 1 - 1e-6 for r in ok)
            vals.append(comb(c, k) / comb(len(ok), k))
        out[f"pass^{k}"] = None if vals is None else round(sum(vals) / len(vals), 4)
    return out


def summarize(rows: list[dict], k: int) -> dict:
    out = {}
    for d in sorted({r["domain"] for r in rows}):
        rs = [r for r in rows if r["domain"] == d]
        stops: dict = {}
        for r in rs:
            stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        out[d] = {"trials": len(rs), "missing": sum(r["missing"] for r in rs),
                  **pass_hat_k(rs, k), "stops": stops}
    return out


COMPARE_KEYS = ("sim_model", "sim_temperature", "judge_llm", "tau2", "tasks", "budget", "trials",
                "agent_temperature", "thinking", "max_steps")


def compare(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    for key in COMPARE_KEYS:
        vals = {json.dumps(r["settings"].get(key), sort_keys=True) for r in runs}
        if len(vals) > 1:
            sys.exit(f"[compare] REFUSED: '{key}' differs across runs: {sorted(vals)}")
    k = runs[0]["settings"]["trials"]
    hdr = "domain    " + "".join(f"{r['settings']['tag']:>22}" for r in runs)
    print(hdr)
    for d in runs[0]["by_domain"]:
        for j in range(1, k + 1):
            cells = "".join(f"{str(r['by_domain'].get(d, {}).get(f'pass^{j}')):>22}" for r in runs)
            print(f"{d:<8}^{j} {cells}")
        miss = "".join(f"{('missing ' + str(r['by_domain'].get(d, {}).get('missing'))):>22}" for r in runs)
        print(f"{'':<10}{miss}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="results name, e.g. base4b / r3")
    ap.add_argument("--served-model", default="", help="vLLM model or LoRA name (e.g. r3)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="HF id: tokenizer + template")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--temperature", type=float, default=0.7, help="agent sampling (sandbox-gym default)")
    ap.add_argument("--domains", default="airline,telecom",
                    help="airline,telecom are mechanical; retail needs --judge-llm")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--sim-model", default=SIM_DEFAULT)
    ap.add_argument("--sim-temperature", type=float, default=0.0, help="tau2 default user temperature")
    ap.add_argument("--judge-llm", default="", help="NL-assertion judge (retail); recorded")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--transcripts", action="store_true")
    ap.add_argument("--compare", nargs="+", metavar="SUMMARY")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
        return
    if not a.tag:
        sys.exit("--tag is required")
    domains = [d.strip() for d in a.domains.split(",") if d.strip()]
    if "retail" in domains and not a.judge_llm:
        sys.exit("retail test tasks are graded with NL assertions (LLM judge): pass --judge-llm")
    if a.judge_llm:
        nl_eval.DEFAULT_LLM_NL_ASSERTIONS = a.judge_llm
    if a.sim_model.startswith("deepseek/") and not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY is not set (user simulator)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = Config()
    cfg.serve.base, cfg.serve.model_name, cfg.serve.served_model = a.base, a.model, a.served_model
    cfg.serve.temperature = a.temperature
    cfg.episode.thinking = True
    tok_lock = threading.Lock()

    def prompt_len(msgs):
        with tok_lock:
            return len(render_chat(tok, msgs, True))

    sim = Sim(a.sim_model, a.sim_temperature)
    tasks = pick_tasks(domains)
    settings = {"tag": a.tag, "served_model": a.served_model or a.model, "model": a.model,
                **sim.settings(), "judge_llm": a.judge_llm or None, "tau2": tau2_version(),
                "tasks": tasks, "budget": a.budget, "trials": a.trials, "thinking": True,
                "agent_temperature": a.temperature, "max_steps": MAX_STEPS}
    if not A.wait_healthy(a.base, max_wait=60):
        sys.exit(f"vLLM at {a.base} is not healthy")
    GB.check_server(a.base, a.budget, cfg.serve.api_key)

    jobs = [(d, t, j) for d in domains for t in tasks[d] for j in range(a.trials)]
    RESULTS.mkdir(exist_ok=True)
    raw = RESULTS / f"{a.tag}.jsonl"
    done = {}
    if raw.exists():   # resumable: keep finished non-missing trials
        for line in raw.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r["missing"] and r["settings"] == settings:
                done[(r["domain"], r["task_id"], r["trial"])] = r
    todo = [j for j in jobs if j not in done]
    print(f"[tau2] {a.tag}: {len(jobs)} trials ({len(done)} cached), sim {a.sim_model} "
          f"T={a.sim_temperature}, tau2 {settings['tau2']}", flush=True)
    lock = threading.Lock()

    def one(job):
        d, t, j = job
        for attempt in range(3):           # request errors are retried, never scored
            try:
                r = run_trial(d, t, j, cfg, prompt_len, A.chat_once,
                              lambda dd, tt: make_env(dd, tt, sim), a.budget, a.transcripts)
            except Exception as e:  # noqa: BLE001
                r = {"domain": d, "task_id": t, "trial": j, "reward": 0.0, "missing": True,
                     "error": f"{type(e).__name__}: {str(e)[:160]}", "stop": "exception"}
            if not r["missing"]:
                break
            time.sleep(30 * (attempt + 1))
        r["settings"] = settings
        with lock:
            with raw.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[tau2] {d} {t} #{j} reward={r['reward']} stop={r['stop']}"
                  + (f" MISSING {r['error']}" if r["missing"] else ""), flush=True)
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(done.values()) + list(ex.map(one, todo))
    summary = {"settings": settings, "by_domain": summarize(rows, a.trials)}
    (RESULTS / f"{a.tag}.summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["by_domain"], indent=1))


if __name__ == "__main__":
    main()
