"""GRPO with qualified public outcome contracts (model execution off by default).

The default is the public development suite with work-contract-v1 instructions.
Legacy keyword graders/mixes are refused for training. This is not a sealed benchmark.
Run ops/qualify_offline.py to check instructions, reward, and integration without models.

When separately authorized: llama-server (base Q4_K_M + current LoRA via --lora) writes the rollouts,
bf16 Qwen3.5-4B + the same LoRA (PEFT, trainable) takes the update.

Per step:
  1. llama-server starts with the current adapter (converted to GGUF by ops/convert_lora_qwen35.py).
  2. TASK_BATCH public development tasks x GROUP attempts, Bionic native tool loop:
     up to 8 requests, the last one without tools, history as Bionic sends it (no reasoning_content),
     tools run by the gym's Bionic mirror. Plain sampler (temperature 1.0 only), random 31-bit seed per
     request, pre-sampler log-probs of every sampled token come back (n_probs=1).
  3. Server stops (frees VRAM for the backward). Reward = qualified outcome contract (0/1).
     Advantage = reward - group mean (groups with one reward value are skipped: no signal).
  4. Each assistant round is one (prompt ids, sampled ids) sequence. One forward with grad gives lp_bf16;
     weight w = min(exp(lp_bf16 - lp_q4), CAP), detached (truncated importance sampling for the Q4/bf16 gap);
     loss = -sum(w * A * lp_bf16) / (trained tokens). One optimizer step per batch.
  5. Guard: if more than STOP_OUTSIDE of trained tokens have a ratio outside [0.5, 2], stop before stepping.
  6. Save the adapter, convert it, next step.

Logs every step to runs/grpo-q4/<run>/steps.jsonl: reward, dead groups, ratio stats, clip fraction,
VRAM (whole card), timings.

  .venv\\Scripts\\python.exe -B ops\\qualify_offline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q4_bf16_gap as G  # noqa: E402
from q4_bf16_gap2 import CardVram, render  # noqa: E402
import real_tasks  # noqa: E402
import contract_tasks
import reward_contract
import work_contract
from recovery_contract import RecoverySession
from run_policy import add_model_flag, require_model_opt_in

ROOT = HERE.parent
BASE_Q4 = ROOT / "q4" / "out" / "4b-base.Q4_K_M.gguf"
LORA0_GGUF = G.OUT / "lora-sft-1008.f16.gguf"          # SFT-1008 adapter, converted + verified exact
CTX = 12288
N_PREDICT = 2048
MAX_ROUNDS = 8                                          # Bionic MAX_TOOL_ITERATIONS
FINAL_ROUND_NOTE = ('Tools are unavailable in this final reply. Do not request or describe another tool call. '
                    'Report only the outcome established by observations. If unfinished, state the blocker and '
                    'unverified work; never claim completion. Follow the task final-output format only when its '
                    'completion conditions are met.')
LAST_ROUND_NOTE = ""                                    # Bionic fix/agent-context (24 rounds) appends this on the
                                                        # tool-less last round; off = the 8-round mirror trained on
MAX_TRAIN_TOKENS = 10240                                # longer sequences are skipped (counted) to fit VRAM
CAP = 2.0
STOP_OUTSIDE = 0.05
LN2 = math.log(2.0)
RNG = random.SystemRandom()
PARSE_RETRIES = [0]                                     # resampled after a server tool-call parse 500
FAMILIES = [contract_tasks.FAMILY]
PROMPT_PROFILE = 'work-contract-v1'
LEGACY_DIAGNOSTIC = False  # historical replay only; never a training reward
MIX: dict[str, float] = {}                              # --mix template weights, e.g. clarify=3,markdown=2
PARALLEL = [1]                                          # --parallel: concurrent episodes = llama-server slots
REAL: list[dict] = []                                   # --mix real=W: tasks from data/bionic-real/tasks
DIAG: list[dict] = []                                   # --mix diag=W: data/diag-eval/tasks (not sealed/, not canary)
NEED_LP = [True]                                        # training needs Q4 log-probs; evals set this False
LP_DROPPED = [0]                                        # attempts dropped for LpMismatch


# ---------------------------------------------------------------- server

def start_server(lora_gguf: Path, log: Path) -> tuple[subprocess.Popen, float]:
    args = [str(G.SERVER), "-m", str(BASE_Q4), "--lora", str(lora_gguf), "--host", "127.0.0.1",
            "--port", str(G.PORT), "-ngl", "99", "-c", str(CTX * PARALLEL[0]), "-np", str(PARALLEL[0]),
            "--no-webui"]
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    for _ in range(1500):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited ({proc.returncode}); see {log}")
        try:
            if requests.get(f"{G.URL}/health", timeout=2).status_code == 200:
                return proc, time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(0.2)
    proc.kill()
    raise SystemExit(f"llama-server not healthy; see {log}")


def stop_server(proc):
    proc.kill()
    proc.wait()
    time.sleep(1)


class LpMismatch(RuntimeError):
    """llama-server's completion_probabilities don't line up with the sampled tokens. Seen consistently on
    multi-byte UTF-8 tokens (emoji): token 10838 sampled, prob entry 222 reported, list shorter. Retrying the
    same prompt keeps hitting it, so the attempt is dropped from its group instead of killing the run."""


def complete(prompt_ids: list[int], n_predict: int) -> dict:
    body = {"prompt": prompt_ids, "n_predict": n_predict, "seed": RNG.getrandbits(31), "cache_prompt": True,
            "samplers": ["temperature"], "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "n_probs": 1 if NEED_LP[0] else 0, "post_sampling_probs": False, "return_tokens": True}
    if not NEED_LP[0]:                                   # evals: no log-probs needed, so no alignment to fail
        j = requests.post(f"{G.URL}/completion", json=body, timeout=3600).json()
        for _ in range(8):
            if not ("error" in j and "Failed to parse input" in str(j["error"])):
                break
            PARSE_RETRIES[0] += 1
            body["seed"] = RNG.getrandbits(31)
            j = requests.post(f"{G.URL}/completion", json=body, timeout=3600).json()
        if "error" in j:
            raise RuntimeError(j["error"])
        return {"gen_ids": j["tokens"], "lp_q4": [], "stop_type": j.get("stop_type"),
                "tps": j.get("timings", {}).get("predicted_per_second", 0.0)}
    for attempt in range(9):
        j = requests.post(f"{G.URL}/completion", json=body, timeout=3600).json()
        # llama-server 500s when it can't parse a sampled tool call (seen: garbled UTF-8 parameter name).
        # That sample is lost, so draw a fresh one; any other error is fatal (request errors are never results).
        if "error" in j and "Failed to parse input" in str(j["error"]) and attempt < 8:
            PARSE_RETRIES[0] += 1
            body["seed"] = RNG.getrandbits(31)
            continue
        # rare (1 in ~2000): `tokens` and `completion_probabilities` ids differ; log-probs can't be aligned,
        # so keep the evidence and draw a fresh sample
        if "error" not in j and [p["id"] for p in (j.get("completion_probabilities") or [])] != j.get("tokens") \
                and attempt < 8:
            PARSE_RETRIES[0] += 1
            with open(ROOT / "runs" / "grpo-q4" / "mismatch-dumps.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"seed": body["seed"], "prompt_len": len(prompt_ids), "tokens": j.get("tokens"),
                                     "prob_ids": [p["id"] for p in j.get("completion_probabilities") or []],
                                     "stop_type": j.get("stop_type")}) + "\n")
            body["seed"] = RNG.getrandbits(31)
            continue
        break
    if "error" in j:
        raise RuntimeError(j["error"])
    probs = j.get("completion_probabilities") or []
    ids = j["tokens"]
    if [p["id"] for p in probs] != ids:
        raise LpMismatch("tokens and completion_probabilities disagree")
    return {"gen_ids": ids, "lp_q4": [p["logprob"] for p in probs], "stop_type": j.get("stop_type"),
            "tps": j.get("timings", {}).get("predicted_per_second", 0.0)}


# ---------------------------------------------------------------- gym

def gym():
    os.chdir(G.GYM)
    sys.path.insert(0, str(G.GYM))
    import dialect, native, repo_gen  # noqa: E401
    from config import Config
    from eval import make_task
    from runner import Runner
    from schema import Action, Turn, Task
    from workspace import Workspace
    return dict(dialect=dialect, native=native, repo_gen=repo_gen, Config=Config, make_task=make_task,
                Runner=Runner, Workspace=Workspace, Action=Action, Turn=Turn, Task=Task)


class TaskStream:
    """Fresh train-seed tasks, families in rotation, holdout seeds refused, never the same seed twice."""
    def __init__(self, M, seed0: int):
        self.M, self.seed, self.i = M, seed0, 0

    def next(self) -> tuple[str, dict]:
        if MIX:                                         # --mix: template by weight, from the next seed that has it
            kind = RNG.choices(list(MIX), weights=list(MIX.values()))[0]
            if kind == "real":                          # a real Bionic fumble task, uniformly
                if not REAL:
                    REAL.extend(real_tasks.load())
                return real_tasks.FAMILY, REAL[RNG.randrange(len(REAL))]
            if kind == "diag":                          # spec-31 code tasks, hidden-test graded; visible split
                if not DIAG:                            # only (sealed/ and the canary are never trained on)
                    DIAG.extend(t for t in real_tasks.load(ROOT / "data" / "diag-eval" / "tasks")
                                if "canary" not in t["name"])
                return real_tasks.FAMILY, DIAG[RNG.randrange(len(DIAG))]
            while True:
                self.seed += 1
                for fam in FAMILIES:
                    if (fam, self.seed) in self.M["sealed"]:
                        continue
                    hit = [s for s in self.M["repo_gen"].make_specs(self.seed, fam)
                           if s["name"].split("-")[1] == kind and "make_pdf" not in s["prompt"]]
                    if hit:
                        return fam, hit[RNG.randrange(len(hit))]
        fam = FAMILIES[self.i % len(FAMILIES)]
        self.i += 1
        while True:
            self.seed += 1
            if (fam, self.seed) in self.M["sealed"]:
                continue
            specs = self.M["repo_gen"].make_specs(self.seed, fam)
            # doc_make t2/t3 ask for the gym-only make_pdf tool, which Bionic (native mode) doesn't have
            specs = [s for s in specs if "make_pdf" not in s["prompt"]]
            if specs:
                return fam, specs[RNG.randrange(len(specs))]


def _dict_args(s: str) -> dict:
    try:
        v = json.loads(s) if s else {}
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def parse_calls(native, content: str, tools: list[dict]) -> tuple[list[dict], str]:
    """llama-server's Qwen XML tool-call parse of the reply (after </think>): every <tool_call> block,
    any function name (Bionic answers an unknown one with "Unknown tool: x"), prose = text before the
    first block. Args typed by the tool schema when the name is known, else JSON where it parses.
    (recover_from_reasoning is Bionic's fallback for calls stuck in <think>: it drops the whole message
    on one unknown name, which llama-server does not.)"""
    schemas = {t["function"]["name"]: t["function"].get("parameters") for t in tools}
    blocks = list(native._BLOCK_RE.finditer(content))
    calls = []
    for b in blocks:
        fn = native._FN_RE.search(b.group(1))
        if not fn:
            continue
        props = ((schemas.get(fn.group(1)) or {}).get("properties") or {})
        args: dict = {}
        for p in native._PARAM_RE.finditer(b.group(1)):
            key, raw = p.group(1), p.group(2)
            if (props.get(key) or {}).get("type") == "string":
                args[key] = raw
                continue
            try:
                args[key] = native._js_parse(raw.strip())
            except ValueError:
                args[key] = raw
        calls.append({"name": fn.group(1), "args": native._js_stringify(args)})
    prose = content[:blocks[0].start()].strip() if blocks else content
    return calls, prose


def proot_runner(M):
    """The docker runner, with `docker run python:3.12-slim` swapped for proot into that image's exported
    rootfs (GRPO_ROOTFS): only the workspace is mounted (/work), the whole process group dies at the timeout.
    No network isolation (docker had --network none) and no memory/pid caps."""
    import signal

    class ProotRunner(M["Runner"]):
        def _docker_proc(self, workdir, cwd, argv, timeout_ms, stdin=None, extra_mounts=None, extra_env=None):
            rel = Path(cwd).resolve().relative_to(Path(workdir).resolve()).as_posix()
            secs = timeout_ms / 1000
            args = ["proot", "-r", os.environ["GRPO_ROOTFS"], "-b", f"{workdir}:{self.mount_point}"]
            for source, dest in extra_mounts or []:
                args += ["-b", f"{source}:{dest.removesuffix(':ro')}"]
            args += ["-w", self.mount_point if rel == "." else f"{self.mount_point}/{rel}",
                     "/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME=/root",
                     "PYTHONIOENCODING=utf-8", *(extra_env or []), "timeout", "-s", "KILL", f"{secs:g}", *argv]
            t0 = time.time()
            p = subprocess.Popen(args, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                out, err = p.communicate(input=stdin, timeout=secs + 10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.communicate()
                return None
            if p.returncode == 137 or time.time() - t0 >= secs:    # proot exits 0 when timeout KILLs the child
                return None
            return out, err, p.returncode

    return ProotRunner("docker")


def episode(M, tok, fam, spec, tag) -> dict:
    """One Bionic user turn (native.run_episode semantics), generated by llama-server from HF-rendered ids."""
    native, Turn, Action = M["native"], M["Turn"], M["Action"]
    if fam == contract_tasks.FAMILY:
        reward_contract.validate_spec(spec)
    elif not LEGACY_DIAGNOSTIC or NEED_LP[0]:
        raise reward_contract.UnqualifiedTask('Legacy graders are diagnostic only and cannot supply training rewards.')
    cfg = M["Config"]()
    cfg.episode.tool_dialect, cfg.episode.bionic_profile = "bionic", "default"
    runner = (proot_runner(M) if os.environ.get("GRPO_RUNNER") == "proot"   # pod: no Docker
              else M["Runner"](os.environ.get("GRPO_RUNNER", "docker")))
    ws = M["Workspace"](cfg.episode.data_root / "workspaces" / f"grpo-{tag}", runner)
    ws.reset()
    execution = None
    if fam == contract_tasks.FAMILY:
        real_tasks.write_files(spec['files'], ws.root)
        if 'recovery' in spec['contract']:
            execution = RecoverySession(spec, ws.root)
        task = M['dialect'].prepare(cfg, contract_tasks.make_task(spec, M['Task'], execution=execution), ws)
    elif fam == real_tasks.FAMILY:                         # real Bionic fumble (data/bionic-real/tasks)
        real_tasks.write_files(spec["files"], ws.root)
        task = M["dialect"].prepare(cfg, real_tasks.make_task(spec, diagnostic=True), ws)
    else:
        M["repo_gen"].write_files(M["repo_gen"].find_spec(spec["seed"], fam, spec["name"])["files"], ws.root)
        task = M["dialect"].prepare(cfg, M["make_task"](spec, native=True), ws)
    tools = native.native_tools()
    if execution is not None:
        tools = [t for t in tools if t['function']['name'] in spec['tools']]
        if {t['function']['name'] for t in tools} != set(spec['tools']):
            raise reward_contract.InvalidReward('recovery native tool schemas unavailable')
    system = work_contract.build_system(tools, PROMPT_PROFILE, legacy_system=native.system_prompt)
    final_system = work_contract.build_system([], PROMPT_PROFILE, legacy_system=native.system_prompt)
    final_system += '\n\n' + FINAL_ROUND_NOTE + ('\n\n' + LAST_ROUND_NOTE if LAST_ROUND_NOTE else '')
    user = task.prompt.strip()
    rounds, seqs, turns, counters, stop = [], [], [], {}, "cap"
    for it in range(MAX_ROUNDS):
        last = it == MAX_ROUNDS - 1
        sys_r = final_system if last else system
        msgs = native.to_api_messages(sys_r, user, rounds)
        for m in msgs:                                  # HF template wants dict arguments
            for c in m.get("tool_calls") or []:
                c["function"]["arguments"] = _dict_args(c["function"]["arguments"])
        ids = render(tok, msgs, None if last else tools)
        if len(ids) + 256 > CTX:
            stop = "ctx_full"
            break
        g = complete(ids, min(N_PREDICT, CTX - len(ids)))
        seqs.append({"prompt_ids": ids, "gen_ids": g["gen_ids"], "lp_q4": g["lp_q4"], "tps": g["tps"]})
        text = tok.decode(g["gen_ids"], skip_special_tokens=False).replace("<|im_end|>", "")
        reasoning, content = (text.split("</think>", 1) if "</think>" in text else (text, ""))
        reasoning, content = reasoning.strip(), content.strip()
        calls, prose = parse_calls(native, content, [] if last else tools)
        if not content and reasoning:                 # Bionic fix: call stuck in an unclosed <think>
            rec = native.recover_from_reasoning(reasoning, [] if last else tools)
            if rec and rec["kind"] == "tool_call":
                calls, reasoning = rec["toolCalls"], rec["reasoning"]
            elif rec:
                prose = rec.get("text", prose)
        truncated = g["stop_type"] != "eos"
        if calls and not last and not truncated:
            done = []
            for k, c in enumerate(calls):
                dispatch = lambda: native.execute(ws, c['name'], c['args'], counters)
                observation = (execution.execute(c['name'], c['args'], dispatch, round_index=it)
                               if execution is not None else dispatch())
                done.append(dict(c, id=f'call_{it}_{k}', result=observation))
            rounds.append({"text": prose, "reasoning": reasoning, "calls": done})
            acts = [(Action(c["name"], _dict_args(c["args"])), c["result"]) for c in done]
            turns.append(Turn(content=prose, resp_text=prose, reasoning=reasoning,
                              action=acts[0][0], obs=acts[0][1], extra=acts[1:]))
            continue
        turns.append(Turn(content=prose, resp_text=prose, reasoning=reasoning, truncated=truncated))
        stop = "truncated" if truncated else ("answer" if prose else "empty")
        break
    reward = float(task.reward(turns, ws))
    return {"family": fam, "seed": spec["seed"], "name": spec["name"], "reward": reward, "stop": stop,
            "root_id": spec.get('root_id', spec['name']),
            "instruction": {'profile': PROMPT_PROFILE,
                            'system_sha256': hashlib.sha256(system.encode('utf-8')).hexdigest(),
                            'final_system_sha256': hashlib.sha256(final_system.encode('utf-8')).hexdigest()},
            "score_kind": 'qualified-development' if fam == contract_tasks.FAMILY else 'legacy-diagnostic',
            "recovery_execution": execution.receipt() if execution is not None else None,
            "reason": (reward_contract.grade(spec, turns, ws.root, execution=execution).reason if fam == contract_tasks.FAMILY
                       else real_tasks.grade(ws.root, spec, turns)[1] if fam == real_tasks.FAMILY
                       else grade_reason(fam, task, turns, ws)), "rounds": len(seqs), "seqs": seqs}


def grade_reason(fam: str, task, turns, ws) -> str:
    """The family grader's reason string (diagnostic only; the reward is task.reward)."""
    try:
        from graders import last_answer
        truth = json.loads(task.ground_truth)
        if fam == "file_work":
            import file_work
            if truth["kind"] in ("clarify", "clarify_one"):       # graded on the final reply, as reward_fn does
                from graders import strip_reasoning
                if not turns or turns[-1].action is not None:
                    return "no-final-reply"
                final = strip_reasoning(turns[-1].resp_text or turns[-1].content)
                return file_work.grade_workspace(ws.root, truth, final)[1]
            return file_work.grade_workspace(ws.root, truth, last_answer(turns))[1]
        if fam == "office_draft":
            import office_draft
            ans = last_answer(turns)
            if ans is None or ans.strip().strip("` ") != truth["path"]:
                return f"answer-path:{ans!r}"[:80]
            return office_draft.grade_draft(ws.root, truth)[1]
        if fam == "doc_make":
            import doc_make
            return doc_make.grade_doc(ws.root, truth)[1]
    except Exception as e:                              # never let bookkeeping kill an attempt
        return f"reason-error:{type(e).__name__}"
    return ""


# ---------------------------------------------------------------- adapter

def base_snapshot() -> str:
    """Cached base snapshot dir (config + tokenizer is all the converter reads; README etc. may be absent)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    snaps = Path(HF_HUB_CACHE) / ("models--" + G.BASE_MODEL.replace("/", "--")) / "snapshots"
    for d in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (d / "config.json").exists():
            return str(d)
    raise SystemExit(f"no cached snapshot with config.json under {snaps}")


def convert(adapter_dir: Path, out: Path) -> float:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "convert_lora_qwen35.py"), "--base", base_snapshot(),
                        "--outtype", "f16", "--outfile", str(out), str(adapter_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"LoRA conversion failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return time.time() - t0


# ---------------------------------------------------------------- train

def ratio_stats(diffs: list[float]) -> dict:
    if not diffs:
        return {"n_tokens": 0}
    a = sorted(abs(d) for d in diffs)
    n = len(a)
    return {"n_tokens": n, "mean_diff": sum(diffs) / n, "mean_abs": sum(a) / n,
            "p95_abs": a[min(n - 1, int(0.95 * n))], "max_abs": a[-1],
            "frac_outside_0.5_2": sum(1 for d in a if d > LN2) / n,
            "frac_capped": sum(1 for d in diffs if d > math.log(CAP)) / n,
            "kl_q4_bf16_per_token": -sum(diffs) / n}


def train_step(model, optim, groups: list[list[dict]]) -> dict:
    import torch
    items = []                                          # (seq, advantage)
    dead = 0
    for grp in groups:                                  # sizes can differ (--extend)
        reward_contract.summarize(grp)                  # no empty/invalid values in advantages
        rs = [e["reward"] for e in grp]
        if len(set(rs)) == 1:
            dead += 1
            continue
        mean = sum(rs) / len(rs)
        for e in grp:
            for s in e["seqs"]:
                items.append((s, e["reward"] - mean))
    out: dict = {"groups": len(groups), "dead_groups": dead, "skipped_long": 0}
    fit = [(s, a) for s, a in items if len(s["prompt_ids"]) + len(s["gen_ids"]) <= MAX_TRAIN_TOKENS]
    out["skipped_long"] = len(items) - len(fit)
    n_tok = sum(len(s["gen_ids"]) for s, _ in fit)
    out["trained_seqs"], out["trained_tokens"] = len(fit), n_tok
    if not fit:
        out["stepped"] = False
        return out
    model.train()
    optim.zero_grad(set_to_none=True)
    diffs, loss_sum = [], 0.0
    ent_sum, nll_sum = 0.0, 0.0                         # plateau check: policy entropy / sampled NLL per token
    torch.cuda.reset_peak_memory_stats()
    for s, adv in fit:
        ids = torch.tensor([s["prompt_ids"] + s["gen_ids"]], device="cuda")
        gen = torch.tensor(s["gen_ids"], device="cuda")
        g = len(s["gen_ids"])
        logits = model(input_ids=ids, logits_to_keep=g + 1).logits[0, :-1].float()
        logp = torch.log_softmax(logits, -1)
        lp = logp.gather(-1, gen[:, None])[:, 0]
        with torch.no_grad():                           # entropy collapse shows up here before it shows in reward
            for i in range(0, g, 128):                  # chunked: a full exp() copy is ~2 GB at 248k vocab x 2k tok
                c = logp[i:i + 128]
                ent_sum += float(-(c.exp() * c).sum())
            nll_sum += float(-lp.sum())
        del logp
        lp_q4 = torch.tensor(s["lp_q4"], device="cuda")
        d = lp.detach() - lp_q4
        diffs += d.tolist()
        w = torch.exp(d).clamp(max=CAP)
        loss = -(w * adv * lp).sum() / n_tok
        loss.backward()
        loss_sum += loss.item()
        del logits, lp, ids, loss
    out["torch_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out["entropy_per_token"] = ent_sum / max(1, n_tok)
    out["nll_per_token"] = nll_sum / max(1, n_tok)
    out["ratio"] = ratio_stats(diffs)
    out["loss"] = loss_sum
    if out["ratio"]["frac_outside_0.5_2"] > STOP_OUTSIDE:
        optim.zero_grad(set_to_none=True)
        out["stepped"] = False
        out["guard"] = f"frac outside [0.5,2] {out['ratio']['frac_outside_0.5_2']:.3%} > {STOP_OUTSIDE:.0%}"
        return out
    out["grad_norm"] = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
    optim.step()
    optim.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    out["stepped"] = True
    return out


# ---------------------------------------------------------------- main

def main():
    global LEGACY_DIAGNOSTIC, PROMPT_PROFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--task-batch", type=int, default=4)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed0", type=int, default=46000)
    ap.add_argument("--families", default=contract_tasks.FAMILY,
                    help='Only contract_v1 is qualified for this trainer; old families are diagnostic only.')
    ap.add_argument("--resume-adapter", default=None, help="PEFT dir to continue from (optimizer state restarts)")
    ap.add_argument("--resume-gguf", default=None, help="that adapter converted to GGUF")
    ap.add_argument("--step0", type=int, default=0, help="steps already done (numbering only)")
    ap.add_argument("--mix", default="", help="template weights kind=w,... (kind = name's 2nd part, e.g. clarify_one)")
    ap.add_argument("--extend", type=int, default=0, help="extra attempts for a group with no success yet")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent episodes (llama-server -np)")
    add_model_flag(ap)
    a = ap.parse_args()
    require_model_opt_in(a.allow_model_run)
    LEGACY_DIAGNOSTIC, PROMPT_PROFILE = False, 'work-contract-v1'
    NEED_LP[0] = True
    if a.families != contract_tasks.FAMILY or a.mix:
        raise SystemExit('Training requires the qualified public contract_v1 suite. Legacy mixes are not training rewards.')
    if a.group < 2 or a.task_batch < 1 or a.steps < 1 or a.extend < 0 or a.parallel < 1:
        raise SystemExit('Require group>=2, task-batch/steps/parallel>=1, extend>=0.')
    contract_tasks.specs()  # validate suite before loading a model
    MIX.update({kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.mix.split(",") if kv})
    PARALLEL[0] = a.parallel
    if bool(a.resume_adapter) != bool(a.resume_gguf):
        raise SystemExit("--resume-adapter and --resume-gguf go together")
    if a.resume_adapter:
        G.ADAPTER = (ROOT / a.resume_adapter).resolve()
    FAMILIES[:] = a.families.split(",")
    M = gym()
    out = ROOT / "runs" / "grpo-q4" / a.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({**vars(a), "cap": CAP, "stop_outside": STOP_OUTSIDE,
                                                 "ctx": CTX, "n_predict": N_PREDICT, "max_rounds": MAX_ROUNDS,
                                                 "max_train_tokens": MAX_TRAIN_TOKENS, "base_q4": BASE_Q4.name,
                                                 "init_adapter": str(G.ADAPTER),
                                                 "prompt": work_contract.prompt_metadata(PROMPT_PROFILE, tools=M['native'].native_tools(),
                                                                                          legacy_system=M['native'].system_prompt),
                                                 "final_round_note": FINAL_ROUND_NOTE,
                                                 "suite": contract_tasks.suite_metadata(), "sealed": False,
                                                 "reward_contract": "outcome-v1"}, indent=1), encoding="utf-8")

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.BASE_MODEL)
    tasks = contract_tasks.TaskStream()
    card = CardVram(out / "vram.txt")
    try:
        model = G.load_policy(trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        lora_gguf = (ROOT / a.resume_gguf).resolve() if a.resume_gguf else LORA0_GGUF
        for step in range(a.step0 + 1, a.step0 + a.steps + 1):
            rec: dict = {"step": step, "adapter": lora_gguf.name}
            t0 = time.time()
            model.to("cpu")                             # whole card to llama-server while it generates:
            torch.cuda.empty_cache()                    # with bf16 resident it spilled to shared memory (5 tok/s)
            rec["offload_s"] = time.time() - t0
            m = card.mark()
            proc, rec["server_start_s"] = start_server(lora_gguf, out / "server.log")
            t0 = time.time()
            eps = []
            try:
                picked = [tasks.next() for _ in range(a.task_batch)]
                groups: list[list[dict]] = [[] for _ in picked]

                def run(job):
                    t, k = job
                    fam, spec = picked[t]
                    try:
                        e = episode(M, tok, fam, spec, f"{a.run}-{t}-{k}")
                    except LpMismatch:
                        LP_DROPPED[0] += 1
                        print(f"[s{step}] {fam} {spec['name']} #{k}: dropped (log-prob mismatch)", flush=True)
                        return t, None
                    print(f"[s{step}] {fam} {spec['name']} #{k}: reward {e['reward']:.0f} stop {e['stop']} "
                          f"{e['reason'][:40]} rounds {e['rounds']} tok {sum(len(s['gen_ids']) for s in e['seqs'])}", flush=True)
                    return t, e

                # wave 1: every task's first --group attempts; wave 2: --extend more for groups with no success
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(PARALLEL[0]) as pool:
                    for t, e in pool.map(run, [(t, k) for t in range(len(picked)) for k in range(a.group)]):
                        if e is not None:
                            groups[t].append(e)
                    more = [(t, a.group + k) for t in range(len(picked)) if not any(e["reward"] for e in groups[t])
                            for k in range(a.extend)]
                    for t, e in pool.map(run, more):
                        if e is not None:
                            groups[t].append(e)
                groups = [g for g in groups if g]
                eps = [e for g in groups for e in g]
            finally:
                rec["vram_gen_peak_gb"] = card.peak_since(m)
                stop_server(proc)
            rec["gen_s"] = time.time() - t0
            rec["parse_retries"] = PARSE_RETRIES[0]
            rec["lp_dropped"] = LP_DROPPED[0]
            # A fully dropped generation wave has no score and must never be
            # converted to zero, NaN, or a synthetic training sample.
            rec.update(reward_contract.summarize(eps))
            rec['root_ids'] = sorted({e['root_id'] for e in eps})
            t1 = time.time()
            model.to("cuda")
            rec["reload_s"] = time.time() - t1
            gen_tok = sum(len(s["gen_ids"]) for e in eps for s in e["seqs"])
            rec["gen_tokens"] = gen_tok
            rec["tps"] = gen_tok / max(1e-9, sum(len(s["gen_ids"]) / max(s["tps"], 1e-9)
                                                 for e in eps for s in e["seqs"]))
            rec["stops"] = {s: sum(1 for e in eps if e["stop"] == s) for s in {e["stop"] for e in eps}}
            with (out / "episodes.jsonl").open("a", encoding="utf-8") as fh:
                for e in eps:
                    fh.write(json.dumps({"step": step, **e}) + "\n")

            m = card.mark()
            t0 = time.time()
            rec.update(train_step(model, optim, groups))
            rec["train_s"] = time.time() - t0
            rec["vram_train_peak_gb"] = card.peak_since(m)
            with (out / "steps.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if rec.get("stepped"):
                ad = out / f"adapter-step{step}"
                model.save_pretrained(str(ad))
                lora_gguf = out / f"lora-step{step}.f16.gguf"
                rec["convert_s"] = convert(ad, lora_gguf)
            r = rec.get("ratio", {})
            print(f"[step {step}] reward {rec['reward_mean']:.3f} dead {rec['dead_groups']}/{rec['groups']} "
                  f"trained {rec['trained_tokens']} tok | ratio |d| {r.get('mean_abs', float('nan')):.4f} "
                  f"out {r.get('frac_outside_0.5_2', float('nan')):.3%} capped {r.get('frac_capped', float('nan')):.3%} "
                  f"| gen {rec['gen_s']:.0f}s {rec['tps']:.0f} tok/s train {rec['train_s']:.0f}s "
                  f"| vram gen {rec['vram_gen_peak_gb']:.1f} train {rec['vram_train_peak_gb']:.1f} GB "
                  f"| stepped {rec['stepped'] if 'stepped' in rec else False}", flush=True)
            if rec.get("guard"):
                print(f"[guard] STOP: {rec['guard']}", flush=True)
                break
    finally:
        card.stop()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)                      # ROCm torch can hang at interpreter exit on Windows
