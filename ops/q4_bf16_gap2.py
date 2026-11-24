"""Q4-vs-bf16 log-prob gap, round 2: the generator the GRPO loop would use, on multi-round episodes.

Generators (llama-server b8683 Vulkan, one at a time):
  lora   = base Q4_K_M + SFT-1008 LoRA converted to GGUF (ops/convert_lora_qwen35.py), --lora   <- the loop's path
  merged = 4b-r3sft.Q4_K_M.gguf (same LoRA merged before quantizing)                         <- shipped model
Scorer: bf16 Qwen3.5-4B + the PEFT adapter on the exact token ids each generator sampled.

Episodes: 3 train-seed tasks x 3 families (file_work, doc_make, office_draft) x 4 attempts, per generator.
Each round: HF-render the conversation (Bionic native system prompt + tools; earlier assistant turns carry
reasoning_content and dict tool-call args), send token ids, sample with a plain chain (temperature 1.0 only,
fresh random 31-bit seed per request), parse Qwen XML tool calls, run them with the gym's Bionic tool mirror,
append the results, continue. Up to MAX_ROUNDS rounds; round >= 1 prompts contain tool results.
Also: speed of log-prob return vs plain sampling vs --backend-sampling, server (re)start time, and card VRAM
via the Windows GPU counters (torch's mem_get_info cannot see llama-server's Vulkan allocations).

  .venv\\Scripts\\python.exe -u ops\\q4_bf16_gap2.py      -> runs/pilots/q4-bf16-gap-1010/gap2/
"""
from __future__ import annotations

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

ROOT = HERE.parent
OUT = G.OUT / ("gap2-smoke" if os.environ.get("GAP2_SMOKE") == "1" else "gap2")
LORA_GGUF = G.OUT / "lora-sft-1008.f16.gguf"
BASE_Q4 = ROOT / "q4" / "out" / "4b-base.Q4_K_M.gguf"
MERGED_Q4 = ROOT / "q4" / "out" / "4b-r3sft.Q4_K_M.gguf"
CTX = 16384
N_PREDICT = 2048
MAX_ROUNDS = 3
SMOKE = os.environ.get("GAP2_SMOKE") == "1"           # 1 task/family, 1 attempt: plumbing check only
TASKS_PER_FAMILY = 1 if SMOKE else 3
ATTEMPTS = 1 if SMOKE else 4
RNG = random.SystemRandom()


# ---------------------------------------------------------------- VRAM (whole card, all APIs)

class CardVram:
    """Windows `GPU Adapter Memory\\Dedicated Usage`, sampled every second by a PowerShell child."""
    def __init__(self, path: Path):
        self.path = path
        ps = ("while ($true) { $v = ((Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples "
              "| Measure-Object CookedValue -Maximum).Maximum; "
              f"Add-Content -Path '{path}' -Value ([string]([math]::Round($v/1e9,3))); Start-Sleep -Milliseconds 500 }}")
        path.write_text("", encoding="utf-8")
        if os.name != "nt":                         # Linux pod: nvidia-smi, MiB -> GB
            sh = (f"while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits "
                  f"| awk '{{printf \"%.3f\\n\", $1*1048576/1e9}}' >> '{path}'; sleep 0.5; done")
            self.proc = subprocess.Popen(["sh", "-c", sh], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def mark(self) -> int:
        return len(self.path.read_text(encoding="utf-8").split())

    def peak_since(self, mark: int) -> float:
        vals = [float(x) for x in self.path.read_text(encoding="utf-8").split()[mark:]]
        return max(vals) if vals else float("nan")

    def stop(self):
        self.proc.kill()


# ---------------------------------------------------------------- server

def start(arm: str, extra: list[str] | None = None, tag: str = "") -> tuple[subprocess.Popen, float]:
    model = BASE_Q4 if arm == "lora" else MERGED_Q4
    args = [str(G.SERVER), "-m", str(model), "--host", "127.0.0.1", "--port", str(G.PORT), "-ngl", "99",
            "-c", str(CTX), "-np", "1", "--no-webui"]
    if arm == "lora":
        args += ["--lora", str(LORA_GGUF)]
    args += extra or []
    log = OUT / f"server-{arm}{tag}.log"
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    for _ in range(600):
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


def stop(proc):
    proc.kill()
    proc.wait()
    time.sleep(1)


def complete(prompt_ids: list[int], seed: int, n_predict: int = N_PREDICT, **over) -> dict:
    body = {"prompt": prompt_ids, "n_predict": n_predict, "seed": seed, "cache_prompt": True,
            "samplers": ["temperature"], "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "n_probs": 1, "post_sampling_probs": False, "return_tokens": True, **over}
    body = {k: v for k, v in body.items() if v is not None}     # None = use the server's default
    j = requests.post(f"{G.URL}/completion", json=body, timeout=3600).json()
    if "error" in j:
        raise RuntimeError(j["error"])
    probs = j.get("completion_probabilities") or []
    return {"gen_ids": j["tokens"], "lp_q4": [p["logprob"] for p in probs], "stop_type": j.get("stop_type"),
            "tps": j.get("timings", {}).get("predicted_per_second", 0.0)}


# ---------------------------------------------------------------- tasks and episodes

def gym():
    os.chdir(G.GYM)
    sys.path.insert(0, str(G.GYM))
    import dialect, native, repo_gen  # noqa: E401
    from config import Config
    from eval import make_task
    from runner import Runner
    from workspace import Workspace
    from q4_probe_raw import holdout_seeds
    return dict(dialect=dialect, native=native, repo_gen=repo_gen, Config=Config, make_task=make_task,
                Runner=Runner, Workspace=Workspace, sealed=holdout_seeds())


def pick_tasks(M) -> list[tuple[str, dict]]:
    out = []
    for fam in G.FAMILIES:
        seed, got = 44000, 0
        while got < TASKS_PER_FAMILY:
            seed += 1
            if (fam, seed) in M["sealed"]:
                continue
            specs = M["repo_gen"].make_specs(seed, fam)
            if not specs:
                continue
            out.append((fam, specs[got % len(specs)]))      # rotate templates
            got += 1
    return out


def render(tok, msgs, tools) -> list[int]:
    ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def episode(M, tok, fam, spec, arm, attempt) -> list[dict]:
    native = M["native"]
    cfg = M["Config"]()
    cfg.episode.tool_dialect, cfg.episode.bionic_profile = "bionic", "default"
    ws = M["Workspace"](cfg.episode.data_root / "workspaces" / f"gap2-{arm}-{fam}-{attempt}", M["Runner"]("docker"))
    ws.reset()
    M["repo_gen"].write_files(M["repo_gen"].find_spec(spec["seed"], fam, spec["name"])["files"], ws.root)
    task = M["dialect"].prepare(cfg, M["make_task"](spec), ws)
    tools = native.native_tools()
    msgs = [{"role": "system", "content": native.system_prompt(tools)},
            {"role": "user", "content": task.prompt.strip()}]
    rounds, counters = [], {}
    for r in range(MAX_ROUNDS):
        ids = render(tok, msgs, tools)
        if len(ids) + 256 > CTX:
            break
        g = complete(ids, seed=RNG.getrandbits(31), n_predict=min(N_PREDICT, CTX - len(ids)))
        rounds.append({"family": fam, "seed": spec["seed"], "name": spec["name"], "arm": arm, "attempt": attempt,
                       "round": r, "prompt_ids": ids, **g})
        text = tok.decode(g["gen_ids"], skip_special_tokens=False).replace("<|im_end|>", "")
        reasoning, content = (text.split("</think>", 1) if "</think>" in text else (text, ""))
        reasoning, content = reasoning.strip(), content.strip()
        calls, prose = [], content
        rec = native.recover_from_reasoning(content, tools) if "<tool_call>" in content else None
        if rec and rec["kind"] == "tool_call":
            calls, prose = rec["toolCalls"], rec["reasoning"]
        elif not content and reasoning:                       # Bionic fix: call stuck in unclosed <think>
            rec = native.recover_from_reasoning(reasoning, tools)
            if rec and rec["kind"] == "tool_call":
                calls, reasoning = rec["toolCalls"], rec["reasoning"]
        rounds[-1]["n_calls"] = len(calls)
        if not calls or g["stop_type"] != "eos":
            break
        msgs.append({"role": "assistant", "content": prose, "reasoning_content": reasoning,
                     "tool_calls": [{"type": "function", "function": {"name": c["name"],
                                                                      "arguments": _args(c["args"])}}
                                    for c in calls]})
        for c in calls:
            msgs.append({"role": "tool", "content": native.execute(ws, c["name"], c["args"], counters)})
    return rounds


def _args(s: str) -> dict:
    try:
        v = json.loads(s) if s else {}
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


# ---------------------------------------------------------------- speed check

def speed_check(prompt_ids: list[int], card: CardVram, report: dict):
    """Same prompt, same seed: does returning log-probs or the plain sampler cost the speed?"""
    res = {}
    seed = 123456789
    for label, over in [("plain T=1, n_probs=1", {}),
                        ("plain T=1, n_probs=0", {"n_probs": 0}),
                        ("Bionic chain (top_k 40, top_p .95, min_p .05, T .7), n_probs=0",
                         {"n_probs": 0, "samplers": None, "temperature": 0.7, "top_k": 40, "top_p": 0.95,
                          "min_p": 0.05}),
                        ("Bionic chain, n_probs=1",
                         {"samplers": None, "temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05})]:
        g = complete(prompt_ids, seed, n_predict=512, ignore_eos=True, **over)
        res[label] = {"tps": g["tps"], "n": len(g["gen_ids"]), "gen_ids": g["gen_ids"], "lp": g["lp_q4"]}
        print(f"[speed] {label}: {g['tps']:.1f} tok/s", flush=True)
    report["speed"] = {k: {"tps": v["tps"], "n": v["n"]} for k, v in res.items()}
    report["speed"]["plain n_probs 1 vs 0 same tokens"] = (res["plain T=1, n_probs=1"]["gen_ids"]
                                                           == res["plain T=1, n_probs=0"]["gen_ids"])
    return res["plain T=1, n_probs=1"]


def backend_sampling_check(prompt_ids, ref: dict, report: dict):
    proc, t = start("lora", ["--backend-sampling"], tag="-bs")
    try:
        g = complete(prompt_ids, 123456789, n_predict=512, ignore_eos=True)
    finally:
        stop(proc)
    same = g["gen_ids"] == ref["gen_ids"]
    k = 0
    while k < min(len(g["gen_ids"]), len(ref["gen_ids"])) and g["gen_ids"][k] == ref["gen_ids"][k]:
        k += 1
    dl = [abs(a - b) for a, b in zip(g["lp_q4"][:k], ref["lp"][:k])]
    report["backend_sampling"] = {"tps": g["tps"], "tps_without": ref["tps"], "same_tokens": same,
                                  "common_prefix": k, "max_abs_lp_diff_on_prefix": max(dl) if dl else None}
    print(f"[speed] --backend-sampling: {g['tps']:.1f} tok/s, same tokens {same}, prefix {k}, "
          f"max |dlp| {report['backend_sampling']['max_abs_lp_diff_on_prefix']}", flush=True)


# ---------------------------------------------------------------- scoring

def stats(diffs: list[float]) -> dict:
    if not diffs:
        return {"n_tokens": 0}
    return G.stats(diffs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.BASE_MODEL)
    M = gym()
    tasks = pick_tasks(M)
    print("[tasks]", [(f, s["seed"], s["name"]) for f, s in tasks], flush=True)
    card = CardVram(OUT / "vram.txt")
    report: dict = {"lora_gguf": LORA_GGUF.name, "base_q4": BASE_Q4.name, "merged_q4": MERGED_Q4.name,
                    "max_rounds": MAX_ROUNDS, "attempts": ATTEMPTS, "restart_s": {}}
    rows: list[dict] = []
    try:
        time.sleep(4)
        report["vram_idle_gb"] = card.peak_since(0)
        for arm in ("lora", "merged"):
            m = card.mark()
            proc, t = start(arm)
            report["restart_s"][arm] = t
            report[f"vram_server_{arm}_gb"] = card.peak_since(m)
            print(f"[server] {arm} healthy in {t:.1f}s", flush=True)
            try:
                if arm == "lora":
                    ref = speed_check(render(tok, [{"role": "system", "content": M["native"].system_prompt(
                        M["native"].native_tools())}, {"role": "user", "content": "Write a 300-word story."}],
                        M["native"].native_tools()), card, report)
                for fam, spec in tasks:
                    for a in range(ATTEMPTS):
                        ep = episode(M, tok, fam, spec, arm, a)
                        rows += ep
                        print(f"[ep] {arm} {fam} {spec['seed']} #{a}: rounds {len(ep)} "
                              f"tok {[len(r['gen_ids']) for r in ep]} "
                              f"{sum(r['tps'] for r in ep) / max(1, len(ep)):.0f} tok/s", flush=True)
            finally:
                stop(proc)
        # restart time with files warm in the OS cache (what a per-step reload costs)
        proc, t = start("lora", tag="-restart")
        report["restart_s"]["lora_warm"] = t
        stop(proc)
        prompt0 = rows[0]["prompt_ids"]
        backend_sampling_check(prompt0, ref, report)
        distinct = {}
        for fam, spec in tasks:
            for arm in ("lora", "merged"):
                seqs = {tuple(r["gen_ids"]) for r in rows if r["family"] == fam and r["seed"] == spec["seed"]
                        and r["arm"] == arm and r["round"] == 0}
                distinct[f"{arm} {fam} {spec['seed']}"] = len(seqs)
        report["distinct_round0_of_4"] = distinct
        (OUT / "rounds-raw.jsonl").write_text("\n".join(json.dumps({k: v for k, v in r.items()}) for r in rows),
                                              encoding="utf-8")

        # bf16 scoring, with the loop's generator resident (as during training)
        import torch
        proc, _ = start("lora", tag="-resident")
        try:
            m = card.mark()
            model = G.load_policy(trainable=True)
            model.eval()
            report["vram_server_plus_bf16_gb"] = card.peak_since(m)
            for i, r in enumerate(rows):
                r["lp_bf16"] = G.bf16_logprobs(model, r["prompt_ids"], r["gen_ids"])
                r["diff"] = [b - q for b, q in zip(r["lp_bf16"], r["lp_q4"])]
                if i % 10 == 0:
                    print(f"[score] {i + 1}/{len(rows)}", flush=True)
            report["vram_peak_scoring_gb"] = card.peak_since(m)
            torch.cuda.empty_cache()

            # one training-shaped backward next to the resident server: longest, then the median sequence
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            lens = sorted(rows, key=lambda r: len(r["prompt_ids"]) + len(r["gen_ids"]))
            report["backward"] = []
            for r in (lens[-1], lens[len(lens) // 2]):
                m = card.mark()
                torch.cuda.reset_peak_memory_stats()
                n = len(r["prompt_ids"]) + len(r["gen_ids"])
                try:
                    ids = torch.tensor([r["prompt_ids"] + r["gen_ids"]], device="cuda")
                    g = len(r["gen_ids"])
                    logits = model(input_ids=ids, logits_to_keep=g + 1).logits[0, :-1].float()
                    lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r["gen_ids"], device="cuda")[:, None])
                    (-lp.mean()).backward()
                    model.zero_grad(set_to_none=True)
                    del logits, lp, ids
                    ok = {"ok": True}
                except torch.OutOfMemoryError as e:
                    ok = {"ok": False, "error": str(e)[:200]}
                torch.cuda.empty_cache()
                time.sleep(1.5)
                report["backward"].append({"seq_tokens": n, **ok,
                                           "torch_peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                                           "card_peak_gb": card.peak_since(m)})
                print("[backward]", report["backward"][-1], flush=True)
        finally:
            stop(proc)
    finally:
        card.stop()

    for arm in ("lora", "merged"):
        A = [r for r in rows if r["arm"] == arm and "diff" in r]
        report[arm] = {"overall": stats([d for r in A for d in r["diff"]]),
                       "by_family": {f: stats([d for r in A if r["family"] == f for d in r["diff"]])
                                     for f in G.FAMILIES},
                       "by_round": {f"round {k}" if k == 0 else "round >=1": stats(
                           [d for r in A if (r["round"] == 0) == (k == 0) for d in r["diff"]]) for k in (0, 1)},
                       "by_family_round": {f"{f} r{'0' if k == 0 else '1+'}": stats(
                           [d for r in A if r["family"] == f and (r["round"] == 0) == (k == 0) for d in r["diff"]])
                           for f in G.FAMILIES for k in (0, 1)},
                       "per_sequence_kl": sorted(-sum(r["diff"]) / len(r["diff"]) for r in A if r["diff"])}
    report["card_total_gb"] = 17.096
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with (OUT / "rounds.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("arm", "family", "seed", "attempt", "round", "stop_type",
                                                   "gen_ids", "lp_q4", "lp_bf16") if k in r}) + "\n")
    print("[done]", json.dumps({a: report[a]["overall"] for a in ("lora", "merged")}), flush=True)


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
