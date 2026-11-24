"""Q4-vs-bf16 log-prob gap: can a Q4 llama-server generate GRPO rollouts for a bf16 trainer?

Measurement only (no training loop). Policy on both sides = Qwen3.5-4B + SFT-1008 LoRA:
  Q4   = q4/out/4b-r3sft.Q4_K_M.gguf (that LoRA merged, then Q4_K_M) on llama-server
  bf16 = HF base + the PEFT adapter, scoring the exact token ids llama-server sampled
The prompt is rendered once with the HF tokenizer (Bionic native system prompt + tools + task) and sent to
llama-server as token ids; generated ids come back with their pre-sampler log-probs. Sampler is plain:
only `temperature` (1.0) in the chain, so samples come from the distribution the log-probs describe.
First assistant round only (thinking + reply/tool call), no tool execution.

  .venv\\Scripts\\python.exe ops\\q4_bf16_gap.py            -> runs/pilots/q4-bf16-gap-1010/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GYM = Path(os.environ.get("GRPO_GYM_DIR", ROOT.parent / "grpo-local-wt-filetools" / "sandbox-gym"))
OUT = ROOT / "runs" / "pilots" / "q4-bf16-gap-1010"
SERVER = Path(os.environ.get("LLAMA_SERVER", Path.home() / "Documents/workspace/llamacpp-prebuilt/llama-server.exe"))
GGUF = ROOT / "q4" / "out" / "4b-r3sft.Q4_K_M.gguf"
ADAPTER = ROOT / "adapters-sft-1008" / "lora-Qwen3.5-4B-sft-hosted-distil-1007"
BASE_MODEL = "Qwen/Qwen3.5-4B"
PORT = 8093
URL = f"http://127.0.0.1:{PORT}"
FAMILIES = ["file_work", "doc_make", "office_draft"]
LN2 = math.log(2.0)


def gpu_used_gb() -> float:
    """Dedicated VRAM in use on the card, all processes (Vulkan llama-server + ROCm torch)."""
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


class Peak:
    """Samples whole-card VRAM use in the background; .max is the peak seen."""
    def __init__(self):
        self.max, self._stop = 0.0, False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop:
            self.max = max(self.max, gpu_used_gb())
            time.sleep(0.05)

    def stop(self) -> float:
        self._stop = True
        self._t.join()
        return self.max


def build_prompts(tok, seed0: int) -> list[dict]:
    """One train-seed task per family (holdout seeds refused), rendered as Bionic native mode sends it."""
    os.chdir(GYM)
    sys.path.insert(0, str(GYM))
    import dialect
    import native
    import repo_gen
    from config import Config
    from eval import make_task
    from runner import Runner
    from workspace import Workspace
    sys.path.insert(0, str(HERE))
    from q4_probe_raw import holdout_seeds

    sealed = holdout_seeds()
    tools = native.native_tools()
    out = []
    for fam in FAMILIES:
        seed = seed0
        while True:
            seed += 1
            if (fam, seed) in sealed:
                continue
            specs = repo_gen.make_specs(seed, fam)
            if specs:
                break
        spec = specs[0]
        cfg = Config()
        cfg.episode.tool_dialect, cfg.episode.bionic_profile = "bionic", "default"
        ws = Workspace(cfg.episode.data_root / "workspaces" / f"gap-{fam}", Runner("docker"))
        ws.reset()
        repo_gen.write_files(repo_gen.find_spec(spec["seed"], fam, spec["name"])["files"], ws.root)
        task = dialect.prepare(cfg, make_task(spec), ws)
        msgs = native.to_api_messages(native.system_prompt(tools), task.prompt.strip(), [])
        ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        out.append({"family": fam, "seed": seed, "name": spec["name"], "prompt_ids": list(ids)})
    return out


def check_vocab(tok) -> dict:
    """GGUF tokenizer.ggml.tokens vs HF ids, token by token."""
    sys.path.insert(0, str(Path.home() / "llama.cpp-source" / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(GGUF))
    f = r.fields["tokenizer.ggml.tokens"]
    gg = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
    hf = tok.convert_ids_to_tokens(list(range(len(gg))))
    mism = [i for i, (a, b) in enumerate(zip(gg, hf)) if a != b]
    return {"gguf_tokens": len(gg), "hf_len_tokenizer": len(tok), "mismatches": len(mism),
            "first_mismatches": [(i, gg[i], hf[i]) for i in mism[:5]]}


def start_server(log: Path) -> subprocess.Popen:
    args = [str(SERVER), "-m", str(GGUF), "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99",
            "-c", "16384", "-np", "1", "--no-webui"]
    proc = subprocess.Popen(args, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    for _ in range(600):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited ({proc.returncode}); see {log}")
        try:
            if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"llama-server not healthy; see {log}")


def generate(prompt_ids: list[int], seed: int, n_predict: int) -> dict:
    body = {"prompt": prompt_ids, "n_predict": n_predict, "seed": seed, "cache_prompt": True,
            "samplers": ["temperature"], "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "dry_multiplier": 0.0, "xtc_probability": 0.0,
            "n_probs": 1, "post_sampling_probs": False, "return_tokens": True}
    r = requests.post(f"{URL}/completion", json=body, timeout=1800)
    r.raise_for_status()
    j = r.json()
    probs = j["completion_probabilities"]
    ids = [p["id"] for p in probs]
    if j.get("tokens") and j["tokens"] != ids:
        raise SystemExit("token ids in `tokens` and `completion_probabilities` disagree")
    return {"gen_ids": ids, "lp_q4": [p["logprob"] for p in probs], "stop_type": j.get("stop_type"),
            "n_prompt": j.get("tokens_evaluated"), "timings": j.get("timings", {})}


def load_policy(trainable: bool):
    import torch
    from peft import PeftModel
    sys.path.insert(0, str(GYM))
    from train import _model_class
    base = _model_class(BASE_MODEL).from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True).to("cuda")
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=trainable)
    return model


def bf16_logprobs(model, prompt_ids: list[int], gen_ids: list[int]) -> list[float]:
    import torch
    ids = torch.tensor([prompt_ids + gen_ids], device="cuda")
    g = len(gen_ids)
    with torch.no_grad():
        logits = model(input_ids=ids, logits_to_keep=g + 1).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(gen_ids, device="cuda")[:, None])[:, 0]
    return lp.tolist()


def stats(diffs: list[float]) -> dict:
    a = sorted(abs(d) for d in diffs)
    n = len(diffs)
    return {"n_tokens": n, "mean_diff": sum(diffs) / n, "mean_abs": sum(a) / n,
            "p95_abs": a[min(n - 1, int(0.95 * n))], "max_abs": a[-1],
            "frac_ratio_outside_0.5_2": sum(1 for d in a if d > LN2) / n,
            # samples come from Q4, so E_q4[log q4 - log bf16] = KL(q4 || bf16) per token (k1 estimator)
            "kl_q4_bf16_per_token": -sum(diffs) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=1536)
    ap.add_argument("--seed0", type=int, default=43000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    report: dict = {"gguf": GGUF.name, "adapter": ADAPTER.name, "server": str(SERVER),
                    "sampler": "samplers=[temperature], T=1.0; log-probs pre-sampling (n_probs=1)"}
    report["vocab"] = check_vocab(tok)
    print("[vocab]", report["vocab"], flush=True)

    prompts = build_prompts(tok, a.seed0)
    print("[prompts]", [(p["family"], p["seed"], len(p["prompt_ids"])) for p in prompts], flush=True)

    report["vram_idle_gb"] = gpu_used_gb()
    proc = start_server(OUT / "llama-server.log")
    try:
        report["vram_server_gb"] = gpu_used_gb()
        ver = requests.get(f"{URL}/props", timeout=10).json().get("build_info")
        report["server_build"] = ver
        rows = []
        for p in prompts:
            for k in range(a.attempts):
                g = generate(p["prompt_ids"], seed=1000 + k, n_predict=a.n_predict)
                if g["n_prompt"] is not None and g["n_prompt"] not in (len(p["prompt_ids"]), 0) and k == 0:
                    print(f"[warn] server evaluated {g['n_prompt']} prompt tokens vs {len(p['prompt_ids'])}")
                rows.append({**p, "attempt": k, **g})
                print(f"[gen] {p['family']} #{k}: {len(g['gen_ids'])} tok, stop={g['stop_type']}, "
                      f"{g['timings'].get('predicted_per_second', 0):.0f} tok/s", flush=True)

        peak = Peak()
        model = load_policy(trainable=True)
        report["vram_server_plus_bf16_gb"] = gpu_used_gb()
        model.eval()
        for r in rows:
            r["lp_bf16"] = bf16_logprobs(model, r["prompt_ids"], r["gen_ids"])
            r["diff"] = [b - q for b, q in zip(r["lp_bf16"], r["lp_q4"])]
            s = stats(r["diff"])
            r["stats"] = s
            print(f"[score] {r['family']} #{r['attempt']}: mean {s['mean_diff']:+.4f} |d| {s['mean_abs']:.4f} "
                  f"p95 {s['p95_abs']:.3f} max {s['max_abs']:.2f} out {s['frac_ratio_outside_0.5_2']:.3%}",
                  flush=True)
        report["vram_peak_scoring_gb"] = peak.stop()

        # one training-shaped backward next to the server: longest sequence, LoRA grads, checkpointing
        longest = max(rows, key=lambda r: len(r["prompt_ids"]) + len(r["gen_ids"]))
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        torch.cuda.reset_peak_memory_stats()
        peak = Peak()
        try:
            ids = torch.tensor([longest["prompt_ids"] + longest["gen_ids"]], device="cuda")
            g = len(longest["gen_ids"])
            logits = model(input_ids=ids, logits_to_keep=g + 1).logits[0, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(longest["gen_ids"], device="cuda")[:, None])
            (-lp.mean()).backward()
            report["backward"] = {"ok": True, "seq_tokens": ids.shape[1],
                                  "torch_peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9}
        except torch.OutOfMemoryError as e:
            report["backward"] = {"ok": False, "seq_tokens": len(longest["prompt_ids"]) + len(longest["gen_ids"]),
                                  "error": str(e)[:300]}
        report["backward"]["card_peak_gb"] = peak.stop()
        report["card_total_gb"] = torch.cuda.mem_get_info()[1] / 1e9
    finally:
        proc.kill()
        proc.wait()

    all_d = [d for r in rows for d in r["diff"]]
    report["overall"] = stats(all_d)
    report["per_family"] = {f: stats([d for r in rows if r["family"] == f for d in r["diff"]]) for f in FAMILIES}
    report["per_attempt"] = [{"family": r["family"], "seed": r["seed"], "attempt": r["attempt"],
                              "prompt_tokens": len(r["prompt_ids"]), "stop": r["stop_type"], **r["stats"]}
                             for r in rows]
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "tokens.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("family", "seed", "attempt", "gen_ids", "lp_q4", "lp_bf16")}) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "per_family", "backward")}, indent=1))


if __name__ == "__main__":
    main()
