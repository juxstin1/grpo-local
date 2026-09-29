# GRPO - state

**Start here: [HANDOFF-2026-09-29.md](HANDOFF-2026-09-29.md)** (state, what's running, gotchas). **Next up: [NEXT-workspace-gym.md](NEXT-workspace-gym.md)** (real file tools, sandbox, Find gym). Best adapter: r3.

**Canon pipeline + rules: `SKILL.md` -> "Training pipeline (canon)"** (baseline -> SFT own wins -> re-baseline -> GRPO; DPO later for uncheckable).

[2026-09-28] Home plan (supersedes anything older)

## Decisions
- One model class: **Qwen3.5**. Pick one size (4B default), load once, leave it running.
  Every trained version is a LoRA adapter hot-loaded via `/v1/load_lora_adapter`. No restarts.
- vLLM in Docker, not LM Studio. Home GPU is AMD (ROCm image + device flags, see step 2). It crashes sometimes; the loop below makes crashes cheap.
- Eval set at home is `evals/data/sets/baseline_20.json` (5 LCB, 5 BFCL, 5 When2Call, 5 holdout).
  It is a strict subset of baseline_v1 with the same item IDs. baseline_v1 (118 items) is the
  full set; do not regenerate either one.

## Qwen3.5-2B result in evals/results/qwen3.5-2b is NOT a baseline
Every BFCL / When2Call / holdout attempt is `request-error` (server was down mid-run).
LCB got 2/30. Treat that folder as a smoke test of the harness, not a number. Re-run.

## The five steps at home
1. `git clone https://github.com/juxstin1/grpo-local.git`
2. Launch vLLM. **Working home command (verified 2026-09-28, RX 9070 XT via Docker Desktop /dev/dxg):**
   Image `vllm-dflash-wsl:v0.27.1` (built from `C:\Users\juxxs\vllm`). Stock
   `vllm/vllm-openai-rocm:v0.30.0` crashes on /dev/dxg; /dev/kfd does not exist here.
   ```powershell
   $shim = "C:\Users\juxxs\vllm\.docker-rocdxg"
   docker run -d --name vllm-qwen35 --restart unless-stopped --device /dev/dxg `
     --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host --shm-size 8g -p 8000:8000 `
     -e HSA_ENABLE_DXG_DETECTION=1 -e VLLM_ROCM_USE_AITER=0 -e VLLM_USE_V2_MODEL_RUNNER=0 `
     -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True -e HF_HUB_OFFLINE=1 `
     -v "${shim}\libdxcore.so:/usr/lib/libdxcore.so:ro" `
     -v "${shim}\librocdxg.so:/usr/lib/librocdxg.so:ro" `
     -v "${shim}\dids.conf:/usr/share/rocdxg/dids.conf:ro" `
     -v vllm_huggingface-cache:/root/.cache/huggingface `
     -v "C:\Users\juxxs\vllm\grpo-local:/work:ro" `
     vllm-dflash-wsl:v0.27.1 Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000 `
     --attention-backend ROCM_ATTN --enforce-eager `
     --max-model-len 16384 --max-num-seqs 40 --gpu-memory-utilization 0.85 `
     --enable-prefix-caching --reasoning-parser qwen3 `
     --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 8
   ```
   Boot ~4 min (weights 8.0 GiB, KV 4.75 GiB). Thinking arrives in `reasoning_content`.
   vLLM holds 85% of VRAM: stop it before `train.py` (the VRAM guard will refuse otherwise).
   Adapters: repo is mounted at `/work`; load with
   `POST /v1/load_lora_adapter {"lora_name":"<name>","lora_path":"/work/lora-Qwen3.5-4B-sft-<name>"}`
   and request `"model": "<name>"`.

   Old generic guess, kept for reference only (/dev/kfd does NOT exist on this box):
   ```powershell
   docker run -d --name vllm-qwen --restart unless-stopped `
     --device=/dev/kfd --device=/dev/dri --group-add video `
     --ipc=host --security-opt seccomp=unconfined -p 8000:8000 `
     -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True `
     -v hf-cache:/root/.cache/huggingface -v vllm-compile:/root/.cache/vllm `
     rocm/vllm:latest vllm serve Qwen/Qwen3.5-4B `
     --enable-lora --max-num-seqs 64 --max-model-len 32768 --gpu-memory-utilization 0.85
   ```
   The runner does not care how vLLM is started; it only needs `http://127.0.0.1:8000/health` = 200.
   Once a command works, paste it here verbatim so it is never rediscovered.
   First launch downloads the weights into hf-cache (10-30 min).
3. Run the 20-question eval (resumable; rerun the same command after any crash):
   ```powershell
   python evals\eval_runner.py --model Qwen/Qwen3.5-4B --slug qwen3.5-4b-base `
     --set evals\data\sets\baseline_20.json
   ```
4. Read `evals/results/<slug>/*.summary.json`. Per-bucket numbers only, never the blend.
5. Append one entry to `runs/LOG.md`. Commit results + log, push.

## Crash resilience (already wired)
- Runner resumes by (item_id, attempt); `request-error` lines are retried, not skipped.
- Runner polls `/health` before every request and blocks until the server is back (15 min cap).
- `--restart unless-stopped` brings the container back on its own.
- If Docker Desktop itself dies it is usually WSL2 RAM: raise `memory=` in `C:\Users\<you>\.wslconfig`.

## Not done
- No real baseline exists yet for any model.
- Training loop (train.py / rl.py) untested against a live server.
