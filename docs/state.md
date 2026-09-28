# GRPO - state

[2026-09-28] Home plan (supersedes anything older)

## Decisions
- One model class: **Qwen3.5**. Pick one size (4B default), load once, leave it running.
  Every trained version is a LoRA adapter hot-loaded via `/v1/load_lora_adapter`. No restarts.
- vLLM in Docker, not LM Studio. It crashes sometimes; the loop below makes crashes cheap.
- Eval set at home is `evals/data/sets/baseline_20.json` (5 LCB, 5 BFCL, 5 When2Call, 5 holdout).
  It is a strict subset of baseline_v1 with the same item IDs. baseline_v1 (118 items) is the
  full set; do not regenerate either one.

## Qwen3.5-2B result in evals/results/qwen3.5-2b is NOT a baseline
Every BFCL / When2Call / holdout attempt is `request-error` (server was down mid-run).
LCB got 2/30. Treat that folder as a smoke test of the harness, not a number. Re-run.

## The five steps at home
1. `git clone https://github.com/juxstin1/grpo-local.git`
2. Launch vLLM (Docker Desktop must be up):
   ```powershell
   docker run -d --name vllm-qwen --restart unless-stopped --gpus all -p 8000:8000 `
     -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True `
     -v hf-cache:/root/.cache/huggingface -v vllm-compile:/root/.cache/vllm `
     vllm/vllm-openai:latest Qwen/Qwen3.5-4B `
     --enable-lora --max-num-seqs 64 --max-model-len 32768 --gpu-memory-utilization 0.85
   ```
   First launch downloads the weights into hf-cache (10-30 min). Wait for `curl http://127.0.0.1:8000/health` = 200.
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
