# sandbox-gym — Agent Environment Rules

## Environment (non-negotiable)
- Windows host, PowerShell only.
- NEVER run commands inside WSL (no `wsl`, no /home/... paths, no Linux-side pip/hf).
- All Linux tooling runs via `docker run` from PowerShell.
- Model downloads happen inside containers into the `hf-cache` Docker volume.
- Always use http://127.0.0.1, never localhost.
- The shared torch venv is `..\.venv\Scripts\python.exe` (ROCm torch,
  peft, transformers 5.x). Use it for eval/collect/train.

## vLLM bake-off loop
Do NOT run a separate model-download step. vLLM downloads the model itself on
first launch, straight into the `hf-cache` Docker volume.

```powershell
docker run --gpus all -p 8000:8000 `
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-compile:/root/.cache/vllm `
  vllm/vllm-openai:latest `
  Qwen/Qwen3.5-4B `
  --enable-lora --max-num-seqs 64 --max-model-len 32768
```

Swap the model argument per run; every model after the first download is cached.
`--enable-lora` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` let trained LoRA
adapters hot-swap via `/v1/load_lora_adapter` without restarting. The eval/collect
harness only needs `/health` = 200 on http://127.0.0.1:8000.

For the home ROCm image (`vllm-dflash-wsl:v0.27.1`), the verified launch command
lives in `..\docs\state.md`; this harness is server-agnostic.

## Run_tests sandbox
`build`/`run_tests` run in a throwaway Docker container (`python:3.12-slim`),
no network, only the episode workspace mounted. The generated suites are
self-contained (no pytest needed). `--runner host` is for `selftest.py` only —
never for a real eval or GRPO reward.