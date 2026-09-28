# GRPO - Agent Environment Rules

## Environment rules (non-negotiable)
- Windows host, PowerShell only.
- NEVER run commands inside WSL (no `wsl`, no /home/... paths, no Linux-side pip/hf).
- All Linux tooling runs via `docker run` from PowerShell.
- Model downloads happen inside containers into the `hf-cache` Docker volume.
- Always use http://127.0.0.1, never localhost.

## vLLM bake-off loop
Do NOT run a separate model-download step. vLLM downloads the model itself on first
launch, straight into the `hf-cache` Docker volume.

```powershell
docker run --gpus all -p 8000:8000 `
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-compile:/root/.cache/vllm `
  vllm/vllm-openai:latest `
  Qwen/Qwen3.5-2B `
  --enable-lora --max-num-seqs 64 --max-model-len 32768
```

Swap the model argument per run. Every model after the first download is cached in
`hf-cache`, so no `hf` CLI, no WSL, no DNS drama. `vllm-compile` persists the
torch.compile cache across container recreates, so the same model boots fast on
re-runs. `--max-num-seqs 64 --max-model-len 32768` shrink CUDA graph capture for
an eval box. `--enable-lora` + runtime adapter updating
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) let trained LoRA adapters hot-swap via
`/v1/load_lora_adapter` without restarting.