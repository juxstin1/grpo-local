# Serve r3 on the home GPU (Docker vLLM, gfx1201 /dev/dxg) at http://127.0.0.1:8000, adapter name "r3".
# Same engine args as the old vllm-qwen35 container; adapter mounted read-only from adapter-backups.
# Usage: .\ops\serve-r3-local.ps1        (waits until /v1/models lists r3)
param([string]$Name = "vllm-r3-local", [int]$Port = 8000)
$ErrorActionPreference = "Stop"
$vllm = "C:\Users\juxxs\vllm"
$ad = "$vllm\adapter-backups\2026-09-29\lora-Qwen3.5-4B-sft-r3"
$sha = (Get-FileHash "$ad\adapter_model.safetensors").Hash.ToLower()
if (-not $sha.StartsWith("f7832d628ae0")) { throw "r3 adapter sha mismatch: $sha" }
if (docker ps -q -f "name=^$Name$") { "[serve] $Name already running" } else {
    docker run -d --name $Name --device /dev/dxg --cap-add SYS_PTRACE --security-opt seccomp=unconfined `
        --ipc host --shm-size 8g -p "${Port}:8000" `
        -e HSA_ENABLE_DXG_DETECTION=1 -e VLLM_ROCM_USE_AITER=0 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 `
        -v "$vllm\.docker-rocdxg\libdxcore.so:/usr/lib/libdxcore.so:ro" `
        -v "$vllm\.docker-rocdxg\librocdxg.so:/usr/lib/librocdxg.so:ro" `
        -v "$vllm\.docker-rocdxg\dids.conf:/usr/share/rocdxg/dids.conf:ro" `
        -v vllm_huggingface-cache:/root/.cache/huggingface -v "${ad}:/adapters/r3:ro" `
        vllm-dflash-wsl:v0.27.1 Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000 --attention-backend ROCM_ATTN `
        --enforce-eager --max-model-len 16384 --max-num-seqs 40 --gpu-memory-utilization 0.85 `
        --enable-prefix-caching --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' `
        --enable-lora --max-lora-rank 8 --lora-modules r3=/adapters/r3 | Out-Null
}
for ($i = 0; $i -lt 120; $i++) {
    try { $m = (Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3).data.id
          if ($m -contains "r3") { "[serve] up: $($m -join ', ')"; exit 0 } } catch {}
    if (-not (docker ps -q -f "name=^$Name$")) { docker logs --tail 30 $Name; throw "$Name exited" }
    Start-Sleep 10
}
throw "r3 not up after 20 min"
