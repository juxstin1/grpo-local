# Stage 4 of the night chain (queue-1002 evals), re-gated 20:56: waits for the tool-episode
# pilot (pids 37452/19948) AND runs/ops/tool-edits.GATE, which Claude writes only after the
# offset + doc-tools agents finish and parity + compact-off + selftest are re-run green.
$root = "C:\Users\juxxs\vllm\grpo-local"
while (Get-Process -Id 37452, 19948 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
"[chain3] $(Get-Date -Format HH:mm:ss) tool pilot exited"
while (-not (Test-Path "$root\runs\ops\tool-edits.GATE")) { Start-Sleep 60 }
"[chain3] $(Get-Date -Format HH:mm:ss) gate: $(Get-Content "$root\runs\ops\tool-edits.GATE" -Raw)"
& pwsh -NoProfile -File "$root\ops\queue-1002.ps1"
"[chain3] $(Get-Date -Format HH:mm:ss) DONE"
