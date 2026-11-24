# Queued home-GPU evals for 2026-10-02: wait for the handoff runs to exit (GPU stays clean),
# then r3 editv_h at cap 20 (is 0/30 a turn-cap artifact?) and mhqa_t base-4B vs r3.
$ErrorActionPreference = "Continue"
$root = "C:\Users\juxxs\vllm\grpo-local"; $py = "$root\.venv\Scripts\python.exe"
$wait = @(8648, 21896)   # handoff-off-1002 / handoff-force-1002
while (Get-Process -Id $wait -ErrorAction SilentlyContinue) { Start-Sleep 60 }
"[queue] handoff runs exited $(Get-Date -Format HH:mm:ss)"
Set-Location "$root\sandbox-gym"
$common = @("--attempts", "2", "--base", "http://127.0.0.1:8000", "--model", "Qwen/Qwen3.5-4B",
            "--runner", "docker", "--workers", "8", "--transcripts")

"[queue] $(Get-Date -Format HH:mm:ss) r3 editv_h cap20"
& $py eval.py --set data/holdout/smoke_editv_h_9001-9005.json --slug r3-docker-cap20-smoke_editv_h_9001-9005 `
    --served-model r3 --max-turns 20 @common 2>&1 | Select-String -NotMatch HF_TOKEN

foreach ($s in "smoke_mhqa_t_h3_9201-9205", "smoke_mhqa_t_h5_9211-9215", "smoke_mhqa_t_h7_9221-9225") {
    foreach ($m in @(@{tag = "base4b"; served = "Qwen/Qwen3.5-4B"}, @{tag = "r3"; served = "r3"})) {
        "[queue] $(Get-Date -Format HH:mm:ss) $($m.tag) $s cap10"
        & $py eval.py --set "data/holdout/$s.json" --slug "$($m.tag)-$s" --served-model $m.served `
            --ws-root "data/workspaces-$($m.tag)" @common 2>&1 | Select-String -NotMatch HF_TOKEN
    }
}
foreach ($m in @(@{tag = "base4b"; served = "Qwen/Qwen3.5-4B"}, @{tag = "r3"; served = "r3"})) {
    "[queue] $(Get-Date -Format HH:mm:ss) $($m.tag) mhqa_t h7 cap14"
    & $py eval.py --set data/holdout/smoke_mhqa_t_h7_9221-9225.json --slug "$($m.tag)-cap14-smoke_mhqa_t_h7_9221-9225" `
        --served-model $m.served --max-turns 14 --ws-root "data/workspaces-$($m.tag)" @common 2>&1 | Select-String -NotMatch HF_TOKEN
}
"[queue] DONE $(Get-Date -Format HH:mm:ss)"
