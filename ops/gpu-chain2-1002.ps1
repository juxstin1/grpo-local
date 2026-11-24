# Stages 3-4 of gpu-chain-1002, gated (user 2026-10-02 20:23): do not start until
#  (a) the h8 3-way pilot (pids 11360/23008/23308) has exited, and
#  (b) runs/ops/compact-builder.GATE exists. Claude writes the gate only after the
#      compaction builder reports done AND its compact-off token-identical test passes
#      on a re-run. No gate = no stage 3/4, ever.
$ErrorActionPreference = "Continue"
$root = "C:\Users\juxxs\vllm\grpo-local"; $py = "$root\.venv\Scripts\python.exe"
$gate = "$root\runs\ops\compact-builder.GATE"
function Stage($s) { "[chain2] $(Get-Date -Format HH:mm:ss) $s" }
while (Get-Process -Id 11360, 23008, 23308 -ErrorAction SilentlyContinue) { Start-Sleep 30 }
Stage "h8 pilot exited"
while (-not (Test-Path $gate)) { Start-Sleep 60 }
Stage "gate present: $(Get-Content $gate -Raw)"

Set-Location "$root\handoff-gym"
& $py -u tool_handoff.py --source live --families mhqa_t,find_h --only mhqat-h5,mhqat-h7,findh-effective `
    --seeds 3100-3119 --states 5 --k 2 --reads 2 --modes blank,oracle,off,force --tag pilot-tool 2>&1 |
    Select-String -NotMatch "HF_TOKEN|HF Hub"
Stage "tool pilot done"

& pwsh -NoProfile -File "$root\ops\queue-1002.ps1"
Stage "DONE"
