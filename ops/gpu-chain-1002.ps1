# Home GPU chain, 2026-10-02 night. One stage at a time on vllm-r3-local :8000.
# 1) wait for h8 blank controls  2) 3-way pilot on h8 (oracle/off/force, 5 states k2)
# 3) tool-episode compaction pilot  4) r3 editv_h cap20 + mhqa_t base4b vs r3.
# The full 20-state h8 run is NOT here: it waits for a human look at the pilot.
$ErrorActionPreference = "Continue"
$root = "C:\Users\juxxs\vllm\grpo-local"; $py = "$root\.venv\Scripts\python.exe"
function Stage($s) { "[chain] $(Get-Date -Format HH:mm:ss) $s" }
while (Get-Process -Id 29192 -ErrorAction SilentlyContinue) { Start-Sleep 30 }   # h8-controls-1002
Stage "h8 controls done"

Set-Location "$root\handoff-gym"
$jobs = foreach ($m in "oracle", "off", "force") {
    Start-Process -PassThru -NoNewWindow -FilePath $py -RedirectStandardOutput "$root\runs\ops\pilot-h8-$m.log" `
        -RedirectStandardError "$root\runs\ops\pilot-h8-$m.err" `
        -ArgumentList "-u", "score.py", "--hops", "8", "--states", "5", "--k", "2", "--workers", "4",
                      "--note-mode", $m, "--tag", "pilot-h8-$m"
}
Stage "3-way pilot started: $($jobs.Id -join ',')"
$jobs | Wait-Process
Stage "3-way pilot done"

& $py -u tool_handoff.py --source live --families mhqa_t,find_h --only mhqat-h5,mhqat-h7,findh-effective `
    --seeds 3100-3119 --states 5 --k 2 --reads 2 --modes blank,oracle,off,force --tag pilot-tool 2>&1 |
    Select-String -NotMatch "HF_TOKEN|HF Hub"
Stage "tool pilot done"

& pwsh -NoProfile -File "$root\ops\queue-1002.ps1"
Stage "DONE"
