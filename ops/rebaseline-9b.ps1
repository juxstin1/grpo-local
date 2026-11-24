# Turn-cap re-baseline, 9B half. 9B on pod :8002 via tunnel 8002; all grading at home (Docker).
# Raised cap (20) on edit/editv_h/fix_h; default cap (10) on find_h. Before each ring, measure the pipeline's
# prescreen rate over 60 s; if it fell >25% vs the first reading, drop workers 16 -> 8.
$ErrorActionPreference = "Continue"
$root = "C:\Users\juxxs\vllm\grpo-local"; $gym = "$root\sandbox-gym"; $py = "$root\.venv\Scripts\python.exe"
$ssh = @("-o", "BatchMode=yes", "-i", "$HOME/.ssh/id_ed25519_runpod", "-p", "18890", "root@<pod-ip>")
$M = "Qwen/Qwen3.5-9B"; $CAP = 20

function Rate {   # prescreen graded/min over 60 s; $null once prescreen is done
    $g = { [int](ssh @ssh "grep -aoE '[0-9]+/4000 graded' /workspace/obj/prescreen.log | tail -1 | cut -d/ -f1") }
    $a = & $g; Start-Sleep 60; $b = & $g
    if ($b -ge 4000 -or $b -eq $a) { return $null } else { return $b - $a }
}
$base = Rate; $workers = 16
function Workers {
    if ($null -eq $base) { return 16 }
    $r = Rate
    if ($null -ne $r -and $r -lt 0.75 * $base) { "[rebase] prescreen $r/min < 75% of $base/min -> workers 8"; $script:workers = 8 }
}
"[rebase] start $(Get-Date -Format HH:mm:ss) prescreen baseline $base/min"

Set-Location $gym
$rings = @(
    @{set = "smoke_find_h_9001-9005"; cap = $null},
    @{set = "smoke_edit_9001-9005"; cap = $CAP},
    @{set = "smoke_editv_h_9001-9005"; cap = $CAP},
    @{set = "smoke_fix_h_9001-9005.gen2"; cap = $CAP}
)
foreach ($r in $rings) {
    Workers
    $tag = if ($r.cap) { "cap$($r.cap)" } else { "cap10" }
    $args_ = @("eval.py", "--set", "data/holdout/$($r.set).json", "--attempts", "2", "--slug", "rebase-9b-$tag-$($r.set)",
               "--base", "http://127.0.0.1:8002", "--served-model", $M, "--model", $M, "--runner", "docker",
               "--workers", $workers, "--transcripts")
    if ($r.cap) { $args_ += @("--max-turns", $r.cap) }
    "[rebase] $(Get-Date -Format HH:mm:ss) $($r.set) $tag workers $workers"
    & $py @args_ 2>&1 | Select-String -NotMatch HF_TOKEN
}
Workers
"[rebase] $(Get-Date -Format HH:mm:ss) mhqa h3/h4 seeds 6000-6009 workers $workers"
Set-Location $root
& $py sft/collect.py --mhqa-seeds 6000-6009 --hops 3,4 --per-hop 2 --attempts 2 --keep-per-task 0 `
    --name rebase-9b-mhqa-h34 --model $M --served-model $M --base http://127.0.0.1:8002 --workers $workers 2>&1 |
    Select-String -NotMatch HF_TOKEN | Select-Object -Last 4
"[rebase] DONE $(Get-Date -Format HH:mm:ss)"
