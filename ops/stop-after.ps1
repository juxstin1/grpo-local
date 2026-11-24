<#
stop-after.ps1 -- ask the pod's GRPO run to stop cleanly (no kill, no relaunch).

Writes /workspace/obj/STOP_AFTER on the pod. train_grpo.py checks it after each step is fully
written (ckpt + steps.jsonl + rollouts.jsonl), exits 0, and run-objective.sh then marks
full.done and runs stage 3 (holdout eval) on the latest ckpt/step-* adapter.

  .\ops\stop-after.ps1 -Step 12     # stop after step 12 completes (or now, if already past 12)
  .\ops\stop-after.ps1              # stop after the step currently running
#>
param(
    [int]$Step = -1,
    [string]$HostIp = "<pod-ip>",
    [int]$Port = 18890,
    [string]$Key = "$HOME\.ssh\id_ed25519_runpod",
    [string]$Flag = "/workspace/obj/STOP_AFTER"
)
$ErrorActionPreference = "Stop"
$content = if ($Step -ge 0) { "$Step" } else { "" }
$remote = 'printf "%s" "{0}" > {1} && echo "[stop-after] wrote {1} = [$(cat {1})]" && tail -1 /workspace/obj/ckpt/steps.jsonl 2>/dev/null | cut -c1-120' -f $content, $Flag
ssh -i $Key -p $Port -o StrictHostKeyChecking=accept-new "root@$HostIp" $remote
if ($LASTEXITCODE -ne 0) { throw "ssh failed ($LASTEXITCODE); flag NOT written" }
$ops = Join-Path $PSScriptRoot "ops.ps1"
if (Test-Path $ops) {
    $what = if ($Step -ge 0) { "after step $Step" } else { "after current step" }
    & $ops note "STOP_AFTER set on pod ($what) via stop-after.ps1"
}
