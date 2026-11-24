# After /workspace/REPORT.md lands: pull the final adapter, load it on the pod vLLM, run the fresh sealed
# holdouts from home in Docker for r3 vs final, append a table to .planning/STATE.md.
# Usage: .\ops\post-report-eval.ps1        (detached: .\ops\ops.ps1 start post-eval <cwd> -Cmd "pwsh -File ops\post-report-eval.ps1")
param([string]$Ip = "<pod-ip>", [int]$Port = 18890)
$ErrorActionPreference = "Stop"
$root = "C:\Users\juxxs\vllm\grpo-local"; $gym = "$root\sandbox-gym"; $py = "$root\.venv\Scripts\python.exe"
$key = "$HOME/.ssh/id_ed25519_runpod"; $ssh = @("-o", "BatchMode=yes", "-i", $key, "-p", $Port, "root@$Ip")

$final = ssh @ssh 'cat /workspace/obj/final.txt'            # "<step> <sha256>"
if (-not $final) { throw "no /workspace/obj/final.txt -- eval stage not done" }
$k, $sha = $final.Split(" ")
$dst = "$root\adapters-grpo\grpo-findh-step-$k"
New-Item -ItemType Directory -Force $dst | Out-Null
scp -q -i $key -P $Port -r "root@${Ip}:/workspace/obj/ckpt/step-$k/adapter/*" $dst
$got = (Get-FileHash "$dst\adapter_model.safetensors" -Algorithm SHA256).Hash.ToLower()
if ($got -ne $sha) { throw "pulled adapter sha $got != pod $sha" }
# grpo-final is loaded on :8000 by the pipeline's eval stage; reach it through tunnel 8001

Set-Location $gym
$rows = @()
foreach ($set in "fresh_find_h_9100-9119", "fresh_edit_9100-9119") {
    foreach ($m in "r3", "grpo-final") {
        $slug = "post-$m-$set"
        & $py eval.py --set "data/holdout/$set.json" --sealed --attempts 4 --slug $slug `
            --base http://127.0.0.1:8001 --served-model $m --runner docker --workers 16 2>&1 |
            Select-String -NotMatch HF_TOKEN | Select-Object -Last 2
        $s = Get-ChildItem "data\results\$slug" -Filter *.summary.json | Get-Content -Raw | ConvertFrom-Json
        $rows += "| $set | $m | $($s.per_attempt_acc) | $($s.accuracy) | $($s.turn_cap) | $($s.files_ok_no_answer) |"
    }
}
$t = Get-Date -Format HH:mm
$tbl = @("", "### Post-report fresh holdouts ($t, final = step $k, sha $($sha.Substring(0,12))…)", "",
         "| set | model | per-attempt | pass@4 | turn-cap | files-ok-no-answer |", "|---|---|---|---|---|---|") + $rows
Add-Content "$root\.planning\STATE.md" ($tbl -join "`n")
Add-Content "$root\.planning\STATE.md" "- $t post-report fresh holdout eval done (table above)."
$tbl
