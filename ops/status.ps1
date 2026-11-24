# One-call dashboard: pod (GRPO, vLLM, GPU via /workspace/status.sh) + home evals + tunnels.
# Usage: .\ops\status.ps1 [-Ip <pod-ip>] [-Port 18890] [-Push]   (-Push re-uploads pod-status.sh)
param([string]$Ip = "<pod-ip>", [int]$Port = 18890, [switch]$Push)

$key = "$HOME/.ssh/id_ed25519_runpod"
if ($Push) { scp -q -i $key -P $Port "$PSScriptRoot\pod-status.sh" "root@${Ip}:/workspace/status.sh" }
ssh -o BatchMode=yes -o ConnectTimeout=10 -i $key -p $Port "root@$Ip" 'bash /workspace/status.sh'

$res = "C:\Users\juxxs\vllm\grpo-local\sandbox-gym\data\results"
Get-ChildItem $res -Directory | Where-Object { $_.Name -match '^(9b|r3)-docker-' } | ForEach-Object {
    $raw = Get-ChildItem "$($_.FullName)\raw" -Filter *.jsonl -ErrorAction SilentlyContinue
    $n = if ($raw) { (Get-Content $raw.FullName | Measure-Object -Line).Lines } else { 0 }
    $s = Get-ChildItem $_.FullName -Filter *.summary.json -ErrorAction SilentlyContinue
    $st = if ($s) { "DONE" } else { "$n/30" }
    "eval {0,-36} {1}" -f ($_.Name -replace '-docker-', ' '), $st
}
$t = foreach ($p in 8001, 8002) { try { $null = Invoke-RestMethod "http://127.0.0.1:$p/v1/models" -TimeoutSec 3; "$p ok" } catch { "$p DOWN" } }
"tunnels " + ($t -join ' | ')
