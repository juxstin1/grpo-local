# fix_h holdout at a raised turn cap (default 2x config.episode.max_turns), Docker grading at home.
# The config default is untouched: the cap is passed as eval.py --max-turns.
# Own workspace root per slug (--ws-root), so concurrent evals of other models can't wipe episodes.
# r3 (local Docker vLLM, ops\serve-r3-local.ps1):  .\ops\eval-fix-capx2.ps1
# 9B re-baseline (pod :8002 via tunnel):           .\ops\eval-fix-capx2.ps1 -Tag 9b -Model Qwen/Qwen3.5-9B -Served Qwen/Qwen3.5-9B -Base http://127.0.0.1:8002
param(
    [string]$Tag = "r3", [string]$Model = "Qwen/Qwen3.5-4B", [string]$Served = "r3",
    [string]$Base = "http://127.0.0.1:8000", [string]$Set = "smoke_fix_h_9001-9005.gen2",
    [int]$Attempts = 4, [int]$Cap = 0, [int]$Workers = 8
)
$ErrorActionPreference = "Continue"
$root = "C:\Users\juxxs\vllm\grpo-local"; $gym = "$root\sandbox-gym"; $py = "$root\.venv\Scripts\python.exe"
Set-Location $gym
$def = [int](& $py -c "from config import Config; print(Config().episode.max_turns)" 2>$null | Select-Object -Last 1)
if ($Cap -le 0) { $Cap = 2 * $def }
$slug = "$Tag-docker-cap$Cap-$Set"
"[fixcap] $(Get-Date -Format HH:mm:ss) $slug  (config cap $def -> $Cap)  base $Base served $Served"
& $py eval.py --set "data/holdout/$Set.json" --attempts $Attempts --slug $slug --base $Base `
    --served-model $Served --model $Model --runner docker --workers $Workers --transcripts `
    --max-turns $Cap --ws-root "data/workspaces-$slug" 2>&1 | Select-String -NotMatch HF_TOKEN

# per-seed passes, cap hits, stop reasons from the raw records
$items = @{}
foreach ($item in (Get-Content "data/holdout/$Set.json" -Raw | ConvertFrom-Json).items) { $items[$item.name] = $item.seed }
$records = @(Get-Content "data/results/$slug/raw/fix_h.jsonl" | ForEach-Object { $_ | ConvertFrom-Json })
$ok = @($records | Where-Object { $null -ne $_.passed })
$passed = @($ok | Where-Object passed).Count
$caps = @($ok | Where-Object cap_hit).Count
$errors = $records.Count - $ok.Count
"[fixcap] pass $passed/$($ok.Count) | errors $errors | cap_hit $caps/$($ok.Count)"
$seeds = [ordered]@{}
foreach ($item in $items.GetEnumerator() | Sort-Object Value) {
    $seeds[[string]$item.Value] = @($ok | Where-Object { $_.name -eq $item.Key -and $_.passed }).Count
}
"[fixcap] per-seed $($seeds | ConvertTo-Json -Compress)"
$stops = [ordered]@{}
foreach ($group in ($ok | Group-Object stop | Sort-Object Name)) { $stops[[string]$group.Name] = $group.Count }
"[fixcap] stop $($stops | ConvertTo-Json -Compress)"
