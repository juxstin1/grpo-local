<# Read-only pod sampler. Detached via ops.ps1; alerts only on stage done, failure,
   lost vLLM, or no graded-count movement for over 20 minutes. #>
param([switch]$Once, [int]$IntervalSeconds = 600)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$baselinePath = Join-Path $root '.planning\leg3-monitor-last.json'
$alertsPath = Join-Path $root 'runs\ops\leg3-monitor-alerts.log'
$registryPath = Join-Path $root 'runs\ops\leg3-diagnostics.json'
$key = 'C:\Users\juxxs\.ssh\id_ed25519_runpod'
$known = 'C:\Users\juxxs\.ssh\known_hosts'

while ($true) {
    $now = Get-Date
    $old = if (Test-Path -LiteralPath $baselinePath) {
        Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
    } else { $null }
    $runner = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json
    $alive = [bool](Get-Process -Id $runner.pid -ErrorAction SilentlyContinue)
    $raw = & ssh -o BatchMode=yes -o "UserKnownHostsFile=$known" -i $key -p 18890 `
        root@<pod-ip> 'bash /workspace/leg3-status.sh' 2>$null
    if ($LASTEXITCODE -eq 0) {
        $sample = $raw | ConvertFrom-Json
    } else {
        $sample = [pscustomobject]@{stage='UNREACHABLE';source='';graded=0;total=0;vllm='UNKNOWN';last_update=0}
    }
    $progressAt = $now
    if ($old -and $old.source -eq $sample.source -and $old.graded -eq $sample.graded -and
        $old.stage -eq $sample.stage) {
        $progressAt = [datetime]$old.progress_at
    }
    $alert = ''
    if ($sample.stage -eq 'DONE') { $alert = 'DONE' }
    elseif ($sample.stage -eq 'BLOCKED') { $alert = 'GATE_FAIL' }
    elseif (-not $alive) { $alert = 'DIAGNOSTIC_JOB_DEAD' }
    elseif ($sample.vllm -eq 'DOWN') { $alert = 'VLLM_DOWN' }
    elseif ($sample.stage -eq 'UNREACHABLE') { $alert = 'POD_UNREACHABLE' }
    elseif (($now - $progressAt).TotalMinutes -gt 20) { $alert = 'PROGRESS_STALLED' }
    $current = [ordered]@{
        at=$now.ToString('s'); stage=$sample.stage; source=$sample.source;
        graded=[int]$sample.graded; total=[int]$sample.total;
        vllm=$sample.vllm; runner_alive=$alive;
        progress_at=$progressAt.ToString('s'); alert=$alert
    }
    $current | ConvertTo-Json | Set-Content -LiteralPath $baselinePath -Encoding utf8
    if ($alert -and (-not $old -or $old.alert -ne $alert)) {
        $line = '{0} {1} stage={2} source={3} graded={4}/{5} vllm={6} runner_alive={7}' -f `
            $now.ToString('s'), $alert, $sample.stage, $sample.source,
            $sample.graded, $sample.total, $sample.vllm, $alive
        Add-Content -LiteralPath $alertsPath -Value $line -Encoding utf8
        Write-Output $line
    }
    if ($Once) {
        Write-Output ('SAMPLE stage={0} source={1} graded={2}/{3} vllm={4} runner_alive={5}' -f `
            $sample.stage, $sample.source, $sample.graded, $sample.total, $sample.vllm, $alive)
        break
    }
    if ($sample.stage -eq 'DONE' -or $sample.stage -eq 'BLOCKED' -or -not $alive) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
