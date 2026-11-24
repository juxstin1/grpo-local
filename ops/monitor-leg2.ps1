# Detached ten-minute monitor. It writes a baseline every tick and emits only alerts.
param([switch]$Once, [int]$IntervalSeconds = 600)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$baseline = Join-Path $repo '.planning\monitor-last.txt'
$alerts = Join-Path $repo 'runs\ops\leg2-monitor-alerts.log'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
if (-not (Test-Path -LiteralPath $ssh)) { $ssh = 'C:\Program Files\Git\usr\bin\ssh.exe' }
$key = 'C:\Users\juxxs\.ssh\id_ed25519_runpod'
$known = 'C:\Users\juxxs\.ssh\known_hosts'

function Last-Sample {
    if (-not (Test-Path -LiteralPath $baseline)) { return $null }
    $lines = @(Get-Content -LiteralPath $baseline -Tail 50)
    [array]::Reverse($lines)
    foreach ($line in $lines) {
        try {
            $item = $line | ConvertFrom-Json
            if ($item.kind -eq 'leg2-monitor') { return $item }
        } catch { }
    }
    return $null
}

function Alert([string]$message) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Add-Content -LiteralPath $alerts -Value $line
    Write-Host $line
}

function Tick {
    $prior = Last-Sample
    $raw = & $ssh -o BatchMode=yes -o "UserKnownHostsFile=$known" -i $key -p 18890 `
        root@<pod-ip> 'bash /workspace/leg2-status.sh' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Alert "ERROR pod status SSH exit=$LASTEXITCODE"
        return $false
    }
    $line = @($raw | Where-Object { $_ -match '^epoch=' }) | Select-Object -Last 1
    if (-not $line) { Alert 'ERROR pod status returned no sample'; return $false }
    $fields = @{}
    foreach ($match in [regex]::Matches($line, '([a-z_]+)=([^ ]+)')) {
        $fields[$match.Groups[1].Value] = $match.Groups[2].Value
    }
    foreach ($required in 'epoch','stage','graded','total','supervisor','vllm','errors') {
        if (-not $fields.ContainsKey($required)) { Alert "ERROR pod status missing $required"; return $false }
    }
    $nowEpoch = [long]$fields.epoch
    $done = [long]$fields.graded
    $total = [long]$fields.total
    $errors = [long]$fields.errors
    $rate = $null
    $etaEpoch = $null
    $etaBasisSeconds = $null
    $slip = $false
    if ($prior -and $prior.stage -eq $fields.stage -and $nowEpoch -gt [long]$prior.epoch) {
        $delta = $done - [long]$prior.graded
        $seconds = $nowEpoch - [long]$prior.epoch
        $minutes = $seconds / 60.0
        if ($delta -gt 0 -and $seconds -ge 300) {
            $rate = [math]::Round($delta / $minutes, 3)
            $etaBasisSeconds = $seconds
        }
        if ($rate -and $done -lt $total) {
            $etaEpoch = [long]($nowEpoch + 60 * ($total - $done) / $rate)
        }
        if ($prior.eta_epoch -and $prior.eta_basis_sec -ge 300 -and $done -lt $total) {
            $expected = [long]$prior.eta_epoch
            $budget = [math]::Max(60, $expected - [long]$prior.epoch)
            $slip = (-not $etaEpoch) -or (($etaEpoch - $expected) -gt 0.20 * $budget)
        }
    }
    $sample = [ordered]@{
        kind='leg2-monitor'; epoch=$nowEpoch; stage=$fields.stage; graded=$done;
        total=$total; supervisor=$fields.supervisor; vllm=$fields.vllm;
        errors=$errors; rate_per_min=$rate; eta_epoch=$etaEpoch;
        eta_basis_sec=$etaBasisSeconds; eta_slip=$slip
    }
    Add-Content -LiteralPath $baseline -Value ($sample | ConvertTo-Json -Compress)
    if ($prior -and $prior.stage -ne $fields.stage -and $prior.stage -notin @('DONE','FAILED')) {
        Alert "STAGE_DONE $($prior.stage) -> $($fields.stage) graded=$done/$total"
    }
    if ($fields.stage -eq 'FAILED' -and (-not $prior -or $prior.stage -ne 'FAILED')) {
        Alert 'GATE_FAIL leg2; see /workspace/REPORT-leg2.md'
    }
    if ($fields.stage -eq 'DONE' -and (-not $prior -or $prior.stage -ne 'DONE')) {
        Alert 'DONE leg2; see /workspace/REPORT-leg2.md'
    }
    if ($fields.stage -notin @('DONE','FAILED') -and $fields.supervisor -eq 'DEAD' -and
        (-not $prior -or $prior.supervisor -ne 'DEAD')) { Alert "SUPERVISOR_DEAD stage=$($fields.stage)" }
    if ($fields.vllm -eq 'DOWN' -and (-not $prior -or $prior.vllm -ne 'DOWN')) { Alert 'VLLM_DOWN port=8000' }
    if ($errors -gt 0 -and (-not $prior -or $errors -gt [long]$prior.errors)) {
        Alert "ERRORS count=$errors stage=$($fields.stage)"
    }
    if ($slip -and (-not $prior -or -not $prior.eta_slip)) {
        Alert "ETA_SLIP_GT20 stage=$($fields.stage) graded=$done/$total rate_per_min=$rate"
    }
    return $fields.stage -in @('DONE','FAILED')
}

do {
    $finished = Tick
    if ($Once -or $finished) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
