<#
ops.ps1 -- the one way to run long work (collection, evals, pilots, tunnels) for grpo-local.

Every job is a DETACHED process (survives agent/harness limits), with:
  runs/ops/<name>.log     stdout+stderr of the job
  runs/ops/<name>.json    registry entry: pid, cmd, cwd, started, host
  runs/ops/events.log     append-only: every start / stop / exit, with timestamps and reasons

  .\ops\ops.ps1 start  <name> <cwd> -Cmd "<command>"  # detached; refuses a name already running
  .\ops\ops.ps1 tunnel <name> <localPort> <remotePort> <ip> <sshPort>
  .\ops\ops.ps1 status                               # every job: alive/exited, log tail, tunnels
  .\ops\ops.ps1 stop   <name> -Reason "<why>" -Confirm   # refuses without -Confirm and a reason
  .\ops\ops.ps1 note   "<text>"                     # free-form event (decisions, observations)
#>
param(
    [Parameter(Position = 0)][string]$Action = "status",
    [Parameter(Position = 1)][string]$Name,
    [Parameter(Position = 2, ValueFromRemainingArguments = $true)][string[]]$Rest,
    [string]$Cmd = "",
    [string]$Reason = "",
    [switch]$Confirm
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot           # grpo-local
$Dir = Join-Path $Root "runs\ops"
New-Item -ItemType Directory -Force $Dir | Out-Null
$Events = Join-Path $Dir "events.log"

function Log-Event([string]$kind, [string]$text) {
    $line = "{0}  {1,-7} {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $kind, $text
    Add-Content -Path $Events -Value $line -Encoding utf8
    Write-Host $line
}

function Alive([int]$procId) { return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue) }

switch ($Action) {
    "start" {
        if (-not $Name -or $Rest.Count -lt 1 -or -not $Cmd) { throw 'usage: start <name> <cwd> -Cmd "<command>"' }
        $reg = Join-Path $Dir "$Name.json"
        if (Test-Path $reg) {
            $old = Get-Content $reg -Raw | ConvertFrom-Json
            if (Alive $old.pid) { throw "job '$Name' is already running (pid $($old.pid))" }
        }
        $cwd = (Resolve-Path $Rest[0]).Path
        $cmd = $Cmd
        $log = Join-Path $Dir "$Name.log"
        # cmd.exe wrapper: detached, unbuffered, both streams to the log, exit code appended
        $wrapped = "set PYTHONUNBUFFERED=1 && $cmd > `"$log`" 2>&1 & echo [ops] exit !ERRORLEVEL! >> `"$log`""
        $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/v:on", "/c", $wrapped -WorkingDirectory $cwd `
            -WindowStyle Hidden -PassThru
        [pscustomobject]@{ name = $Name; pid = $p.Id; cmd = $cmd; cwd = $cwd; log = $log;
            started = (Get-Date -Format "s") } | ConvertTo-Json | Set-Content $reg -Encoding utf8
        Log-Event "START" "$Name pid=$($p.Id) cwd=$cwd :: $cmd"
    }
    "tunnel" {
        if ($Rest.Count -lt 4) { throw "usage: tunnel <name> <localPort> <remotePort> <ip> <sshPort>" }
        $lp, $rp, $ip, $sp = $Rest[0], $Rest[1], $Rest[2], $Rest[3]
        $ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
        if (-not (Test-Path $ssh)) { $ssh = "C:\Program Files\Git\usr\bin\ssh.exe" }
        $args = @("-N", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
            "-i", "$HOME\.ssh\id_ed25519_runpod", "-p", $sp, "-L", "${lp}:127.0.0.1:${rp}", "root@$ip")
        $p = Start-Process -FilePath $ssh -ArgumentList $args -WindowStyle Hidden -PassThru
        [pscustomobject]@{ name = $Name; pid = $p.Id; cmd = "ssh tunnel $lp->$ip`:$rp"; kind = "tunnel";
            port = [int]$lp; started = (Get-Date -Format "s") } | ConvertTo-Json |
            Set-Content (Join-Path $Dir "$Name.json") -Encoding utf8
        Log-Event "TUNNEL" "$Name pid=$($p.Id) localhost:$lp -> $ip`:$rp (ssh port $sp)"
    }
    "status" {
        Get-ChildItem $Dir -Filter *.json | ForEach-Object {
            $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
            $state = if (Alive $j.pid) { "RUNNING" } else { "exited" }
            $tail = ""
            if ($j.kind -eq "tunnel") {
                try { Invoke-WebRequest "http://127.0.0.1:$($j.port)/v1/models" -TimeoutSec 5 | Out-Null; $tail = "answers" }
                catch { $tail = "NOT ANSWERING" }
            } elseif ($j.log -and (Test-Path $j.log)) {
                $tail = ((Get-Content $j.log -Tail 1) -replace "\s+", " ")
                if ($tail.Length -gt 90) { $tail = $tail.Substring(0, 90) }
            }
            "{0,-28} {1,-8} pid={2,-6} {3}" -f $j.name, $state, $j.pid, $tail
        }
    }
    "stop" {
        if (-not $Confirm -or -not $Reason) { throw "refused: stop needs -Confirm and -Reason (ask the user first)" }
        $reg = Join-Path $Dir "$Name.json"
        if (-not (Test-Path $reg)) { throw "no job '$Name'" }
        $j = Get-Content $reg -Raw | ConvertFrom-Json
        if (Alive $j.pid) { & taskkill /PID $j.pid /T /F | Out-Null }
        Log-Event "STOP" "$Name pid=$($j.pid) reason: $Reason"
    }
    "note" { Log-Event "NOTE" (@($Name) + $Rest -join " ") }
    default { throw "unknown action '$Action' (start|tunnel|status|stop|note)" }
}
