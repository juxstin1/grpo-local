# sandbox-gym — runbook (PowerShell)

$py = "..\.venv\Scripts\python.exe"

Write-Host "== sandbox-gym ==" -ForegroundColor Cyan
Write-Host "1) offline selftest:        .\cli.py selftest"
Write-Host "2) serve vLLM (see AGENTS.md for the docker command)"
Write-Host "3) baseline eval (sealed):  .\cli.py eval --family find --attempts 4 --slug r3-find --served-model r3 --sealed"
Write-Host "4) SFT collection:          .\cli.py collect --family find --seeds 2000-2019 --name find-r3-s1 --served-model r3"
Write-Host "5) GRPO:                    .\cli.py train --family find --seeds 1000-1009 --steps 50 --group-size 6"

$cmd = Read-Host "`nCommand to run (or Enter to exit)"
if ($cmd) { & $py $cmd }