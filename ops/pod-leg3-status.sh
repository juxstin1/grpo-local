#!/usr/bin/env bash
# Read-only one-line JSON sampler for the detached leg-3 diagnostic job.
python3 - <<'PY'
import json, re, urllib.request
from pathlib import Path

out = Path('/workspace/obj/leg3-diagnostics')
report = Path('/workspace/REPORT-leg3-diagnostics.md')
logs = list(out.glob('leg3-*.log'))
latest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
graded = total = 0
if latest:
    matches = re.findall(r'\[collect\] (\d+)/(\d+) graded', latest.read_text(errors='replace'))
    if matches:
        graded, total = map(int, matches[-1])
text = report.read_text(errors='replace') if report.exists() else ''
if (out / 'diagnostics.done').exists():
    stage = 'DONE'
elif 'BLOCKED ' in text:
    stage = 'BLOCKED'
elif (out / 'screen.done').exists():
    stage = 'DIAGNOSTICS'
else:
    stage = 'SCREEN'
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5) as r:
        vllm = 'UP' if r.status == 200 else 'DOWN'
except Exception:
    vllm = 'DOWN'
print(json.dumps({'stage': stage, 'source': latest.name if latest else '',
                  'graded': graded, 'total': total, 'vllm': vllm,
                  'last_update': int(latest.stat().st_mtime) if latest else 0}))
PY
