#!/usr/bin/env bash
# Approved diagnostics only: step-20 prescreens and fresh, unsealed checkpoint probes.
# No optimizer, sealed-test read, server restart, or pod lifecycle operation.
set -euo pipefail

PY=/root/train-venv/bin/python
ROOT=/workspace/grpo-local
GYM=$ROOT/sandbox-gym
SFT=$ROOT/sft
CK=/workspace/obj/ckpt
OUT=/workspace/obj/leg3-diagnostics
REPORT=/workspace/REPORT-leg3-diagnostics.md
STEP20=leg2-final-step20
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
mkdir -p "$OUT"
if [ ! -e "$REPORT" ]; then
  printf '# Leg-3 pool and fresh checkpoint diagnostics\n\nStarted %s UTC. No training or sealed-test evaluation.\n' "$(date -u +%FT%TZ)" > "$REPORT"
fi
say() { printf '%s\n' "$*" | tee -a "$REPORT"; }
fail() { say "BLOCKED $(date -u +%FT%TZ): $*"; exit 3; }

cd "$GYM"
$PY - "$CK" <<'PY' || fail 'checkpoint/hash preflight failed'
import hashlib, json, sys
from pathlib import Path
c = Path(sys.argv[1])
rows = {r['step']: r for r in map(json.loads, (c / 'steps.jsonl').read_text().splitlines())}
assert all(k in rows for k in range(10, 21))
for k in range(10, 21):
    p = c / f'step-{k}' / 'adapter' / 'adapter_model.safetensors'
    assert p.is_file() and (c / f'step-{k}' / 'state.json').is_file()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == rows[k]['sha256'], k
print('PREFLIGHT_OK checkpoints 10-20 SHA256 match steps.jsonl')
PY
curl -fsS http://127.0.0.1:8000/v1/models | $PY -c 'import json,sys; d=json.load(sys.stdin); assert any(x["id"]=="leg2-final-step20" for x in d["data"]); print("MODEL_OK leg2-final-step20")' \
  || fail 'step-20 model unavailable; no server restart attempted'

check_stats() {
  local path=$1 expected=$2 attempts=$3 kind=$4
  $PY - "$path" "$expected" "$attempts" "$kind" <<'PY' || fail "stats gate failed: $path"
import json, sys
from pathlib import Path
p, expected, attempts, kind = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = json.loads(p.read_text())
assert len(d['per_task']) == expected, (p, len(d['per_task']), expected)
assert not d.get('refused_seeds') and not d.get('refused_leak_gate')
assert not d.get('skipped_prompt_too_long')
for uid, s in d['per_task'].items():
    assert s['error'] == 0, (uid, s)
    assert s['pass'] + s['fail'] + s['truncated'] == attempts, (uid, s)
print(f'STATS_OK {p.name} tasks={expected} attempts={attempts} errors=0 kind={kind}')
PY
}

gym_collect() {
  local name=$1 family=$2 seeds=$3 expected=$4 attempts=$5 workers=$6 model=$7
  local stats="$GYM/data/collect/$name.stats.json"
  if [ ! -f "$stats" ]; then
    test ! -e "$GYM/data/collect/$name.jsonl" || fail "partial immutable output: $name"
    (cd "$GYM" && $PY collect.py --family "$family" --seeds "$seeds" --name "$name" \
      --attempts "$attempts" --keep-per-task 0 --workers "$workers" \
      --served-model "$model" --runner host --temperature 0.8 \
      > "$OUT/$name.log" 2>&1) || fail "collector failed: $name"
  fi
  check_stats "$stats" "$expected" "$attempts" gym
}

mhqa_collect() {
  local name=$1 seeds=$2 per_hop=$3 expected=$4 attempts=$5 workers=$6 model=$7 temperature=$8
  local stats="$ROOT/sft/data/$name.stats.json"
  if [ ! -f "$stats" ]; then
    test ! -e "$ROOT/sft/data/$name.jsonl" || fail "partial immutable output: $name"
    (cd "$ROOT" && $PY sft/collect.py --mhqa-seeds "$seeds" --hops 3,4 \
      --per-hop "$per_hop" --attempts "$attempts" --keep-per-task 0 \
      --name "$name" --served-model "$model" --workers "$workers" \
      --temperature "$temperature" > "$OUT/$name.log" 2>&1) \
      || fail "collector failed: $name"
  fi
  check_stats "$stats" "$expected" "$attempts" mhqa
}

if [ ! -f "$OUT/screen.done" ]; then
  say '## Step-20 prescreens (T=0.8, 8 attempts per task)'
  gym_collect leg3-smoke-step20-findh find_h 2200 5 1 8 "$STEP20"
  gym_collect leg3-smoke-step20-edit edit 2200 3 1 8 "$STEP20"
  mhqa_collect leg3-smoke-step20-mhqa 7300 1 2 1 8 "$STEP20" 0.8
  say 'SMOKE_OK find_h=5 edit=3 mhqa=2; zero request errors.'

  gym_collect leg3-screen-step20-findh find_h 2200-2319 600 8 48 "$STEP20"
  gym_collect leg3-screen-step20-edit edit 2200-2259 180 8 32 "$STEP20"
  mhqa_collect leg3-screen-step20-mhqa 7300-7359 2 240 8 48 "$STEP20" 0.8
  $PY - "$GYM" "$ROOT" "$OUT" <<'PY' || fail 'pool extraction failed'
import json, sys
from pathlib import Path
gym, root, out = map(Path, sys.argv[1:])
sources = {
    'find_h': gym / 'data/collect/leg3-screen-step20-findh.stats.json',
    'edit': gym / 'data/collect/leg3-screen-step20-edit.stats.json',
    'mhqa': root / 'sft/data/leg3-screen-step20-mhqa.stats.json',
}
manifest = {'adapter_step': 20, 'attempts': 8, 'keep_passes': [1, 6], 'families': {}}
for family, path in sources.items():
    stats = json.loads(path.read_text())['per_task']
    keep = sorted(uid for uid, s in stats.items() if 1 <= s['pass'] <= 6)
    assert len(keep) >= 16, (family, len(keep))
    pool = out / f'pool-{family}.txt'
    payload = '\n'.join(keep) + '\n'
    if pool.exists():
        assert pool.read_text() == payload
    else:
        pool.write_text(payload)
    manifest['families'][family] = {
        'source_stats': str(path), 'screened_tasks': len(stats), 'mixed_tasks': len(keep),
        'histogram': {str(k): sum(s['pass'] == k for s in stats.values()) for k in range(9)},
    }
    print(f'POOL {family} screened={len(stats)} mixed_1_to_6={len(keep)}')
path = out / 'pool-manifest.json'
payload = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
if path.exists():
    assert path.read_text() == payload
else:
    path.write_text(payload)
PY
  cat "$OUT/pool-manifest.json" >> "$REPORT"
  touch "$OUT/screen.done"
fi

load_adapter() {
  local name=$1 path=$2
  if ! curl -fsS -X POST http://127.0.0.1:8000/v1/load_lora_adapter \
       -H 'Content-Type: application/json' \
       -d "{\"lora_name\":\"$name\",\"lora_path\":\"$path\"}" \
       > "$OUT/load-$name.json" 2> "$OUT/load-$name.err"; then
    curl -fsS http://127.0.0.1:8000/v1/models | grep -q "\"id\":\"$name\"" \
      || fail "cannot load $name"
  fi
}
unload_adapter() {
  local name=$1
  curl -fsS -X POST http://127.0.0.1:8000/v1/unload_lora_adapter \
    -H 'Content-Type: application/json' -d "{\"lora_name\":\"$name\"}" \
    > "$OUT/unload-$name.json" 2> "$OUT/unload-$name.err" \
    || fail "cannot unload $name"
}

diagnose() {
  local tag=$1 model=$2
  # Fresh diagnostic set 7400-7459 is not a sealed test or a training pool.
  mhqa_collect "leg3-diag-$tag-mhqa" 7400-7459 2 240 2 32 "$model" 0.7
  gym_collect "leg3-diag-$tag-findh" find_h 8000-8009 50 2 24 "$model"
  say "DIAG_OK $tag: 240 fresh MHQA items x2; 50 fresh find_h items x2; errors 0."
}

if [ ! -f "$OUT/diagnostics.done" ]; then
  say '## Fresh diagnostics (no sealed test seeds, no training)'
  diagnose step20 "$STEP20"
  diagnose r3 r3
  for step in 10 11 12 13 14 15 16 17 18 19; do
    name="leg3-diag-s$step"
    load_adapter "$name" "$CK/step-$step/adapter"
    diagnose "step$step" "$name"
    unload_adapter "$name"
  done
  $PY - "$ROOT" "$GYM" "$OUT" >> "$REPORT" <<'PY' || fail 'diagnostic report failed'
import json, sys
from pathlib import Path
root, gym, out = map(Path, sys.argv[1:])
tags = ['r3'] + [f'step{k}' for k in range(10, 21)]
ref_mhqa = ref_find = None
print('\n| model | MHQA fresh pass/480 | find_h fresh pass/100 | errors |')
print('|---|---:|---:|---:|')
for tag in tags:
    m = json.loads((root / f'sft/data/leg3-diag-{tag}-mhqa.stats.json').read_text())['per_task']
    f = json.loads((gym / f'data/collect/leg3-diag-{tag}-findh.stats.json').read_text())['per_task']
    keys_m, keys_f = set(m), set(f)
    if ref_mhqa is None:
        ref_mhqa, ref_find = keys_m, keys_f
    assert keys_m == ref_mhqa and keys_f == ref_find
    errors = sum(s['error'] for s in m.values()) + sum(s['error'] for s in f.values())
    assert len(m) == 240 and len(f) == 50 and errors == 0
    print(f'| {tag} | {sum(s["pass"] for s in m.values())}/480 | '
          f'{sum(s["pass"] for s in f.values())}/100 | {errors} |')
PY
  touch "$OUT/diagnostics.done"
fi
say "DONE $(date -u +%FT%TZ). Pod left running; no leg-3 training launched."
