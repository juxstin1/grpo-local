#!/usr/bin/env bash
# Continue the proven find_h checkpoint on a fresh mixed-task pool.
# Run on the pod from /workspace after stage 3 wrap.done; every stage has a marker.
set -euo pipefail

PY=/root/train-venv/bin/python
G=/workspace/grpo-local/sandbox-gym
O=/workspace/obj/leg2
C=/workspace/obj/ckpt
R=/workspace/REPORT-leg2.md
PRE=leg2-pre-e5822331
COL=leg2-prescreen-findh-step10-2000-2199
SMOKE=leg2-prescreen-smoke-seed2000
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
mkdir -p "$O"
cd "$G"

say() { printf '%s\n' "$*" | tee -a "$R"; }
stop() { say "**STOPPED $(date -u +%FT%TZ): $***"; exit 3; }
done_() { touch "$O/$1.done"; }

if [ ! -f "$R" ]; then
  printf '# GRPO find_h leg 2 — new seeds 2000-2199\n\nStarted %s UTC; resume step 10, target step 20; group 8 x batch 8.\n' \
    "$(date -u +%FT%T)" > "$R"
fi
test -f /workspace/obj/wrap.done || stop 'step-10 stage-3 report is incomplete'
test -f "$C/step-10/state.json" || stop 'step-10 checkpoint is missing'
test "$(sha256sum "$C/step-10/adapter/adapter_model.safetensors" | cut -c1-8)" = e5822331 \
  || stop 'step-10 adapter hash changed'

# The exact decision gate recorded on 2026-09-30: holdout gain and no >10pt regression.
$PY - <<'PY' || stop 'stage-3 metric or request-error gate failed'
import json
from pathlib import Path
p = Path('data/results')
def metric(slug, family):
    rows = [json.loads(x) for x in (p / slug / 'raw' / f'{family}.jsonl').read_text().splitlines()]
    expected = 100 if family == 'find_h' else 60
    assert len(rows) == expected and all(x.get('passed') is not None for x in rows)
    return json.loads((p / slug / f'{family}.summary.json').read_text())
a, b = metric('obj-r3-find_h', 'find_h'), metric('obj-grpo-final-find_h', 'find_h')
assert b['per_attempt_acc'] > a['per_attempt_acc'] and a['n'] == b['n'] == 25
e3, eg = metric('obj-r3-edit', 'edit'), metric('obj-grpo-final-edit', 'edit')
assert eg['per_attempt_acc'] >= e3['per_attempt_acc'] - 0.10 and e3['n'] == eg['n'] == 15
def mh(name):
    d = json.loads(Path(f'/workspace/grpo-local/sft/data/obj-{name}-mhqa-h34.stats.json').read_text())
    assert len(d['per_task']) == 40 and sum(x['error'] for x in d['per_task'].values()) == 0
    assert all(x['pass'] + x['fail'] + x['truncated'] == 2 for x in d['per_task'].values())
    x = d['by_bucket']['mhqa']
    return (x['always'] + x['sometimes'] / 2) / x['tasks']
assert mh('grpo-final') >= mh('r3') - 0.10
print('LEG2_GATE_OK find_h', a['per_attempt_acc'], b['per_attempt_acc'],
      'edit', e3['per_attempt_acc'], eg['per_attempt_acc'],
      'mhqa', mh('r3'), mh('grpo-final'))
PY

load_adapter() {
  local name=$1 path=$2
  if ! curl -fsS -X POST http://127.0.0.1:8000/v1/load_lora_adapter \
       -H 'Content-Type: application/json' \
       -d "{\"lora_name\":\"$name\",\"lora_path\":\"$path\"}" \
       > "$O/load-$name.json" 2> "$O/load-$name.err"; then
    curl -fsS http://127.0.0.1:8000/v1/models | grep -q "$name" \
      || stop "cannot load adapter $name"
  fi
}

if [ ! -f "$O/prescreen.done" ]; then
  $PY - <<'PY' || stop 'holdout seal or seed-overlap gate failed'
from pathlib import Path
import seal
hold = set()
for p in Path('data/holdout').glob('*.json'):
    if p.name == 'smoke_fix_h_9001-only-9b-cap20.json':
        assert not p.with_suffix('.sha256').exists(), 'test fixture unexpectedly sealed'
        continue
    assert p.with_suffix('.sha256').exists(), f'missing seal sidecar: {p}'
    ok, m = seal.verify(p)
    assert ok, f'invalid seal: {p}'
    hold.update(m.get('seeds', []))
assert not hold.intersection(range(2000, 2200))
print('HOLDOUT_GATE_OK sealed files', len(list(Path('data/holdout').glob('*.json'))),
      'overlap', 0)
PY
  load_adapter "$PRE" "$C/step-10/adapter"
  if [ ! -f "data/collect/$SMOKE.stats.json" ]; then
    test ! -e "data/collect/$SMOKE.jsonl" || stop 'partial smoke collection exists'
    $PY collect.py --family find_h --seeds 2000 --name "$SMOKE" \
      --attempts 8 --keep-per-task 1 --workers 8 --served-model "$PRE" \
      --runner host --temperature 0.8 >> "$O/smoke.log" 2>&1 \
      || stop 'one-seed prescreen smoke failed'
  fi
  $PY - "$SMOKE" <<'PY' || stop 'one-seed prescreen smoke gate failed'
import json, sys
d = json.load(open(f'data/collect/{sys.argv[1]}.stats.json'))
assert len(d['per_task']) == 5 and not d['refused_seeds']
assert all(s['error'] == 0 and s['pass'] + s['fail'] + s['truncated'] == 8
           for s in d['per_task'].values())
print('SMOKE_OK 5 tasks x8, 0 request errors')
PY
  say 'Prescreen smoke: 5 tasks x8, 0 request errors; step-10 adapter.'

  if [ ! -f "data/collect/$COL.stats.json" ]; then
    test ! -e "data/collect/$COL.jsonl" || stop 'partial full prescreen exists'
    $PY collect.py --family find_h --seeds 2000-2199 --name "$COL" \
      --attempts 8 --keep-per-task 1 --workers 48 --served-model "$PRE" \
      --runner host --temperature 0.8 >> "$O/prescreen.log" 2>&1 \
      || stop 'full prescreen failed'
  fi
  $PY - "$COL" "$O/pool.txt" >> "$R" <<'PY' || stop 'prescreen/data audit gate failed'
import json, statistics, sys
from collections import Counter
from pathlib import Path
name, pool = sys.argv[1:]
d = json.loads(Path(f'data/collect/{name}.stats.json').read_text())
st = d['per_task']
assert len(st) == 1000 and not d['refused_seeds']
assert all(2000 <= int(s['root_id']) <= 2199 and s['family'] == 'find_h' for s in st.values())
assert all(s['error'] == 0 and s['pass'] + s['fail'] + s['truncated'] == 8 for s in st.values())
keep = sorted(k for k, s in st.items() if 0 < s['pass'] < 8)
assert len(keep) >= 16
pool_text = '\n'.join(keep) + '\n'
if Path(pool).exists():
    assert Path(pool).read_text() == pool_text
else:
    Path(pool).write_text(pool_text)
hist = dict(sorted(Counter(s['pass'] for s in st.values()).items()))
tokens = [n for s in st.values() for n in s['win_tokens']]
turns = [sum(z['kind'] == 'assistant' for z in x['segments'])
         for x in map(json.loads, Path(f'data/collect/{name}.jsonl').read_text().splitlines())]
print(f'Prescreen: {len(st)} tasks x8, errors 0, refused seeds 0; pass histogram {hist}')
print(f'Mixed pool: {len(keep)} tasks with 1-7/8 passes; seeds 2000-2199, no holdout seeds.')
print(f'Saved winners: {len(tokens)}; median tokens {statistics.median(tokens) if tokens else "n/a"}; '
      f'median turns {statistics.median(turns) if turns else "n/a"}.')
PY
  sha256sum agent.py config.py repo_gen.py >> "$O/source-sha256.txt"
  done_ prescreen
fi

if [ ! -f "$O/train.done" ]; then
  say "Training from step 10 through step 20 on $(wc -l < "$O/pool.txt") mixed tasks."
  $PY train_grpo.py --family find_h --seeds 2000-2199 --pool "$O/pool.txt" \
    --init-adapter "$C/step-10/adapter" --ckpt "$C" --resume --steps 20 \
    --group-size 8 --task-batch 8 --runner host --tag leg2 >> "$O/train.log" 2>&1 \
    || { say 'Training exited nonzero; checkpoint can resume without removing data.'; exit 1; }
  test -f "$C/step-20/state.json" || stop 'trainer exited without step-20 checkpoint'
  done_ train
fi

if [ ! -f "$O/eval.done" ]; then
  load_adapter leg2-final-step20 "$C/step-20/adapter"
  for fam in find_h edit; do
    $PY eval.py --set "data/holdout/smoke_${fam}_9001-9005.json" --attempts 4 \
      --slug "leg2-step20-$fam" --base http://127.0.0.1:8000 \
      --served-model leg2-final-step20 --runner host --workers 32 \
      >> "$O/eval-$fam.log" 2>&1 || stop "step-20 $fam eval failed"
  done
  if [ ! -f /workspace/grpo-local/sft/data/leg2-step20-mhqa-h34.stats.json ]; then
    (cd /workspace/grpo-local && $PY sft/collect.py --mhqa-seeds 6000-6009 \
      --hops 3,4 --per-hop 2 --attempts 2 --keep-per-task 0 \
      --name leg2-step20-mhqa-h34 --served-model leg2-final-step20 --workers 32 \
      >> "$O/eval-mhqa.log" 2>&1) || stop 'step-20 mhqa eval failed'
  fi
  done_ eval
fi

if [ ! -f "$O/wrap.done" ]; then
  $PY - "$C" >> "$R" <<'PY' || stop 'leg-2 report failed'
import json, sys
from pathlib import Path
c = Path(sys.argv[1]); p = Path('data/results')
def metric(slug, fam):
    rows = [json.loads(x) for x in (p / slug / 'raw' / f'{fam}.jsonl').read_text().splitlines()]
    expected = 100 if fam == 'find_h' else 60
    assert len(rows) == expected and all(x.get('passed') is not None for x in rows)
    return json.loads((p / slug / f'{fam}.summary.json').read_text())
a, b = metric('obj-grpo-final-find_h', 'find_h'), metric('leg2-step20-find_h', 'find_h')
e_base = metric('obj-r3-edit', 'edit')
e0, e1 = metric('obj-grpo-final-edit', 'edit'), metric('leg2-step20-edit', 'edit')
def mh(name):
    d = json.loads(Path(f'/workspace/grpo-local/sft/data/{name}.stats.json').read_text())
    assert len(d['per_task']) == 40 and sum(x['error'] for x in d['per_task'].values()) == 0
    assert all(x['pass'] + x['fail'] + x['truncated'] == 2 for x in d['per_task'].values())
    x = d['by_bucket']['mhqa']; return (x['always'] + x['sometimes']/2)/x['tasks']
m_base = mh('obj-r3-mhqa-h34')
m0, m1 = mh('obj-grpo-final-mhqa-h34'), mh('leg2-step20-mhqa-h34')
rows = [json.loads(x) for x in (c / 'steps.jsonl').read_text().splitlines()]
assert [x['step'] for x in rows] == list(range(1, 21))
assert sum(x['errors'] for x in rows) == 0
print('\n## Step-20 gate (same holdout and attempts)')
print('| metric | r3 | step 10 | step 20 |\n|---|---:|---:|---:|')
print(f'| find_h per attempt | — | {a["per_attempt_acc"]} | {b["per_attempt_acc"]} |')
print(f'| edit per attempt | {e_base["per_attempt_acc"]} | {e0["per_attempt_acc"]} | {e1["per_attempt_acc"]} |')
print(f'| mhqa h3+h4 win | {m_base:.4f} | {m0:.4f} | {m1:.4f} |')
print(f'| training request errors | — | 0 | 0 |')
print('Holdout improvement:', b['per_attempt_acc'] > a['per_attempt_acc'])
print('Edit DROP vs r3:', e1['per_attempt_acc'] < e_base['per_attempt_acc'] - .10)
print('MHQA DROP vs r3:', m1 < m_base - .10)
print('MHQA DROP vs step 10:', m1 < m0 - .10)
print('CUMULATIVE GATE PASS:', b['per_attempt_acc'] > a['per_attempt_acc'] and e1['per_attempt_acc'] >= e_base['per_attempt_acc'] - .10 and m1 >= m_base - .10)
print('Step-20 adapter:', c / 'step-20/adapter')
PY
  done_ wrap
  say "**DONE $(date -u +%FT%TZ)** — pod left UP."
fi
