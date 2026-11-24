#!/usr/bin/env bash
# Walk-away pipeline: prescreen -> smoke (gated) -> full GRPO (ckpt+resume) -> holdout eval -> REPORT.md.
# Lives at /workspace/run-objective.sh. Idempotent: finished stages leave /workspace/obj/<stage>.done,
# so a restart (supervisor.sh) skips them and GRPO resumes from its latest checkpoint.
# Anything the plan does not cover -> written to REPORT.md, exit 3 (supervisor does not retry exit 3).
set -uo pipefail
STEPS=${STEPS:?set STEPS (full-run optimizer steps)}
FAM=find_h SEEDS=1000-1099 GROUP=8 BATCH=8
O=/workspace/obj R=/workspace/REPORT.md
PY=/root/train-venv/bin/python G=/workspace/grpo-local/sandbox-gym
export HF_HOME=/root/hf PYTHONUNBUFFERED=1
mkdir -p $O; cd $G
say()  { echo "$*" | tee -a $R; }
stop() { say ""; say "**STOPPED $(date -u +%FT%TZ): $***"; exit 3; }
done_() { touch $O/$1.done; }
[ -f $R ] || printf "# GRPO find_h from r3 — objective report\n\nstarted $(date -u +%FT%TZ), STEPS=$STEPS, group $GROUP x batch $BATCH, seeds $SEEDS\n" > $R

# holdout seeds must never be trained or prescreened
$PY - <<'EOF' || stop "holdout overlap check failed"
import json, glob
train = set(range(1000, 1100))
hold = set()
for f in glob.glob("data/holdout/*.json"):
    d = json.load(open(f))
    hold.update(int(s) for s in d.get("seeds", []))
    hold.update(int(it["seed"]) for it in d.get("items", []) if "seed" in it)
bad = train & hold
print(f"holdout seeds {min(hold) if hold else '-'}..{max(hold) if hold else '-'} | overlap {sorted(bad)}")
raise SystemExit(1 if bad else 0)
EOF

# ---- stage 0: prescreen at GRPO's exact sampling (T 0.8, top_p 0.95, served r3) ----
if [ ! -f $O/prescreen.done ]; then
  say ""; say "## 0. Prescreen ($FAM $SEEDS, r3, 8 attempts, T 0.8)"
  NAME=prescreen-findh-r3-t08
  [ -f data/collect/$NAME.stats.json ] || $PY collect.py --family $FAM --seeds $SEEDS --name $NAME \
      --attempts 8 --keep-per-task 2 --workers 48 --served-model r3 --runner host --temperature 0.8 \
      > $O/prescreen.log 2>&1 || stop "prescreen crashed (see $O/prescreen.log)"
  $PY - $NAME $O/pool.txt >> $R <<'EOF' || stop "prescreen gate failed"
import json, sys
st = json.load(open(f"data/collect/{sys.argv[1]}.stats.json"))["per_task"]
hist, keep, err = {}, [], 0
for t, s in st.items():
    if s["error"]: err += 1; continue            # failed request != result
    n = s["pass"] + s["fail"] + s["truncated"]
    hist[s["pass"]] = hist.get(s["pass"], 0) + 1
    if 0 < s["pass"] < n: keep.append(t)
open(sys.argv[2], "w").write("\n".join(sorted(keep)) + "\n")
print(f"- tasks {len(st)}, with request errors (excluded) {err}")
print(f"- passes-per-task (of 8): {dict(sorted(hist.items()))}")
print(f"- **pool: {len(keep)} mixed tasks** -> /workspace/obj/pool.txt")
raise SystemExit(0 if len(keep) >= 16 else 1)
EOF
  done_ prescreen
fi

# ---- stage 1: smoke on the pool (2 steps) + resume test (step 3 must start from step-2) ----
if [ ! -f $O/smoke.done ]; then
  say ""; say "## 1. Smoke (2 steps + resume to 3)"
  C=$O/smoke-ckpt
  $PY train_grpo.py --family $FAM --seeds $SEEDS --pool $O/pool.txt --init-adapter /workspace/lora-Qwen3.5-4B-sft-r3 \
     --ckpt $C --resume --steps 2 --group-size $GROUP --task-batch $BATCH --runner host --tag smoke --stop-file $C/NO_STOP > $O/smoke.log 2>&1 \
     || stop "smoke crashed (see $O/smoke.log)"
  $PY train_grpo.py --family $FAM --seeds $SEEDS --pool $O/pool.txt --init-adapter /workspace/lora-Qwen3.5-4B-sft-r3 \
     --ckpt $C --resume --steps 3 --group-size $GROUP --task-batch $BATCH --runner host --tag smoke --stop-file $C/NO_STOP > $O/smoke-resume.log 2>&1 \
     || stop "resume crashed (see $O/smoke-resume.log)"
  grep -q "start step 3/3" $O/smoke-resume.log || stop "resume did not continue at step 3"
  $PY - $C/steps.jsonl >> $R <<'EOF' || stop "smoke gate failed (>=50% mixed groups, 0 errors, reward>0)"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
g = sum(r["groups"] for r in rows); dead = sum(r["dead"] for r in rows)
err = sum(r["errors"] for r in rows); rew = max(r["mean_reward"] for r in rows)
mixed = 1 - dead / max(1, g)
for r in rows: print(f"- step {r['step']}: dead {r['dead_pct']:.0%} solve-none {r['solve_none_pct']:.0%} reward {r['mean_reward']} errors {r['errors']} time {r['time']}s")
print(f"- **gate: mixed {mixed:.0%} (need >=50%), errors {err} (need 0), max reward {rew} (need >0)**; resume OK")
raise SystemExit(0 if mixed >= 0.5 and err == 0 and rew > 0 else 1)
EOF
  done_ smoke
fi

# ---- stage 2: full GRPO, checkpoint every step, resumes on restart ----
if [ ! -f $O/full.done ]; then
  grep -q "^## 2. Full" $R || { say ""; say "## 2. Full GRPO ($STEPS steps, ckpt every step -> $O/ckpt)"; }
  $PY train_grpo.py --family $FAM --seeds $SEEDS --pool $O/pool.txt --init-adapter /workspace/lora-Qwen3.5-4B-sft-r3 \
     --ckpt $O/ckpt --resume --steps $STEPS --group-size $GROUP --task-batch $BATCH --runner host --tag grpo --stop-file $O/STOP_AFTER >> $O/full.log 2>&1 \
     || { echo "full run exited nonzero at $(date -u +%T)" >> $R; exit 1; }   # exit 1: supervisor may resume
  # exit 0 = all STEPS done OR STOP_AFTER hit (train_grpo renamed the flag): both mark stage 2 done,
  # and stage 3 evals the latest ckpt/step-* adapter.
  H=$(grep '^\[stop\] STOP_AFTER hit at step' $O/full.log | tail -1)
  [ -n "$H" ] && say "- stopped early via STOP_AFTER: ${H#\[stop\] }"
  done_ full
fi

# ---- stage 3: eval final adapter vs r3 on the unfiltered find_h holdout (never trained/prescreened) ----
if [ ! -f $O/eval.done ]; then
  say ""; say "## 3. Holdout eval (smoke_find_h_9001-9005, 4 attempts, same settings both)"
  LAST=$(ls -d $O/ckpt/step-* | sort -t- -k2 -n | tail -1); K=${LAST##*-}
  curl -sf -X POST 127.0.0.1:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
       -d "{\"lora_name\":\"grpo-final\",\"lora_path\":\"$LAST/adapter\"}" >/dev/null || true
  for M in r3 grpo-final; do
    $PY eval.py --set data/holdout/smoke_find_h_9001-9005.json --attempts 4 --slug obj-$M-find_h \
       --base http://127.0.0.1:8000 --served-model $M --runner host --workers 32 > $O/eval-$M.log 2>&1 \
       || stop "eval $M crashed (see $O/eval-$M.log)"
  done
  # regression (KL is off): edit ring (content checks only, no code runs) + fresh multi-hop h3/h4
  for M in r3 grpo-final; do
    $PY eval.py --set data/holdout/smoke_edit_9001-9005.json --attempts 4 --slug obj-$M-edit \
       --base http://127.0.0.1:8000 --served-model $M --runner host --workers 32 > $O/eval-$M-edit.log 2>&1 \
       || stop "edit regression eval $M crashed"
    (cd /workspace/grpo-local && $PY sft/collect.py --mhqa-seeds 6000-6009 --hops 3,4 --per-hop 2 \
       --attempts 2 --keep-per-task 0 --name obj-$M-mhqa-h34 --served-model $M --workers 32 \
       > $O/eval-$M-mhqa.log 2>&1) || stop "mhqa regression eval $M crashed"
  done
  echo "$K $(sha256sum $LAST/adapter/adapter_model.safetensors | cut -c1-64)" > $O/final.txt
  done_ eval
fi

# ---- stage 4: wrap ----
if [ ! -f $O/wrap.done ]; then
  $PY - $O >> $R <<'EOF'
import json, sys, pathlib
O = pathlib.Path(sys.argv[1])
def summ(m):
    return json.load(open(f"data/results/obj-{m}-find_h/find_h.summary.json"))
a, b = summ("r3"), summ("grpo-final")
k, sha = (O / "final.txt").read_text().split()
print("\n| holdout find_h 9001-9005 x4 | r3 | GRPO final |\n|---|---|---|")
for key in ("per_attempt_acc", "accuracy", "turn_cap", "files_ok_no_answer", "truncated"):
    print(f"| {key} | {a.get(key)} | {b.get(key)} |")
print("\n## 4. Per-step curve\n\n| step | mean reward | dead % | solve-none % | errors | grad | time s |\n|---|---|---|---|---|---|---|")
rows = [json.loads(l) for l in open(O / "ckpt" / "steps.jsonl")]
for r in rows:
    print(f"| {r['step']} | {r['mean_reward']} | {r['dead_pct']:.0%} | {r['solve_none_pct']:.0%} | {r['errors']} | {r['grad_norm']} | {r['time']} |")
print("\n## Regression (final vs r3; KL off)\n\n| check | r3 | GRPO final | flag |\n|---|---|---|---|")
S3 = json.load(open("data/results/obj-r3-edit/edit.summary.json"))
SG = json.load(open("data/results/obj-grpo-final-edit/edit.summary.json"))
e3, eg = S3["per_attempt_acc"], SG["per_attempt_acc"]
n = S3["n"] * 4    # 5 items x 3 specs x 4 attempts; one attempt ~ 1/n
print(f"| edit per-attempt ({round(e3*n)}/{n} vs {round(eg*n)}/{n}) | {e3} | {eg} | "
      f"{'**DROP**' if eg < e3 - 0.10 else '(within 10 pt noise)' if eg < e3 else ''} |")
def mh(m):   # h3+h4 combined (collector reports per bucket); approx win rate from always/sometimes
    b = json.load(open(f"/workspace/grpo-local/sft/data/obj-{m}-mhqa-h34.stats.json"))["by_bucket"]["mhqa"]
    return round((b["always"] + b["sometimes"] / 2) / b["tasks"], 3), b
(w3, b3), (wg, bg) = mh("r3"), mh("grpo-final")
print(f"| mhqa h3+h4 fresh seeds 6000-6009 (~win) | {w3} | {wg} | {'**DROP**' if wg < w3 - 0.10 else ''} |")
print(f"| mhqa never-solved tasks | {b3['never']}/{b3['tasks']} | {bg['never']}/{bg['tasks']} | |")
print(f"\n- final adapter: {O}/ckpt/step-{k}/adapter (sha256 {sha})")
r3sha = __import__('hashlib').sha256(open('/workspace/lora-Qwen3.5-4B-sft-r3/adapter_model.safetensors','rb').read()).hexdigest()
print(f"- r3 untouched: {r3sha.startswith('f7832d628ae0')} ({r3sha[:12]})")
print("- odd: " + ", ".join(f"step {r['step']} errors {r['errors']}" for r in rows if r["errors"]) or "- odd: none flagged")
print("\n## Next proposal (not done this run)\n- editv_h/edit turn cap: both models do the edit and run out of turns "
      "(r3 files-ok-no-answer 24/30, 9B 20/30). Test a higher cap for ALL models, re-baseline, before GRPO on code rings (needs Docker reward).")
EOF
  done_ wrap
  say ""; say "**DONE $(date -u +%FT%TZ)** — pod left UP; decide keep/stop."
fi
