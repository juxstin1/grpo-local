#!/usr/bin/env bash
# One machine-readable, read-only sample for monitor-leg2.ps1.
O=/workspace/obj/leg2
G=/workspace/grpo-local/sandbox-gym
P=$G/data/collect/leg2-prescreen-findh-step10-2000-2199
epoch=$(date -u +%s)
pid=$(cat "$O/supervisor.pid" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then sup=ALIVE; else sup=DEAD; fi
if curl -fsS -m 4 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then vllm=UP; else vllm=DOWN; fi

progress() {
  local file=$1 total=$2 value
  value=$(grep -oE "^\[collect\] [0-9]+/$total" "$file" 2>/dev/null | tail -1)
  value=${value#* }
  printf '%s' "${value%%/*}"
}
if [ -f "$O/wrap.done" ]; then
  stage=DONE graded=1 total=1
elif grep -q '^\*\*STOPPED' /workspace/REPORT-leg2.md 2>/dev/null; then
  stage=FAILED graded=0 total=1
elif [ -f "$O/train.done" ]; then
  stage=eval total=240
  a=$(wc -l < "$G/data/results/leg2-step20-find_h/raw/find_h.jsonl" 2>/dev/null) || a=0
  b=$(wc -l < "$G/data/results/leg2-step20-edit/raw/edit.jsonl" 2>/dev/null) || b=0
  c=$(progress "$O/eval-mhqa.log" 80); c=${c:-0}
  graded=$((a+b+c))
elif [ -f "$O/prescreen.done" ]; then
  stage=train total=10
  n=$(wc -l < /workspace/obj/ckpt/steps.jsonl 2>/dev/null) || n=10
  graded=$((n-10))
elif [ -f "$O/prescreen.log" ]; then
  stage=prescreen total=8000
  graded=$(progress "$O/prescreen.log" 8000); graded=${graded:-0}
else
  stage=smoke total=40
  graded=$(progress "$O/smoke.log" 40); graded=${graded:-0}
fi

err=0
if [ -f "$P.fail.jsonl" ]; then
  n=$(grep -c '"status": "error"' "$P.fail.jsonl") || n=0
  err=$((err+n))
fi
if [ -f /workspace/obj/ckpt/steps.jsonl ]; then
  n=$(grep -cE '"errors": [1-9]' /workspace/obj/ckpt/steps.jsonl) || n=0
  err=$((err+n))
fi
for fam in find_h edit; do
  f="$G/data/results/leg2-step20-$fam/raw/$fam.jsonl"
  if [ -f "$f" ]; then
    n=$(grep -c '"passed": null' "$f") || n=0
    err=$((err+n))
  fi
done
printf 'epoch=%s stage=%s graded=%s total=%s supervisor=%s vllm=%s errors=%s\n' \
  "$epoch" "$stage" "$graded" "$total" "$sup" "$vllm" "$err"
