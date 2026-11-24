#!/usr/bin/env bash
# Pod dashboard for the objective pipeline. Lives at /workspace/status.sh; called by ops/status.ps1.
O=/workspace/obj
S=$(cat /workspace/supervisor.pid 2>/dev/null)
ps -p "$S" >/dev/null 2>&1 && sup=ALIVE || sup=DEAD
st=$(for x in prescreen smoke full eval wrap; do [ -f $O/$x.done ] && printf "%s:done " $x || printf "%s:- " $x; done)
echo "pipeline supervisor=$sup | $st"
echo "running: $(ps aux | grep -oE '[c]ollect.py --family [a-z_]+|[t]rain_grpo.py .*--tag [a-z]+|[e]val.py --set [^ ]+|sft/[c]ollect.py' | sed 's/--seeds.*--tag/…tag/' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
[ -f $O/prescreen.log ] && echo "prescreen: $(grep -aE '^\[collect\]' $O/prescreen.log | tail -1)"
for c in $O/smoke-ckpt $O/ckpt; do
  [ -f $c/steps.jsonl ] && { echo "$(basename $c): $(wc -l < $c/steps.jsonl) steps"; tail -3 $c/steps.jsonl | python3 -c '
import json,sys
for l in sys.stdin:
    r=json.loads(l); print("  step {step}: reward {mean_reward} dead {dead_pct:.0%} solve-none {solve_none_pct:.0%} errors {errors} {time}s via {v}".format(v=r.get("rollout_adapter"), **r))'; }
done
grep -aE '^\*\*STOPPED|^\*\*DONE|supervisor:' /workspace/REPORT.md 2>/dev/null | tail -2
for p in 8000 8002; do curl -sf -m 3 127.0.0.1:$p/v1/models >/dev/null && s=up || s=DOWN; printf "vLLM %s %s | " $p $s; done; echo
sm=$(timeout 6 nvidia-smi dmon -s u -d 1 -c 5 | awk '!/^#/{s+=$2;n++} END{if(n) printf "%d", s/n}')
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -v sm="$sm" '{printf "GPU sm %s%% | mem %d/%d MiB\n", sm, $1, $2}'
