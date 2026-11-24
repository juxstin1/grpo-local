#!/usr/bin/env bash
# Runs /workspace/run-objective.sh; if it dies (exit != 0 and != 3), restarts it (it resumes from
# stage markers + the latest GRPO checkpoint). At most 2 restarts. exit 3 = deliberate stop: no retry.
# Start detached:  STEPS=N setsid nohup bash /workspace/supervisor.sh > /workspace/supervisor.log 2>&1 < /dev/null &
echo $$ > /workspace/supervisor.pid
for try in 0 1 2; do
  echo "[sup] $(date -u +%T) attempt $try"
  bash /workspace/run-objective.sh; rc=$?
  echo "[sup] $(date -u +%T) exit $rc"
  [ $rc -eq 0 ] && exit 0
  [ $rc -eq 3 ] && exit 3
  [ $try -lt 2 ] && echo "- supervisor: pipeline died (exit $rc), restart $((try+1))/2 from checkpoint" >> /workspace/REPORT.md
  sleep 30
done
echo "- **supervisor: gave up after 2 restarts**" >> /workspace/REPORT.md
exit 1
