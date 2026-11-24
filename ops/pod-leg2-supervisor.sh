#!/usr/bin/env bash
# Retry a transient trainer exit from the latest complete checkpoint.
set -uo pipefail
O=/workspace/obj/leg2
mkdir -p "$O"
echo "$$" > "$O/supervisor.pid"
for attempt in 0 1 2; do
  printf '[leg2-sup] %s attempt %s\n' "$(date -u +%FT%TZ)" "$attempt"
  bash /workspace/leg2.sh
  rc=$?
  if [ "$rc" -eq 0 ]; then
    test -f "$O/wrap.done" && exit 0
    printf '[leg2-sup] script exited 0 without wrap.done\n'
    exit 3
  fi
  printf '[leg2-sup] %s exit %s\n' "$(date -u +%FT%TZ)" "$rc"
  [ "$rc" -eq 3 ] && exit 3
  [ "$attempt" -eq 2 ] && exit "$rc"
  sleep 30
done
