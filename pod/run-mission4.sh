#!/bin/bash
# Run launcher modes in order, stop on first failure. Logs to $W/results/mission4-<modes>.log via the caller.
# Usage: /opt/grpo/bin/run-mission4.sh <workdir> sanity groups train eval
set -euo pipefail
W=$1; shift
G=/opt/grpo
export LLAMA_SERVER=$G/llama.cpp/build/bin/llama-server GRPO_GYM_DIR=$W/sandbox-gym
export USE_HUB_KERNELS=NO HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=$W/triton-cache
export PYTHONPATH=$G/llama.cpp/gguf-py
cd "$W/repo"
for m in "$@"; do
  echo "=== $m $(date -u +%FT%TZ)"
  $G/venv/bin/python -B -u runs/audits/mission4-pod-2026-10-27/mission4_pod.py --mode "$m" --allow-model-run
done
echo "=== ALL DONE $(date -u +%FT%TZ)"
