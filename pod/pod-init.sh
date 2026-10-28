#!/bin/bash
# On a fresh pod: pull the pinned Qwen3.5-4B snapshot, verify it, build the Q4 rollout model.
# Idempotent. Usage: /opt/grpo/bin/pod-init.sh [workdir]   (default /workspace/grpo)
set -euo pipefail
W=${1:-/workspace/grpo}
G=/opt/grpo
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
M=$W/repo/runs/mission4-assets/Qwen3.5-4B/$REV
Q4=$W/repo/q4/out/4b-base.Q4_K_M.gguf
mkdir -p "$W/results" "$W/tmp" "$M" "$(dirname "$Q4")"

HF_HUB_OFFLINE=0 HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$W/hf-home \
  $G/venv/bin/hf download Qwen/Qwen3.5-4B --revision $REV --local-dir "$M"
(cd "$M" && sha256sum -c --quiet $G/model.sha256) && echo "model: 11/11 sha256 OK"

if [ ! -f "$Q4" ]; then
  PYTHONPATH=$G/llama.cpp/gguf-py $G/venv/bin/python $G/llama.cpp/convert_hf_to_gguf.py "$M" \
    --outtype bf16 --outfile "$W/tmp/4b-base.bf16.gguf"
  $G/llama.cpp/build/bin/llama-quantize "$W/tmp/4b-base.bf16.gguf" "$Q4" Q4_K_M
  rm -f "$W/tmp/4b-base.bf16.gguf"
fi
# 2026-10-27 pod build was e6483763...; local PC Q4 is c7c31f31.... A different hash is a different
# rollout model: the launcher's vocab checks still gate it, but record it before comparing runs.
echo "q4: $(sha256sum "$Q4" | cut -d' ' -f1)  (2026-10-27 pod: e648376330e09be2ac8b8b76a225a30eeba008ec214a8c6e45167975090cba8a)"
