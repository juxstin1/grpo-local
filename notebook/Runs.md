# Runs

One row per launched run. Append a row at launch; update only the Status cell when it ends (RUNNING → DONE / FAILED / KILLED + one-word reason). See [[README]], [[Decisions]], [[Claims]].

| Time (local) | Who | What | Where | Log path | Status |
|---|---|---|---|---|---|
| 2026-09-30 14:58 | claude | GRPO pipeline under supervisor, STEPS=40 (prescreen → train_grpo.py) | pod | see [[Log/2026-09-30]] | CLAIMED RUNNING (seeded from STATE.md log) |
| 2026-09-30 16:08 | claude | GRPO full stage RELAUNCH with rollout saving (supervisor PID 28399, train_grpo 28406), resume step 7/40 from step-6; then stage 3 eval + REPORT.md | pod /workspace/obj/ckpt (rollouts.jsonl) | /workspace/obj/full.log, /workspace/supervisor.log | STOPPED step 10 (stage 3 relaunch blocked, pending user) |
| 2026-09-30 18:20 | codex | Stage-3 eval + REPORT from step-10 adapter; supervisor PID 33966, retry 1 active | pod /workspace/obj | /workspace/supervisor.log, /workspace/REPORT.md | FAILED missing-evals-deps |
| 2026-09-30 18:36 | codex | Stage-3 relaunch after missing-dependency fix and 1-item mhqa smoke; supervisor PID 37260 | pod /workspace/obj | /workspace/supervisor.log, /workspace/REPORT.md | DONE gate-pass |
| 2026-09-30 19:36 | codex | find_h leg 2: smoke, 8000-attempt prescreen, steps 11-20, holdout/report from step10 | pod /workspace/obj/leg2 | /workspace/obj/leg2-supervisor.log, /workspace/REPORT-leg2.md | DONE cumulative-MHQA-gate-HOLD |
| 2026-09-30 19:42 | codex | leg-2 10-minute graded-rate monitor | home detached PID 18332 | runs/ops/leg2-monitor.log, .planning/monitor-last.txt | DONE exit-0 |
| 2026-09-30 21:53 | codex | step-20 leg-3 pool prescreen and fresh checkpoint MHQA/find_h diagnostics; no training | pod /workspace/obj/leg3-diagnostics via detached home ops PID 34660 | runs/ops/leg3-diagnostics.log, /workspace/REPORT-leg3-diagnostics.md | FAILED find_h-pool-13-below-16 |
| 2026-09-30 22:06 | codex | leg-3 diagnostic ten-minute monitor; alerts on completion, blocked result, dead runner, vLLM down, or 20-minute stall | home detached PID 26752 | .planning/leg3-monitor-last.json, runs/ops/leg3-monitor-alerts.log | DONE gate-fail-alert |
| 2026-09-30 23:33 | codex | step-20 interactive workspace one-question smoke on copied handoff-gym source; no grading or training | home runs/interactive/step20-handoff-2026-09-30, pod vLLM via existing :8001 tunnel | runs/interactive/step20-handoff-2026-09-30.chat.jsonl | DONE accurate-3-tool-answer |
| 2026-09-30 23:44 | codex | copy full pod obj training state, collected screens, reports, and pod scripts before user closes pod | home adapter-backups/2026-09-30/pod-close-2344 | adapter-backups/2026-09-30/pod-close-2344/MANIFEST.md | RUNNING |
| 2026-10-02 19:35 | claude | handoff compaction rerun, note-mode off vs force, 20 states x4 notes x2 reads | home vllm-r3-local :8000 | runs/ops/handoff-off-1002.log, runs/ops/handoff-force-1002.log; handoff-gym/data/stats-{off,force}-1002.json | RUNNING |
| 2026-10-02 19:41 | claude | queued after handoff: r3 editv_h cap20; mhqa_t h3/h5/h7 base4b vs r3 cap10 + h7 cap14 | home vllm-r3-local :8000 | runs/ops/queue-1002.log; sandbox-gym/data/results/{r3-docker-cap20-*,base4b-*,r3-smoke_mhqa_t*} | QUEUED |
| 2026-10-02 19:50 | claude | h8 blank-note controls (score.py --hops 8 --baseline-n 8 --control-only), 48 continuations | home vllm-r3-local :8000 | harness bg job; handoff-gym/data/controls.json | RUNNING |
| 2026-10-02 20:00 | claude | h8 blank-note controls resumed (score.py --hops 8 --baseline-n 8 --control-only) | home vllm-r3-local :8000 | runs/ops/h8-controls-1002.log; handoff-gym/data/controls.json | RUNNING |
| 2026-10-02 20:08 | claude | gpu-chain-1002: h8 3-way pilot -> tool-episode pilot -> r3 editv_h cap20 + mhqa_t evals | home vllm-r3-local :8000 | runs/ops/gpu-chain-1002.log, runs/ops/pilot-h8-*.log | RUNNING |
| 2026-10-02 20:25 | claude | gpu-chain2-1002: gated stages 3-4 (tool-episode pilot, queue-1002 evals) | home vllm-r3-local :8000 | runs/ops/gpu-chain2-1002.log | WAITING-GATE |
| 2026-10-02 20:32 | claude | diag-oracle-1002: oracle reads with finish reasons, same 5 h8 states k2 | home vllm-r3-local :8000 | runs/ops/diag-oracle-1002.log | WAITING-PILOT |
| 2026-10-02 20:56 | claude | gpu-chain3-1002: queue-1002 evals gated on tool pilot exit + tool-edits.GATE | home vllm-r3-local :8000 | runs/ops/gpu-chain3-1002.log | WAITING-GATE |
