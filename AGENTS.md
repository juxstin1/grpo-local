# GRPO - Agent Environment Rules

## Active ownership and repair policy (2026-10-26)
- Recovery task: `docs/RECOVERY-TASK-2026-10-26.md`. Preserve the live
  harness-session requirement and separate observation rounds; exported receipts
  and model claims cannot establish recovery. Backend failures are invalid attempts.
- Teacher admission changes: `docs/TEACHER-REPAIR-2026-10-26.md`. Historical
  teacher rows remain legacy/unreplayed. Never strip hints or refresh prompts
  while retaining the old passing verdict. Endpoint SFT requires registered
  inputs, bound collection evidence, token fidelity and verifier replay.
  `ops/check_suite_coverage.py --require-broad` currently fails by design;
  five public roots do not cover the required capabilities.
- Endpoint/dataset changes are documented in `docs/EVAL-REPAIR-2026-10-26.md`.
  SFT collection/training requires an explicit `GRPO_DATASET_REGISTRY` or `--registry`;
  original pools and historical wins are not automatically qualified. Public code
  execution requires an explicitly pinned CPU sandbox image; never fall back to
  executing candidates on the Windows host. Private evaluation remains disabled.
- Codex owns the current reward, instruction, and evaluation repair. Read
  `docs/REPAIR-2026-10-26.md` and `suite-v2/README.md` before continuing it.
- **The user explicitly said not to run models.** Do not train, infer, download
  models, or start/restart model services until the user changes that instruction.
  A CLI `--allow-model-run` flag is a technical guard, not user authorization.
- Default learner instructions are `prompts/work-contract-v1.md`; task authoring
  uses `prompts/task-contract-template.md`. Do not alter mirrored Bionic tools or
  silently append the old contradictory fallback-to-memory system instruction.
- Offline acceptance: `.venv\Scripts\python.exe -B ops\qualify_offline.py`.
  New task roots need correct controls, adversarial failures, source-preservation
  checks, infrastructure-invalid handling and repeatability before reward use.
- Legacy keyword/prose graders are diagnostic only, not qualified training
  rewards. Five public development roots are not evidence of general competence.
- Do not read/reseal private task contents to repair evaluation. The local Q4
  holdout mode intentionally refuses operation until an independent evaluator
  identity/storage boundary exists. Hashes alone do not provide confidentiality.

## Environment rules (non-negotiable)
- Windows host, PowerShell only.
- NEVER run commands inside WSL (no `wsl`, no /home/... paths, no Linux-side pip/hf).
- All Linux tooling runs via `docker run` from PowerShell.
- Model downloads happen inside containers into the `hf-cache` Docker volume.
- Always use http://127.0.0.1, never localhost.

## vLLM bake-off loop
Do NOT run a separate model-download step. vLLM downloads the model itself on first
launch, straight into the `hf-cache` Docker volume.

```powershell
docker run --gpus all -p 8000:8000 `
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-compile:/root/.cache/vllm `
  vllm/vllm-openai:latest `
  Qwen/Qwen3.5-2B `
  --enable-lora --max-num-seqs 64 --max-model-len 32768
```

Swap the model argument per run. Every model after the first download is cached in
`hf-cache`, so no `hf` CLI, no WSL, no DNS drama. `vllm-compile` persists the
torch.compile cache across container recreates, so the same model boots fast on
re-runs. `--max-num-seqs 64 --max-model-len 32768` shrink CUDA graph capture for
an eval box. `--enable-lora` + runtime adapter updating
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) let trained LoRA adapters hot-swap via
`/v1/load_lora_adapter` without restarting.
## Ops (added 2026-09-30 after a rough night -- see docs/HANDOFF-2026-09-30.md)
- **Long work runs through `ops\ops.ps1`, never as a harness background job** (those die at 2 h):
  `.\ops\ops.ps1 start <name> <cwd> -Cmd "<command>"`, `.\ops\ops.ps1 tunnel <name> 8001 8000 <ip> <sshport>`,
  `.\ops\ops.ps1 status`. Logs: `runs/ops/<name>.log`; every start/stop/decision: `runs/ops/events.log`
  (`.\ops\ops.ps1 note "<decision>"`). `stop` requires `-Confirm -Reason` -- ask the user first.
- **Never stop/remove the pod, kill runs or delete data without an explicit instruction.** Hedged
  phrasing ("I guess") gets a one-line question.
- **No server start/restart on a GPU while a scored eval or pilot is using it.**
- **Request errors are never results** (eval.py / handoff-gym score.py already enforce this).
- **Delegate** (project agents in `C:\Users\juxxs\vllm\.claude\agents\`):
  `ops-monitor` (haiku, read-only health check every ~20 min of a long run),
  `smoke-tester` (haiku, one item end-to-end -> GO/NO-GO before ANY batch),
  `trace-reader` (sonnet, failure patterns from transcripts/wins; keeps traces out of the main context),
  `data-auditor` (haiku, GO/NO-GO before every training run: shares, turns/win, prompt version, leaks, dry-run).
- Pod vLLM: launch with `/root/vllm-venv/bin` on PATH (ninja). Rebuild: `..\runpod-r4\pod-snapshot\REBUILD.sh`.

## Notebook logging (added 2026-09-30) -- applies to Codex
- **Every turn**, append at least one line to `grpo-local/notebook/Log/YYYY-MM-DD.md` (today, local; create if absent), append-only:
  `- HH:MM [codex] <LAUNCH|RESULT|DECISION|FINDING|FIX> <what, numbers, path> [VERIFIED|CLAIMED]`
  (`VERIFIED` only if checked this turn). Never edit or delete existing lines.
- **Before editing any file**, add `- YYYY-MM-DD HH:MM [codex] OPEN <path> -- <why>` to `grpo-local/notebook/Claims.md`;
  when done change `OPEN` to `DONE HH:MM` on that line (never delete). Don't edit a file with someone else's `OPEN` claim.
- Launched a run? Add a row to `grpo-local/notebook/Runs.md`. Conventions: `grpo-local/notebook/README.md`.

## Bionic parity (added 2026-10-02)
- Gym tools mirror the Bionic rebuild (C:\Users\juxxs\bionic\llama-studio) byte-for-byte; parity scripts
  (sandbox-gym/*_parity.py) run Bionic's own code from env BIONIC_DIR.
- Until branch feat/fetch-url-offset merges into Bionic, BIONIC_DIR = C:\Users\juxxs\bionic\llama-studio-fetch-offset
  (set in C:\Users\juxxs\vllm\.claude\settings.json env and the parity hook default).
- **Never revert or edit gym web-tool (or any mirrored-tool) changes to make a parity failure go away.
  Report the drift to the user instead.**
