# GRPO local

> Previous README, kept for reference. Links to `swe-gym-bridge/`, `compaction-eval/` and `docs/EXP-2026-10-21-*` point to work that is not in this repository.

Training and evaluation workspace for Qwen models: supervised fine-tuning (SFT),
group relative policy optimization (GRPO), sandbox tool use, and Bionic behavior checks.

## Start here

For offline checks on a fresh checkout, use Python 3.12 and install only
`requirements-offline.txt` into your virtual environment. Then run
`.venv\Scripts\python.exe -B ops\qualify_offline.py`.
This path needs no model libraries, datasets, sibling gym checkout, or model service.
The [staged-source qualification receipt](docs/verification/offline-2026-10-26.json)
records 223 passing tests and their source hashes.

- **Recovery task:** [deterministic edit failure and verification](docs/RECOVERY-TASK-2026-10-26.md).
  The fifth public root requires observed recovery and saved-file readback.
  It is qualified with mocked tools; models remain off and broad coverage is still incomplete.
- **Teacher data and admission:** [evidence, quarantine and next offline task](docs/TEACHER-REPAIR-2026-10-26.md).
  Historical wins need replay; prompt rewrites invalidate their grades. The
  [coverage inventory](suite-v2/coverage.json) explicitly records missing capabilities.
- **Endpoint and dataset repair:** [current behavior and limits](docs/EVAL-REPAIR-2026-10-26.md).
  Evaluation summaries now require complete valid attempts and a bound run identity.
  SFT collection requires an explicit versioned dataset registry; old wins need requalification.
- **Current work, 2026-10-26:** [repair and ownership](docs/REPAIR-2026-10-26.md),
  [public outcome-contract suite](suite-v2/README.md), and
  [learner work instructions](prompts/work-contract-v1.md).
- **Models remain off by user instruction.** Run
  `.venv\Scripts\python.exe -B ops\qualify_offline.py` for offline acceptance.
  The Q4 path now defaults to qualified artifact contracts; historical keyword
  graders are labelled diagnostics. A private final evaluator is still unprovisioned.

- [Environment and ownership rules](AGENTS.md): Windows PowerShell, Docker, file claims, and run controls.
- [Latest recorded experiments, 2026-10-21](docs/EXP-2026-10-21-0.8B-THINKING-OFF.md): thinking-off probes, invalid evaluation subsets, and incomplete long-context runs.
- [SWE-Gym integration pilot](swe-gym-bridge/README.md) and [qualification results](swe-gym-bridge/QUALIFICATION.md): real repository tasks through the existing harness, with upstream test grading.
- [Daily notebook](notebook/Log/2026-10-21.md), [run register](notebook/Runs.md), and [file claims](notebook/Claims.md): dated evidence and ownership.
- [Documentation index](docs/README.md) and [operations index](ops/README.md).
- [Training methodology](SKILL.md): baseline, smoke, training, and evaluation gates. Its original defaults must be read alongside the dated experiment configuration.

The latest notebook records a stop on GPU work and subsequent individually
approved probes. That is recorded history, not a live process check. Do not infer
permission to launch a new evaluation from a command in an older document.

## Repository map

| Path | Purpose |
| --- | --- |
| Root Python files | Original training loop, configuration, rollout, loss, context, and memory helpers |
| [sandbox-gym](sandbox-gym/README.md) | File-tool workspace harness, graders, evaluation, and tool parity checks |
| `handoff-gym/` | Handoff and continuation experiments |
| `evals/` | Evaluation datasets, sealing, runners, and saved results |
| `sft/` | Supervised training and collection tooling |
| [compaction-eval](compaction-eval/README.md) | Direct handoff-note scoring |
| [tau2-bridge](tau2-bridge/README.md) | Evaluation-only tau2 benchmark integration |
| [swe-gym-bridge](swe-gym-bridge/README.md) | Pinned real-issue tasks, repository splits, CPU verifier controls, existing Bionic loop adapter |
| [ops](ops/README.md) | Job supervision and experiment-specific orchestration |
| `q4/` | GGUF build tooling; `out/`, `work/`, and `tools/` are ignored local artifacts |
| `data/` | Task definitions and collected data; preserve provenance and splits |
| `adapters-sft-1008/`, `lora-*/` | Saved adapters, training metadata, and logs |
| `runs/` | Job logs, pilot reports, and experiment outputs |
| [notebook](notebook/README.md) | Shared claims, decisions, and append-only daily logs |
| [docs](docs/README.md), `research/` | Handoffs, plans, audits, and research notes |
| `_archive/` | Historical material retained as evidence |

## Checkout dependencies

`ops/eval_q4.py` uses `../grpo-local-wt-filetools/sandbox-gym`, rather than the
gym directory in this checkout. `compaction-eval` also defaults to that sibling
tree. Check the script's actual paths before running it; moving or deleting a
worktree can break these workflows.

Read-only orientation from this directory:

```powershell
git status --short
git worktree list
Get-Content notebook\Claims.md -Tail 30
Get-Content notebook\Log\2026-10-21.md -Tail 30
```

Before an authorized long run, use the [operations workflow](ops/README.md).
Model downloads belong in Docker's `hf-cache` volume through the serving launch;
follow `AGENTS.md` for the environment constraints.

## Artifact policy

Keep source, task definitions, adapter configuration, manifests, and experiment
evidence reviewable. Generated adapter weights under `adapters-sft-*/` are
ignored by extension; their JSON metadata, READMEs, and logs remain visible to
Git. Ignoring a file does not remove it or reclaim disk space.

Do not treat untracked files, old dates, or an `OPEN` claim as proof that data is
disposable. Resolve ownership before editing, and obtain an explicit deletion
instruction before removing saved data or stopping runs.
