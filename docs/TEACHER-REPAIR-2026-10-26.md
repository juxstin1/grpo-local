# Teacher data and admission repair, 2026-10-26

Models remain off. These changes qualify checks on records and invented fixtures,
not model behavior or learning. Codex owns this repair. Original teacher data,
source traces, earlier evaluation results and mirrored tools remain unchanged.

## Verified historical corpus

`../runpod-sft-1007/dataset/train.jsonl` contains 617 rows from 497
`(teacher_model, teacher_trace_id)` episodes and 482 roots. All 497 admitted
source-line hashes match across 62 public source files. The corpus SHA-256 is
`b91f70578332353366bc3676430d788990f560f714e907d1dbbf0b1e4b44a651`.

The manifest attributes hints to 25 episodes: 99 rows and 97,465 of 292,714
completion tokens (33.30%). Quarantining complete hinted roots also includes
seven unhinted rows: 106 rows total. The remaining 511 are still unreplayed.
`hint_texts_stripped=2` counts configured hint strings, not measured removals.

All 617 original grades are generic win strings. They lack per-episode task,
fixture, output-artifact and grader hashes supporting an executable replay.
Source correspondence does not establish correctness. The audit records all
617 as legacy/unreplayed; it neither deletes them nor declares them incorrect.
Per-row dispositions and input hashes are in
`runs/audits/teacher-repair-2026-10-26/teacher-audit.json`.

Reproduce without tokenizer or model loading, choosing a new output filename:

```powershell
.venv\Scripts\python.exe -B sft\audit_teacher.py --train ..\runpod-sft-1007\dataset\train.jsonl --manifest ..\runpod-sft-1007\dataset\manifest.json --source-dir ..\grpo-local-wt-filetools\sandbox-gym\data\teacher --output runs\audits\teacher-repair-2026-10-26\teacher-audit-new.json
```

## Enforcement

- SFT schema requires exact booleans and nonnegative integer token IDs, consistent
  counts, explicit context masks, recognized segment types and EOS-terminated
  trainable assistant spans. Trainer and tests use the same mask helper.
- Gym conversion records old/new prompt and token hashes, original verdicts and
  hint provenance. Changed prompts or root mappings invalidate the win and
  require replay. Outputs are exclusive; no existing dataset is overwritten.
- Endpoint collection accepts only exact tasks from the hash-verified registry.
  Synthetic generated tasks must first be frozen and registered. It grades the
  decoded training token target, then binds the record, task, context and verifier.
- Training rejects missing/stale evidence before model-library imports, verifies
  tokenizer/template and target decoding, and replays the registered verifier
  before building the policy. Rehashed false win strings still fail replay.
  Verifier errors remain invalid. Tool trajectories are deliberately not admitted
  by this endpoint-only replay path.

Local hashes are correspondence checks, not signatures or a security boundary.
A trusted operator can still modify source and registry together. The original
teacher builder outside this repository is not changed; its exports cannot bypass
this repository's admission checks.

## Coverage and next work

`suite-v2/coverage.json` declares the capabilities exercised by the four public
development roots. `ops/check_suite_coverage.py` checks the pinned suite and
mapping, collapses duplicate fixtures and requires two independent roots per
declared required capability. Human review must still judge semantic independence.
This is an inventory, not proof that models can perform those tasks.

Five capabilities have no tasks: tool-error recovery, long-context constraint
retention, conflicting source/injected instruction handling, necessary
clarification, and dependent stateful tool work. Three existing capabilities
have only one independent root. Protected-content preservation has four.
`--require-broad` must fail until the structural gaps are filled. The suite is
public and the separate private evaluator remains unprovisioned.

There is no blocker to continuing offline. Next artifact: one public tool-error
recovery task with a deterministic failure injection, saved-artifact verifier,
correct control and adversarial failures. Owner: Codex. Input: existing work
contract and mocked tool transport. Acceptance: recovery reaches the correct
artifact without unauthorized changes, while skipped/incorrect recovery fails.
Next action: author that task and its verifier controls, then run
`.venv\Scripts\python.exe -B ops\qualify_offline.py`.

Teacher replay remains unfinished: it needs preserved task specifications,
fixtures, outputs and the historical grading implementation. Native CPU Docker
execution remains unverified because the daemon was unavailable in the preceding
repair. Neither condition blocks offline fixture work. No models or services
were launched for this repair.
