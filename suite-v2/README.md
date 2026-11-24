# Public outcome-contract suite

Five development roots replace the legacy keyword graders in the Q4 training and
evaluation entrypoints. These fixtures are intentionally public and fixed. They
are a small foundation for qualification, not a sealed benchmark or a measure of
general model understanding. Repetition does not increase root-task coverage.

| Root | Required behavior | Verification |
| --- | --- | --- |
| Pricing rename | Rename a function, import, and caller without changing logic | Python ASTs parsed from source bytes match the specified refactor; all other files preserved |
| Approved decisions | Resolve each field from the latest approved revision, excluding newer drafts | Exact typed JSON values; source bytes and file inventory preserved |
| Open orders | Select latest revisions, exclude cancelled orders, calculate decimal totals | Exact structured output, decimal strings, sorted records; source preservation |
| Scoped section edit | Remove the requested line in Beta while retaining Alpha and archive | Exact target bytes and protected source bytes |
| Edit-conflict recovery | Observe an injected edit failure, read, retry, and read back the saved change | Live harness evidence across separate rounds plus exact target and protected bytes |

Correct reference transformations are calculated from input files by the tests,
not copied from grader expectations. Alternate JSON formatting and equivalent
Python formatting/comments pass. No-op, partial repair, constant output, stale
facts, source damage, extra files, encoding-cookie tricks, wrong final delivery,
blank/truncated replies and unrelated questions fail.

The refactor verifier does not execute candidate Python. It qualifies a constrained
structural rename, not general code repair. The other tasks require structured
artifacts or byte preservation; their verifiers do not claim to understand
arbitrary prose. Tasks specify a strict final delivery line to avoid asking a
keyword matcher to interpret completion claims. This is an explicit protocol,
not an inferred requirement applied to older tasks.

New task instructions come from [work-contract-v1](../prompts/work-contract-v1.md).
Use the [task contract template](../prompts/task-contract-template.md) when adding
roots. Add positive controls, independent input-derived expected behavior,
adversarial negatives, invalid-infrastructure cases, and repeatability checks
before enabling a new reward. Do not copy a held-out task into this public file.

Offline command, from the project root:

```powershell
.venv\Scripts\python.exe -B ops\qualify_offline.py
```

The runner blocks network, subprocess launches, and model-library imports. It
does not start a model, container or service. Tests use temporary authored files.
It exits nonzero on a test failure, skipped test, changed source, or attempted
external action. Use `--output runs/audits/<new-name>.json` for a hashed receipt;
existing receipts are never overwritten.

Source, suite, instruction and rendered-system hashes are recorded for future
authorized runs. Model entrypoints require `--allow-model-run`; the flag is a
technical guard, not a substitute for the user's explicit authorization. The
current user instruction is **do not run models**.

The sibling Bionic tool implementations and original prompt remain untouched.
`GRPO_GYM_DIR` may explicitly locate the existing tool gym when testing from an
isolated worktree; the active checkout retains its existing sibling default.

Legacy free-text, edit and office graders remain accessible only as labelled
diagnostics through the updated Q4 evaluation path. Their old counterexamples
still describe real defects; they have been excluded from training, not declared
semantically repaired. Open-ended clarification, email honesty, general code
repair, and recovery beyond the one scripted edit conflict need independently
qualified contracts before expansion. See [recovery task details](../docs/RECOVERY-TASK-2026-10-26.md).

Local `--holdout` now refuses to claim confidentiality. The integrity module can
verify development/exclusion registries against an explicitly supplied external
manifest, but a final private evaluator still needs its own identity and storage
boundary. No holdout was read, regenerated, resealed, or transferred here.
