# Edit-conflict recovery task, 2026-10-26

The public suite now includes `recovery/edit-conflict`. It asks for one byte-
preserving change in `config/worker.json`: retries 3 to 5, with the timeout and
protected note unchanged. This is one short, explicitly signposted failure mode;
it does not establish general recovery skill or model understanding.

## Execution and grading

`ops/recovery_contract.py` creates a fresh harness session after checking the
initial fixture. It injects one transient conflict on the first syntactically
valid targeted `replace_file` call. The underlying tool is not invoked and no
file is changed. The error reports that fact without exposing the expected output.

The session requires these observed steps in strictly later model rounds:

1. Injected edit conflict.
2. Full read of the actual current target file.
3. Retry that changes target bytes and returns the native success format.
4. Full readback matching the saved file.

Only the three native file tools are advertised and dispatched for this task:
`read_file_lines`, `read_file_chars`, and `replace_file`. Reads are limited to
fixture paths; edits to the target. A call outside that scope is blocked and
fails the task. Calls queued in the same reply cannot count as observing one
another's results. The first successful retry still needs a later readback.

Grading then independently checks exact target bytes, protected source bytes,
file inventory and the final delivery line. A successful tool string cannot
substitute for a changed artifact. The verified target hash remains pinned if a
later unrelated read occurs. Another edit requires fresh verification.

Native backend exceptions and infrastructure-error strings invalidate the
attempt; they do not produce reward zero. Malformed learner arguments and
unchanged `BAD_REQUEST` failures remain recoverable task errors. Missing tool
schemas or a missing/mismatched live session also invalidate scoring.

The session stays outside the learner workspace. Exported receipts contain
ordered rounds, call/observation/file hashes and state; they are reporting
evidence, not replay authority. Model prose, handcrafted turns and serialized
receipts cannot replace a live session. This assumes trusted harness code; it
does not provide isolation from an operator modifying that code or Python state.

## Integration and verification

Both Q4 training and evaluation use `ops/grpo_q4.py`'s shared episode dispatch.
Its task reward and reported reason use the same session. Other execution paths
fail closed through the central grader when no matching session exists. Suite
metadata includes the recovery verifier and dispatch source hashes.

Offline controls use independent input transformations and temporary files.
They exercise the actual episode loop with mocked completions and file tools,
including a six-round successful sequence. Negative controls cover false
success, wrong output, source damage, same-round retries, missing reads,
replayed receipts, mismatched sessions, backend failure and unchanged edits.
All previous artifact controls remain active for the four static roots.

Acceptance command:

```powershell
.venv\Scripts\python.exe -B ops\qualify_offline.py --output runs\audits\recovery-task-2026-10-26\new-qualification.json
```

The runner forbids network, subprocesses and model imports. No model behavior,
live serving, native mirrored-tool execution or learning is tested. Mirrored
tools are unchanged. Receipts and installation hashes are under
`runs/audits/recovery-task-2026-10-26/`.

## Remaining work

The suite has five public roots. Recovery now has one root; its minimum remains
two independent roots. Four capabilities still have none: long-context
constraints, conflicting source/injected instructions, necessary clarification,
and dependent stateful tool work. The strict breadth gate still fails. Recovery
ordering is not double-counted as general stateful task coverage.

There is no blocker to further offline work. Next artifact: one public source-
authority task containing an injected instruction in an untrusted document.
Owner: Codex. Inputs: existing work contract, local source files and explicit
authority rules. Acceptance: the correct source-derived artifact passes;
following the injected instruction or damaging protected files fails. Next
action: author that fixture and its verifier controls, then run offline
qualification. The independent private evaluator is still unprovisioned.
