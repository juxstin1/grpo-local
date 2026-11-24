# Suite and seal audit — 2026-10-26

The suite needs both stronger task contracts and a stronger evaluation boundary.
The existing gym seal does not establish confidentiality from Claude/Codex or an
independent source of truth. No real sealed task bodies were opened for this
audit, and no production files, permissions, services, or holdouts were changed.

The user's phrase "sealed from Claude grader" remains ambiguous at report time.
The proposed default below keeps the final holdout inaccessible to training and
development agents, including Claude and Codex, and uses executable checks where
the outcome is objectively testable. This is a proposed boundary, not a claim
that such isolation exists today. No Claude model judge was found in the audited
Q4 reward dispatch; current scoring uses Python graders.

## New findings

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Q4 holdout loader never verifies the seal | `ops/grpo_eval.py:43`; executable invented-fixture probe | A fixture rejected by actual `seal.verify` is still loaded |
| Gym seal trusts a checksum inside the same JSON | sibling `sandbox-gym/seal.py::verify`; executable probe | Altered answers with a recomputed checksum pass; the untouched `.sha256` sidecar is ignored |
| Holdout transcripts are permitted in the Q4 runner | `ops/grpo_eval.py:62`, `:70`, `:133` | `--holdout --transcripts` has no mutual-exclusion guard; code can save the conversation |
| Holdout reports expose task-level feedback | `ops/grpo_eval.py:115` onward | Task names, reasons and per-task rewards can guide repeated tuning against the holdout |
| Training exclusion discovery fails open | `ops/q4_probe_raw.py::holdout_seeds` catches every exception and continues | A missing directory or unreadable/corrupt file can silently produce an incomplete exclusion registry |
| Frozen task specifications do not freeze the workspace implementation | `ops/grpo_q4.py:283` regenerates files with current `repo_gen.find_spec` | Generator changes can alter evaluated workspaces independently of sealed JSON |
| Seed separation does not establish task independence | sibling `file_work.py::specs`, hardcoded clarification files/prompts | New seed labels reproduce the same task; prior audit measured 40 clarification items collapsing to two exact tasks |

The transcript exposure and exclusion findings are source findings; no actual
holdout run or leakage incident was reproduced. The fresh execution probes used
only invented data in a temporary directory. They executed the actual function
bodies extracted by AST, avoiding imports that might read real holdouts or launch
services. Five checks yielded three agreeing controls and two contract failures.

The broader `evals/holdout/seal.py` is more thorough than the gym seal: it checks
root-task identities, file hashes and a hash-chained ledger. This is useful
integrity/provenance machinery, but the ledger and expected data remain local
files. It does not by itself create a separate principal that can keep answers
unreadable to a development agent. Do not generalize the gym's exact checksum
flaw to this separate implementation without testing it.

## What can be reused

The existing SWE-Gym bridge is a stronger starting point for code repair:
task inputs exclude reference/test patches, candidate patches are graded in a
fresh container, environment images are pinned, and unavailable infrastructure
has a separate classification. Its recorded qualification accepted nine of
twelve candidates. That is historical qualification evidence, not a rerun today.

However, the pilot explicitly declares `evaluation_sealed=false` and
`training_release=false`. Its public tasks and previous debugging outcomes make
it development material. Its own qualification document says deliberate grader
tampering remains unqualified. Tests checking omitted fields/mount arguments do
not prove the entire host agent cannot read the locally stored judge material.

The verifier's generated script and test output share a process/container with
candidate code; output markers and parsed test logs are part of its trust model.
Before using it for reward, test fabricated success output, pytest hooks,
modified test configuration, missing collection, test deletion, import-time exit,
and changes to setup scripts. These are required adversarial controls, not
exploits demonstrated by this audit.

## Proposed suite: public qualification first, separate final holdout

Create **12 public development scenarios**, four per category below, with complete
fixtures, reference outcomes and intentionally wrong solutions. This is an initial
coverage target for validating the harness, not a sample large enough to claim
general performance. Do not call these tasks sealed after we have inspected them.

| Category | Four development scenarios | Correctness evidence |
| --- | --- | --- |
| Repository changes | boundary-value parser repair; cross-module rename; configuration precedence repair; package API change with caller updates | Independent behavioral tests, required regression tests, protected-file hashes |
| File artifacts | reconcile conflicting dated facts; edit one section without changing adjacent content; CSV transformation with missing/duplicate rows; save a draft without changing its sources or claiming delivery | Parsed artifact/schema checks, exact required values, preserved bytes and file set; independently reviewed text rubric where necessary |
| Completion and recovery | recover from actionable wrong-path error; recover from oldText mismatch; resolve two-file ambiguity via a scripted user response; report an unavailable external dependency without claiming success | Required final artifact plus final reply; preserved files; separate backend-invalid classification; recovery trace used diagnostically |

Longer prompts or more files do not count as stronger coverage on their own.
Vary dependency structure, input edge cases, conflicting evidence and preservation
constraints. Assign every task a root scenario and repository/project identity;
split by those identities before generating variants. Near-duplicate detection
must compare prompt, files and requirements, not just names/seeds. A separate
curator should create a fresh holdout from distinct roots after contracts freeze.

For each development task, require:

1. The original/no-op submission fails at least one intended behavior check.
2. Two materially different correct solutions pass every required check.
3. Every applicable bad solution fails: comment-only, constant/hardcoded output,
   partial repair, unrelated deletion, weakened tests, forged success logs,
   early exit, wrong delivery path, empty final reply, or false completion claim.
4. Missing verifier infrastructure produces `invalid`, never model reward 0.
   A candidate's ordinary assertion failure remains a valid failure. Candidate
   attempts to break the verifier are separately classified, not silently dropped.
5. Repeat the same candidate three times with identical classification/results.
   Freeze source, environment and test hashes in a receipt before model evaluation.

Report correctness, preservation, completion, valid/invalid attempt counts and
task-root coverage separately. Turns/tokens/retries are costs and diagnostics,
not success rewards. Do not average away the real-versus-synthetic gap found in
the reward audit. Keep free-text judgments outside binary training reward until
their rubric has been checked against agreed positive and negative examples.

## What "sealed" should mean here

Four distinct properties need separate evidence:

- **Confidentiality:** the training/development agent cannot read the hidden
  task bank, expected outputs, test source, private results or authoring history.
- **Integrity:** that agent cannot replace the bank, grader, trusted manifest,
  environment, result ledger or evaluator's signing key.
- **Independence:** holdout roots and structures do not duplicate training/dev;
  final acceptance tests are independently reviewed rather than generated only
  from the same implementation assumptions as the solver.
- **Limited feedback:** freeze model/checkpoint and evaluation budget before
  submission; return aggregate category results to developers, not task answers,
  per-item reasons or transcripts. Repeated adaptive evaluation can still overfit
  aggregate feedback, so record all submissions and reserve an untouched final set.

At evaluation time the model necessarily sees the current task prompt. It should
not see expected outputs/tests, the full task bank, or past private evaluation
traces. The curator may see secrets; the training/development role may not.
If Claude authors private tests, that must be a separate curator context and
credential boundary, without shared transcripts/memory with the solving agent.
Merely changing an agent's role name does not enforce this.

Two concrete deployment options:

| Option | Cost/runtime tradeoff | Required verification |
| --- | --- | --- |
| Dedicated Windows evaluator account, restricted file ACLs and narrow submission process | Can reuse installed host software; no new model/API dependency; still shares host/admin boundary | Actual development identity denied reads/writes; no evaluator credentials, privilege escalation or Docker authority exposed |
| Separate evaluator machine/VM or service with its own account/storage and container runner | Extra CPU/RAM/storage and possibly hosting expense; more operational setup | Development agent has only submit/status capability; cannot mount or enumerate private storage; private verifier remains outside candidate execution privileges |

Recommend the separate evaluator boundary for the final holdout if development
agents retain broad host shell/Docker access. Windows ACLs are a lighter option
only after tests prove the agent's actual identity lacks the necessary rights.
This is an architectural recommendation, not an installed service or cost quote.
No additional package/model choice is required for this audit.

Windows grants file access according to identities and access-control entries,
not a directory name or a "do not read" instruction:
[Microsoft access-control documentation](https://learn.microsoft.com/en-us/windows/security/identity-protection/access-control/access-control).
Docker's own documentation explains why daemon access and host mounts matter:
[Docker Engine security](https://docs.docker.com/engine/security/),
[bind mounts](https://docs.docker.com/engine/storage/bind-mounts/).
A read-only mount prevents writes; it deliberately permits reads.

Before declaring the boundary sealed, use a disposable fake secret and prove:
direct reads fail; path traversal/archive tricks fail; no private host mount or
Docker socket is available; evaluator changes/signature substitution fail; forged
candidate output cannot create a pass; invalid infrastructure is separated;
developer-visible logs contain no fixture-secret canary. Do not test access by
reading the real holdout and then claim it remained unseen.

## Artifacts, owner and next action

Completed by Codex: this audit, `ops/audit_seal_20261026.py`, and the hashed receipt
`runs/audits/seal-2026-10-26/seal-contracts.json`. Independent one-case smoke passed.
Full fixture audit exited 1 because two required integrity checks failed.
Production graders, loader, holdout and account permissions remain unchanged.

Reproduce without overwriting receipts:

```powershell
.venv\Scripts\python.exe -B ops\audit_seal_20261026.py
```

Next artifact: a public `suite-v2` contract/qualification package, owned by Codex,
starting with one code-repair scenario and its correct/bad solutions. Inputs:
a public fixture snapshot, permitted edits, observable required behavior, and
protected files. Acceptance is the five-part qualification gate above, before
scaling to twelve scenarios. Existing SWE-Gym fixtures may supply development
inputs but must not be relabeled unseen holdout.

The investigation has no blocker. Actual final sealing is unfinished and depends
on choosing the intended Claude role and an evaluator identity/storage boundary.
No claim of a newly built or sealed suite is made by this document.
