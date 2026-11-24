# Endpoint and dataset repair — 2026-10-26

Codex implemented the evaluation/data review with models kept off. No training,
inference, downloads, server starts/stops, private task-body reads, or resealing
were performed. Original datasets and historical results remain intact.

## Delivered behavior

- **Trusted comparison:** `evals/graders.py` keeps expected outputs and verdict
  logic in the parent. `evals/code_executor.py` sends only candidate code and test
  inputs to an explicitly configured CPU container. Candidate stdout is either
  raw program output or an exact outputs-only JSON object; a `passed/total` receipt
  cannot establish success. The old host Python execution path is removed.
- **Execution constraints:** the selected image must be an existing repository
  digest in `GRPO_CODE_SANDBOX_IMAGE`. The command disables pulls, network, host
  mounts, extra capabilities, and GPU access; it uses a nonroot user, read-only
  root, bounded temporary storage, memory/process/CPU limits and bounded captured
  output. Cleanup targets only the uniquely named container of that invocation.
  Missing infrastructure, ambiguous nonzero exits, and executor timeouts are
  invalid outcomes, not rewards. Same-interpreter assertion suites remain refused.
- **Endpoint correctness:** empty suites and malformed test inputs are invalid;
  bools do not count as integer answers. Math boxes must be at the end of delivery;
  multihop answers must be on the final nonempty line. Retracted/blank final
  answers cannot reuse an earlier success. LCB delegates to the same executor,
  removing its shared `solution.py` race.
- **Result accounting:** every selected ID and attempt slot must have exactly one
  valid outcome before headline `accuracy`/`pass_at_k` is emitted. Otherwise the
  fields are null with invalid/missing counts. Foreign IDs, malformed rows,
  duplicate attempts, and mismatched bucket/run identities fail closed.
- **Resume identity:** dataset bytes, selected IDs, rendered messages/tools,
  grader/executor source, Python version, sandbox image/profile, sampling/budget,
  and the supplied serving-identity receipt are bound together. Resume verifies
  identity before requests, tokenizer loading, and result mutation. A per-run
  lock prevents concurrent writers. Existing unbound legacy result directories
  are refused, not upgraded in place. `pass4_items` is replaced with explicit k.
- **Serving identity limit:** the receipt must identify the endpoint, alias,
  artifact hashes, timestamp and verification method. It is an external operator
  assertion bound to the run, not a fresh authenticated check of the live server.
- **Private evaluation:** this public runner refuses private/holdout options and
  paths. A separate identity/storage boundary remains required for final evaluation.
- **SFT gates:** collector and trainer require an explicit dataset registry.
  Duplicate IDs are refused before prompt mapping. Legacy stored wins lacking
  the matching registry hash and a valid grading receipt require requalification.
  Invalid grading does not become a training failure. The model opt-in gate runs
  before tokenizer/model imports, including tokenizer-loading dry runs.
- **Preparation:** first-activity upper bounds no longer establish post-cutoff
  dates. New IDs incorporate source identity and content; JSONL hashes match raw
  LF bytes; seal-verification failure aborts publication; partial rebuilds retain
  prior source provenance. No source download/rebuild/seal command was run here.
- **Tau2:** comparisons now include agent temperature, thinking and max steps.
  Its simulator/judge was not run.

## Derived public registry

The new artifact is
[`evals/data/registry-v2-20261026/registry.json`](../evals/data/registry-v2-20261026/registry.json).
It pins all four public evaluation files, original training pools, both required
endpoint sealed manifests as exclusion metadata, and available provenance logs.
Private task bodies and `roots.json` were not opened. The registry is local
development integrity metadata; it is not independent sealed authority.

| Pool | Original rows | Exact duplicates removed | Evaluation overlaps quarantined | Derived rows |
| --- | ---: | ---: | ---: | ---: |
| Math | 3,254 | 128 | 0 | 3,126 |
| Code | 3,525 | 0 | 43 | 3,482 |

All derived IDs are unique. Every normalized row retains original identity and
contributing source lineage. The loader reconstructs normalized content from the
pinned original bytes and verifies it against the derived files, including the
quarantine receipt. Changed source bytes, missing exclusion inputs, modified
derived content, removed exclusions and corrupt manifests all fail closed.

The 43 quarantines are legacy evaluation overlaps discovered by extending the
previous endpoint-only root comparison to legacy LCB question/title identities.
They do not contradict the earlier zero exact-root overlap against the newer
math/code/multihop endpoint sets. The registry has 790 exclusion identities/aliases
and evidence for 32 source names; 790 is not a count of independent task roots.

**The derived pools are not a training release.** They retain 2,457 code rows
tagged `unsupported-assert-grader`; collector preflight rejects unsupported tasks
before model loading. Other rows remain `candidate-unqualified`. The 617-row
hosted-teacher dataset was not rewritten or declared requalified; its historical
fixture/provenance gaps and hinted episodes still need review.

The original `post_cutoff_v1_5pb.json` still has five code and five multihop items,
with no math. It was not expanded under the same version name. Neither this data
repair nor the four-root outcome suite establishes broad model competence.

## Verification and remaining dependency

The final isolated-worktree suite passed **141 tests**: 64 prior outcome/instruction/
pipeline tests, 20 endpoint tests, 26 result-accounting tests, and 31 registry/SFT
gate tests. The installed checkout has its own receipt beside the source-hash
installation record in `runs/audits/eval-repair-2026-10-26/`.

Tests include two independently written correct function fixtures, an incorrect
function, counterfeit verdicts, output-schema tampering, answer-key exclusion,
crashes, missing executors, transport limits, stale answers, all-invalid results,
foreign/duplicate IDs, complete/missing trial slots, resume mutations, concurrent
writers, source collisions, raw hash checks, quarantine, metadata tampering and
no-model entrypoints. The runner blocks network, model imports and real process
launches. Only invented fixture code runs in memory; Docker transport is mocked.

**Native container execution is unverified and currently blocked.** The read-only
Docker image query failed because `dockerDesktopLinuxEngine` was unavailable.
No daemon was started and no image was selected or pulled. Configuring a pinned
image alone does not qualify the execution boundary. Model behavior and training
improvements are also untested.

Current no-model acceptance command:

```powershell
.venv\Scripts\python.exe -B ops\qualify_offline.py
```

Explicit registry selection for a future separately authorized workflow:

```powershell
$env:GRPO_DATASET_REGISTRY = (Resolve-Path evals\data\registry-v2-20261026\registry.json).Path
```

To derive a later version offline, use `evals/datasets/build_registry.py --root
<checkout> --out <new-directory>`. Existing destinations are refused. Any changed
input invalidates the earlier registry; never edit its hashes to silence a failure.

Next artifact: a CPU sandbox verification receipt, owned by Codex, using a pinned
existing Python image and only invented correct/adversarial fixtures. It must
prove isolation and trusted comparison through the actual container path before
any model execution is considered. Docker availability is the resource dependency;
additional dataset quality/coverage work can continue offline.
