# Reward audit — 2026-10-26

The current GRPO reward can give full credit to incorrect work. Saved training
reward also conceals a large gap between synthetic tasks and tasks derived from
observed Bionic failures. Increasing that reward is not sufficient evidence that
the model completes the user's work.

Audit completed offline; no blocker prevented it. No model calls, GPU work,
server changes, production edits, or sealed-task reads were performed. Existing
dirty work was preserved. This is an audit, not a grader repair or retraining run.

## Exact path and mechanism

`ops/grpo_q4.py:322` calls `task.reward(turns, workspace)`. Its gym comes from
`ops/q4_bf16_gap.py:29`: **../grpo-local-wt-filetools/sandbox-gym**, not the gym
in this checkout. `eval.make_task(native=True)` attaches `graders.reward_for`;
`bionic_real` instead uses `ops/real_tasks.py::grade` through its own adapter.

The reward is binary 0/1. `train_step`, lines 394–435, skips groups whose rewards
are identical. Otherwise each episode gets `reward - group_mean`, shared by
all its assistant tokens. For rewards [1,0,0,0], advantages are
[0.75,-0.25,-0.25,-0.25]. A false accept therefore reinforces its entire sampled
trajectory in a mixed group. This is the mechanism; the audit does not establish
that a saved model deliberately exploited these particular counterexamples.

Loss sums tokens across the batch, with detached Q4/bf16 importance weights capped
at 2.0, then divides by the total trained tokens. Longer trajectories contribute
more token terms; there is no separate correctness, preservation, or completion
component. Sequences over 10,240 prompt-plus-generation tokens are excluded after
advantages are calculated. Saved records inspected here show no such exclusions.

The older root `tasks.py` has a 0.5 answer-without-tool reward and root `rl.py`
standardizes advantages. Those are a different training path; they do not explain
the r1–r8 Q4 records audited here.

## Executed counterexamples

The existing 19-case grader audit reproduced **11 false accepts**, with seven
positive controls and one correctly rejected unchanged-workspace control.
These deliberately selected fixtures do not estimate the frequency of bad model
outputs. The same cases include comment-only edits, constant-return code,
invalid Python, negated facts, an unrelated clarification, and a false sent claim.

A new audit exercised the actual reward adapter dispatch for file-work and office
tasks, and injected an unavailable verifier into the real-task grading path:

| Submission / condition | Expected | Observed |
| --- | ---: | ---: |
| Correct file-work artifact and reply | 1 | 1 |
| Artifact denies all required facts | 0 | **1** |
| Earlier tool round contains ANSWER; final reply is empty | 0 | **1** |
| Earlier tool round contains ANSWER; final reply retracts completion | 0 | **1** |
| Truncated final reply with otherwise correct output | 0 | 0 |
| Correct artifact but wrong delivery path | 0 | 0 |
| Sandwich question containing the candidate names | 0 | **1** |
| Correct reference email draft | 1 | 1 |
| Draft falsely claims the email was sent | 0 | **1** |
| Correct draft, every original source file deleted | 0 | **1** |
| Verifier executable unavailable (injected FileNotFoundError) | Invalid attempt | **0** |

Thus the 11 wrapper/verifier checks have six false positive rewards and one
infrastructure-classification failure. These overlap the older audit; do not add
the mismatch totals as though they were independent defects.

Causes and repair locations:

1. **Text matching is weaker than the requested behavior.** Sibling
   `file_work.py:100` accepts required strings even if denied and clarification
   questions even if irrelevant. `office_draft.py:133` does not reject the tested
   sent claim or protect source files. These need explicit behavior/preservation
   contracts, not merely more reward for passing existing checks.
2. **Completion is not consistently tied to the final reply.** Sibling
   `graders.py:44` searches backwards for any ANSWER. Non-clarification file-work
   and office wrappers reuse it, allowing an earlier tool round to override an
   empty or retracted final reply. The truncation control correctly gets zero.
3. **Unavailable verification is scored as model failure.**
   `ops/real_tasks.py:177` catches OSError and returns false with `run-error`;
   the reward adapter turns that into zero. The probe mocked the missing
   executable: no submitted code was executed. This finding is distinct from
   inference request failures, which the Q4 loop generally raises rather than scores.
4. **Per-family metrics omit real tasks.** `ops/grpo_q4.py:555` iterates only
   FAMILIES, while `--mix real`/`diag` inject `bionic_real`. Across r7's 11 completed
   step records, 80 episode outcomes are absent from `by_family`. They are included
   in the overall reward. The episode file has additional attempts beyond those
   completed step records; it contains 100 real-task attempts in total.

## What happened in saved training

These are saved training-attempt rewards, not fresh evaluations or regrading of
historical artifacts with today's grader.

| Run | Positive rewards / saved attempts | Dead groups / logged groups | Step records |
| --- | ---: | ---: | ---: |
| r1 | 9/16 | unavailable | 0 |
| r2 | 2/16 | unavailable | 0 |
| r3 | 46/96 | 10/24 | 6 |
| r4 | no episode file | unavailable | 0 |
| r4b | 152/224 | 15/56 | 14 |
| r5 | 66/80 | 10/16 | 4 |
| r6 | 25/53 | 3/8 | 2 |
| r6pod | 439/668 | 55/160 | 40 |
| r7 | 135/224 | 24/44 | 11 |
| r8 | 13/39 | unavailable | 0 |

- r6pod rises from 87/176 (49.4%, steps 21–30) to 132/160 (82.5%, steps 51–60).
  Tasks and mixture changed; this is not a controlled learning curve.
- r7's 60.3% overall reward hides **107/124 synthetic successes (86.3%) versus
  28/100 real-task successes (28.0%)**. Real-task labels denote locally curated
  Bionic tasks, not an external repository benchmark.
- r7 steps 62 and 71 have four dead groups each and `stepped:false`. Step 62
  scored 100%, yet generated no training update. All-success and all-failure
  groups both give zero relative signal.
- Across every inspected episode file, no positive reward ended with `empty`
  or `truncated`. The blank-final loophole is an executable counterexample,
  not an observed exploit in these records.
- r8 has 39 saved attempts and **no steps.jsonl**. The source writes episodes
  before `train_step`; these records cannot prove an optimizer update happened.
  r5/r6/r7 also have attempts beyond their final completed step record.
- Diagnostic `reason` is not authoritative: r6pod step 54 has a clarification
  episode with `reason:ok`, `stop:truncated`, and reward 0. Use the actual reward
  and completion fields together.
- `--extend` adds attempts only after all-failure groups. Parse retries resample
  some failures and log-probability mismatches can drop attempts. Consequently
  aggregate training reward is conditional on these collection rules. Missing
  historical drop-count fields mean unknown, not zero. Current config snapshots
  also do not reconstruct every earlier step's mixture.

## Evidence and reproduction

- `ops/audit_grader_contracts.py` — existing broader counterexamples.
- `ops/audit_reward_20261026.py` — new reward-adapter and unavailable-verifier checks.
- `runs/audits/reward-2026-10-26/grader-contracts.json` — 19 cases, source hashes.
- `runs/audits/reward-2026-10-26/reward-wrappers.json` — 11 cases, source hashes.
- `runs/audits/reward-2026-10-26/summarize_training.py` and `training-summary.json`
  — reproducible training aggregates and hashes of saved inputs.

Independent delegated one-case smoke checks passed before each counterexample
suite. Both full suites exited 1 as documented: reproduced contract mismatches,
not harness crashes. Aggregation exited 0. The scripts checked that input sources
did not change during execution.

```powershell
.venv\Scripts\python.exe -B ops\audit_grader_contracts.py
.venv\Scripts\python.exe -B ops\audit_reward_20261026.py
```

Omitting `--output` preserves existing receipts. The aggregation script creates
its receipt exclusively and deliberately refuses to overwrite it.

Audit run rows are in the isolated receipt directory's `Runs.md`: the shared
`notebook/Runs.md` has unresolved OPEN ownership claims and was preserved.

## Next concrete change

Codex can own a repair of these reward contracts using the saved failing fixtures
as inputs. Acceptance: preserve correct controls, reject every deliberately wrong
submission, classify an unavailable verifier separately, require valid final
completion, and report every sampled family plus non-dead groups. The immediate
verification command is the wrapper audit above; it currently exits 1.

For code tasks, prefer executable behavioral checks and source-preservation checks.
For free-text clarification/contradiction, define a reviewed rubric before treating
text matching as correctness. Only after those checks pass should a paired,
frozen-task model evaluation test whether the repaired signal improves behavior.
