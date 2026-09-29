# Next: workspace gym (real tools, real files) - plan

## North star (the goal, written 2026-09-29)

**A 4B that keeps a 200-turn episode alive by writing its own notes and forgetting the rest.**
GRPO on compaction/note-taking, graded only by whether the notes actually got her to the finish line. Not a bigger
context window - her deciding what matters. That's the one that removes her ceiling: the intern who knows one office
becomes the intern who never goes home.

Built on the rebuilt-context design below (every call a fresh, small prompt: task + her notes + recent results),
with long episodes graded by final outcome plus checkpoints.

**Reward-hacking guard for notes (expect it in the first runs):** she will try writing "TODO: everything is fine"
and claiming victory. So:
- Notes are **never** graded directly and never read by the grader. Reward comes only from the task's mechanical end
  state (exact answer / hidden pristine tests / end-state checks) - which a note can't fake.
- The final answer must stand on its own: the grader re-derives success from the workspace, never from what she
  says she did.
- Canary items where the honest move is "not done yet": a note claiming completion there = flagged run.
- Read the notes in the rollouts every run (canon: read 20 transcripts). Rising reward + nonsense notes = hacking.

First sandbox numbers (r3, never trained on these tools): Find 7/7 on seed 9000; on seeds 9001-9005
62/70 attempts (89%), 34/35 tasks; zero jail escapes attempted.

Written 2026-09-29, after round 3. Current best adapter: **r3** (see runs/LOG.md).
Goal: get her working in a real workspace with file tools, so she can be driven from a terminal CLI.

## 1. Freeze the tool set first (everything downstream depends on it)

She learns the harness, so these names and schemas must be identical in the gym and in the CLI she'll be driven with.

| Tool | Args | Notes |
|---|---|---|
| `read_file` | path, (offset, limit) | output capped |
| `list_dir` | path | output capped |
| `search` | pattern, (path, glob) | ripgrep; output capped (first N hits + "N more") |
| `edit_file` | path, oldString, newString | exact-string replace; oldString must match exactly once |
| `write_file` | path, content | new files only |
| `run_tests` | (target) | runs in the container |
| `answer` | text | ends the episode |

- **edit_file is exact-string replace, no line numbers.** Small models miscount lines and every edit shifts later lines.
  Errors teach: "0 matches" / "3 matches, add more context". Arg names match OpenCode (`oldString` / `newString`).
- Tool calls in Qwen's native format via vLLM's tools API, so any OpenAI-compatible client can drive her later.
- **Later (C++ / JUCE buckets):** LSP `definition`, `references`, `diagnostics` + `build`.
  - LSP is a hint, the compiler is the judge; same definition of "error" in both.
  - clangd: one static index built up front with clangd-indexer; per workspace `--background-index=false
    --index-file=<path>` with synchronous diagnostics, so results are deterministic and fast.
  - Cap builds at 3-4 per episode (teaches think-before-compile).
  - If we lift a CLI's terminal/tool runtime as ours: check the license, pin the version.

## 2. Sandbox: split tools by risk

- **File tools** (`read_file`, `list_dir`, `search`, `edit_file`, `write_file`): plain Python, Windows paths,
  **path jail**: every path resolved and checked to be inside the episode's workspace folder. Nothing outside.
- **Execution** (`run_tests`, later `build`): Docker container, **no network**, only the workspace mounted,
  CPU/memory/time limits, thrown away after. Linux is fine for Python tests (Windows Home can't run Windows Sandbox).
- The same container becomes the **code grader for GRPO** (today model-written code runs as the user on the host).

## 3. Workspace + repo generator

- Each episode: fresh copy of a seeded, generated repo in a scratch folder, wiped after.
- Generator: small codebases with planted targets; contamination-free by construction, like the multi-hop worlds.
- Eval = 5 verified items per gym from a held-out seed (+ ideally one held-out task family).

## 4. Gym ladder (each ring adds one tool she must get right)

1. **Find**: "where is X defined / who calls it / what does it return". Exact answer. (read_file, list_dir, search, answer)
2. **Find + edit**: make a specified change. Graded on the end state. (+ edit_file, write_file)
3. **Find-and-fix**: injected bug (flip a comparison, off-by-one, drop a null check). Keep only if the original
   passes all tests AND the bugged version fails at least one (else "no change" passes). Graded by hidden tests. (+ run_tests)
4. Later: scripting chores (hidden second fixture, no-op guard), PDFs (generated text + scanned variant), C++ with LSP/build.

## 5. Harness changes this forces

- **Multi-turn rollout/collector**: she calls a tool, harness runs it, result appended as a NON-trainable
  observation segment, repeat until `answer` or budget. Raw token ids throughout (rollout.py invariants).
- **Observations capped** so one huge search/read can't blow the budget.
- **SFT record v2**: per-segment trainable mask (her turns trainable, observations/prompt not).
- **Budget: 8192 per call for the agent bucket** (decided 2026-09-29). First tasks are small; safe on both machines.
  - Measure, don't guess: one training step at **16k and 32k on the pod**, read the real peak (vram.step_report).
    Qwen3.5 is 24 linear-attention + 8 full-attention layers and modern kernels don't build the square matrix, so memory
    should grow ~linearly; vram.py's T^2 attention term is likely pessimistic - recalibrate it from the measurement.
- **Compaction as a harness turn**, built before tasks get long enough to need it.
  - First version: harness elides old tool outputs (`[read_file src/x.py: 3,200 tokens, elided]`), keeps her own turns.
  - Later: she writes the summary herself (`compact` turn).
  - **Split training at every compaction point.** A trajectory is never trained as one sequence across a compaction:
    each segment trains with the context she actually saw, and the summary is the prompt of the next segment.
  - **Compaction becomes gradable:** a summary is good if she can still finish the task after it. That's the
    summarization skill with a real checker (credit flows through the task's final grade in GRPO).

## 6. Tomorrow's order

1. Freeze tool schema (table above).
2. Path-jailed file tools + container runner for run_tests.
3. Repo generator + Find gym (5 eval items, fresh seeds).
4. Multi-turn collector + SFT v2 record.
5. Baseline r3 on Find. Read the transcripts. Then the usual: format failure -> own-wins SFT; sometimes right -> GRPO.

## Target design: rebuilt 8k context per call (decided 2026-09-29; compaction is the fallback)

Every call gets a fresh, hand-built prompt instead of an ever-growing history. The small-model way to do long jobs.

- **Prompt layout, fixed order:** task (stable, first -> vLLM prefix cache hits) -> her notes file -> her plan ->
  last few tool results (capped). Never her earlier thinking (matches Qwen's template, which drops prior reasoning).
- **Notes file:** lives in the jailed workspace, hard cap ~1.5k tokens, written with the normal file tools.
  What she writes down decides whether step 80 still knows what step 3 found -> note-taking becomes a graded skill
  (graded by whether she finishes).
- **Why it fits a 4B:** every call looks like training (same size/shape, inside the budget she's good at);
  episodes can run for hundreds of steps; each call is its own token-exact training segment - no compaction seams.
- **GRPO credit:** one pass/fail after 100 calls is almost no signal. Start with short episodes (<~15 calls) and
  grow them; give long tasks graded checkpoints ("found the file", "module A tests pass").
- 32k + compaction stays as a fallback lane for tasks that need raw long context.

### Call format: the stateful loop (decided 2026-09-29)

Never spend a call only to update state - every call does work AND updates state. Minimum loop = **1 + k calls**
(k = tool actions); a find task is ~3-4 calls, a real fix ~10-15.

| Call | Harness injects | She outputs |
|---|---|---|
| 1. start + plan | task, notes file, prior handoff (if resuming) | short plan + first tool call + `NOTE:` line |
| 2..k. run | task, notes, plan, last few tool results | one tool call + `NOTE:` line |
| last. answer + handoff | same | `ANSWER:` + handoff summary (written to the handoff file) |

- **`NOTE: <one line>`** ends every output; the harness appends it to the notes file (capped, oldest dropped first).
  Fixed format so parsing is reliable; no separate "update notes" calls.
- The plan is written once in call 1 and kept in the notes/plan slot of every later prompt.
- A **separate handoff call** only when the task pauses unfinished (budget/time out) - that handoff is what the next
  session resumes from. **Graded:** does a fresh session finish from the handoff alone?
- Every call is one token-exact training segment. **Headline metrics: success rate AND calls-per-success.**

### Sub-agents (same lane, later)

- **Orchestrator** call: holds the task + notes, splits work, writes the plan. **Workers**: each starts in a clean
  context on one sub-task (e.g. "find where X is computed", "fix module A") and returns a short report.
- Only the report re-enters the orchestrator's context, never the worker's history.
- Each worker call is its own token-exact, gradable segment: worker graded on its sub-task (exact answer / tests),
  orchestrator graded on the final task result. Reports are graded the same way notes are: can the orchestrator
  finish with them.
- Same model (same adapter) in both roles, with role-specific system prompts; one lineage.

### Context-size tiers (same design, three budgets)

Run the rebuilt-context design at three per-call budgets and compare on the same tasks - the budget is a knob, not a fork:

| Tier | Per-call budget | Notes cap | Recent tool results | Use |
|---|---|---|---|---|
| 8k | 8,192 | ~1.5k | last 2-3, capped | default; cheapest; closest to what she was trained on |
| 16k | 16,384 | ~3k | last 5-6 | bigger files / more evidence per decision |
| 32k | 32,768 | ~6k | last 10+ | raw long-context tasks; also the base for the 32k+compaction fallback |

- Before 16k/32k training: measure real VRAM for one training step at 16k and 32k on the pod (see Budget above).
- Serving: the pod runs 8192 today; 16k/32k tiers need `--max-model-len` raised (costs KV cache / fewer parallel requests).
- Eval each tier on the same gym items: success rate, calls per episode, tokens per success. Pick per task family,
  not globally - small tasks may be best at 8k even if big ones need 32k.
- Train tiers in one lineage (mixed budgets in the SFT/GRPO data) so one adapter works at every size.

## Lineage decision (decided 2026-09-29)

Same lineage, not a new adapter: workspace tools are core agent skills and need her clean thinking from r1-r3.
1. Baseline **r3** on the workspace gyms first.
2. If she fumbles the tool calls (bad format, never stops calling tools): SFT round **r4 = fresh from base on
   r3's 700 records + her workspace wins** (old data rides along so multi-hop and code don't regress).
3. Regression suite on r4: w3 multi-hop, code real eval, sealed code holdout, BFCL/When2Call, canary. Nothing drops.
4. Then GRPO on everything live.
- If r3 handles the tools cleanly out of the gate, skip SFT and go straight to GRPO.

## Standing rules (from tonight)

- **One lineage.** Each new skill trains on its new data plus a slice of old skills' data (replay), so she doesn't forget.
- **Every old eval is a regression test** on every round: w3 multi-hop, code real eval, sealed code holdout, BFCL, When2Call, canary.
- **Merge when settled.** Once a skill set is solid, merge the adapter into base; the next LoRA starts from there.
- **Keep every checkpoint.** A regression rolls back one step, not to zero.
- **LoRA rank:** 8 now; if a skill looks capped, try 16/32 before anything drastic.
- **GRPO** sharpens what she sometimes does; it doesn't add capacity and it exploits grader loopholes, so the
  container grader comes before GRPO. Also still to do: wire `graders.make_task` into `train.py`.
- **Train on the pod, eval at home** (parity verified in round 2).

## Using her from a terminal

- Now: any OpenAI-compatible client at `http://127.0.0.1:8000/v1`, `model: r3` (thinking arrives in `reasoning_content`).
- OpenCode can use her as a custom provider, but its tools and its multi-thousand-token system prompt are off-distribution
  until the gyms use the same interface. Pick the CLI, then build gyms around its tools and prompt size.
- Without vLLM: merge LoRA into base -> GGUF -> quantize; re-run the evals on the quantized file before trusting it.

## Open decisions

- Our six tools (table above) vs OpenCode's full tool set as the canonical interface.
- Rotate the pod's vLLM API key (it was pasted into chat).
- ~~8192 budget~~ decided: 8192 now, measure 16k/32k on the pod, compaction turn before long tasks.
