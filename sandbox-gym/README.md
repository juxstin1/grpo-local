# sandbox-gym — the opencode-style sandbox loop for evals and GRPO

A cleaner, self-contained harness for the plan in
`..\docs\NEXT-workspace-gym.md`: an agent (a served model) works a
real, seeded workspace through a path-jailed tool set, ends every episode with
a plain `ANSWER:` line, and is graded mechanically. Same lineage, same numbers
as `grpo-local`, but the loop runs against real files instead of a mock env.

## The ten tools, in three groups (frozen)

| Group | Tools | Ring that unlocks it |
|---|---|---|
| **Files** (jailed) | `list_dir`, `read_file(path, offset, limit)`, `search(pattern, path, glob)` (ripgrep, capped), `edit_file(path, old_string, new_string)` (exact match, once), `write_file` (new files only) | Find gym: 1–3. Find + edit: add `edit_file`/`write_file` |
| **Code intelligence** (LSP, hint never grade) | `definition(symbol)`, `references(symbol)`, `diagnostics(path)` | C++/JUCE repos, last |
| **Execution** (container only, capped) | `build()`, `run_tests(filter)` | Find and fix: add `build`/`run_tests` |

Tool names/args copy OpenCode where they overlap. Wire format is Qwen-native
JSON — `{"name": ..., "arguments": {...}}` — so served eval == local GRPO.

Rollout order by feedback speed: **Python** (find → find+edit → find-and-fix),
then Rust, then C++/JUCE. Python comes first because it's cheapest to grade.

## Layout

```
config.py      one budget (8192), the tool rings, train hyperparams
schema.py      Action / Turn / Trajectory / Task
tools.py       the frozen tool set + tolerant action parsing
workspace.py   the path jail + tool executors + output caps
runner.py      build/run_tests in a throwaway no-network container (host=selftest)
lsp.py         LSP seam (Python stopgap; clangd arrives with the C++ bucket)
repo_gen.py    seeded Python workspace generator + find-and-fix guard
graders.py     mechanical graders, shared by eval and GRPO reward
gyms.py        spec -> Task + system prompt + workspace lifecycle
agent.py       the opencode loop, served side (eval + collect)
encoding.py    token-exact segment encoding (local GRPO path)
eval.py        eval runner over a sealed holdout (aggregate-only when sealed)
seal.py        holdout seal: canary + sha256 + read-only
collect.py     SFT collector: multi-turn, token-exact, leak-gated
rollout.py     local-torch GRPO trajectory sampling (fresh workspace per roll)
rl.py          GRPO loss (k3 KL, chunked logprobs, trainable masks)
vram.py        VRAM ceiling guard + PID lock (ported from grpo-local)
train.py       GRPO training loop
selftest.py    offline tests (no torch/GPU/network)
```

## Tomorrow's order (eval + GRPO)

1. **Selftest** — prove plumbing: `.\cli.py selftest`
2. **Serve** the base or an adapter on vLLM (see AGENTS.md).
3. **Baseline** r3 on the Find holdout:
   `.\cli.py eval --family find --attempts 4 --slug r3-find --served-model r3 --sealed`
4. **Read the transcripts** (results are written under `data/results/<slug>/`).
   Fumbles tool format → `collect` + SFT; sometimes right → `train` (GRPO).
5. **GRPO**: `.\cli.py train --family find --seeds 1000-1009 --steps 50 --group-size 6`
6. Re-eval, write one `runs/LOG.md` entry.

## Rules (inherited from grpo-local/SKILL.md)

- Baseline before training; the holdout is sealed; one change per cycle.
- Reward is mechanical and binary; reward the endpoint, never the path.
- A truncated final turn earns 0. Cap-hit % is reported, never hidden.
- Split by **seed**, never by string; holdout seeds are never generated for train.
- `--runner host` is selftest-only; production grading is container-only.
- Run `.\cli.py selftest` after any change to tools/workspace/repo_gen/graders.