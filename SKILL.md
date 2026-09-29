---
name: grpo-loop
description: Run, evaluate, and grow Justin's local GRPO agent honestly. Use for any training run, eval, baseline, model comparison, tool stack / LSP / compiler tooling setup, or "is it actually getting better?" question on the right-hand agent.
---

# GRPO Loop: train it, grade it honestly, keep it growing

This skill is the runbook for one full cycle: **baseline → smoke → short run → eval → decide**.
Every cycle changes ONE thing. Every claim of "it got better" must be backed by the sealed holdout.

---

## Core rules (never break these)

1. **Baseline before training.** Every run starts with the untrained model's holdout score. No baseline = no run.
2. **The holdout is sealed.** The agent never reads, edits, or trains on holdout tasks. It may only run `eval_holdout` and see the aggregate number. If the holdout leaks, it's dead: rebuild it, don't reuse it.
3. **One change per cycle.** New bucket, new reward, new hyperparam, new model: pick one. Two changes = you can't tell which one did it.
4. **Reward curve over loss.** Loss in RL means almost nothing. Progress = mean reward on train + score on holdout.
5. **Read the rollouts.** Every run, read 20 raw trajectories (10 best, 10 worst). A rising reward with gibberish transcripts is reward hacking.
6. **Fixes only count when the original demonstrably failed.** No "improving" things that weren't broken.
7. **Per-bucket numbers, always.** Never trust a blended average. One bucket can quietly drop while the mean climbs.
8. **Check what it forgot.** Run the off-task set (general coding, chat, a few lyrics) before and after. A win that breaks everything else is a loss.
9. **Write it down.** Every cycle ends with a run log entry, including failed and killed runs. Especially those.

---

## Training pipeline (canon, decided 2026-09-28)

Change only by explicit decision; note the date and why.

**Order**
1. **Baseline** first. The failure type picks the tool:
   - *Format / termination* (right idea, wrong shape, no answer line, runs out of budget) → SFT on own wins.
   - *Sometimes right* → straight to GRPO.
   - *Never right* → make the task easier first (fewer hops, easier items). Own-wins SFT can't help with no wins.
2. **SFT on own wins** (rejection-sampled): keep only graded wins that finish under budget, max 2 per root task. Small: a few hundred records, 1 epoch, low LR. If wins are rare on the real items, collect on easier generator settings; "wrap up and answer" transfers.
   - **Salvage** capped failures (collector saves them to `<name>.fail.jsonl`): cut the thinking at a conclusion point, insert only `</think>`, let the model generate its own answer from there, keep it only if that answer grades correct. The ending is the model's own tokens except one inserted stop; a cut made mid-hypothesis filters itself out. Never write the answer line ourselves. Tag records `salvaged` with the cut index, so natural vs natural+salvaged can be compared.
3. **Re-baseline** on the same items. SFT should cut cap-hits without dropping accuracy.
4. *Optional:* if SFT helped but wins are still rare, collect again with the SFT'd model and SFT again.
5. **GRPO** for checkable buckets (code, multi-hop, find-and-fix). Rollouts are collected live during GRPO, graded, and learned from immediately. No separate collection step.
6. **DPO**, later and separate, only for things with no checker (summary quality, style), from human-picked pairs.

**Buckets**

| Bucket | Checker | SFT | GRPO | DPO |
|---|---|---|---|---|
| Code | tests pass | own wins | yes | - |
| Multi-hop (invented world) | full chain, every hop exact | own wins | yes | - |
| Find-and-fix (injected bugs) | tests pass | own wins | yes | - |
| Summarization | behavior only | behavior filter | **no** | quality, later |

- **Find-and-fix guard:** keep an injected bug only if the original passes every test **and** the bugged version fails at least one, or "no change" counts as a fix.
- **Summarization:** baseline first (Qwen's post-training likely covers it); SFT only on a specific observed failure. The SFT filter checks behavior, not quality: finished under budget, requested format and length, every name and number appears in the source. Never use it as a GRPO reward; it is easy to game.

**Rules**
- **One budget:** 4096 tokens total (prompt + completion, thinking included) for eval, SFT collection, and GRPO. Eval: `--total-cap 4096`.
- **Thinking on** everywhere.
- **Headline metrics:** per-attempt accuracy and cap-hit %, per bucket, never blended. pass@k is secondary. Multi-hop is reported next to its guess floor.
- **Eval items:** post-cutoff (published on or after 2026-03-02 for Qwen3.5) or contamination-free by construction (invented worlds). 5 verified items per bucket.
- **Split by root task**, never by difficulty; the leak gate fails closed (`sft/collect.py`, `sft/train_sft.py`, `evals/holdout/seal.py`).
- **Token-exact SFT:** train on the raw `completion_ids` (EOS included), never re-tokenized text. The loss covers completion tokens only, via the same `rl._masked_logprobs` GRPO uses.
- **One phase on the GPU at a time:** stop vLLM before training; for eval, start vLLM and hot-load adapters with `/v1/load_lora_adapter`.

**Harness backlog (priority order):** (1) one budget constant read by eval, collection, and GRPO; (2) cap-hit % in every training log; (3) `graders.make_task` wired into `train.py`; (4) off-task canary of 5 tool-calling items (When2Call/BFCL) run with every eval; (5) training refuses to start while vLLM or a stray process holds VRAM, and names it.

---

## Token + loss invariants (the silent-bug zone)

Check these before any real run. Each one fails silently.

- [ ] **Never re-encode generated text.** Append sampled `gen_ids` straight to the running `tok_ids`. `gen_text` is for parsing and logs only.
- [ ] **Never tokenize a joined string.** Prompt, each generation, each observation: encode separately, concatenate id lists.
- [ ] **Every id has a `trainable` flag.** Model-sampled = True. Prompt, observations, template/role wrappers = False.
- [ ] **EOS / end-of-turn stays in and stays trainable.** The model must learn when to stop.
- [ ] **Parse actions from ids or with special tokens kept.** Never `skip_special_tokens=True` before parsing.
- [ ] **Chat template applied** to the prompt, and observations wrapped in the tool-role tokens as their own non-trainable segment.
- [ ] **One optimizer step per rollout batch.** The no-importance-ratio shortcut is only valid under this. 2+ steps on the same rollouts = bring back ratio + clipping.
- [ ] **KL uses the k3 estimator:** `exp(ref - pi) - (ref - pi) - 1`. Not `pi - ref`.
- [ ] **log_softmax in float32.** Cast logits `.float()` first.
- [ ] **Reference = base with adapter disabled** (`policy.disable_adapter()`), not a second loaded model.
- [ ] **Tokenizer + chat template pinned** in one frozen config shared by sampling and training.

---

## The cycle

### 0. Pick what this cycle tests
Write one sentence: *"This cycle tests whether ___ improves ___."*
If you can't write it in one sentence, the cycle is too big.

### 1. Baseline (no training)
- Run the current model on the **holdout** → record score per bucket.
- Run **pass@8 per prompt** on the training pool.
- Keep only prompts that land between **10% and 90%**. Always-solved and never-solved prompts give zero GRPO signal.
- Run the **off-task set** → record.
- If fewer than ~30 prompts survive the 10-90% filter, the bucket is too easy or too hard. Adjust the tasks, not the model.

### 2. Smoke run (~10 steps, tiny model or tiny set)
Checking plumbing only, not learning.
- Loss is finite, grad norm is sane, nothing NaNs.
- Rewards **vary within groups**. If every group is all-0 or all-1, nothing will train.
- Action parse rate is logged and non-zero.
- Save a checkpoint, kill the process, resume. Find out now, not at step 400.

### 3. Short run (100–200 steps)
Log every step:
- mean reward (per bucket)
- dead-group fraction (groups with identical rewards)
- action parse rate
- mean trajectory length / turns
- KL to reference

Expect a **two-phase curve**: parse rate climbs first (format), then reward climbs (skill). Don't judge phase 1 by reward.

**Kill the run early if:**
- dead-group fraction > ~70% for 20+ steps
- KL shoots up while reward is flat (drifting without learning)
- trajectory length explodes or collapses to 1 token
- reward climbs but the 20 transcripts look wrong

### 4. Eval
- Holdout score per bucket vs baseline.
- Off-task set vs baseline.
- For quantized deployment (QAT / GGUF), **eval the quantized model too**, not just bf16.
- Read 20 rollouts. Write 2-3 sentences on what changed in behavior, in plain words.

### 5. Decide
| Holdout | Off-task | Decision |
|---|---|---|
| up | flat/up | **Keep.** Checkpoint becomes the new baseline. |
| up | down | Keep the adapter separate, or lower LR / raise KL. Don't merge. |
| flat | flat | The change didn't matter. Revert, try a different one. |
| down | any | Revert. Log why. |
| up on train, flat on holdout | any | **Memorization.** Tasks aren't varied enough. Add perturbation / procedural generation. |

---

## Bucket order (add one at a time)

1. **Tool-call correctness** on your MCP tools (most transferable, start here)
2. **C++ edits** on plugins: reward = applies + compiles + tests
3. **Multi-hop** in repos: reward = correct file/line/answer
4. **JUCE realtime safety**: reward = no allocs/locks in processBlock
5. *(separate adapter later)* MIDI grooves, lyrics

When adding a bucket, re-run every earlier bucket's holdout. Earlier buckets must hold.

## Keeping tasks un-memorizable

- **Split by repo, not by task.** Holdout = plugins it never trains on.
- **Bug injection.** Break working code at random spots each episode; the original is the answer key.
- **Procedural prompts.** Random combos of params so the task space is effectively infinite.
- **Time split.** Anything written after the run goes straight to the holdout.

## Reward rules

- Mechanical only. No LLM judge.
- Sparse is fine (1.0 / 0.0). Partial credit only when it blocks a specific hack (e.g. correct-but-no-tool = 0.5).
- Reward the **result**, not the behavior. "Called a tool" alone = spam. "Called a tool and the answer was right" = skill.
- Any task with a guessable answer (yes/no, audio/ui) must require a tool call for full reward.

## Agent tools (the six, nothing more)

Small tool list or a 9-12B model chokes. Six tools, descriptions one line each:

- `definition(symbol)` — LSP. Exact location.
- `references(symbol)` — LSP. The 4 real uses, not 40 grep hits. This is what makes multi-hop cheap.
- `diagnostics(file)` — LSP. ~1s feedback before a full build.
- `build()` — compiler. The mid-rollout ground truth: read errors, fix, rebuild.
- `read(file)`, `edit(file)` — the file surface.

**LSP is a hint, never the reward.** clangd can be stale, especially mid-edit. Reward stays terminal: applies + compiles + tests. Diagnostics-clean ≠ done; the compile reward already punishes skipping the build.

**Compiler tool differs from compiler reward, same definition of "error":**
- Same flags, same rules, both. If W4 matters it shows in both. Only differences: the tool build is incremental/cached/trimmed for speed; the reward build adds checks the tool doesn't run (clean build, tests, realtime-safety scan).
- Never lenient-tool + strict-grader on the SAME check — the model then fails on warnings it never saw. Unexplainable failure, unlearnable. Stricter by *adding* checks, not by moving goalposts.
- Trim output: first 3-5 errors, deduped, `file:line` + message. MSVC spew drowns a small model and eats context.
- Budget builds: 3-4 per episode, with timeout. The cap is shaping, not resource management — it's what makes "diagnostics first, build only when clean" the discoverable ordering.

**Reward may read the trajectory ONLY to block a specific hack, never as a quality proxy.** "Used a tool before a guessable answer" (kills the audio/ui guess path) = fine. "Called build 3 times = good process" = banned. Same rule as `reward_tool_discipline`.

**Train it with, deploy it with.** Same six tools, same output format, same rendering. Pin clangd and the MSVC toolset like you pin the tokenizer. Difference = distribution shift.

**JUCE/Windows specifics:**
- `compile_commands.json` needs CMake + Ninja (`CMAKE_EXPORT_COMPILE_COMMANDS=ON`). MSBuild/VS generator doesn't produce it.
- Build one static index via `clangd-indexer` on the compile DB. Each rollout worktree points clangd at it with `--background-index=false --index-file=<path>`, so only opened/edited files get live parse. Stale-file staleness is covered by the open-file parse; verify worktree paths line up once before trusting it.
- Each rollout gets its own git worktree + build dir. Two rollouts in one folder clobber each other into garbage rewards.

**Did LSP earn its seat?** One cycle: same bucket, same model, ONLY change = LSP tools added. Holdout vs the grep-only baseline. That's the experiment.

---

## Run log format

Append one entry per cycle to `runs/LOG.md`:

```
## <date> — <one-sentence cycle goal>
model: <name + quant>          adapter: <name or none>
change: <the ONE thing changed>
baseline holdout: <bucket: score, ...>
result holdout:   <bucket: score, ...>
off-task:         <before → after>
steps: <n>   dead-group %: <avg>   parse rate: <start → end>
rollout read: <2-3 sentences of what actually changed>
decision: keep | revert | separate adapter | kill
why: <one line>
```

---

## Tonight's messy run (first time through)

Goal: get **one full cycle** working end to end, badly. Clean comes later.

1. Fix the scaffold blockers: circular import, re-encode, EOS drop, reward logging, hop2 guessable.
2. Make a **tiny holdout** (5-10 tasks) in a folder the training code never touches.
3. Baseline Qwen2.5-0.5B on holdout + pass@8 on the 3 seed tasks.
4. Smoke run: 10 steps, group 4. Confirm the smoke checklist.
5. Short run: 50 steps.
6. Eval, read 20 rollouts, write the first `runs/LOG.md` entry.

Success tonight = the log entry exists and the numbers are real, even if they're bad.
Once the loop works on 0.5B, the only change next cycle is swapping in the real model (Gemma 4 12B QAT or Qwen 3.5 9B, whichever wins its baseline).
