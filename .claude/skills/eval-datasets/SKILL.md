---
name: eval-datasets
description: Build or extend post-cutoff eval sets (math, code, multi-hop) for the Qwen3.5-4B GRPO loop. Use whenever picking eval sources, dating items, writing eval questions, splitting train/eval/holdout, or sealing a holdout.
---

# Eval datasets: post-cutoff, few, verified

Full rules and source notes: `evals/datasets/RULES.md`. Pipeline: `evals/datasets/prepare.py`.

## Hard rules
1. **Numbers are exact.** 5 per bucket means 5. No 7/7/6, no "about 200".
2. **A review step is a hard stop.** "Show me X" means show X and wait.
3. **Write items yourself.** 5 good items beat 100 bad ones. No agent fan-out to author items.
4. **Search before building.** Only build after confirming no existing post-cutoff set fits.
5. **Stay in `evals/`.** Ask before touching training code, rollout or repo config.
6. Cutoff 2026-03-02. Eval and holdout are post-cutoff only; everything pre-cutoff goes to training. Split by root task. Seal and hash the holdout before training.

## Where post-cutoff data comes from (Sep 2026)
- **Math:** MathArena ArXivMath 03–08/2026, dated by arXiv v1. Not AIME 2026 or HMMT Feb 2026 (pre-cutoff). Apex is training only.
- **Code:** LeetCode by release date (`whiskwhite/leetcode-complete`). Tests come from statement examples plus an accepted-solution oracle, so scores run generous. LiveCodeBench tops out in 2025-04 and is training only.
- **Multi-hop:** use `evals/datasets/synth_multihop.py`, an invented world that's contamination-free by construction. Eval seed 20260302; train on other seeds; set difficulty by hop count. Nothing post-cutoff is downloadable. HotpotQA, 2Wiki, MuSiQue, MultiHop-RAG, MoreHopQA and MINTQA are all pre-cutoff.

## Checklist for every item
- [ ] The fact is dated on or after 2026-03-02: the fact itself, not the article, upload or `lastModified`. That includes the answer fact.
- [ ] Supporting text is in the prompt and states the answer.
- [ ] The answer is short and canonical, with aliases. No lists and no two-part answers.
- [ ] No false bridge (same string meaning different things). Hop 2 isn't answerable alone. No riddle phrasing and no year slips.
- [ ] Closed-book check against Qwen3.5-4B: if it answers without context, swap the item.
- [ ] The set varies topic and answer type (name, place, number, date).
- [ ] Graders are self-tested: known-good gets 1, known-bad gets 0, an infinite loop gets 0.

## Windows
Split JSONL on `\n`, not `splitlines()`. Use `PYTHONIOENCODING=utf-8`. Keep LF: write bytes, not `write_text`. For math-verify, pin antlr 4.13.2 and pass `timeout=None`.
