# Run log

One entry per eval or training run. Newest at top.

## 2026-09-29 round 3 - SFT adapter r3 (own wins, 700 records, 88 steps, trained on RunPod 4090)
model: Qwen3.5-4B bf16 + LoRA r=8 (all 32 layers), fresh from base   adapter: lora-Qwen3.5-4B-sft-r3
change: SFT on 700 natural wins (r2's 260 + 440 r2-sampled on the pod: mhqa w3-size 3/4-hop + code pool).
  Salvaged records NOT used (salvage run died in the wifi outage; separate round if ever needed).
r3 collection (r2 sampling, pod): mhqa h3 81% / h4 57% win, cap 18% / 42%, wrong 2 | code win 51% cap 16% wrong 33%
real eval (5+5 x4, cap 4096) base -> r1 -> r2 -> r3:  mhqa 5% -> 60% -> 85% -> 90% (cap 95->40->15->10%)
                                                      code 10% -> 20% -> 40% -> 30% (n=20, noise)
pod fresh tasks (140 never-seen, x2, same tasks both): r2 -> r3
  mhqa 3/4-hop  wins 64% -> 74%  cap 35% -> 24%  solved 61 -> 70/80  median win 1795 -> 1595 tok
  code          wins 35% -> 42%  cap 13% -> 18%  wrong 52% -> 40%  solved 25 -> 29/60
out-of-domain (home, x2, cap 4096)            base    r1     r2     r3
  sealed code-v1 holdout (61 post-cutoff)     18%     50%    52%    45%   (cap 74 / 31 / 23 / 29%)
  BFCL 50                                     83%     80%    82%    82%
  When2Call 30                                80%     85%    87%    85%
  BFCL5 canary                                75%     80%    80%    80%
read: SFT's big win is base -> r1 (holdout 18% -> 50%); r2/r3 refine. r3 is clearly better on multi-hop
  (fresh 3/4-hop +10 pts, fewer caps). Code: r3 answers more carefully (fewer wrong, more caps); on the
  holdout it lands 45% vs r2 52% - inside the +/-12 pt tie zone for two 122-attempt runs, not a regression
  signal, but no gain either. Tool calling flat across all rounds. SFT is tapped out for code.
decision: r3 = new baseline and the GRPO starting point. Next: wire graders.make_task into train.py,
  sandbox code grading in a container, GRPO smoke on "sometimes right" mhqa + code tasks.

## 2026-09-28 round 2 - SFT adapter r2 (own wins, 260 records, 33 steps) + pod parity
model: Qwen3.5-4B bf16 + LoRA r=8 (all 32 layers), fresh from base   adapter: lora-Qwen3.5-4B-sft-r2
change: SFT on 260 own wins (34 pilot + 226 r1-sampled: w3-size mhqa 2/3-hop + code pool), 1 epoch, lr 1e-4, batch 8
r2 collection (r1 sampling, 160 tasks x4): mhqa win 57% cap 43% wrong 0 | code win 43% cap 23% wrong 33%
real eval (post_cutoff_v1_5pb, cap 4096), base -> r1 -> r2 (home) / r2 (pod-trained):
  mhqa w3  per-attempt 5% -> 60% -> 85% / 75%   cap-hit 95% -> 40% -> 15% / 25%   pass@4 1/5 -> 5/5 -> 5/5 / 5/5
  code     per-attempt 10% -> 20% -> 40% / 50%  cap-hit 65% -> 30% -> 30% / 20%   pass@4 1/5 -> 4/5 -> 4/5 / 5/5
off-task: BFCL canary 15/20 base -> 16/20 r1 -> 16/20 r2 (home and pod). No regression.
pod parity: RunPod 4090 (CUDA) trained the same 260 records in ~7 min vs ~58 min home (RX 9070 XT, ROCm).
  Loss curves match to ~3e-4 per step; weight updates cosine 0.978; eval within noise (each leads one bucket by 2/20).
  -> train on the pod, eval at home.
decision: keep r2 as the new baseline. Round 3: harder tasks (mhqa 3/4-hop w3-size + code), sampled on the pod.

## 2026-09-28 round 1 - SFT adapter r1-pilot34 (own wins, 34 records, 5 steps)
model: Qwen3.5-4B bf16 + LoRA r=8 (all 32 layers: attn, linear-attn, MLP)   adapter: lora-Qwen3.5-4B-sft-r1-pilot34
change: SFT on 34 own wins (20 easy 2-hop mhqa w3-12x6x6x6, 14 code pool), 1 epoch, lr 1e-4, batch 8
fresh easy tasks (20 mhqa seeds 2000-2004 + 20 code not in SFT data), 4 attempts, base -> adapter:
  mhqa  win/attempt 25% -> 56%   cap-hit 75% -> 44%   tasks solved 13 -> 18/20   median win 2503 -> 2085 tok
  code  win/attempt 15% -> 34%   cap-hit 71% -> 36%   tasks solved  6 -> 8/20    median win 1746 -> 1103 tok
real eval (post_cutoff_v1_5pb, cap 4096), base -> adapter:
  mhqa w3  per-attempt 1/20 (5%) -> 12/20 (60%) vs guess floor 8.3%   cap-hit 95% -> 40%   pass@4 1/5 -> 5/5
  code     per-attempt 2/20 (10%) -> 4/20 (20%)   cap-hit 65% -> 30%   pass@4 1/5 -> 4/5
rollout read: mhqa passes quote the facts then follow them (721-2619 tok); still zero wrong chains, all
  mhqa failures are cap hits. Code now finishes more often and exposes real bugs (wrong output, runtime errors)
  that the cap used to hide; code gain is small n and could be noise.
off-task: BFCL canary (baseline_20 bfcl, 5 items x4, cap 4096) base 15/20 -> r1 16/20, cap-hit 0 both.
  No regression. bfcl-003 is 0/4 for both (right function, args mismatch every time) - check that item's gold.
decision: keep r1 as the new baseline; next = more own-wins data (+ salvage), then GRPO smoke.

## 2026-09-28 qwen3.5-4b base - mhqa w3 (12/12/16/16 world, full-chain grading), total cap 4096
- 5 items (3x 2-hop, 2x 3-hop), prompt ~1.39k tokens, completion budget ~2.7k. pass@4, 20 concurrent.
  Results: evals/results/qwen3.5-4b-base-mhqa-w3-cap4096.
- per-attempt 1/20 (5.0%) vs follow-the-facts guess floor 8.3% (blind floor 0.5% / 0.03%).
- cap-hit 19/20 (95%). Zero wrong chains: every failure is "ran out of budget thinking".
  The one attempt that finished (2,337 tok) was a correct full chain.
- Read: termination failure, not reasoning/grader failure. DeepSeek recheck not needed (no chain rejected).

## 2026-09-28 qwen3.5-4b base - post_cutoff_v1_5pb, code+mhqa, total cap 4096 (matches SFT/GRPO)
- Same 10 items, pass@4, 40 attempts concurrent. `--total-cap 4096`: max_tokens = 4096 - prompt.
  Results: evals/results/qwen3.5-4b-base-pc5-cap4096.
- code: pass@4 1/5 (per-attempt 2/20). Hit cap 13/20 (budgets 2790-3616).
- mhqa: pass@4 3/5 (per-attempt 4/20). Hit cap 16/20 (budget ~1250: prompt is ~2.85k of facts).
- Every attempt that finished under the cap on mhqa was correct (4/4).

## 2026-09-28 qwen3.5-4b base (home, RX 9070 XT) - post_cutoff_v1_5pb, code+mhqa only - CAP MISMATCH
- Budgets (code 8192, mhqa 2048 completion) do not match training's 4096 total; superseded by the run above.
- Server: vllm-dflash-wsl:v0.27.1, bf16, thinking on (reasoning parser), max-num-seqs 40.
- Sampling T=0.7 top_p=0.8, pass@4, all 40 attempts concurrent. Results: evals/results/qwen3.5-4b-base-pc5.
- code: pass@4 4/5 (per-attempt 4/20). 9/20 attempts hit the 8192 cap -> no-code.
- mhqa: pass@4 2/5 (per-attempt 2/20). 18/20 hit the 2048 cap -> no ANSWER line;
  the 2 attempts that finished (~600 tok) were both correct.
- Math not run: arXivMath items are research-level; no post-cutoff contest source yet.
- Read: failures are mostly budget/termination (runs out of tokens thinking), not wrong answers.

## 2026-09-28 qwen3.5-2b (office) - INVALID
- Set: baseline_v1 (118 items). Server dropped mid-run.
- LCB 2/30. BFCL, When2Call, holdout: all request-error. Not a baseline.
- Outcome: fixed runner (retry request-errors on resume, health-wait, --set flag).
