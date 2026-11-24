# Eval & training data rules — Qwen3.5-4B GRPO

## Active repair policy — 2026-10-26

This section supersedes conflicting historical claims below. See
[`docs/EVAL-REPAIR-2026-10-26.md`](../../docs/EVAL-REPAIR-2026-10-26.md).

- No model runs are authorized. The original datasets remain unchanged historical inputs.
- First-activity upper bounds cannot prove post-cutoff publication, even with a
  seven-day margin or larger problem ID. Such code dates are ambiguous.
- Public development evaluation uses all registered evaluation exclusions, not
  just the five-per-bucket subset. Missing/corrupt registry sources fail closed.
- New derived pools carry unique content/source identities and explicit overlap
  quarantine. An identity/provenance check is not a task-quality qualification.
- Candidate code and expected answers run in separate trust domains. Code grading
  requires the configured pinned CPU container; legacy same-interpreter assertion
  suites remain disabled. Missing infrastructure is invalid, never reward zero.
- Final/private evaluation requires a separate evaluator identity. Do not read or
  reseal private task bodies as part of a repair. Existing manifests are used only
  as exclusion metadata here.
- The 5pb file currently contains five code and five synthetic multihop items,
  with no math. Historical counts and grader self-test claims below are not current
  qualification. Only new offline receipts establish repaired-source behavior.

## Historical dataset policy and source notes

1. **One model.** `Qwen/Qwen3.5-4B`. Cutoff = **2026-03-02**, its public release date. Qwen publishes no training cutoff, so the release date is the only defensible bound.
2. **Eval = post-cutoff only.** Every eval and holdout item is dated on or after 2026-03-02. No pre-cutoff item appears in any eval.
3. **Train = pre-cutoff only.** Everything dated before the cutoff goes to the training pool. Post-cutoff items never train.
4. **Dates must be proven, or the item is dropped (fail closed).**
   - arXiv: v1 submission date from the arXiv API. Competitions: exam date.
   - LeetCode: the first-activity date is only an upper bound. Eval requires it to be at least cutoff + 7 days, and the problem ID must be above every pre-cutoff ID.
   - Anything that can't be proven to fall on one side is dropped.
5. **Split by root task, not string.**
   - A problem, its renamed-variable and reworded variants, its I/II versions, and its copies in other datasets all share one `root_id`.
   - Roots are recovered from lineage IDs (contest + problem number, arXiv ID, LeetCode slug, Wikidata/MuSiQue hop IDs). Skeleton near-duplicate matching adds more links: numbers, single-letter variables and declared identifiers are masked before shingle Jaccard comparison.
   - A whole root sits on exactly one side.
   - If any member of a root is pre-cutoff, the whole root is pre-cutoff and goes to training. Its post-cutoff variants are dropped.
6. **Holdout sealed before any training.**
   - 40% of post-cutoff roots per dataset are sealed by `evals/holdout/seal.py`. Sealing stamps a canary, makes the files read-only, and writes a SHA-256 manifest, per-root hashes and a merkle root to a hash-chained ledger (`evals/holdout/SEALS.log`).
   - Every training pool and the eval set must pass `seal.py check` against every seal. Unrooted records fail.
   - The holdout is scored aggregate-only (`eval_runner.py --holdout-seal`) and never read, printed or trained on.
7. **Reward the endpoint, not the path.** `evals/graders.py` is shared by eval and GRPO reward and is binary (1/0):
   - math: the last `\boxed{}` checked with math-verify;
   - code: every test passes;
   - multi-hop: normalized exact match of the final `ANSWER:` line against the gold answer or an alias.
   - Reasoning is stripped and never scored. A truncated final turn scores 0.
8. **Standalone graders only.** No dataset that needs its own agent loop or an LLM judge. Everything runs in the existing `eval_runner.py` and `rl.py` harness.
9. **Everything is logged.** Each run writes `logs/prepare-<UTC>.log` and `.json`: source revisions, file hashes, per-stage drop reasons, clustering stats, counts and seal hashes. The table below is regenerated from that log.

## Known limits

- **Math eval is research-level.** It uses ArXivMath: closed-form answers from arXiv papers, verified by MathArena. The only post-cutoff competitions with public machine-readable answers would be SMT, BRUMO and CMIMC 2026, and MathArena hasn't published them. Expect a 4B model to score near zero on ArXivMath.
- **Code tests are the statement examples.** Expected outputs come from an OJ-accepted solution and are cross-checked against the stated outputs. That's usually 2–3 tests per problem, so some wrong solutions will pass. Treat code scores as generous.
- **Multi-hop eval is generated, not downloaded.** It uses the AntiLeakBench method (Wikidata facts that changed after the cutoff, plus Wikipedia context), run against the live Wikidata query service because the 103 GB dump doesn't fit on disk. HotpotQA, 2Wiki and MuSiQue are all pre-cutoff and are training data only.
- **Legacy sets predate these rules.** `baseline_v1` and `baseline_20` are pre-cutoff (LCB 2025, BFCL, When2Call). They are not evals under these rules; report them separately as legacy.

## Working rules

- **Numbers are exact.** "5 per bucket" means 5. Don't invent splits like 7/7/6 and don't round up to "about 200". The baseline set is `data/sets/post_cutoff_v1_5pb.json`.
- **A review step is a hard stop.** "Show me X" means show X and wait. No building, scaling or sealing past a checkpoint.
- **Write eval items yourself and keep them few.** 5 good items beat 100 bad ones. No agent fan-out to author items.
- **Search before building.** Build only after confirming no existing post-cutoff set fits.
- **Stay inside `evals/`.** Anything outside it (training code, `.gitignore`, rollout) needs an explicit ask.

## Source knowledge (as of 2026-09-28)

**Multi-hop**
- The famous sets are all pre-cutoff, so they're training pool only: HotpotQA (2018), 2Wiki (2020), MuSiQue (2021), MultiHop-RAG (2023 news), MoreHopQA (2024), and MINTQA. MINTQA's "new" facts come from a 2024 Wikidata dump, despite the ACL 2026 venue.
- No downloadable post-cutoff multi-hop set exists. MedHopQA withholds its answers, so it can't be graded locally. "BRIDGE" on Hugging Face is an unrelated bias-classification set.
- Post-cutoff multi-hop has to be built, in one of two ways:
  - the AntiLeakBench method: Wikidata claims whose start time falls after the cutoff, plus Wikipedia context. The dump route needs 103 GB; the SPARQL route avoids it.
  - dated news QA: `youdotcom/LiveQA`, `liveqa-deep`, `liveqa-standard`, covering 2026-08-18 onward.

- **Chosen: invented-world multi-hop** (`evals/datasets/synth_multihop.py`, vendored from llamacpp-prebuilt `tuning/fleet/multihop.py`). The world is invented per seed, so it's contamination-free by construction and needs no dating. The eval uses seed 20260302: three 2-hop and two 3-hop items, each about a different person. GRPO pools use any other seed, with difficulty set by hop count (2–5). The hand-chained LiveQA questions were dropped.

**Math**
- Source is MathArena. The ArXivMath monthly sets (0326–0826) are post-cutoff when judged by each paper's arXiv v1 date. For example, `2603.00693` was submitted 2026-02-28, so it's pre-cutoff.
- AIME 2026 and HMMT Feb 2026 are pre-cutoff. The Apex shortlist is 2025 competitions, so training only.
- SMT, BRUMO and CMIMC 2026 aren't published on Hugging Face.
- License is CC BY-NC-SA. Expect a 4B model to score near zero on ArXivMath.

**Code**
- LiveCodeBench's public release lags; its newest problem is from 2025-04. It's training only.
- Post-cutoff code means LeetCode problems selected by release date (`whiskwhite/leetcode-complete`). Tests are built from the statement examples, with expected outputs from an accepted solution. That's only 2–3 tests per problem, so scores run generous.
- `neulab/leetcode` has real assert tests but is pre-cutoff, so it's training only.

## Dating

- **Date the fact.** Not the article, the upload, or the repo's `lastModified`: new articles restate old facts. For every hop, the answer fact itself must be post-cutoff.
- **Proof comes from the fact's own record.** arXiv v1 date, exam date, Wikidata start time, or LeetCode problem release (plus a 7-day margin, because that date is an upper bound). If a date can't be proven, the item is dropped.

## Building multi-hop items

- **Failure modes to reject:**
  - false bridges: the same string meaning different things, e.g. "Austen" the storm vs Jane Austen, or two different "Song of the Year" awards;
  - hop 2 answerable without hop 1;
  - list or two-part answers;
  - verbose answers that break exact match;
  - truncated context;
  - wrong-year slips;
  - riddle phrasing.
- **Include supporting text in the prompt.** It must state the answer. Without it the model scores zero and the eval measures nothing.
- **Answers are short and canonical, with aliases.**
- **Two checks before an item is kept:**
  - hop-1-removed: drop the item if it's still answerable without the first article;
  - closed-book against Qwen3.5-4B: swap out any item it answers without context.
- **Vary topic and answer type.** Mix names, places, numbers and dates across domains. Five money-in-sports questions is one skeleton five times.

## Graders

- **Self-test every grader** with a known-good answer, a known-bad answer and an infinite loop before trusting it. `graders.py` passes all three: 1, 0, and a timeout scored as 0.

## Windows gotchas

- `str.splitlines()` also splits on U+2028/0x1c inside JSON strings. Split on `\n`.
- Set `PYTHONIOENCODING=utf-8` when printing dataset text.
- Keep LF line endings. `Path.write_text` writes CRLF on Windows, so use `write_bytes`.
- math-verify:
  - needs `antlr4-python3-runtime==4.13.2`;
  - its built-in timeout spawns processes, which fails in the sandbox, so pass `parsing_timeout=None` / `timeout_seconds=None` and cap the input length instead.

## Datasets

<!-- BEGIN GENERATED -->
| dataset | eval source (post-cutoff) | train pool (pre-cutoff) | train | eval | holdout | holdout merkle root |
|---|---|---|---|---|---|---|
| math | MathArena ArXivMath 03–08/2026, arXiv v1 date >= cutoff | AIME/HMMT/BRUMO/CMIMC/SMT/Apex 2023–Feb 2026, ArXivMath <= Feb 2026, arxivmath-training | 3254 | 112 | 73 | `ea3cff7fe0363473` |
| code | LeetCode problems first seen >= cutoff+7d, tests = statement examples w/ accepted-oracle outputs | LiveCodeBench v1–v6 (2023-05..2025-04), neulab/leetcode | 3525 | 87 | 61 | `81161350ee783402` |

Generated by prepare.py at 20260928T223913Z; full log: `logs/prepare-20260928T223913Z.log`.
<!-- END GENERATED -->
