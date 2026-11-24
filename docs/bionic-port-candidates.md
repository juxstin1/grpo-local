# Bionic -> sandbox-gym port candidates (ranked)

Spec paths: S = C:\Users\juxxs\agent-re\llama-studio\specs\, F = C:\Users\juxxs\bionic\. "Parity" = does it change tokens she sees; three loops must match: agent.py run_episode (eval+collect), rollout.py (GRPO), collect.py segments.

## Ranked
1. **Pre-seeded workspace tree in the task prompt** [b] - S/M
   - What: Bionic injects an `<environment>`/workspace block every turn so the model never spends rounds discovering it (F13 s5). A "find" subagent returns file lists so the caller skips exploring (F6 BONUS 1).
   - Change: append a capped tree (`list_dir` of root, depth 2, <=60 entries, LIST_DIR_CAP) to `task.prompt` where it is built (gyms.spec_to_task / make_workspace), NOT in agent.py, so all 3 loops share it.
   - Parity: YES, changes the prompt. Old SFT wins have no tree -> recollect or strip; eval must use it too. Kills the 3x list_dir directly.
2. **Retry-once on format failure (no-action / empty)** [c] - M
   - What: parse fail -> feed error as a tool result + retry exactly once per turn; leaked-markup detector picks "clean" (drop bad turn) vs "dirty" (keep) retry (S26 s3-5; S11 s5 layer 1).
   - Change: today agent.py:339 and rollout.py:76-77 END the episode on `action is None` (stop=no-action / empty). Add one retry observation there in both.
   - Parity: YES (new obs text). Caution: tombstoning hides the failure from GRPO (no negative advantage). Use DIRTY retry for training (keep failed turn, trainable) or charge a reward penalty per retry. Unclosed `<think>`: Bionic has no analog (llama-server strips reasoning); use the S26 s5 regex idea on the reasoning text (tool-call JSON inside reasoning -> retry).
3. **Make rollout.py share agent.py's error feedback** [c] - S (bug, not a port)
   - agent.py:349-357 catches KeyError/TypeError/ValueError/JailError -> error observation. rollout.py:79 calls `execute_tool` bare: a call missing an arg crashes/diverges in GRPO but is feedback in eval. This is Bionic's handleInvalidToolRequest (S11 s5) - half ported. Extract one `run_action(ws, action, tools)` used by both. Parity: fixes an existing mismatch.
4. **Force final answer on last turn** [b][c] - S
   - What: last round strips tools so the model must answer (S11 s2). `cap_hit` = no ANSWER = 0 reward today.
   - Change: at turn N append "[turn N of N] final turn: give ANSWER now" (encoding.observation_body already owns the counter line); do not remove tools (prompt-only, keeps tokens stable).
   - Parity: YES, single shared function. Ablate: may teach guessing; still outcome-graded so fine.
5. **Actionable error/result hints** [b][c] - S/M
   - What: "the hint in the result" (S19 s3, S17): missing file -> name nearest paths; edit_file no-match -> show closest line; `read_file` on a dir -> suggest list_dir.
   - Change: workspace.py tool methods (shared by all loops -> automatic parity in tokens, but a behavior change vs old SFT data).
6. **Middle-truncation with warning header** [a] - S
   - What: cap item, keep head+tail, prefix `Warning: truncated output (original token count: N)` (S25 s2b; F13 s6, 20000-token walk newest->oldest).
   - Change: workspace.py `_cap` is head-only (`s[:MAX_TOOL_OUT]`); run_tests failure summaries sit at the END and get cut. Switch to head+tail for run_tests/search only. Parity: shared code, automatic.
7. **Ratio trigger for compaction** [a] - S/M
   - What: compact when server-reported prompt tokens >= ratio of ctx (F13 s6: 15/16; S25 s2d: 90%), pre-model, not by turn count.
   - Change: trigger when `len(prompt_ids) >= budget - NOTE_CALL_MAX - margin` (1000 reserve = frozen-state rule already in notes.py). Deterministic from ids, so eval/collect/GRPO agree if all use the same threshold const in notes.py.
8. **Lenient arg parse (JSON5-ish repair)** [c] - S
   - S11 s5 layer 2. T.parse_action uses strict json.loads; add trailing-comma/single-quote repair in tools.py. Shared -> auto parity; low value (grammar failures are mostly wrong tool names / think, not JSON).
9. **Multi-call turns / read_files batching** [b] - L. Bionic runs all calls of a round, results in one message (S11 s9). Our "exactly one tool per turn" + parse_action-first-only + turn counter + trainable-mask logic would all change. Defer; item 1 gets most of the win.
10. **Completion self-check** (S11 s6) - M, opt-in in Bionic. Adds a round; skip unless answer-without-run_tests is a measured failure.

## Skip
Grammar-constrained tool calls (S10): vLLM could, but rollout.py uses HF generate with no grammar -> unfixable parity break. Session queue/cancel (S24), normalization/orphans (S25 s2c), deterministic IDs, sibling-file fallback (S19: config.json-specific), approval guard (S12), skills/hooks.

## Bionic vs our note spec
- **Improves ours:** (i) "temporary changes that still need cleanup" - our sections lack it; in an edit gym stray debug edits / scratch write_file are real debt. Fold into "Current state" or add "Cleanup" (cap pressure: 400 tokens). (ii) "important files/paths + significance" - ours has no home for it; add to Done/Current state wording, not a 7th section. (iii) Ratio trigger (item 7).
- **Ours is better/keep:** Hypothesis and Don't repeat (Bionic only has consumer-side "do not repeat"); resume_prompt "check the claim against the files" (Bionic: "read carefully, understand their reasoning" = trust); hard 400 cap counted as failure (Bionic: "be concise", no cap; "say nothing of note" for empty sections wastes tokens under a cap - keep our skeleton but let empty sections be one word).
- **Contradicts:** Bionic keeps recent tail verbatim and only summarizes the oldest turns (S13 s3); we give the fresh instance task+note only. Ours makes the note the sole carrier (cleaner GRPO signal) but worse at losing the last observation. Cheap ablation: keep last tool result verbatim after the note. Would change resume tokens in all loops.
- **Different mechanics:** Bionic appends the producer prompt to the live context (KV-cache hit, model sees own raw turns); ours re-renders the trace inside `<work>` (cache miss, text copy). Not needed now; note it if note-call latency matters.
- **User-approved vs self-decided** split: no user in the gym; drop. Thinking budget: Bionic has none; no conflict.
- Bionic's handoff never verified as scored by outcome; its only test is category retention (S13 header). Our outcome-graded GRPO is strictly stronger evidence.

## Parity checklist for any item that touches tokens
One shared function in encoding.py/gyms.py/workspace.py/notes.py; run selftest.py + a collect.py prompt-id fidelity dry run; re-strip/recollect SFT data; eval-set unchanged (only the harness changes).
