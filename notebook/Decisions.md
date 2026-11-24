# Decisions

Append-only. Format: `- YYYY-MM-DD HH:MM [who] <decision> — why — [[Log/YYYY-MM-DD]]`. Superseded? Append a new line referencing the old one.
See [[README]], [[Runs]], [[Claims]].

- 2026-09-30 15:32 [claude] edit + editv_h dropped as training targets; fix kept (0 = turn cap, all fix episodes capped before test step); fix rerun at 2x cap pending — [[Log/2026-09-30]] [CLAIMED]
- 2026-09-30 15:37 [claude] notebook vault created as the single shared log; Claude Code Stop hook blocks turns that don't append to today's log; Codex rule in grpo-local/AGENTS.md — [[Log/2026-09-30]] [VERIFIED]
- 2026-10-02 19:46 [claude] tau2/FuncBenchGen = post-training transfer checks only, never on the critical path; compaction gym is the main eval — why: no public bench measures 200-turn note survival — [[Log/2026-10-02]]
- 2026-10-02 20:40 [user] budget 8192 -> 32768 for training/working size — 8k starved tasks (2 fetches fill it, tau2 telecom unloadable, oracle reader possibly budget-starved); switch between runs, all baselines re-measured at 32k, compaction tasks > 32k — [[Log/2026-10-02]]
