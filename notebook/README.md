# grpo-local notebook

Shared, append-only log for every agent working in this repo (Claude Code agents, Codex, the user).
Open `grpo-local/notebook/` as a vault in Obsidian.

## Notes
- [[Log/2026-09-30|Log/YYYY-MM-DD]] — daily log. **One line per turn, minimum.** Append only.
- [[Decisions]] — decisions with rationale (also logged as `DECISION` lines).
- [[Runs]] — one row per launched run.
- [[Claims]] — who is editing which file right now.

## Line format (Log)
```
- HH:MM [claude|codex|user] <LAUNCH|RESULT|DECISION|FINDING|FIX> <what, numbers, path> [VERIFIED|CLAIMED]
```
- `HH:MM` = local time. Tag = who wrote it.
- `VERIFIED` only if you checked it this turn (command output, file read, sha). Otherwise `CLAIMED`.
- Numbers and paths, not adjectives. Wrong earlier line? Append a correction; never edit/delete old lines.

## Rules
1. **Every turn appends at least one line** to today's `Log/YYYY-MM-DD.md` (create the file if absent).
   Claude Code enforces this with a Stop hook (`C:\Users\juxxs\vllm\.claude\settings.json` →
   `.claude/hooks/notebook_stop.js`); Codex follows `grpo-local/AGENTS.md`.
2. **Before editing a file, add a line to [[Claims]]**; when done, mark it `DONE` in place. Never delete claim lines.
   If a file has an open claim by someone else, don't edit it — log a `FINDING` and ask.
3. **Every launched run gets a row in [[Runs]]**; update its status cell when it ends.
4. Nothing in this vault is deleted. Append, or mark superseded.
