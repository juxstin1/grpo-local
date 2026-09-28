# Run log

One entry per eval or training run. Newest at top.

## 2026-09-28 qwen3.5-2b (office) - INVALID
- Set: baseline_v1 (118 items). Server dropped mid-run.
- LCB 2/30. BFCL, When2Call, holdout: all request-error. Not a baseline.
- Outcome: fixed runner (retry request-errors on resume, health-wait, --set flag).
