"""THE episode token budget: one constant for all of grpo-local.

Default 8192 (prompt + completion tokens per episode / per call). Override, in order:
  --budget N on any script's command line (read here at import, so every module sees it)
  env GYM_BUDGET=N
Everything else derives from BUDGET: sandbox-gym Episode.budget, handoff-gym BUDGET,
sft MAX_TOKENS_MULTITURN, tau2 --budget, compaction thresholds (notes.compact_*_tokens).
Results/stats record it ("budget"); compare/merge steps refuse mixed budgets
(require_same). check_server() fails fast when the server's max_model_len is too small.
"""
from __future__ import annotations

import os
import sys

DEFAULT_BUDGET = 8192
MIN_BUDGET = 1024
ENDPOINT_DEFAULT = 4096      # single-turn endpoint cap (sft_schema.MAX_TOKENS); unchanged by default


def _cli_budget(argv: list[str]) -> int | None:
    for i, a in enumerate(argv):
        if a == "--budget" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--budget="):
            return int(a.split("=", 1)[1])
    return None


def resolve(argv: list[str] | None = None, env: dict | None = None) -> int:
    env = os.environ if env is None else env
    v = _cli_budget(sys.argv[1:] if argv is None else argv)
    if v is None and env.get("GYM_BUDGET", "").strip():
        v = int(env["GYM_BUDGET"])
    if v is None:
        return DEFAULT_BUDGET
    if v < MIN_BUDGET:
        raise SystemExit(f"[budget] {v} is below the minimum {MIN_BUDGET}")
    return v


BUDGET = resolve()
OVERRIDDEN = BUDGET != DEFAULT_BUDGET
# Endpoint (single-turn mhqa) cap: 4096 as before unless the budget is overridden.
ENDPOINT_MAX_TOKENS = BUDGET if OVERRIDDEN else ENDPOINT_DEFAULT


def require_same(budgets, what: str = "results") -> int | None:
    """Refuse to compare/merge records made under different budgets. None entries
    (legacy files with no recorded budget) count as the default 8192."""
    seen = {DEFAULT_BUDGET if b is None else b for b in budgets}
    if len(seen) > 1:
        raise SystemExit(f"[budget] REFUSED: {what} were made under different budgets "
                         f"{sorted(seen)} -- not comparable")
    return next(iter(seen), None)


def server_max_model_len(base: str, model: str = "", api_key: str = "", timeout: int = 10):
    """max_model_len the server reports on /v1/models (None if unreachable/unreported)."""
    import requests
    try:
        r = requests.get(f"{base.rstrip('/')}/v1/models", timeout=timeout,
                         headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
        data = r.json().get("data", [])
    except Exception:
        return None
    lens = [m.get("max_model_len") for m in data if m.get("max_model_len")
            and (not model or m.get("id") == model or m.get("root") == model)]
    lens = lens or [m.get("max_model_len") for m in data if m.get("max_model_len")]
    return min(lens) if lens else None


def check_server(base: str, budget: int = None, model: str = "", api_key: str = "") -> None:
    """Fail fast if budget > the server's max_model_len (else vLLM truncates/rejects
    silently deep inside a run). Unknown max_model_len is a warning, not a failure."""
    budget = BUDGET if budget is None else budget
    n = server_max_model_len(base, model, api_key)
    if n is None:
        print(f"[budget] note: {base} reports no max_model_len; cannot verify budget {budget}")
    elif budget > n:
        raise SystemExit(f"[budget] budget {budget} > server max_model_len {n} at {base}: "
                         f"relaunch vLLM with --max-model-len >= {budget} (or lower GYM_BUDGET)")
