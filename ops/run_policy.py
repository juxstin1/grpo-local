"""Model execution requires an explicit operator opt-in; offline checks never opt in."""
from __future__ import annotations


def require_model_opt_in(allowed):
    if not allowed:
        raise SystemExit('Model execution is disabled. Use offline qualification; --allow-model-run requires separate operator authorization.')


def add_model_flag(parser):
    parser.add_argument('--allow-model-run', action='store_true',
                        help='Explicitly authorize model execution for this invocation (off by default).')
