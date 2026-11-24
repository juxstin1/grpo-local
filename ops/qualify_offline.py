"""Run contract qualification with network, subprocess and model-import tripwires."""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN_IMPORTS = {'torch', 'transformers', 'peft', 'llama_cpp', 'openai', 'anthropic'}


def source_hashes():
    paths = list((ROOT / 'tests').glob('test_*.py'))
    paths += list((ROOT / 'prompts').glob('*.md')) + [ROOT / 'suite-v2/tasks.json']
    paths += [ROOT / 'ops' / name for name in (
        'qualify_offline.py', 'reward_contract.py', 'recovery_contract.py', 'contract_tasks.py', 'work_contract.py',
        'evaluation_integrity.py', 'run_policy.py', 'grpo_q4.py', 'grpo_eval.py',
        'real_tasks.py', 'q4_probe_raw.py', 'q4_bf16_gap.py', 'q4_bf16_gap2.py',
        'eval_sft_1008.py', 'eval_q4.py')]
    paths += [ROOT / p for p in (
        'evals/graders.py', 'evals/code_executor.py', 'evals/result_contract.py',
        'evals/eval_runner.py', 'evals/datasets/prepare.py', 'evals/datasets/registry.py',
        'evals/datasets/build_registry.py', 'sft/collect.py', 'sft/train_sft.py',
        'tau2-bridge/run_tau2.py', 'sft/admission.py', 'sft/audit_teacher.py',
        'sft/sft_schema.py', 'sft/from_gym.py', 'ops/check_suite_coverage.py', 'suite-v2/coverage.json',
        'gym_budget.py', 'evals/datasets/roots.py', 'evals/datasets/synth_multihop.py',
        'evals/holdout/seal.py', 'requirements-offline.txt')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if any(name.split('.')[0] in FORBIDDEN_IMPORTS for name in sys.modules):
        raise SystemExit('Use a fresh Python process without model libraries loaded.')
    before = source_hashes()
    attempted = []
    original_import = builtins.__import__

    def blocked(operation):
        def stop(*a, **kw):
            attempted.append(operation)
            raise RuntimeError('Offline qualification forbids ' + operation)
        return stop

    def guarded_import(name, *a, **kw):
        if name.split('.')[0] in FORBIDDEN_IMPORTS:
            attempted.append('model-import:' + name.split('.')[0])
            raise ImportError('Model libraries are forbidden during offline qualification')
        return original_import(name, *a, **kw)

    with patch('socket.socket.connect', blocked('socket-connect')), \
         patch('socket.create_connection', blocked('socket-create-connection')), \
         patch('subprocess.Popen', blocked('subprocess')), \
         patch('builtins.__import__', guarded_import):
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = before == source_hashes()
    ok = result.wasSuccessful() and not result.skipped and unchanged and not attempted
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   mode='offline-contract-qualification', qualified=ok,
                   tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   skipped=len(result.skipped), blocked_external_attempts=attempted,
                   source_unchanged=unchanged, source_sha256=before,
                   limitations=['No model behavior, learning, or live serving tested.',
                                'Public development fixtures only; not a sealed benchmark.',
                                'Only invented fixture programs run in memory; Docker transport is mocked.',
                                'Native container execution requires separate CPU verification; no model calls.',
                                'Legacy free-text graders remain unqualified for training.'])
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / 'runs/audits'):
            raise ValueError('Receipt must be within this checkout runs/audits')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as fh:
            fh.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('qualified', 'tests_run', 'failures', 'errors', 'blocked_external_attempts')}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
