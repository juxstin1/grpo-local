"""Probe actual seal/loader function bodies using invented data only.

AST extraction avoids module import side effects and never opens a real holdout.
Exit 1 means the tested isolation/integrity contract is not enforced.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT.parent / 'grpo-local-wt-filetools/sandbox-gym/seal.py'
LOADER = ROOT / 'ops/grpo_eval.py'


def extract(path, name, env):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    sources = [SEAL, LOADER, Path(__file__)]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    tree = ast.parse(SEAL.read_text(encoding='utf-8'))
    canary = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'CANARY' for t in n.targets))
    verify = extract(SEAL, 'verify', dict(Path=Path, json=json, hashlib=hashlib, CANARY=canary))
    cases = []

    def record(name, expected, actual, meaning):
        cases.append(dict(name=name, expected=expected, actual=actual,
                          agrees=expected == actual, meaning=meaning))

    def seal(body):
        content = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False)
        return {**body, 'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest()}

    with tempfile.TemporaryDirectory(prefix='seal-contract-audit-') as tmp:
        folder = Path(tmp)
        holdout = folder / 'data/holdout'
        holdout.mkdir(parents=True)
        path = holdout / 'invented.json'
        body = dict(schema='gym_v1', family='audit_only', canary=canary, seeds=[123],
                    items=[dict(name='invented', family='audit_only', seed=123,
                                prompt='Invented audit fixture', ground_truth='alpha', tools=[])])
        original = seal(body)
        path.write_text(json.dumps(original), encoding='utf-8')
        sidecar = path.with_suffix('.sha256')
        sidecar.write_text(original['sha256'], encoding='utf-8')
        record('untouched_control', True, verify(path)[0], 'A correctly hashed fixture verifies.')
        if not args.smoke:
            changed = json.loads(json.dumps(original))
            changed['items'][0]['ground_truth'] = 'beta'
            path.write_text(json.dumps(changed), encoding='utf-8')
            record('stale_hash_control', False, verify(path)[0], 'Unrehashed edit is detected.')
            loader = extract(LOADER, 'holdout_tasks', dict(json=json,
                             Q=SimpleNamespace(G=SimpleNamespace(GYM=folder))))
            loaded = loader(['audit_only'])
            record('loader_rejects_invalid_seal', 0, len(loaded),
                   'Actual Q4 loader accepts a fixture that actual seal.verify rejects.')
            changed.pop('sha256')
            rewritten = seal(changed)
            path.write_text(json.dumps(rewritten), encoding='utf-8')
            record('rehash_cannot_replace_trusted_seal', False, verify(path)[0],
                   'Embedded checksum can be replaced; unchanged sidecar is never consulted.')
            record('sidecar_preserved_control', True,
                   sidecar.read_text() == original['sha256'], 'Audit did not change the original sidecar.')

    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed during audit: ' + path)
    report = dict(scope='Invented fixtures only; no sealed task contents or model calls',
                  source_sha256=hashes, cases=cases,
                  summary=dict(cases=len(cases), mismatches=sum(not c['agrees'] for c in cases)))
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / 'runs/audits'):
            raise ValueError('Output must be under runs/audits')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as fh:
            fh.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))
    for c in cases:
        print(c['name'], 'expected=', c['expected'], 'actual=', c['actual'])
    return int(bool(report['summary']['mismatches']))


if __name__ == '__main__':
    raise SystemExit(main())
