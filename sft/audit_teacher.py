"""Metadata-only teacher provenance audit. Never imports a tokenizer or model.

Source-line hashes establish correspondence, not task success. This auditor
does not admit historical tool trajectories: executable replay is still needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(train_path, manifest_path, source_dir):
    train_path, manifest_path, source_dir = map(Path, (train_path, manifest_path, source_dir))
    raw, manifest_raw = train_path.read_bytes(), manifest_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    manifest = json.loads(manifest_raw)
    admitted = {}
    source_cache = {}
    for entry in manifest['admitted']:
        key = (entry['model'], entry['id'])
        if key in admitted:
            raise ValueError('duplicate admitted episode')
        if type(entry['hinted']) is not bool:
            raise ValueError('hinted flag must be boolean')
        path = (source_dir / entry['file']).resolve()
        if not path.is_relative_to(source_dir.resolve()):
            raise ValueError('source path escapes source directory')
        if path not in source_cache:
            source_cache[path] = path.read_text(encoding='utf-8').splitlines()
        line = entry['line']
        if type(line) is not int or not 1 <= line <= len(source_cache[path]):
            raise ValueError('invalid source line')
        if sha(source_cache[path][line - 1].encode()) != entry['row_sha256']:
            raise ValueError('source line hash mismatch')
        admitted[key] = entry
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate training ID')
    pairs = [(row['teacher_model'], row['teacher_trace_id']) for row in rows]
    if set(pairs) != set(admitted):
        raise ValueError('training rows and admitted episodes differ')
    hinted_roots = {row['root_id'] for row, key in zip(rows, pairs) if admitted[key]['hinted']}
    dispositions = []
    for row, key in zip(rows, pairs):
        reasons = ['legacy-tool-trajectory-needs-executable-replay']
        if admitted[key]['hinted']:
            reasons.append('hinted-source-episode')
        if row['root_id'] in hinted_roots:
            reasons.append('hinted-root-quarantine')
        dispositions.append({'id': row['id'], 'root_id': row['root_id'], 'teacher_model': key[0],
                             'teacher_trace_id': key[1], 'source_row_sha256': admitted[key]['row_sha256'],
                             'training_row_sha256': sha(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                                       separators=(',', ':'), allow_nan=False).encode()),
                             'admitted': False, 'reasons': reasons})
    hinted = [row for row, key in zip(rows, pairs) if admitted[key]['hinted']]
    return {'version': 'teacher-provenance-audit-v1', 'inputs': {
                'train_sha256': sha(raw), 'manifest_sha256': sha(manifest_raw)},
            'rows': len(rows), 'episodes': len(admitted), 'roots': len({r['root_id'] for r in rows}),
            'source_lines_verified': len(admitted), 'source_files': len(source_cache),
            'completion_tokens': sum(len(r['completion_ids']) for r in rows),
            'hinted_episodes': sum(e['hinted'] for e in admitted.values()), 'hinted_rows': len(hinted),
            'hinted_completion_tokens': sum(len(r['completion_ids']) for r in hinted),
            'hinted_root_rows': sum(r['root_id'] in hinted_roots for r in rows),
            'by_teacher': dict(Counter(r['teacher_model'] for r in rows)),
            'admitted_rows': 0, 'quarantined_rows': len(rows),
            'limitations': ['Source linkage is not evidence of task success.',
                            'No task bodies, tokenizer decoding, artifact replay or model behavior inspected.',
                            'Hint flags are manifest attribution, not proof of unassisted behavior.',
                            'Historical hint_texts_stripped is the number of configured hint strings, not removals.'],
            'dispositions': dispositions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.train, args.manifest, args.source_dir)
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('dispositions', 'by_teacher', 'limitations')}))


if __name__ == '__main__':
    main()
