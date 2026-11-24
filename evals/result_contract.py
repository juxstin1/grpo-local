"""Strict public evaluation accounting and immutable resume identity.

No service discovery, candidate execution, or private dataset access occurs here.
The serving identity receipt is verified by an operator outside this process; its
claims are bound to the run, not independently established by this module.
"""
from __future__ import annotations

from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re


class ResultIntegrityError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ResultIntegrityError('duplicate JSON key')
            result[key] = value
        return result
    def constant(_):
        raise ResultIntegrityError('non-finite JSON value')
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise ResultIntegrityError('malformed JSON') from exc


def refuse_private_path(path):
    parts = re.split(r'[/\\]', str(path).lower())
    if any('holdout' in p or 'sealed' in p or p == 'private' for p in parts):
        raise ResultIntegrityError('Private evaluation requires an independent evaluator; no private inputs were read.')


def selected_ids(items):
    ids = []
    for item in items:
        value = item.get('id') if isinstance(item, dict) else None
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ResultIntegrityError('every selected item needs a nonempty exact string id')
        if item.get('split') in ('sealed', 'holdout', 'private'):
            raise ResultIntegrityError('private items are unavailable to this development evaluator')
        ids.append(value)
    if not ids or len(set(ids)) != len(ids):
        raise ResultIntegrityError('selected item ids must be nonempty and unique')
    return ids


def validate_served_identity(receipt, model, base):
    if not isinstance(receipt, dict) or receipt.get('verified') is not True:
        raise ResultIntegrityError('an explicitly verified serving identity receipt is required')
    if receipt.get('served_model') != model or receipt.get('base_url') != base:
        raise ResultIntegrityError('serving receipt alias/endpoint does not match requested run')
    hashes = receipt.get('artifacts_sha256')
    if not isinstance(hashes, dict) or not hashes or any(
            not isinstance(name, str) or not name or not isinstance(sha, str)
            or not re.fullmatch(r'[0-9a-f]{64}', sha) for name, sha in hashes.items()):
        raise ResultIntegrityError('serving receipt needs SHA-256 for all served weight/adapter artifacts')
    if not isinstance(receipt.get('verification_method'), str) or not receipt['verification_method'].strip():
        raise ResultIntegrityError('serving receipt must explain how identity was verified')
    try:
        timestamp = datetime.fromisoformat(receipt['verified_at'].replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('timezone absent')
    except (KeyError, AttributeError, ValueError) as exc:
        raise ResultIntegrityError('serving receipt needs timezone-qualified verified_at') from exc
    return receipt


def make_identity(dataset_bytes, selected, rendered, grader_sources, settings, served_identity):
    all_ids = [value for values in selected.values() for value in values]
    if not all_ids or len(set(all_ids)) != len(all_ids):
        raise ResultIntegrityError('selected IDs must also be unique across buckets')
    payload = dict(version='endpoint-result-v2', dataset_sha256=digest(dataset_bytes),
                   selected=selected, rendered_prompt_tools_sha256=digest(canonical(rendered)),
                   grader_source_sha256={name: digest(body) for name, body in sorted(grader_sources.items())},
                   settings=settings, served_identity=served_identity)
    return {**payload, 'run_id': digest(canonical(payload))}


def validate_rows(rows, ids, k, bucket, run_id):
    if type(k) is not int or k < 1 or not ids or len(set(ids)) != len(ids):
        raise ResultIntegrityError('nonempty unique selection and positive integer k required')
    allowed, indexed = set(ids), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ResultIntegrityError('attempt row is not an object')
        item_id, attempt = row.get('item_id'), row.get('attempt')
        if not isinstance(item_id, str) or item_id not in allowed:
            raise ResultIntegrityError('foreign or missing item id in attempts')
        if type(attempt) is not int or not 1 <= attempt <= k:
            raise ResultIntegrityError('attempt index outside selected k')
        if row.get('bucket') != bucket or row.get('run_id') != run_id:
            raise ResultIntegrityError('attempt bucket/run identity mismatch')
        if 'passed' not in row or (row['passed'] is not None and type(row['passed']) is not bool):
            raise ResultIntegrityError('passed must be bool or explicit null for invalid attempts')
        if row['passed'] is None and not any(isinstance(row.get(key), str) and row[key].strip()
                                            for key in ('note', 'error')):
            raise ResultIntegrityError('invalid attempt needs an error reason')
        if row['passed'] is not None and row.get('error'):
            raise ResultIntegrityError('graded attempt cannot also contain an error')
        if 'hit_cap' in row and type(row['hit_cap']) is not bool:
            raise ResultIntegrityError('hit_cap must be a bool')
        key = item_id, attempt
        if key in indexed:
            raise ResultIntegrityError('duplicate item/attempt row, including identical duplicates')
        indexed[key] = row
    return indexed


def read_rows(path, ids, k, bucket, run_id):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding='utf-8')
        if text and not text.endswith('\n'):
            raise ResultIntegrityError('unterminated attempt row; inspect interrupted write before resume')
        lines = text.split('\n')
        if lines[-1] == '':
            lines.pop()  # JSONL terminator only; Unicode separators belong to strings.
    except (OSError, UnicodeError) as exc:
        raise ResultIntegrityError('attempt log is unreadable') from exc
    if any(not line.strip() for line in lines):
        raise ResultIntegrityError('blank or incomplete attempt row')
    return validate_rows([strict_json(line) for line in lines], ids, k, bucket, run_id)


def summarize(rows, ids, k, bucket, run_id):
    indexed = validate_rows(rows, ids, k, bucket, run_id)
    expected = len(ids) * k
    valid = [r for r in indexed.values() if r['passed'] is not None]
    invalid = len(indexed) - len(valid)
    missing = expected - len(indexed)
    complete = invalid == 0 and missing == 0
    per_item = {}
    observed_solved = 0
    for item in ids:
        attempts = [indexed.get((item, a)) for a in range(1, k + 1)]
        solved = any(r is not None and r['passed'] is True for r in attempts)
        observed_solved += solved
        per_item[item] = bool(solved) if all(r is not None and r['passed'] is not None for r in attempts) else None
    accuracy = observed_solved / len(ids) if complete else None
    return dict(status='complete' if complete else 'incomplete', bucket=bucket, run_id=run_id,
                n=len(ids), k=k, expected_attempts=expected, observed_attempts=len(indexed),
                valid_attempts=len(valid), invalid_attempts=invalid, missing_attempts=missing,
                accuracy=accuracy, pass_at_k=accuracy, solved_items=observed_solved if complete else None,
                observed_solved_items=observed_solved, per_item=per_item,
                per_attempt_acc=sum(r['passed'] for r in valid) / expected if complete else None,
                cap_hit_rate=sum(bool(r.get('hit_cap')) for r in valid) / expected if complete else None)


def validate_resume(result_dir, identity, selected, k):
    """Read-only preflight; callers must finish this before requests or writes."""
    result_dir = Path(result_dir)
    identity_file = result_dir / 'run.identity.json'
    if not result_dir.exists():
        return {bucket: {} for bucket in selected}
    if not identity_file.is_file():
        raise ResultIntegrityError('existing result directory has no bound run identity; use a new slug')
    if strict_json(identity_file.read_text(encoding='utf-8')) != identity:
        raise ResultIntegrityError('resume identity mismatch; use a new slug')
    raw = result_dir / 'raw'
    if raw.exists() and any(p.name not in {b + '.jsonl' for b in selected} or not p.is_file()
                            for p in raw.iterdir()):
        raise ResultIntegrityError('unexpected raw bucket/file in resume directory')
    return {bucket: read_rows(raw / (bucket + '.jsonl'), ids, k, bucket, identity['run_id'])
            for bucket, ids in selected.items()}


@contextmanager
def locked_run(result_dir, identity, selected, k):
    """One writer per result directory; a stale lock requires operator inspection."""
    result_dir = Path(result_dir)
    validate_resume(result_dir, identity, selected, k)
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=False)
        with (result_dir / 'run.identity.json').open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(identity, indent=2) + '\n')
    lock = result_dir / '.run.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise ResultIntegrityError('result directory is locked by another run; inspect stale locks before recovery') from exc
    try:
        handle.write(identity['run_id'])
        handle.flush()
        yield validate_resume(result_dir, identity, selected, k)
    finally:
        handle.close()
        lock.unlink()
