"""Harness-owned, per-episode edit failure and observed recovery evidence.

No mirrored tools are modified. A live session owns the dispatch record outside
the learner workspace; transcript claims and exported receipts cannot substitute
for it. This is a trusted-harness boundary, not isolation from malicious host code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import reward_contract as R

VERSION = 'edit-conflict-v1'
TOOLS = ('read_file_lines', 'read_file_chars', 'replace_file')
ERROR_PREFIX = 'Builtin tool error (backend unavailable?): '
CONFLICT = 'Builtin tool error (backend unavailable?): Injected transient edit conflict; no bytes changed. Read the current file before retrying the edit.'


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                            separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def validate_policy(spec):
    policy = spec['contract'].get('recovery')
    if not isinstance(policy, dict) or set(policy) != {'version', 'path'} or policy['version'] != VERSION:
        raise R.InvalidReward('unsupported recovery policy')
    path = policy['path']
    R.safe_rel(path)
    if path not in spec['files'] or path not in spec['contract']['outputs']:
        raise R.InvalidReward('recovery target must be an existing contracted output')
    if set(spec.get('tools', [])) != set(TOOLS):
        raise R.InvalidReward('recovery requires the scoped native file tools')


class RecoverySession:
    def __init__(self, spec, root):
        R.validate_spec(spec)
        validate_policy(spec)
        self._spec = copy.deepcopy(spec)
        self._root = Path(root).resolve()
        self._path = spec['contract']['recovery']['path']
        self._task_hash = digest(spec)
        self._events = []
        self._phase = 'await-failure'
        self._boundary = -1
        self._last_round = -1
        self._violation = False
        self._invalid = False
        self._verified_hash = None
        try:
            inventory = R.file_inventory(root)
            if inventory is None or set(inventory) != set(spec['files']):
                raise R.InvalidReward('recovery requires an untouched fixture')
            if any(inventory[p].read_bytes() != text.encode() for p, text in spec['files'].items()):
                raise R.InvalidReward('recovery fixture changed before session creation')
        except OSError as exc:
            raise R.InvalidReward('recovery fixture unavailable') from exc

    def _bytes(self):
        try:
            inventory = R.file_inventory(self._root)
            if inventory is None or self._path not in inventory:
                raise R.InvalidReward('recovery target unavailable or linked')
            return inventory[self._path].read_bytes()
        except OSError as exc:
            raise R.InvalidReward('recovery target unavailable') from exc

    def _record(self, name, args, result, before, after, round_index, kind):
        self._events.append(dict(seq=len(self._events), round=round_index, tool=name, kind=kind,
                                 args_sha256=digest(args), observation_sha256=digest(result.encode()),
                                 before_sha256=digest(before), after_sha256=digest(after)))
        return result

    def execute(self, name, args, delegate, *, round_index):
        if self._invalid:
            raise R.InvalidReward('recovery session already invalid')
        try:
            return self._execute(name, args, delegate, round_index=round_index)
        except R.InvalidReward:
            self._invalid = True
            raise
        except Exception as exc:
            self._invalid = True
            raise R.InvalidReward('recovery dispatch unavailable') from exc

    def _execute(self, name, args, delegate, *, round_index):
        if type(round_index) is not int or round_index < 0 or round_index < self._last_round:
            raise R.InvalidReward('invalid recovery round ordering')
        self._last_round = round_index
        try:
            args = R.strict_json(args) if isinstance(args, str) else copy.deepcopy(args)
            if not isinstance(args, dict):
                raise ValueError('not an object')
            digest(args)
        except (ValueError, TypeError):
            return 'Builtin tool error (backend unavailable?): Invalid arguments; use a JSON object.'
        before = self._bytes()
        path = args.get('path')
        if not isinstance(path, str) or not path:
            return self._record(name, args, ERROR_PREFIX + 'BAD_REQUEST: path must be nonempty text.',
                                before, before, round_index, 'invalid-tool-arguments')
        allowed = name in TOOLS and path in self._spec['files']
        if name == 'replace_file' and path != self._path:
            allowed = False
        if not allowed:
            self._violation = True
            return self._record(name, args, 'Builtin tool error (backend unavailable?): Outside this task tool/path scope.',
                                before, before, round_index, 'blocked-scope')
        is_edit = name == 'replace_file'
        numeric = {}
        if not is_edit:
            keys = ('startLine', 'endLine') if name == 'read_file_lines' else ('offset', 'length')
            if set(args) - {'path', *keys}:
                return self._record(name, args, ERROR_PREFIX + 'BAD_REQUEST: Unknown read parameter.',
                                    before, before, round_index, 'invalid-tool-arguments')
            for key in keys:
                if key not in args:
                    continue
                value = args[key]
                if (type(value) not in (int, float) or value < 0 or value > 2**31 - 1
                        or not math.isfinite(value) or int(value) != value):
                    return self._record(name, args, ERROR_PREFIX + f'BAD_REQUEST: {key} must be a nonnegative integer.',
                                        before, before, round_index, 'invalid-tool-arguments')
                numeric[key] = int(value)
        if is_edit:
            edits = args.get('edits')
            if (set(args) != {'path', 'edits'} or not isinstance(edits, list) or not edits
                    or any(not isinstance(e, dict) or set(e) != {'oldText', 'newText'}
                           or not isinstance(e['oldText'], str) or not e['oldText']
                           or not isinstance(e['newText'], str) or e['oldText'] == e['newText'] for e in edits)):
                return self._record(name, args, 'Builtin tool error (backend unavailable?): A nonempty exact replacement is required.',
                                    before, before, round_index, 'invalid-edit')
            if self._phase == 'await-failure':
                self._phase, self._boundary = 'await-read', round_index
                return self._record(name, args, CONFLICT, before, before, round_index, 'injected-conflict')
            if self._phase == 'await-read' or round_index <= self._boundary:
                return self._record(name, args, 'Builtin tool error (backend unavailable?): Observe a full current-file read in a later reply before retrying.',
                                    before, before, round_index, 'unobserved-retry')
        result = delegate()
        if not isinstance(result, str):
            raise R.InvalidReward('native tool observation must be text')
        after = self._bytes()
        if result.startswith(ERROR_PREFIX):
            if not result.startswith(ERROR_PREFIX + 'BAD_REQUEST:') or before != after:
                raise R.InvalidReward('native backend failure cannot be scored as learner failure')
            if is_edit:
                self._phase, self._boundary = 'await-read', round_index
            return self._record(name, args, result, before, after, round_index, 'invalid-tool-arguments')
        kind = 'tool-observation'
        if is_edit:
            if not result.startswith(f'Applied {len(edits)} edit(s) to {path}.'):
                raise R.InvalidReward('unexpected native edit observation')
            success = result.startswith('Applied ') and before != after
            if success:
                self._phase, self._boundary = 'await-verification', round_index
                kind = 'applied-retry'
            else:
                self._phase, self._boundary = 'await-read', round_index
                kind = 'failed-retry'
        elif path == self._path:
            if before != after:
                raise R.InvalidReward('read tool changed the recovery target')
            text = after.decode('utf-8')
            full = False
            if name == 'read_file_lines':
                start, end = numeric.get('startLine', 1), numeric.get('endLine', max(1, len(text.splitlines())))
                full = (type(start) is int and start == 1 and type(end) is int and end >= len(text.splitlines())
                        and set(args) <= {'path', 'startLine', 'endLine'}
                        and result == '\n'.join(text.splitlines()))
            elif name == 'read_file_chars':
                offset, length = numeric.get('offset', 0), numeric.get('length', 2500) or 2500
                full = (type(offset) is int and offset == 0 and type(length) is int and length >= len(text)
                        and set(args) <= {'path', 'offset', 'length'} and result == text)
            if full and round_index > self._boundary:
                if self._phase == 'await-read':
                    self._phase, self._boundary = 'await-retry', round_index
                    kind = 'observed-current-file'
                elif self._phase == 'await-verification':
                    self._phase, self._boundary = 'complete', round_index
                    self._verified_hash = digest(after)
                    kind = 'verified-readback'
        return self._record(name, args, result, before, after, round_index, kind)

    def verify(self, spec, root):
        if self._invalid or digest(spec) != self._task_hash or Path(root).resolve() != self._root:
            raise R.InvalidReward('invalid or mismatched recovery session')
        if self._violation:
            return R.Verdict('fail', 'recovery-tool-scope')
        if self._phase != 'complete':
            return R.Verdict('fail', 'recovery-incomplete')
        if digest(self._bytes()) != self._verified_hash:
            return R.Verdict('fail', 'changed-after-verification')
        return R.Verdict('pass', 'verified-recovery')

    def receipt(self):
        return dict(version=VERSION, task_sha256=self._task_hash, phase=self._phase,
                    events=copy.deepcopy(self._events), invalid=self._invalid, scope_violation=self._violation)
