"""Outcome contracts for the public development suite; no model judges or code execution.

Legacy substring/keyword graders are not qualified rewards. They remain available
only through an explicitly labelled diagnostic path, never as training targets.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat


class InvalidReward(RuntimeError):
    """Verifier/environment invalidity: abort scoring; never substitute zero."""


class UnqualifiedTask(InvalidReward):
    """No validated outcome contract exists for this task."""


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str

    @property
    def reward(self):
        if self.status not in ('pass', 'fail'):
            raise InvalidReward(self.reason)
        return float(self.status == 'pass')


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def constant(_):
        raise ValueError('non-finite JSON number')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def final_reply(turns):
    if not turns:
        return None
    turn = turns[-1]
    if getattr(turn, 'action', None) is not None or getattr(turn, 'extra', None) or getattr(turn, 'truncated', False):
        return None
    text = getattr(turn, 'resp_text', '') or getattr(turn, 'content', '') or ''
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        return None
    return text.strip() or None


def final_answer(turns):
    reply = final_reply(turns)
    if reply is None:
        return None
    matches = re.findall(r'^\s*ANSWER\s*:\s*(\S[^\r\n]*)$', reply, re.MULTILINE)
    return matches[0].strip().strip('` ') if len(matches) == 1 else None


def safe_rel(name):
    if not isinstance(name, str):
        raise InvalidReward('contract path must be text')
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise InvalidReward('invalid contract path')
    return path


def _linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def file_inventory(root):
    """Reject observed links/junctions; caller must stop workspace writers first."""
    root = Path(root)
    if not root.is_dir():
        raise InvalidReward('workspace unavailable')
    if any(_linked(p) for p in (root.absolute(), *root.absolute().parents)):
        return None
    result = {}
    pending = [root]
    while pending:
        current = pending.pop()
        for path in current.iterdir():
            if _linked(path):
                return None
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                result[path.relative_to(root).as_posix()] = path
            else:
                return None
    return result


def validate_spec(spec):
    if not isinstance(spec, dict):
        raise InvalidReward('task specification must be an object')
    contract = spec.get('contract')
    if not isinstance(contract, dict) or contract.get('version') != 'outcome-v1':
        raise UnqualifiedTask('Task has no qualified outcome-v1 contract; legacy keyword scores are diagnostic only.')
    if spec.get('split') != 'development':
        raise UnqualifiedTask('This local evaluator supports public development tasks only, not sealed evaluation.')
    files, outputs = spec.get('files'), contract.get('outputs')
    if not isinstance(files, dict) or not isinstance(outputs, dict) or not outputs:
        raise InvalidReward('contract files/outputs missing')
    for path, text in files.items():
        safe_rel(path)
        if not isinstance(text, str):
            raise InvalidReward('contract source must be text')
    for path, rule in outputs.items():
        safe_rel(path)
        if not isinstance(rule, dict) or rule.get('kind') not in ('text_exact', 'json_exact', 'python_ast') or 'expected' not in rule:
            raise InvalidReward('unsupported output contract')
        if rule['kind'] != 'json_exact' and not isinstance(rule['expected'], str):
            raise InvalidReward('text/AST contract expectation must be text')
        if rule['kind'] == 'python_ast':
            try:
                compile(rule['expected'].encode('utf-8'), path, 'exec', flags=ast.PyCF_ONLY_AST)
            except (SyntaxError, ValueError) as exc:
                raise InvalidReward('invalid reference Python') from exc
        if rule['kind'] == 'json_exact':
            try:
                json.dumps(rule['expected'], allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise InvalidReward('invalid reference JSON') from exc
    if not isinstance(contract.get('final_answer'), str) or not contract['final_answer']:
        raise InvalidReward('contract final answer missing')
    if 'recovery' in contract:
        from recovery_contract import validate_policy
        validate_policy(spec)


def _json_equal(actual, expected):
    # bool is a subclass of int in Python; True must not satisfy expected 1.
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_json_equal(actual[k], v) for k, v in expected.items())
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_json_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, float) and not math.isfinite(actual):
        return False
    return actual == expected


def grade(spec, turns, root, *, execution=None):
    validate_spec(spec)
    if 'recovery' in spec['contract']:
        from recovery_contract import RecoverySession
        if type(execution) is not RecoverySession:
            raise InvalidReward('recovery task requires a matching live harness session')
        recovery = execution.verify(spec, root)
        if recovery.status != 'pass':
            return recovery
    if final_reply(turns) is None:
        return Verdict('fail', 'no-complete-final-reply')
    # Public tasks explicitly require a single terminal delivery record. Do not
    # attempt to infer semantic completion from arbitrary prose or old rounds.
    expected_reply = 'ANSWER: ' + spec['contract']['final_answer']
    if final_reply(turns) != expected_reply:
        return Verdict('fail', 'final-delivery-record')
    try:
        actual = file_inventory(root)
        if actual is None:
            return Verdict('fail', 'unexpected-link-or-file-type')
        originals, outputs = spec['files'], spec['contract']['outputs']
        if set(actual) != set(originals) | set(outputs):
            return Verdict('fail', 'unexpected-or-missing-file')
        for name, text in originals.items():
            if name not in outputs and actual[name].read_bytes() != text.encode('utf-8'):
                return Verdict('fail', 'source-preservation')
        for name, rule in outputs.items():
            if actual[name].stat().st_size > 1_000_000:
                return Verdict('fail', 'output-too-large')
            data = actual[name].read_text(encoding='utf-8')
            try:
                if rule['kind'] == 'text_exact':
                    matches = actual[name].read_bytes() == rule['expected'].encode('utf-8')
                elif rule['kind'] == 'json_exact':
                    matches = _json_equal(strict_json(data), rule['expected'])
                else:
                    # Python interprets encoding cookies before parsing a file.
                    # Parse bytes exactly as Python would; parsing decoded text
                    # alone can overlook executable content in another encoding.
                    parsed = compile(actual[name].read_bytes(), name, 'exec', flags=ast.PyCF_ONLY_AST)
                    expected_ast = compile(rule['expected'].encode('utf-8'), name, 'exec', flags=ast.PyCF_ONLY_AST)
                    matches = ast.dump(parsed, include_attributes=False) == ast.dump(expected_ast, include_attributes=False)
            except (ValueError, SyntaxError, TypeError):
                matches = False
            if not matches:
                return Verdict('fail', 'output-contract')
    except UnicodeError:
        return Verdict('fail', 'invalid-output-encoding')
    except OSError as exc:
        raise InvalidReward('workspace verification unavailable') from exc
    return Verdict('pass', 'verified-outcome')


def summarize(episodes):
    if not episodes:
        raise InvalidReward('No valid episodes; no reward mean exists.')
    rewards = [e['reward'] for e in episodes]
    if any(type(r) not in (int, float) or not math.isfinite(r) or r not in (0, 1) for r in rewards):
        raise InvalidReward('Invalid/unqualified attempt cannot enter reward aggregation.')
    families = sorted({e['family'] for e in episodes})
    return dict(reward_mean=sum(rewards) / len(rewards),
                by_family={f: [e['reward'] for e in episodes if e['family'] == f] for f in families},
                valid_attempts=len(episodes))
