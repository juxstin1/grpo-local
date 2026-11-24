"""Rejection-sampled SFT record: one kept win, token-exact.

  {"id", "root_id", "bucket", "source", "split": "train",
   "messages": [{"role", "content", "train": false}, ...],   # prompt side, never a loss target
   "prompt_ids": [...],        # exactly what vLLM fed the model (verified == local template)
   "completion_ids": [...],    # raw sampled ids, EOS (<|im_end|>) included; the ONLY loss target
   "completion_text": "...",   # decode(completion_ids), for reading only - never re-tokenize it
   "n_tokens": int,            # len(prompt_ids) + len(completion_ids) <= MAX_TOKENS
   "grade": {"pass": true, "detail": "..."},
   "model": "Qwen/Qwen3.5-4B@<rev>", "tokenizer_template_sha": "...",
   "sampling": {"temperature", "top_p", "max_tokens"}}

root_id is the leak key: a task's wins never straddle train and eval/holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

MAX_TOKENS = 4096
sys.path.append(str(Path(__file__).resolve().parent.parent))
from gym_budget import BUDGET as MAX_TOKENS_MULTITURN  # noqa: E402  gym episodes: the episode budget (8192 default)
REQUIRED = ("id", "root_id", "bucket", "source", "split", "messages", "prompt_ids",
            "completion_ids", "completion_text", "n_tokens", "grade", "model",
            "tokenizer_template_sha", "sampling")


def template_sha(tokenizer) -> str:
    """Hash of the chat template + vocab. If this differs between SFT data and the
    trainer/GRPO tokenizer, the stored ids no longer mean the same thing."""
    h = hashlib.sha256()
    h.update((tokenizer.chat_template or "").encode())
    h.update(json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False).encode())
    return h.hexdigest()


def _token_ids(value, label, *, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label}: expected nonempty token-id list")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{label}: token IDs must be nonnegative exact integers")


def validate_structure(r: dict, eos_id: int, *, max_tokens: int | None = None) -> None:
    """Validate loss boundaries, scalar types and token counts, including quarantines.

    This proves structural consistency only, not tokenizer semantics or grading
    evidence. ``validate`` additionally requires an exact true passing verdict.
    Explicit max_tokens supports offline inspection of differently sized corpora.
    """
    if not isinstance(r, dict):
        raise ValueError('record must be an object')
    missing = [key for key in REQUIRED if key not in r]
    if missing:
        raise ValueError(f"{r.get('id')}: missing {missing}")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError('eos_id must be a nonnegative exact integer')
    for key in ('id', 'root_id', 'bucket', 'source', 'model'):
        if not isinstance(r[key], str) or not r[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if r['split'] != 'train':
        raise ValueError('only train records are SFT data')
    if not isinstance(r['completion_text'], str):
        raise ValueError('completion_text must be a string')
    if not isinstance(r['tokenizer_template_sha'], str) or not re.fullmatch(r'[0-9a-f]{64}', r['tokenizer_template_sha']):
        raise ValueError('tokenizer_template_sha must be a SHA-256 digest')
    grade = r['grade']
    if not isinstance(grade, dict) or type(grade.get('pass')) is not bool:
        raise ValueError('grade.pass must be an exact boolean')
    if any(key in grade and not isinstance(grade[key], str) for key in ('detail', 'status')):
        raise ValueError('grade detail/status must be strings when supplied')
    if not isinstance(r['messages'], list):
        raise ValueError('messages must be a list')
    for message in r['messages']:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
            raise ValueError('prompt message has unsupported role')
        if type(message.get('train')) is not bool or message['train']:
            raise ValueError('every prompt message must have exact train=false')
        if not isinstance(message.get('content'), str):
            raise ValueError('prompt message content must be a string')
    _token_ids(r['prompt_ids'], 'prompt_ids')
    _token_ids(r['completion_ids'], 'completion_ids')
    if type(r['n_tokens']) is not int or r['n_tokens'] < 1:
        raise ValueError('n_tokens must be a positive exact integer')
    sampling = r['sampling']
    if not isinstance(sampling, dict):
        raise ValueError('sampling must be an object')
    if 'max_tokens' not in sampling:
        raise ValueError('sampling.max_tokens must record a positive token cap')
    for key in ('temperature', 'top_p', 'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty'):
        if key in sampling and (type(sampling[key]) not in (int, float) or not math.isfinite(sampling[key])):
            raise ValueError(f'sampling.{key} must be a finite number')
    for key in ('max_tokens', 'budget'):
        if key in sampling and (type(sampling[key]) is not int or sampling[key] < 1):
            raise ValueError(f'sampling.{key} must be a positive exact integer')
    if 'temperature' in sampling and sampling['temperature'] < 0:
        raise ValueError('temperature cannot be negative')
    if 'top_p' in sampling and not 0 < sampling['top_p'] <= 1:
        raise ValueError('top_p must be in (0,1]')
    # Reject nonfinite values even in extension metadata; no stringification of
    # arbitrary Python objects or silent JSON scalar coercion is allowed.
    try:
        json.dumps(r, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('record contains non-JSON or nonfinite metadata') from exc
    segmented = 'segments' in r
    limit = max_tokens if max_tokens is not None else (MAX_TOKENS_MULTITURN if segmented else MAX_TOKENS)
    if type(limit) is not int or limit < 1:
        raise ValueError('max_tokens must be a positive exact integer')
    if segmented:
        segments = r['segments']
        if not isinstance(segments, list) or not segments:
            raise ValueError('segments must be nonempty when supplied')
        context_kinds = {'prompt', 'obs', 'observation', 'system', 'user', 'tool', 'context'}
        trainable = []
        seen_assistant = False
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or segment.get('kind') not in context_kinds | {'assistant'}:
                raise ValueError('unknown segment kind')
            kind = segment['kind']
            if type(segment.get('train')) is not bool:
                raise ValueError('segment train flag must be an exact boolean')
            _token_ids(segment.get('ids'), f'segment {index}')
            if index == 0 and (kind != 'prompt' or segment['train']):
                raise ValueError('first segment must be nontrainable prompt context')
            if index > 0 and kind == 'prompt':
                raise ValueError('only the initial segment can be a prompt')
            if kind in context_kinds and segment['train']:
                raise ValueError('context/observation/user/system/tool segments never train')
            if kind in {'obs', 'observation', 'tool'} and not seen_assistant:
                raise ValueError('tool observations need a preceding assistant segment')
            if 'role' in segment:
                allowed_roles = {'assistant'} if kind == 'assistant' else ({'user', 'tool'} if kind in {'obs', 'observation'} else {kind})
                if segment['role'] not in allowed_roles:
                    raise ValueError('segment role contradicts its kind')
            if kind == 'assistant':
                seen_assistant = True
            if segment['train']:
                if segment['ids'][-1] != eos_id:
                    raise ValueError('trainable assistant segment must end in EOS')
                trainable.extend(segment['ids'])
        if r['prompt_ids'] != segments[0]['ids']:
            raise ValueError('prompt_ids must equal the initial prompt segment')
        if not trainable or r['completion_ids'] != trainable:
            raise ValueError('completion_ids must equal nonempty trainable assistant segments')
        actual = sum(len(segment['ids']) for segment in segments)
    else:
        if not r['messages']:
            raise ValueError('non-segment records need nonempty prompt messages')
        if r['completion_ids'][-1] != eos_id:
            raise ValueError('completion must end in EOS')
        actual = len(r['prompt_ids']) + len(r['completion_ids'])
    if actual != r['n_tokens'] or actual > limit:
        raise ValueError(f"n_tokens={r['n_tokens']} actual={actual} max={limit}")


def validate(r: dict, eos_id: int, *, max_tokens: int | None = None) -> None:
    validate_structure(r, eos_id, max_tokens=max_tokens)
    if r['grade']['pass'] is not True:
        raise ValueError(f"{r['id']}: not a win; quarantined records require replay")


def training_tokens_and_mask(r: dict, eos_id: int, *, max_tokens: int | None = None) -> tuple[list[int], list[bool]]:
    """The trainer's shared loss boundary: validated IDs and exact boolean mask."""
    validate(r, eos_id, max_tokens=max_tokens)
    if 'segments' in r:
        ids = [token for segment in r['segments'] for token in segment['ids']]
        mask = [segment['train'] for segment in r['segments'] for _ in segment['ids']]
        return ids, mask
    return list(r['prompt_ids']) + list(r['completion_ids']), [False] * len(r['prompt_ids']) + [True] * len(r['completion_ids'])


def strict_json_object(text: str) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f'duplicate JSON key: {key}')
            value[key] = item
        return value
    def constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    record = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(record, dict):
        raise ValueError('JSON record must be an object')
    json.dumps(record, allow_nan=False)  # also rejects numeric overflow such as 1e999
    return record


def load(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding='utf-8')
    if not text:
        return []
    lines = text.split('\n')
    if lines[-1] == '':
        lines.pop()
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f'blank JSONL row {number}')
        records.append(strict_json_object(line))
    return records
