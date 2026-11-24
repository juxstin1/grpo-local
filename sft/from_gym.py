"""Convert gym token records without concealing prompt transformations.

Tokenizer loading is gated by --allow-model-run. Transformed records retain the
old grade in provenance but receive grade.pass=False until independent replay.
Original inputs are never overwritten; outputs are exclusive UTF-8/LF JSONL.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sft.sft_schema import load, strict_json_object, template_sha, validate_structure
from ops.run_policy import add_model_flag, require_model_opt_in


OLD_FINISH = ("A result line follows each call. When you know the answer, output:\n"
              "  ANSWER: <your answer>\n")
NEW_FINISH = ("A result line follows each call. When you know the answer, or the requested "
              "change is done, end with one line:\n"
              "  ANSWER: <your answer, or a one-line summary of the change>\n")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _prompt(r, tok, render_chat):
    messages, segments = r.get('messages'), r.get('segments')
    if not isinstance(messages, list) or not messages or messages[0].get('role') != 'system':
        raise ValueError('prompt transformation requires a leading system message')
    if not isinstance(segments, list) or not segments or segments[0].get('kind') != 'prompt' or segments[0].get('train') is not False:
        raise ValueError('prompt transformation requires an initial nontrainable prompt segment')
    if any(not isinstance(m, dict) or m.get('train') is not False for m in messages):
        raise ValueError('all prompt messages must explicitly be nontrainable')
    if not isinstance(r.get('grade'), dict) or type(r['grade'].get('pass')) is not bool:
        raise ValueError('original grade.pass must be an exact boolean')
    if 'n_tokens' in r and (type(r['n_tokens']) is not int or r['n_tokens'] != sum(len(s['ids']) for s in segments)):
        raise ValueError('original token count contradicts segments')
    rendered_messages = [{k: v for k, v in m.items() if k != 'train'} for m in messages]
    if render_chat(tok, rendered_messages, True) != segments[0]['ids']:
        raise ValueError('stored prompt tokens do not match the original rendered messages')
    if 'prompt_ids' in r and r['prompt_ids'] != segments[0]['ids']:
        raise ValueError('original prompt_ids contradict initial segment')
    return rendered_messages


def _transform(r, tok, new_system, kind, render_chat, *, hint=None):
    old_messages = _prompt(r, tok, render_chat)
    new_messages = copy.deepcopy(old_messages)
    new_messages[0]['content'] = new_system
    ids = render_chat(tok, new_messages, True)
    out = copy.deepcopy(r)
    out['segments'][0]['ids'] = ids
    out['messages'][0]['content'] = new_system
    out['prompt_ids'] = list(ids)
    out['n_tokens'] = sum(len(segment['ids']) for segment in out['segments'])
    provenance = out.setdefault('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('provenance must be an object')
    transformations = provenance.setdefault('prompt_transformations', [])
    if not isinstance(transformations, list):
        raise ValueError('prompt_transformations must be a list')
    event = dict(kind=kind, original_prompt_sha256=_hash(old_messages), new_prompt_sha256=_hash(new_messages),
                 original_prompt_ids_sha256=_hash(r['segments'][0]['ids']), new_prompt_ids_sha256=_hash(ids),
                 original_grade=copy.deepcopy(r.get('grade')), requires_replay=True)
    if hint is not None:
        event['hint_sha256'] = _hash(hint)
        out['hinted'] = True
        provenance['hinted'] = True
        out['hint_stripped'] = True
    transformations.append(event)
    provenance['requires_replay'] = True
    out['grade'] = {'pass': False, 'status': 'replay-required',
                    'detail': 'Prompt changed; original grade applies only to the original context.'}
    if kind == 'refresh-finish-rule':
        out['prompt_refreshed'] = True
    return out


def refresh_prompt(r: dict, tok, *, render_chat) -> dict:
    """Return an unchanged record or a quarantined copy with both prompt hashes."""
    old = _prompt(r, tok, render_chat)
    if OLD_FINISH not in old[0]['content']:
        return copy.deepcopy(r)
    return _transform(r, tok, old[0]['content'].replace(OLD_FINISH, NEW_FINISH),
                      'refresh-finish-rule', render_chat)


def strip_hint(r: dict, tok, hint: str, *, render_chat) -> dict:
    """Strip context but preserve hinted provenance and invalidate its old grade."""
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError('hint must be nonempty')
    old = _prompt(r, tok, render_chat)
    line = hint.rstrip('\n') + '\n'
    if line not in old[0]['content']:
        raise ValueError('hint line absent from the original system prompt')
    return _transform(r, tok, old[0]['content'].replace(line, '', 1), 'strip-hint', render_chat, hint=hint)


def convert_record(r, source, tok, tsha, eos, *, root_mapping=None, max_tokens=None):
    """Attach serialization metadata without changing a supplied canonical root.

    Legacy integer/missing roots require an explicit mapping keyed by record ID.
    This conversion never upgrades a grade or asserts independent verification.
    """
    if not isinstance(r, dict) or not isinstance(r.get('segments'), list) or not r['segments']:
        raise ValueError('not a segmented gym record')
    if not isinstance(r.get('grade'), dict) or type(r['grade'].get('pass')) is not bool:
        raise ValueError('original grade.pass must be an exact boolean')
    out = copy.deepcopy(r)
    original_root = r.get('root_id')
    if not isinstance(original_root, str) or not original_root.strip():
        mapped = (root_mapping or {}).get(r.get('id'))
        if not isinstance(mapped, str) or not mapped.strip():
            raise ValueError('missing/nonstring root requires an explicit root mapping and requalification')
        out['root_id'] = mapped
        provenance = out.setdefault('provenance', {})
        if not isinstance(provenance, dict):
            raise ValueError('provenance must be an object')
        provenance['original_root_id'] = original_root
        provenance['root_mapping'] = {'record_id': r.get('id'), 'mapped_root_id': mapped}
        provenance['original_grade_before_root_mapping'] = copy.deepcopy(out.get('grade'))
        provenance['requires_replay'] = True
        out['grade'] = {'pass': False, 'status': 'replay-required', 'detail': 'Root identity mapped; requires requalification.'}
    elif root_mapping and r.get('id') in root_mapping and root_mapping[r['id']] != original_root:
        raise ValueError('explicit mapping cannot replace an existing canonical root')
    segments = out['segments']
    # Do not repair contradictory IDs or masks: retain them for schema rejection.
    completion = [token for segment in segments if segment.get('train') is True for token in segment.get('ids', [])]
    if 'completion_ids' in out and out['completion_ids'] != completion:
        raise ValueError('stored completion_ids contradict loss mask')
    out.setdefault('prompt_ids', segments[0].get('ids'))
    out['completion_ids'] = completion
    out['completion_text'] = tok.decode(completion)
    out['source'] = source
    actual_tokens = sum(len(segment.get('ids', [])) for segment in segments)
    if 'n_tokens' in out and (type(out['n_tokens']) is not int or out['n_tokens'] != actual_tokens):
        raise ValueError('stored n_tokens contradict segment lengths')
    out['n_tokens'] = actual_tokens
    if 'tokenizer_template_sha' in out and out['tokenizer_template_sha'] != tsha:
        raise ValueError('tokenizer/template identity differs from the stored record')
    out['tokenizer_template_sha'] = tsha
    sampling = out.setdefault('sampling', {})
    if 'max_tokens' not in sampling and 'budget' in sampling:
        sampling['max_tokens'] = sampling['budget']
    validate_structure(out, eos, max_tokens=max_tokens)
    return out


def write_new_records(path, records):
    """Exclusive UTF-8/LF export; never overwrite a source or previous artifact."""
    if not records:
        raise ValueError('no structurally valid records to export')
    data = ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
                   for record in records).encode('utf-8')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3.5-4B')
    ap.add_argument('--strip-hint', default='')
    ap.add_argument('--max-turns', type=int)
    ap.add_argument('--max-tokens', type=int)
    ap.add_argument('--refresh-prompt', action='store_true')
    ap.add_argument('--root-map', type=Path, help='Explicit record ID to canonical string root mapping; mapped records require replay.')
    add_model_flag(ap)
    a = ap.parse_args()
    require_model_opt_in(a.allow_model_run)
    output = Path(a.out)
    if output.exists():
        raise ValueError('output already exists; choose a new artifact path')
    if a.max_turns is not None and a.max_turns < 1:
        raise ValueError('max-turns must be positive')
    records = load(a.src)
    root_mapping = None
    if a.root_map:
        root_mapping = strict_json_object(a.root_map.read_text(encoding='utf-8'))
        if any(not isinstance(value, str) or not value.strip() for value in root_mapping.values()):
            raise ValueError('root-map values must be nonempty canonical root strings')
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE.parent / 'sandbox-gym'))
    import encoding
    tok = AutoTokenizer.from_pretrained(a.model)
    tsha, eos = template_sha(tok), tok.convert_tokens_to_ids('<|im_end|>')
    converted, too_long = [], 0
    for original in records:
        record = convert_record(original, a.source, tok, tsha, eos,
                                root_mapping=root_mapping, max_tokens=a.max_tokens)
        if a.strip_hint:
            record = strip_hint(record, tok, a.strip_hint, render_chat=encoding.render_chat)
        if a.refresh_prompt:
            record = refresh_prompt(record, tok, render_chat=encoding.render_chat)
        validate_structure(record, eos, max_tokens=a.max_tokens)
        converted_record = record
        if a.max_turns and sum(s['kind'] == 'assistant' for s in converted_record['segments']) > a.max_turns:
            too_long += 1
            continue
        converted.append(converted_record)
    write_new_records(output, converted)
    replay = sum(record['grade']['pass'] is False for record in converted)
    print(f'{len(converted)} records -> {output}; {replay} require replay; {too_long} over turn limit')


if __name__ == '__main__':
    main()
