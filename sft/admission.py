"""Endpoint SFT admission: pinned task, unchanged context, token fidelity, replay.

Local hashes establish correspondence, not independent authority. Historical
tool trajectories need task/fixture/artifact replay and are not admitted here.
No tokenizer or model is imported by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import graders
from evals.code_executor import EXECUTOR_PROFILE, InvalidGrading

VERSION = 'endpoint-sft-evidence-v1'
SOURCES = ('evals/graders.py', 'evals/code_executor.py', 'sft/admission.py', 'sft/sft_schema.py')


class AdmissionError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def verifier_identity():
    return {'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
            'executor_profile': EXECUTOR_PROFILE, 'sandbox_image': os.environ.get('GRPO_CODE_SANDBOX_IMAGE'),
            'python': sys.version}


def bound_record(record):
    # Bind the training target, all prompt text/IDs, grading input and provenance.
    return {k: v for k, v in record.items() if k != 'admission_evidence'}


def build_task_index(registry):
    index = {}
    for pool in registry.document['pools']:
        for task in registry.load_pool(registry.path.parent / pool['path']):
            if task['uid'] in index:
                raise AdmissionError('duplicate registered task identity')
            index[task['uid']] = task
    return index


def require_registered_tasks(tasks, index):
    for task in tasks:
        known = index.get(task.get('uid'))
        if known is None or digest(known) != digest(task):
            raise AdmissionError('task is not a frozen registered input; register reproducible tasks before collection')


def make_evidence(task, record, registry_sha256):
    """Collector-owned receipt after grading; trainer still checks tokens/replays."""
    if record.get('segments') or record.get('teacher_trace_id'):
        raise AdmissionError('tool/teacher trajectory requires a separately qualified artifact replay')
    return {'version': VERSION, 'scope': 'registered-public-endpoint',
            'task_sha256': digest(task), 'record_sha256': digest(bound_record(record)),
            'registry_sha256': registry_sha256, 'verifier': verifier_identity(),
            'context': {'hinted': False, 'transformed': False}, 'status': 'collected-not-replayed'}


def check_record(record, registry, index):
    evidence = record.get('admission_evidence')
    if not isinstance(evidence, dict) or evidence.get('version') != VERSION:
        raise AdmissionError('missing reproducible admission evidence; historical wins are quarantined')
    provenance = record.get('provenance', {})
    if not isinstance(provenance, dict):
        raise AdmissionError('invalid context provenance')
    if (record.get('segments') or record.get('teacher_trace_id') or record.get('hint_stripped') or record.get('hinted')
            or provenance.get('hinted') or provenance.get('requires_replay') or provenance.get('prompt_transformations')
            or record.get('prompt_refreshed') or record.get('context_transformations')):
        raise AdmissionError('trajectory/transformed context requires separate replay qualification')
    if evidence.get('scope') != 'registered-public-endpoint' or evidence.get('context') != {'hinted': False, 'transformed': False}:
        raise AdmissionError('hinted/unknown context cannot be admitted as an unassisted demonstration')
    if evidence.get('status') != 'collected-not-replayed':
        raise AdmissionError('unexpected collection evidence status')
    if record.get('split') != 'train' or record.get('grade', {}).get('pass') is not True or record.get('grade', {}).get('status') != 'valid':
        raise AdmissionError('record is not a valid observed win')
    if evidence.get('registry_sha256') != registry.sha256 or record.get('dataset_registry_sha256') != registry.sha256:
        raise AdmissionError('training registry changed')
    task = index.get(record.get('uid'))
    if task is None or evidence.get('task_sha256') != digest(task) or record.get('root_id') != task['root_id']:
        raise AdmissionError('task/identity differs from registered input')
    expected_messages = [{**m, 'train': False} for m in graders.messages(task)]
    if record.get('messages') != expected_messages:
        raise AdmissionError('collected prompt differs from the registered task/instructions')
    if evidence.get('record_sha256') != digest(bound_record(record)):
        raise AdmissionError('training record changed after grading')
    if evidence.get('verifier') != verifier_identity():
        raise AdmissionError('grader/runtime changed; explicit requalification required')
    if not isinstance(record.get('verification_response'), str) or not record['verification_response'].strip():
        raise AdmissionError('missing exact graded decoding')
    registry.validate_training_rows([record])
    return task


def check_dataset(records, registry, index=None):
    index = build_task_index(registry) if index is None else index
    if not records:
        raise AdmissionError('empty training dataset')
    ids = [r.get('id') for r in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise AdmissionError('training record IDs must be unique')
    for record in records:
        check_record(record, registry, index)
    return index


def verify_tokens_and_replay(record, tokenizer, registry, index):
    task = check_record(record, registry, index)
    messages = [{k: v for k, v in message.items() if k != 'train'} for message in record['messages']]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    prompt = prompt['input_ids'] if hasattr(prompt, 'keys') else prompt
    if list(prompt) != record['prompt_ids']:
        raise AdmissionError('prompt token IDs do not encode the bound instructions/task')
    if tokenizer.decode(record['completion_ids'], skip_special_tokens=False) != record['completion_text']:
        raise AdmissionError('training completion IDs do not encode the stored completion')
    if tokenizer.decode(record['completion_ids'], skip_special_tokens=True) != record['verification_response']:
        raise AdmissionError('graded response is not the training target decoding')
    # Re-evaluate actual decoded target using the pinned registered task. A string
    # claiming pass, even paired with freshly recomputed hashes, cannot replace this.
    try:
        passed, detail = graders.grade(task, record['verification_response'])
    except Exception as exc:
        raise AdmissionError('replay unavailable or invalid; no training admission') from exc
    if passed is not True:
        raise AdmissionError('recorded win did not pass independent replay')
    return {'status': 'replayed', 'record_sha256': digest(bound_record(record)),
            'task_sha256': digest(task), 'verifier': verifier_identity()}
