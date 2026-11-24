"""Public development code execution: isolated candidate, trusted host comparison.

Only candidate code and inputs enter a CPU container. Expected outputs and grading
verdicts never do. This is not the private evaluator; no sealed data is supported.
An existing Python image pinned by repository digest must be explicitly selected.
There is no image pull, daemon start, host execution fallback, or model import.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid


class InvalidGrading(RuntimeError):
    """Unavailable verifier or invalid task; must not be converted to reward zero."""


MAX_OUTPUT = 1_048_576
MAX_JOB = 4_194_304
EXECUTOR_PROFILE = 'public-docker-cpu-v1'

# Deliberately no answers or trusted passed/total fields. stdout is untrusted.
WORKER = r'''
import json,sys,io,contextlib
job=json.load(sys.stdin)
if job['kind']=='stdin':
    sys.stdin=io.StringIO(job['input'])
    exec(compile(job['code'],'<candidate>','exec'),{'__name__':'__main__'})
else:
    g={'__name__':'__candidate__'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(job['code'],'<candidate>','exec'),g)
        cls,meth=job['entry'].split('.')
        fn=getattr(g[cls](),meth)
        outputs=[fn(*args) for args in job['inputs']]
    print(json.dumps({'outputs':outputs},allow_nan=False))
'''


def image_reference():
    image = os.environ.get('GRPO_CODE_SANDBOX_IMAGE', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]*@sha256:[a-f0-9]{64}', image):
        raise InvalidGrading('Set GRPO_CODE_SANDBOX_IMAGE to an existing Python image pinned by @sha256 digest')
    return image


def command(image, name):
    return ['docker', 'run', '--rm', '--pull=never', '--name', name,
            '--network=none', '--read-only', '--cap-drop=ALL',
            '--security-opt=no-new-privileges', '--user=65534:65534',
            '--cpus=1', '--memory=256m', '--memory-swap=256m', '--pids-limit=32',
            '--ulimit=nofile=64:64', '--ulimit=fsize=1048576:1048576',
            '--log-driver=none', '--tmpfs=/tmp:rw,noexec,nosuid,size=16m',
            '--workdir=/tmp', '-i', '--entrypoint=python', image, '-I', '-B', '-c', WORKER]


def _run_container(job, timeout):
    image = image_reference()
    data = json.dumps(job, allow_nan=False).encode()
    if len(data) > MAX_JOB:
        raise InvalidGrading('candidate job exceeds byte limit')
    name = 'grpo-public-grade-' + uuid.uuid4().hex
    try:
        proc = subprocess.Popen(command(image, name), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise InvalidGrading('Docker executable unavailable') from exc
    chunks = [bytearray(), bytearray()]
    overflow = threading.Event()
    read_errors = []

    def consume(stream, index):
        try:
            while True:
                chunk = stream.read1(8192)
                if not chunk:
                    break
                remaining = MAX_OUTPUT - len(chunks[index])
                chunks[index].extend(chunk[:max(0, remaining)])
                if len(chunk) > remaining:
                    overflow.set()
                    break
        except OSError as exc:
            read_errors.append(str(exc))

    def feed():
        try:
            proc.stdin.write(data)
            proc.stdin.close()
        except (OSError, BrokenPipeError):
            pass

    threads = [threading.Thread(target=consume, args=(proc.stdout, 0), daemon=True),
               threading.Thread(target=consume, args=(proc.stderr, 1), daemon=True),
               threading.Thread(target=feed, daemon=True)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    limit = None
    while proc.poll() is None:
        if overflow.wait(0.02):
            limit = 'output-limit'
            break
        if time.monotonic() >= deadline:
            # Host timeout can also be slow/unavailable Docker startup; never blame candidate.
            limit = 'executor-timeout'
            break
    if limit:
        # Only the unique, temporary container created by this invocation is targeted.
        try:
            cleanup = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            proc.kill()
            proc.wait(timeout=5)
            raise InvalidGrading('owned sandbox cleanup failed') from exc
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        if cleanup.returncode != 0:
            raise InvalidGrading('owned sandbox cleanup failed')
    for thread in threads:
        thread.join(timeout=2)
    if any(thread.is_alive() for thread in threads) or read_errors:
        raise InvalidGrading('incomplete executor transport')
    if overflow.is_set():
        limit = 'output-limit'
    if limit == 'executor-timeout':
        raise InvalidGrading(limit)
    if limit:
        return {'status': 'candidate-error', 'reason': limit}
    if proc.returncode in (125, 126, 127):
        raise InvalidGrading('sandbox unavailable or incorrectly configured')
    if proc.returncode != 0:
        # Ambiguous nonzero exit could be Docker/daemon/worker failure, not candidate.
        raise InvalidGrading('nonzero sandbox exit; no verified candidate outcome')
    try:
        return {'status': 'ok', 'stdout': bytes(chunks[0]).decode('utf-8')}
    except UnicodeDecodeError:
        return {'status': 'candidate-error', 'reason': 'non-utf8-output'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def execute(code, *, kind, inputs=None, entry=None, stdin=None, timeout=20):
    if kind == 'functional':
        if not isinstance(inputs, list) or not inputs or not all(isinstance(x, list) for x in inputs):
            raise InvalidGrading('functional tests require nonempty argument lists')
        if not isinstance(entry, str) or not re.fullmatch(r'[A-Za-z_]\w*\.[A-Za-z_]\w*', entry):
            raise InvalidGrading('invalid functional entry')
        job = {'kind': kind, 'code': code, 'entry': entry, 'inputs': inputs}
    elif kind == 'stdin':
        if not isinstance(stdin, str):
            raise InvalidGrading('stdin must be text')
        job = {'kind': kind, 'code': code, 'input': stdin}
    else:
        raise InvalidGrading('unsupported execution kind')
    result = _run_container(job, timeout)
    if result['status'] != 'ok' or kind == 'stdin':
        return result
    try:
        payload = json.loads(result['stdout'], object_pairs_hook=_unique_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite number')))
        if not isinstance(payload, dict) or set(payload) != {'outputs'} or not isinstance(payload['outputs'], list) or len(payload['outputs']) != len(inputs):
            raise ValueError('not an outputs-only receipt')
        return {'status': 'ok', 'outputs': payload['outputs']}
    except (ValueError, TypeError, RecursionError):
        return {'status': 'candidate-error', 'reason': 'malformed-candidate-output'}
