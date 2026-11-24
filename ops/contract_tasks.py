"""Public development roots. Not a sealed benchmark or evidence of model ability."""
from __future__ import annotations
import copy
import json
import hashlib
from pathlib import Path
import reward_contract as R

FAMILY = 'contract_v1'
ROOT = Path(__file__).resolve().parents[1]


def suite_metadata():
    paths = [ROOT / 'suite-v2/tasks.json', Path(__file__), ROOT / 'ops/reward_contract.py',
             ROOT / 'ops/recovery_contract.py', ROOT / 'ops/grpo_q4.py']
    return {'version': json.loads(paths[0].read_text(encoding='utf-8'))['version'], 'sealed': False,
            'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def specs():
    tasks = json.loads((ROOT / 'suite-v2/tasks.json').read_text(encoding='utf-8'))['tasks']
    seen = set()
    for task in tasks:
        if task['root_id'] in seen:
            raise R.InvalidReward('duplicate development root')
        seen.add(task['root_id'])
        R.validate_spec(task)
    return copy.deepcopy(tasks)


def make_task(spec, Task, *, execution=None):
    R.validate_spec(spec)
    return Task(name=spec['name'], family=FAMILY, prompt=spec['prompt'],
                tools=spec['tools'], ground_truth='',
                reward=lambda turns, ws: R.grade(spec, turns, ws.root, execution=execution).reward,
                meta={'seed': spec['seed'], 'native': True, 'root_id': spec['root_id'],
                      'split': 'development', 'reward_contract': 'outcome-v1'})


class TaskStream:
    """Explicit finite development roots; repetitions do not create new coverage."""
    def __init__(self):
        self.tasks = specs()
        self.index = 0

    def next(self):
        task = copy.deepcopy(self.tasks[self.index % len(self.tasks)])
        self.index += 1
        return FAMILY, task
