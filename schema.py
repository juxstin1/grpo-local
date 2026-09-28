"""Shared data types.

Turn / Trajectory / Task live here so that tasks.py (reward logic) and
rollout.py (sampling) can both reference them without a circular import:
tasks.py must read turns to score them, rollout.py must build them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Turn:
    gen_text: str = ""
    gen_ids: list[int] = field(default_factory=list)     # policy-sampled tokens
    obs_text: str = ""                                   # tool result (not policy tokens)


@dataclass
class Trajectory:
    task: Task
    turns: list[Turn]
    tok_ids: list[int] = field(default_factory=list)      # running stream, concatenated
    trainable: list[bool] = field(default_factory=list)   # per-id: policy token?
    reward: float = 0.0
    done: bool = False


RewardFn = Callable[[list[Turn], bool], float]


@dataclass
class Task:
    name: str
    prompt: str
    reward_fn: RewardFn
    ground_truth: str = ""
    meta: dict = field(default_factory=dict)