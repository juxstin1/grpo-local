"""Pure, bounded task selection for dynamic GRPO groups.

The caller supplies complete rollout groups. Request errors make a group invalid;
they are never converted to zero rewards. A live group has full size and reward
variance. Candidate tasks are unique within a step, but may recur at later steps.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


class DynamicGroupSampler:
    def __init__(self, tasks: list[Any], quotas: dict[str, int], group_size: int,
                 oversample_factor: float, max_candidates: int, rng: Any):
        if not quotas or any(n < 1 for n in quotas.values()):
            raise ValueError("every family quota must be positive")
        if group_size < 2 or oversample_factor < 1 or max_candidates < sum(quotas.values()):
            raise ValueError("invalid group size, oversample factor, or candidate budget")
        self.tasks = list(tasks)
        self.quotas = dict(quotas)
        self.group_size = group_size
        self.oversample_factor = oversample_factor
        self.max_candidates = max_candidates
        self.rng = rng
        keys = [(t.family, t.name) for t in self.tasks]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate (family, task name) in candidate pool")
        for family, quota in self.quotas.items():
            if sum(t.family == family for t in self.tasks) < quota:
                raise ValueError(f"{family}: fewer candidates than live-group quota {quota}")
        self.sampled: set[tuple[str, str]] = set()
        self.observed: set[tuple[str, str]] = set()
        self.live: list[tuple[Any, list[Any]]] = []
        self.rewards: list[float] = []
        self.dead_groups = 0
        self.zero_pass_groups = 0
        self.all_pass_groups = 0
        self.error_groups = 0
        self.rounds = 0

    @property
    def target(self) -> int:
        return sum(self.quotas.values())

    @property
    def ready(self) -> bool:
        counts = Counter(t.family for t, _ in self.live)
        return all(counts[family] >= quota for family, quota in self.quotas.items())

    @property
    def candidate_groups(self) -> int:
        return len(self.sampled)

    @property
    def resampled_groups(self) -> int:
        """Extra task groups evaluated beyond the target live-group count."""
        return max(0, self.candidate_groups - self.target)

    @property
    def unused_live_groups(self) -> int:
        return max(0, len(self.live) - self.target)

    def next_candidates(self) -> list[Any]:
        """Oversample each family still short of its quota, within a hard budget."""
        if self.ready:
            return []
        budget = self.max_candidates - self.candidate_groups
        if budget <= 0:
            return []
        counts = Counter(t.family for t, _ in self.live)
        selected: list[Any] = []
        for family, quota in self.quotas.items():
            needed = max(0, quota - counts[family])
            if not needed:
                continue
            remaining = [t for t in self.tasks if t.family == family
                         and (t.family, t.name) not in self.sampled]
            if not remaining:
                continue
            draw = min(len(remaining), budget, math.ceil(needed * self.oversample_factor))
            if draw:
                picked = self.rng.sample(remaining, draw)
                selected.extend(picked)
                self.sampled.update((t.family, t.name) for t in picked)
                budget -= draw
        if selected:
            self.rounds += 1
        return selected

    def observe(self, task: Any, outputs: list[Any]) -> str:
        """Record a sampled group and return live, dead, or error."""
        key = (task.family, task.name)
        if key not in self.sampled or key in self.observed:
            raise ValueError(f"group was not sampled once: {key}")
        self.observed.add(key)
        if len(outputs) != self.group_size or any(isinstance(o, str) for o in outputs):
            self.error_groups += 1
            return "error"
        rewards = [float(o.reward) for o in outputs]
        self.rewards.extend(rewards)
        passes = sum(r > 0 for r in rewards)
        if passes == 0:
            self.zero_pass_groups += 1
        if passes == self.group_size:
            self.all_pass_groups += 1
        if passes in (0, self.group_size) or len(set(rewards)) == 1:
            self.dead_groups += 1
            return "dead"
        self.live.append((task, outputs))
        return "live"

    def selected_groups(self) -> list[tuple[Any, list[Any]]]:
        if not self.ready:
            raise RuntimeError("cannot select groups before all family quotas are live")
        selected = []
        for family, quota in self.quotas.items():
            selected.extend([(t, group) for t, group in self.live if t.family == family][:quota])
        return selected
