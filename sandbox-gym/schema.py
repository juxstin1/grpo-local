"""Shared data types for the sandbox gym.

One vocabulary across the three callers -- the served eval loop (agent.py),
the SFT collector (collect.py) and the local GRPO trainer (rollout.py):

  Action      a parsed tool call (name + arguments)
  Turn        one assistant generation + the observation that followed it
  Trajectory  the token-exact stream of a whole episode (GRPO only)
  Task        a gym item: prompt + enabled tools + mechanical reward

Reward signature is `reward(turns, workspace)` -- the workspace is passed so
edit/fix rewards can grade the *end state* (mutated files, test suite), not
just the final answer line. Find rewards only read the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from workspace import Workspace


@dataclass
class Action:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Turn:
    content: str = ""                          # full generation (reasoning + response)
    resp_text: str = ""                        # response section only (post " response" marker)
    reasoning: str = ""                        # served path: vLLM reasoning_content, sent back in history
    gen_ids: list[int] = field(default_factory=list)  # policy-sampled tokens (GRPO path)
    obs: str = ""                              # tool result shown to the model
    truncated: bool = False                    # hit the token budget without EOS
    action: Optional[Action] = None            # parsed tool call, if any


@dataclass
class Trajectory:
    task: Task
    turns: list[Turn]
    tok_ids: list[int] = field(default_factory=list)      # concatenated segment stream
    trainable: list[bool] = field(default_factory=list)   # per-id: policy-sampled token?
    reward: float = 0.0
    done: bool = False
    truncated: bool = False                               # any turn hit the budget
    stop: str = ""                                        # agent.Episode.stop (answer | cap | no-action | empty)
    # compaction: one (ids, trainable) per context window / note call, same reward.
    # Empty = the single stream above (every non-compact trajectory).
    windows: list = field(default_factory=list)


RewardFn = Callable[[list[Turn], "Workspace"], float]


@dataclass
class Task:
    name: str
    family: str            # "find" | "edit" | "fix"
    prompt: str
    tools: list[str]       # tool names enabled for this ring
    ground_truth: str      # find: the exact answer; edit/fix: description
    reward: RewardFn
    meta: dict = field(default_factory=dict)   # seed, workspace files, etc.