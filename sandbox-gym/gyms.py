"""Gym: turn a repo spec into a Task (prompt, tools, reward) + the system prompt.

Also owns the workspace lifecycle for an episode: a fresh scratch dir under
data/workspaces/<name>/ is created from the spec's files, wiped after.
"""

from __future__ import annotations

import repo_gen
import tools as T
from config import Config
from graders import reward_for
from runner import Runner
from schema import Task
from workspace import Workspace

SYSTEM = """\
You are a coding agent working in a sandboxed Python project. You find facts by
reading files and tools, you make changes by editing files, and you verify by
running the test suite. Never invent values or file contents that you did not
read. Work efficiently: read what you need, then answer.
{TOOLS}
"""


def system_prompt(enabled: list[str]) -> str:
    return SYSTEM.format(TOOLS=T.describe_tools(enabled))


def spec_to_task(spec: dict) -> Task:
    """repo_gen spec -> schema.Task. Reward is the family's mechanical grader."""
    task = Task(
        name=spec["name"],
        family=spec["family"],
        prompt=spec["prompt"],
        tools=spec["tools"],
        ground_truth=spec["ground_truth"],
        reward=lambda turns, ws: 0.0,  # placeholder; bound below
        meta={"seed": spec["seed"],
              "checks": spec.get("checks", [])},
    )
    # reward_for reads task.ground_truth / meta, so bind after construction.
    task.reward = reward_for(task)
    return task


def make_workspace(cfg: Config, task: Task, seed_override: int | None = None) -> Workspace:
    """Fresh sandbox for an episode: spec files into a scratch dir, wiped on reset."""
    seed = seed_override if seed_override is not None else task.meta["seed"]
    spec = repo_gen.find_spec(seed, task.family, task.name)
    root = cfg.episode.data_root / "workspaces" / f"{task.name}-{seed}"
    ws = Workspace(root, Runner(cfg.episode.runner))
    ws.reset()
    ws.builds_left = cfg.episode.max_builds
    repo_gen.write_files(spec["files"], root)
    return ws