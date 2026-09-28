"""Sealed holdout for the GRPO bake-off. 8 tasks, mechanical rewards.

SEALED: never read, edited, or trained on. The runner may only evaluate these
and report the aggregate. Do not paste items into prompts shown during training,
do not use them as reward-tuning examples.

These are procedural variants of the mock-repo family in tasks.py, frozen with
fresh config keys/values the seed training pool has never seen.
"""

from __future__ import annotations

import re
from typing import Optional

from schema import Task, Turn

TOOL_RE = re.compile(
    r'\{"tool"\s*:\s*"(lookup|grep)"\s*,\s*"arg"\s*:\s*"([^"]+)"\}'
)
ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
SYSTEM = """\
You are a coding agent working on a small C++ audio project.
Before answering you MUST use the tools:
  {"tool": "lookup", "arg": "<config key>"}   -> value of that key
  {"tool": "grep",   "arg": "<pattern>"}      -> files/lines matching
Emit exactly one action per turn. A result line follows each action.
When you know the answer, output: ANSWER: <your answer>
Never invent values. Answer in the fewest words possible.
"""

CONFIG = {"eq": 0.25, "mix": -3.0, "thresh": 512, "fade_ms": 850, "seed": 7}

REPO = {
    "config.json": '{"eq": 0.25, "mix": -3.0, "thresh": 512, "fade_ms": 850, "seed": 7}',
    "audio_thread.cpp": [
        'auto t = cfg.read("thresh");   // audio thread',
        'auto f = cfg.read("fade_ms");  // audio thread',
        'auto s = cfg.read("mix");      // audio thread',
    ],
    "dsp.cpp": [
        'auto e = cfg.read("eq");       // dsp thread',
        'auto seed = cfg.read("seed");  // dsp thread',
    ],
    "ui_thread.cpp": [
        'auto m = cfg.read("mix");      // ui thread',
    ],
}


class HoldoutEnv:
    """Sealed mirror of MockEnv; reads the holdout CONFIG/REPO only."""

    def step(self, tool: str, arg: str) -> str:
        if tool == "lookup":
            if arg in CONFIG:
                return f"value of {arg} = {CONFIG[arg]}"
            return f"not found: {arg}"
        if tool == "grep":
            hits = [
                f"{path}:{i}: {line}"
                for path, lines in REPO.items()
                if isinstance(lines, list)
                for i, line in enumerate(lines)
                if arg.lower() in line.lower()
            ]
            return "\n".join(hits) if hits else f"no matches for {arg}"
        return f"unknown tool: {tool}"


def parse_action(text: str) -> Optional[tuple[str, str]]:
    m = TOOL_RE.search(text)
    return (m.group(1), m.group(2)) if m else None


def used_tool(turns: list[Turn]) -> bool:
    return any(parse_action(t.gen_text) is not None for t in turns)


def last_answer(turns: list[Turn]) -> Optional[str]:
    for t in reversed(turns):
        m = ANSWER_RE.search(t.gen_text)
        if m:
            return m.group(1).strip().strip(".")
    return None


def normalize(s: str) -> str:
    return s.strip().lower().rstrip(".")


def reward_answer(turns: list[Turn], ground_truth: str) -> float:
    ans = last_answer(turns)
    return 1.0 if ans is not None and normalize(ans) == normalize(ground_truth) else 0.0


def reward_tool_discipline(turns: list[Turn], ground_truth: str) -> float:
    ans = last_answer(turns)
    correct = ans is not None and normalize(ans) == normalize(ground_truth)
    if correct and used_tool(turns):
        return 1.0
    if correct and not used_tool(turns):
        return 0.5
    return 0.0


def make_holdout() -> list[Task]:
    return [
        Task(
            name="hold-1-lookup-thresh",
            prompt=(
                SYSTEM
                + "\nUSER TASK: What value does the audio thread read for thresh?\n"
                "Report only the number."
            ),
            reward_fn=lambda turns, done: reward_answer(turns, "512"),
            ground_truth="512",
        ),
        Task(
            name="hold-2-lookup-fade",
            prompt=(
                SYSTEM
                + "\nUSER TASK: What is the current value of the config key fade_ms?\n"
                "Report only the number."
            ),
            reward_fn=lambda turns, done: reward_answer(turns, "850"),
            ground_truth="850",
        ),
        Task(
            name="hold-3-grep-two-attrs",
            prompt=(
                SYSTEM
                + "\nUSER TASK: The config key 'mix' is read by two of the project "
                "threads. Name the two.\nReport only the two names, comma separated."
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, "audio,ui"),
            ground_truth="audio,ui",
        ),
        Task(
            name="hold-4-grep-seed-thread",
            prompt=(
                SYSTEM
                + "\nUSER TASK: Which thread reads the 'seed' config key?\n"
                "Report only the thread name."
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, "dsp"),
            ground_truth="dsp",
        ),
        Task(
            name="hold-5-tool-form-mix",
            prompt=(
                SYSTEM
                + f"\nUSER TASK: Using the tools, find the value of the config key mix.\n"
                f"Report only the number."
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, "-3.0"),
            ground_truth="-3.0",
        ),
        Task(
            name="hold-6-two-hop-eq",
            prompt=(
                SYSTEM
                + "\nUSER TASK: The eq value and the seed value are both read in the "
                "same thread. The current seed is the number 7, and eq is 0.25. "
                "Name that thread and confirm eq's value.\nReport only: <thread>:<eq>"
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, "dsp:0.25"),
            ground_truth="dsp:0.25",
        ),
        Task(
            name="hold-7-absent-key",
            prompt=(
                SYSTEM
                + "\nUSER TASK: What is the value of the config key 'tempo'?\n"
                "Report only the number."
            ),
            reward_fn=lambda turns, done: (
                lambda ans: 1.0
                if ans is not None and normalize(ans) == "not found" and used_tool(turns)
                else 0.0
            )(last_answer(turns)),
            ground_truth="not found",
        ),
        Task(
            name="hold-8-list-threads",
            prompt=(
                SYSTEM
                + "\nUSER TASK: How many distinct project threads exist here, and which "
                "one spreads its reads across the most config keys?\n"
                "Report only: <count>,<thread>"
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, "3,audio"),
            ground_truth="3,audio",
        ),
    ]