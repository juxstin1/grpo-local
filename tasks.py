"""Tasks + reward functions for the scaffolding stage.

Everything here is a stand-in. The two seams designed to be replaced:

- `MockEnv`  -> a real executor: opencode running the agent loop, with
                verifier MCP servers providing tool results.
- `Task.reward_fn` -> the MCP verifier itself (`applies+compiles+tests` for
                the C++ bucket, `no allocs/locks in processBlock` for the
                JUCE bucket, diff-vs-ground-truth for multi-hop, etc.)

Rules for good RL rewards (so future buckets hold the line):
- mechanical: no LLM judge, no free-form "quality" scoring
- sparse is fine: 1.0 correct / 0.0 wrong beats fuzzy mid-values
- the harder it is to guess, the denser the useful signal it gives
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


# --- mock repo (tiny stand-in for your JUCE plugins) ------------------------

CONFIG = {"gain": 0.7, "pan": 0.0, "delay_ms": 240}

REPO = {
    "config.json": '{"gain": 0.7, "pan": 0.0, "delay_ms": 240}',
    "audio_thread.cpp": [
        # 0
        'auto gain = cfg.read("gain");  // audio thread',
        # 1
        'auto d = cfg.read("delay_ms"); // audio thread',
    ],
    "ui_thread.cpp": [
        # 0
        'auto pan = cfg.read("pan");    // ui thread',
    ],
}


class MockEnv:
    """Deterministic stand-in for a real repo + MCP tool layer."""

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


def response_section(text: str, thinking: bool) -> str:
    """In thinking mode the prompt ends with <think>, so a generation is
    reasoning, then </think>, then the actual output. Only the text after the
    last </think> is the model's action / answer - reasoning is not action, and
    rewarding a tool call mentioned in reasoning would teach it to think about
    calling instead of calling. No </think> yet = the answer never came.

    Non-thinking mode: everything is output.
    """
    if not thinking:
        return text
    idx = text.rfind("</think>")
    if idx == -1:
        return ""
    return text[idx + len("</think>"):].lstrip("\n")


def used_tool(turns: list[Turn]) -> bool:
    return any(parse_action(t.resp_text) is not None for t in turns)


def turns_parsed(turns: list[Turn]) -> float:
    """Fraction of turns whose RESPONSE section contained a parseable action."""
    if not turns:
        return 0.0
    return sum(1 for t in turns if parse_action(t.resp_text) is not None) / len(turns)


def last_answer(turns: list[Turn]) -> Optional[str]:
    for t in reversed(turns):
        m = ANSWER_RE.search(t.resp_text)
        if m:
            return m.group(1).strip().strip(".")
    return None


def incomplete(turns: list[Turn]) -> bool:
    """Honest truncation check: the final turn was cut off by the token budget
    (no EOS sampled), so any ANSWER/tool text in it is partial. An incomplete
    trajectory never earns credit, even if a substring would match."""
    return bool(turns) and turns[-1].truncated


def normalize(s: str) -> str:
    return s.strip().lower().rstrip(".")


def reward_answer(turns: list[Turn], done: bool, ground_truth: str) -> float:
    """1.0 if the final ANSWER equals ground truth, else 0.0. Never for a
    truncated final turn."""
    if incomplete(turns):
        return 0.0
    ans = last_answer(turns)
    return 1.0 if ans is not None and normalize(ans) == normalize(ground_truth) else 0.0


def reward_tool_discipline(turns: list[Turn], done: bool, ground_truth: str) -> float:
    """Tool-correctness bucket: must use >=1 valid action AND answer right.

    Punishes guessing without a tool: an answer with zero tool calls caps at
    the format reward but never reaches the full 1.0.
    """
    if incomplete(turns):
        return 0.0
    ans = last_answer(turns)
    correct = ans is not None and normalize(ans) == normalize(ground_truth)
    if correct and used_tool(turns):
        return 1.0
    if correct and not used_tool(turns):
        return 0.5
    return 0.0


def make_tasks() -> list[Task]:
    """The seed task set. Buckets from the plan become new entries here.

    Task.prompt is the user-turn content; the SYSTEM message is applied by
    context.encode_prompt, so the two never double-wrap in chat mode.
    """

    def hop1() -> Task:
        return Task(
            name="hop1-lookup",
            prompt=(
                "What is the value of the config key delay_ms?\n"
                "Report only the number."
            ),
            reward_fn=lambda turns, done: reward_answer(turns, done, "240"),
            ground_truth="240",
        )

    def hop2_grep() -> Task:
        # Two-way answer: MUST be tool-discipline, otherwise a random guess
        # solves it half the time and the model learns to guess, not search.
        return Task(
            name="hop2-grep",
            prompt=(
                "The config key 'pan' is read by one of the project "
                "threads (audio or ui). Find where it is read and name the thread.\n"
                "Report only: audio or ui"
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, done, "ui"),
            ground_truth="ui",
        )

    def tool_form() -> Task:
        g = "0.7"
        return Task(
            name="tool-form",
            prompt=(
                "Using the tools, find the value of the config key gain.\n"
                "Report only the number."
            ),
            reward_fn=lambda turns, done: reward_tool_discipline(turns, done, g),
            ground_truth=g,
        )

    return [hop1(), hop2_grep(), tool_form()]