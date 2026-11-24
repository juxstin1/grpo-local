"""Deterministic stub model + char-level fake tokenizer for compaction tests.

No server, no torch model: chat_once / complete_ids are replaced by a script.
Ids are character codes of a toy chat render, so prompt-prefix fidelity can be
checked exactly (a served prompt must start with the previous prompt + her ids).
"""

from __future__ import annotations

import json
from pathlib import Path

import agent
from runner import Runner
from schema import Task
from workspace import Workspace

END = "<|end|>"
THINK_CLOSE = "</think>"


def render(messages: list[dict], thinking: bool = True) -> list[int]:
    """Toy template: <|role|>content<|end|> per message + generation prompt.
    Thinking off appends an empty think block, like Qwen's template."""
    s = "".join(f"<|{m['role']}|>{m['content']}{END}" for m in messages) + "<|assistant|>"
    if not thinking:
        s += "<think></think>"
    return [ord(c) for c in s]


def encode(text: str) -> list[int]:
    return [ord(c) for c in text]


class FakeTok:
    """Just enough tokenizer for compact.py: encode text, find </think>."""
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": encode(text)}

    def encode(self, text, add_special_tokens=False):
        return encode(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=True):
        return render(messages, enable_thinking)

    def convert_tokens_to_ids(self, token):
        return 0xE000 + sum(map(ord, token)) % 4096     # one private-use id per special


class Adaptive:
    """Model stub that answers note calls with `note()` and work calls from a queue
    of actions. A note call = single user message asking for a HANDOFF."""

    def __init__(self, work: list, note=lambda i: NOTE, note_reasoning: str = ""):
        self.work, self.note, self.note_reasoning = list(work), note, note_reasoning
        self.base = Script([])
        self.notes = 0

    @property
    def calls(self):
        return self.base.calls

    def chat_once(self, cfg, messages, max_tokens, return_ids=False, timeout=600,
                  thinking=None):
        if len(messages) == 1 and "HANDOFF" in messages[0]["content"]:
            self.notes += 1
            self.base.replies = [{"content": self.note(self.notes),
                                  "reasoning": self.note_reasoning}]
        else:
            self.base.replies = [self.work.pop(0) if self.work else "ANSWER: none"]
        return self.base.chat_once(cfg, messages, max_tokens, return_ids, timeout, thinking)

    def complete_ids(self, cfg, prompt_ids, max_tokens, timeout=600):
        self.base.replies = [self.note(self.notes)]
        return self.base.complete_ids(cfg, prompt_ids, max_tokens, timeout)


NOTE = ("Goal: collect every code\nDone: read the first files\nCurrent state: codes so far "
        "noted\nHypothesis: none\nNext step: read the remaining files\nDon't repeat: re-reading")


class Script:
    """Replies in order. A reply is a str (work turn or note) or a dict
    {"content": str, "reasoning": str}. Records every call for assertions."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat_once(self, cfg, messages, max_tokens, return_ids=False, timeout=600,
                  thinking=None):
        r = self.replies.pop(0) if self.replies else "ANSWER: out of script"
        r = r if isinstance(r, dict) else {"content": r, "reasoning": ""}
        think = cfg.episode.thinking if thinking is None else thinking
        gen = encode((r["reasoning"] + THINK_CLOSE if think and r["reasoning"] else "")
                     + r["content"] + END)
        finish = "stop"
        if len(gen) > max_tokens:
            gen, finish = gen[:max_tokens], "length"
        text = "".join(chr(i) for i in gen)
        content = text.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else (
            "" if think and r["reasoning"] and finish == "length" else text)
        content = content.replace(END, "")
        self.calls.append({"messages": json.loads(json.dumps(messages)), "max_tokens": max_tokens,
                           "thinking": think, "finish": finish})
        return {"content": content, "reasoning": r["reasoning"], "finish_reason": finish,
                "token_ids": gen if True else [],
                "prompt_token_ids": render(messages, think)}

    def complete_ids(self, cfg, prompt_ids, max_tokens, timeout=600):
        r = self.replies.pop(0) if self.replies else ""
        r = r if isinstance(r, str) else r["content"]
        gen = encode(r + END)
        finish = "stop"
        if len(gen) > max_tokens:
            gen, finish = gen[:max_tokens], "length"
        self.calls.append({"prompt_ids_len": len(prompt_ids), "max_tokens": max_tokens,
                           "finish": finish, "complete": True})
        return {"text": "".join(chr(i) for i in gen).replace(END, ""), "finish": finish,
                "ids": gen}


def install(script: Script, monkey) -> None:
    """Patch agent's network calls with the script (monkey = setattr-like helper)."""
    monkey(agent, "chat_once", script.chat_once)
    monkey(agent, "wait_healthy", lambda base, max_wait=900: True)
    try:
        import compact
        monkey(compact, "complete_ids", script.complete_ids)
    except ImportError:
        pass


def make_ws(root: Path, files: dict[str, str]) -> Workspace:
    ws = Workspace(root, Runner("host"))
    ws.reset()
    for rel, text in files.items():
        p = ws.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return ws


def make_task(prompt: str, answer: str, tools=("list_dir", "read_file", "search")) -> Task:
    t = Task(name="stub-task", family="find", prompt=prompt, tools=list(tools),
             ground_truth=answer, reward=lambda turns, ws: 0.0)
    t.reward = lambda turns, ws: float(
        agent.answer_from_turns(turns) == answer)
    t.meta = {"seed": 0}
    return t


def call(name: str, **args) -> str:
    return json.dumps({"name": name, "arguments": args})
