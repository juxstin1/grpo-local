"""The opencode-style agent loop, served-model side (eval + SFT collection).

Loop per episode: send messages -> model emits content -> if it contains an
`ANSWER:` line we are done; else if it contains a tool call we execute it in
the sandbox and append the result as the next observation; else it ended.
Repeat until answer / max_turns / budget.

Wire format is Qwen-native JSON tool calls in content (tolerant parsing), so
this path sees exactly what the local GRPO path trains on. `return_token_ids`
lets collect.py record token-exact completions and verify prompt-id fidelity
against the local template (encoding.render_chat).

The observation is shown to the model as a user turn wrapping
<tool_response>, matching encoding.encode_observation for the local path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import requests

import tools as T
from config import Config
from graders import ANSWER_RE, strip_reasoning
from gyms import system_prompt
from lsp import make_lsp
from schema import Action, Task, Turn
from workspace import JailError, Workspace


@dataclass
class Episode:
    task: Task
    turns: list[Turn] = field(default_factory=list)
    answer: str | None = None
    done: bool = False
    truncated: bool = False
    error: str | None = None
    token_ids_per_turn: list[list[int]] = field(default_factory=list)
    prompt_ids_per_turn: list[list[int]] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    n_tokens: int = 0
    ws: Workspace | None = None


def _auth(cfg: Config) -> dict:
    return {"Authorization": f"Bearer {cfg.serve.api_key}"} if cfg.serve.api_key else {}


def wait_healthy(base: str, max_wait: int = 900) -> bool:
    if "127.0.0.1" not in base and "localhost" not in base:
        return True
    t0 = time.time()
    while time.time() - t0 < max_wait:
        for path in ("/health", "/v1/models"):
            try:
                if requests.get(f"{base}{path}", timeout=5).status_code == 200:
                    return True
            except Exception:
                pass
        time.sleep(5)
    return False


def chat_once(
    cfg: Config,
    messages: list[dict],
    max_tokens: int,
    return_ids: bool = False,
    timeout: int = 600,
) -> dict:
    """One chat completion. Returns the raw vLLM choice dict, or None on error."""
    body = {
        "model": cfg.serve.served_model or cfg.serve.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": cfg.serve.temperature,
        "top_p": cfg.serve.top_p,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": cfg.episode.thinking},
    }
    if return_ids:
        body["return_token_ids"] = True
    try:
        r = requests.post(f"{cfg.serve.base}/v1/chat/completions", json=body,
                          headers=_auth(cfg), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        return {
            "content": ch["message"].get("content") or "",
            "finish_reason": ch.get("finish_reason", ""),
            "token_ids": ch.get("token_ids") or [],
            "prompt_token_ids": data.get("prompt_token_ids") or [],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "finish_reason": "",
                "content": "", "token_ids": [], "prompt_token_ids": []}


def execute_tool(ws: Workspace, action: Action) -> str:
    a = action.arguments
    n = action.name
    if n == "list_dir":
        return ws.list_dir(str(a["path"]))
    if n == "read_file":
        return ws.read_file(str(a["path"]),
                            int(a["offset"]) if a.get("offset") is not None else None,
                            int(a["limit"]) if a.get("limit") is not None else None)
    if n == "search":
        return ws.search(str(a["pattern"]),
                         str(a["path"]) if a.get("path") else None,
                         str(a["glob"]) if a.get("glob") else None)
    if n == "edit_file":
        return ws.edit_file(str(a["path"]), str(a["old_string"]), str(a["new_string"]))
    if n == "write_file":
        return ws.write_file(str(a["path"]), str(a["content"]))
    if n in ("definition", "references", "diagnostics"):
        lsp = make_lsp()
        if n == "definition":
            return lsp.definition(ws.root, str(a["symbol"]))
        if n == "references":
            return lsp.references(ws.root, str(a["symbol"]))
        return lsp.diagnostics(ws.root, str(a["path"]))
    if n == "build":
        return ws.build()
    if n == "run_tests":
        return ws.run_tests(str(a["filter"]) if a.get("filter") else None)
    return f"unknown tool: {n}"


def observation_text(tool: str, args: dict, result: str) -> str:
    argstr = ", ".join(f"{k}={v}" for k, v in args.items()) or ""
    return f"<tool_response>\nResult of {tool}({argstr}):\n{result}\n</tool_response>"


def run_episode(
    cfg: Config,
    task: Task,
    ws: Workspace,
    prompt_len_fn: Callable[[list[dict]], int] | None = None,
    return_ids: bool = False,
) -> Episode:
    """Serve one episode against the sandbox workspace. Returns an Episode."""
    ep = Episode(task=task, ws=ws)
    messages = [{"role": "system", "content": system_prompt(task.tools)},
                {"role": "user", "content": task.prompt}]
    lsp = make_lsp()
    if not wait_healthy(cfg.serve.base):
        ep.error = "server not healthy"
        return ep

    for _ in range(cfg.episode.max_turns):
        max_tokens = cfg.episode.budget
        if prompt_len_fn is not None:
            prompt_len = prompt_len_fn(messages)
            max_tokens = max(64, cfg.episode.budget - prompt_len)
        ch = chat_once(cfg, messages, max_tokens, return_ids=return_ids)
        if "error" in ch:
            ep.error = ch["error"]
            break
        content = ch["content"]
        truncated = ch["finish_reason"] == "length"
        if return_ids:
            ep.prompt_ids_per_turn.append(ch["prompt_token_ids"])
            ep.prompt_ids = ch["prompt_token_ids"]
            ep.token_ids_per_turn.append(ch["token_ids"])
            ep.n_tokens += len(ch["prompt_token_ids"]) + len(ch["token_ids"])

        action = T.parse_action(content)
        ans_m = ANSWER_RE.search(strip_reasoning(content))
        if ans_m:
            ep.answer = ans_m.group(1).strip().strip(".")
            ep.done = True
            ep.turns.append(Turn(content=content, resp_text=content,
                                 truncated=truncated, action=action))
            break
        if action is None:
            ep.turns.append(Turn(content=content, resp_text=content,
                                 truncated=truncated, action=None))
            ep.done = False
            break

        # A malformed call is feedback for her, never a harness crash.
        try:
            result = execute_tool(ws, action)
        except KeyError as e:
            required = [n for n, req, _ in T.TOOLS.get(action.name, {}).get("args", []) if req]
            result = (f"error: {action.name} is missing required argument {e}. "
                      f"Required: {', '.join(required) or 'none'}")
        except (TypeError, ValueError) as e:
            result = f"error: bad arguments for {action.name}: {e}"
        except JailError as e:
            result = f"error: {e}"
        ep.turns.append(Turn(content=content, resp_text=content, action=action,
                             obs=result, truncated=truncated))
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": observation_text(action.name, action.arguments, result)})

    else:  # max_turns exhausted without terminating
        ep.done = False
    if ep.turns and ep.turns[-1].truncated:
        ep.truncated = True
    return ep


def answer_from_turns(turns: list[Turn]) -> str | None:
    for t in reversed(turns):
        m = ANSWER_RE.search(strip_reasoning(t.resp_text or t.content))
        if m:
            return m.group(1).strip().strip(".")
    return None