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

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

import compact as C
import encoding
import tools as T
import web_tools
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
    cap_hit: bool = False                    # max_turns exhausted without an ANSWER line
    stop: str = ""                           # answer | cap | no-action | empty (unclosed <think>) | error
    error: str | None = None
    token_ids_per_turn: list[list[int]] = field(default_factory=list)
    prompt_ids_per_turn: list[list[int]] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    n_tokens: int = 0
    ws: Workspace | None = None
    # compaction (config Episode.compact); empty / None when it is off
    initial_messages: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)   # {start_turn, messages} per reset
    resets: list[dict] = field(default_factory=list)    # compact.Compactor.reset records
    compact_fail: str | None = None                     # thrash | context


def _hosted(base: str) -> bool:
    return "127.0.0.1" not in base and "localhost" not in base


def _auth(cfg: Config) -> dict:
    """Bearer key: --api-key, else (hosted only) DEEPINFRA_API_KEY from the env or
    grpo-local/evals/.env -- read here, never printed or logged."""
    key = cfg.serve.api_key
    if not key and _hosted(cfg.serve.base):
        import os
        from pathlib import Path
        key = os.environ.get("DEEPINFRA_API_KEY", "")
        env = Path(__file__).resolve().parent.parent / "evals" / ".env"
        if not key and env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPINFRA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


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
    thinking: bool | None = None,
) -> dict:
    """One chat completion. Returns the raw vLLM choice dict, or None on error.
    thinking=None follows cfg.episode.thinking (compaction note calls override it)."""
    body = {
        "model": cfg.serve.served_model or cfg.serve.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": cfg.serve.temperature,
        "top_p": cfg.serve.top_p,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": cfg.episode.thinking
                                 if thinking is None else thinking},
    }
    if return_ids:
        body["return_token_ids"] = True
    base = cfg.serve.base.rstrip("/")
    url = f"{base}/v1/chat/completions"
    if _hosted(base):
        # Reference models on a hosted API (e.g. DeepInfra's OpenAI route
        # <base>/v1/openai/chat/completions): no vLLM-only fields, and no
        # reasoning_content in history (some providers reject it; the model's own
        # template decides what prior thinking it sees).
        body.pop("chat_template_kwargs", None)
        body.pop("return_token_ids", None)
        body["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"}
                            for m in messages]
        # Native tool channel: models like DeepSeek emit their own tool-call tokens,
        # which the provider strips to EMPTY content when no tools are declared.
        # Each native call is written back as our {"name","arguments"} text, so
        # history and grading stay identical to the served path.
        body["tools"] = _openai_tools(messages)
        if base.endswith("/openai"):
            url = f"{base}/chat/completions"
    try:
        r = requests.post(url, json=body,
                          headers=_auth(cfg), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        msg = ch["message"]
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        if _hosted(base) and "DSML" in content:
            content = _dsml_to_json(content)
        if calls and not T.parse_action(content):
            fn = calls[0].get("function", {})
            args = _unwrap_args(fn.get("arguments") or "{}")
            call = json.dumps({"name": fn.get("name", ""), "arguments": args})
            content = f"{content.rstrip()}\n\n{call}" if content.strip() else call
        return {
            "content": content,
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish_reason": ch.get("finish_reason", ""),
            "token_ids": ch.get("token_ids") or [],
            "prompt_token_ids": data.get("prompt_token_ids") or [],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "finish_reason": "", "reasoning": "",
                "content": "", "token_ids": [], "prompt_token_ids": []}


_DSML_NAME = re.compile(r'invoke name="(\w+)"|"name"\s*:\s*"(\w+)"')
_DSML_PARAM = re.compile(r'parameter name="(\w+)"[^>]*>(.*?)</', re.S)


def _unwrap_args(raw) -> dict:
    """Hosted args arrive as JSON text, sometimes double-wrapped as
    {"arguments": "<json text>"}; peel until it's a plain dict."""
    for _ in range(3):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return {}
        if isinstance(raw, dict) and set(raw) == {"arguments"}:
            raw = raw["arguments"]
            continue
        break
    return raw if isinstance(raw, dict) else {}


def _dsml_to_json(content: str) -> str:
    """Hosted DeepSeek sometimes leaks its native DSML tool markup (or a JSON/DSML
    mix) into content. Rewrite the first call as our {"name","arguments"} JSON.
    Reference-model path only; the served 4B is never normalized."""
    start = content.find("<｜DSML")
    head = content[:start] if start >= 0 else ""
    m = _DSML_NAME.search(content)
    if not m:
        return content
    name = m.group(1) or m.group(2)
    args: dict = {}
    for key, val in _DSML_PARAM.findall(content):
        val = val.strip()
        if key == "arguments":
            u = _unwrap_args(val)
            if u:
                args.update(u)
                continue
        args[key] = val
    # a JSON prefix like {"name": "x"> belongs to the call, not the prose
    j = head.find('{"name"')
    prose = (head[:j] if j >= 0 else head).rstrip()
    call = json.dumps({"name": name, "arguments": args})
    return f"{prose}\n\n{call}" if prose else call


def _openai_tools(messages: list[dict]) -> list[dict]:
    """OpenAI tool schemas for the tools named in this episode's system prompt."""
    sys_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    out = []
    for name, spec in T.TOOLS.items():
        if f"  {name}(" not in sys_text:          # enabled tools only (describe_tools lines)
            continue
        props = {a: {"type": "string", "description": d} for a, _, d in spec["args"]}
        params = spec.get("schema") or {"type": "object", "properties": props,
                                        "required": [a for a, req, _ in spec["args"] if req]}
        out.append({"type": "function", "function": {
            "name": name, "description": spec.get("help", ""), "parameters": params}})
    return out


def run_action(ws: Workspace, action: Action, enabled: list[str]) -> str:
    """THE way a tool call becomes an observation -- served loop AND local GRPO
    rollout use this, so a malformed call is identical feedback in both (never a
    harness crash, never a train/eval divergence)."""
    try:
        return T.unavailable(action, enabled) or execute_tool(ws, action)
    except KeyError as e:
        required = [n for n, req, _ in T.TOOLS.get(action.name, {}).get("args", []) if req]
        return (f"error: {action.name} is missing required argument {e}. "
                f"Required: {', '.join(required) or 'none'}")
    except (TypeError, ValueError) as e:
        return f"error: bad arguments for {action.name}: {e}"
    except JailError as e:
        return f"error: {e}"


def replay_edits(ws: Workspace, turns: list[Turn]) -> None:
    """Rebuild a workspace's state after `turns`: re-apply her file edits in
    order (reads/searches/tests don't change files). ws must hold the seed files."""
    for t in turns:
        if t.action and t.action.name in ("edit_file", "write_file"):
            try:
                execute_tool(ws, t.action)
            except Exception:  # noqa: BLE001  -- a failed edit changed nothing then too
                pass


def assistant_message(content: str, reasoning: str) -> dict:
    return {"role": "assistant", "content": content, "reasoning_content": reasoning}


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
    if n in ("web_search", "fetch_url"):
        # Frozen-corpus web tools (web_tools.py). The family's make_workspace attaches
        # ws.corpus; ws.client_model drives Bionic's hint choice (unset = h2, as for a 4B).
        corpus = getattr(ws, "corpus", None)
        if n == "web_search":
            return ws._cap(web_tools.web_search(corpus, a, getattr(ws, "client_model", "")))
        return ws._cap(web_tools.fetch_url(corpus, a))
    if n == "make_pdf":
        import doc_tools
        return ws._cap(doc_tools.make_pdf(ws, a))
    return f"unknown tool: {n}"


def counter_turns(cfg: Config) -> int | None:
    """N for the turn counter, or None when it is off (the one switch)."""
    return cfg.episode.max_turns if cfg.episode.turn_counter else None


def observation_text(tool: str, args: dict, result: str,
                     turn: int | None = None, max_turns: int | None = None) -> str:
    body = encoding.observation_body(tool, args, result, turn, max_turns)
    return f"<tool_response>\n{body}\n</tool_response>"


def run_episode(
    cfg: Config,
    task: Task,
    ws: Workspace,
    prompt_len_fn: Callable[[list[dict]], int] | None = None,
    return_ids: bool = False,
    prefix: "Episode | None" = None,
    prefix_len: int = 0,
    tokenizer=None,
) -> Episode:
    """Serve one episode against the sandbox workspace. Returns an Episode.

    Branching: with `prefix` (an earlier Episode of the same task) and
    `prefix_len` t, her first t turns are replayed verbatim (history + ids) and
    sampling continues from turn t+1. max_turns still bounds the WHOLE episode.
    The caller rebuilds `ws` to its state after turn t (replay_edits).

    Compaction (cfg.episode.compact, see compact.py): the harness asks for a note
    when the prompt reaches 75% of the budget, and a `checkpoint` tool lets her
    reset herself; both rebuild the context. Needs prompt_len_fn + tokenizer."""
    ep = Episode(task=task, ws=ws)
    N = counter_turns(cfg)
    cp = None
    if cfg.episode.compact:
        if prefix is not None:
            raise ValueError("compact does not support branching prefixes")
        cp = C.Compactor(cfg, task, ws, prompt_len_fn, tokenizer, N)
    enabled = C.enabled_tools(task) if cp else task.tools
    messages = [{"role": "system", "content": cp.system if cp else
                 system_prompt(task.tools, N, cfg.episode.hint)},
                {"role": "user", "content": task.prompt}]
    ep.initial_messages = [dict(m) for m in messages]
    if prefix is not None and prefix_len:
        for i, t in enumerate(prefix.turns[:prefix_len]):
            ep.turns.append(t)
            messages.append(assistant_message(t.content, t.reasoning))
            messages.append({"role": "user", "content": observation_text(
                t.action.name, t.action.arguments, t.obs, i + 1, N)})
            if return_ids and i < len(prefix.token_ids_per_turn):
                ep.prompt_ids_per_turn.append(prefix.prompt_ids_per_turn[i])
                ep.token_ids_per_turn.append(prefix.token_ids_per_turn[i])
                ep.n_tokens += len(prefix.prompt_ids_per_turn[i]) + len(prefix.token_ids_per_turn[i])
    lsp = make_lsp()
    if not wait_healthy(cfg.serve.base):
        ep.error = "server not healthy"
        return ep

    for _ in range(cfg.episode.max_turns - len(ep.turns)):
        max_tokens = cfg.episode.budget
        if prompt_len_fn is not None:
            prompt_len = prompt_len_fn(messages)
            if cp is not None and cp.should_reset(ep, prompt_len):
                rebuilt = cp.reset(ep, "harness")
                if rebuilt is None:
                    break
                messages = rebuilt
                prompt_len = prompt_len_fn(messages)
            max_tokens = max(64, cfg.episode.budget - prompt_len)
        ch = chat_once(cfg, messages, max_tokens, return_ids=return_ids)
        if "error" in ch:
            ep.error = ch["error"]
            break
        content = ch["content"]
        reasoning = ch.get("reasoning", "")
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
            ep.stop = "answer"
            ep.turns.append(Turn(content=content, resp_text=content, reasoning=reasoning,
                                 truncated=truncated, action=action))
            break
        if action is None:
            ep.turns.append(Turn(content=content, resp_text=content, reasoning=reasoning,
                                 truncated=truncated, action=None))
            ep.done = False
            # empty = she never closed </think>: the reasoning parser filed the whole
            # reply (often a tool call) as reasoning. Same rule as encoding.response_section.
            ep.stop = "empty" if not content.strip() else "no-action"
            break

        if cp is not None and action.name == "checkpoint":
            note = action.arguments.get("note")
            missing = (C.notes.missing_sections(note) if isinstance(note, str)
                       else list(C.notes.SECTIONS))
            if not missing:
                ep.turns.append(Turn(content=content, resp_text=content, reasoning=reasoning,
                                     action=action, obs="", truncated=truncated))
                plan = action.arguments.get("plan")
                rebuilt = cp.reset(ep, "self", note, plan if isinstance(plan, str) else None)
                if rebuilt is None:
                    break
                messages = rebuilt
                continue
            result = ("error: checkpoint note must use the sections "
                      f"{', '.join(C.notes.SECTIONS)}; missing: {', '.join(missing)}. "
                      "No reset happened.")
        else:
            result = run_action(ws, action, enabled)
            if cp is not None:
                cp.tools_since_reset += 1
        ep.turns.append(Turn(content=content, resp_text=content, reasoning=reasoning, action=action,
                             obs=result, truncated=truncated))
        # Send her thinking back: the template renders reasoning_content for every
        # turn after the last real user message. Dropping it renders empty
        # <think></think> history (-> she starts writing calls inside an open think)
        # and diverges from local GRPO / SFT, which keep the raw generated ids.
        messages.append(assistant_message(content, reasoning))
        messages.append({"role": "user", "content": observation_text(
            action.name, action.arguments, result, len(ep.turns), N)})

    else:  # max_turns exhausted without terminating
        ep.done = False
        ep.cap_hit = True
        ep.stop = "cap"
    if ep.turns and ep.turns[-1].truncated:
        ep.truncated = True
    return ep


def answer_from_turns(turns: list[Turn]) -> str | None:
    for t in reversed(turns):
        m = ANSWER_RE.search(strip_reasoning(t.resp_text or t.content))
        if m:
            return m.group(1).strip().strip(".")
    return None