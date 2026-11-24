"""Chat with the served step-20 agent in an isolated, persistent file workspace.

The file tools and vLLM request format are the same as sandbox-gym/agent.py.
Nothing here grades, collects, or trains a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sandbox-gym"))

import agent  # noqa: E402
import encoding  # noqa: E402
import tools as agent_tools  # noqa: E402
from config import Config  # noqa: E402
from graders import ANSWER_RE, strip_reasoning  # noqa: E402
from gyms import system_prompt  # noqa: E402
from runner import Runner  # noqa: E402
from workspace import Workspace  # noqa: E402


def append_trace(path: Path, event: str, **fields: object) -> None:
    row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def final_answer(content: str) -> str | None:
    display = strip_reasoning(content)
    match = ANSWER_RE.search(display)
    if not match:
        return None
    reply = match.group(1).strip()
    if display[match.end():].strip():
        reply += "\n" + display[match.end():].strip()
    return reply


def chat_turn(cfg: Config, tokenizer, workspace: Workspace, enabled: list[str],
              messages: list[dict], transcript: Path, prompt: str) -> None:
    messages.append({"role": "user", "content": prompt})
    append_trace(transcript, "user", text=prompt)
    for turn in range(1, cfg.episode.max_turns + 1):
        prompt_len = len(encoding.render_chat(tokenizer, messages, cfg.episode.thinking))
        room = cfg.episode.budget - prompt_len
        if room < 256:
            print("Context budget reached. Type /reset to clear chat history; files stay in place.")
            append_trace(transcript, "context_full", prompt_tokens=prompt_len)
            return
        choice = agent.chat_once(cfg, messages, min(2048, room), timeout=180)
        if "error" in choice:
            print(f"Request error: {choice['error']}")
            append_trace(transcript, "request_error", detail=choice["error"])
            return
        content, reasoning = choice["content"], choice["reasoning"]
        action = agent_tools.parse_action(content)
        answer = final_answer(content)
        append_trace(transcript, "assistant", turn=turn, content=content,
                     action=agent_tools.tool_call_text(action) if action else None,
                     finish_reason=choice["finish_reason"])
        messages.append(agent.assistant_message(content, reasoning))
        if answer is not None:
            print(f"\nStep 20: {answer}\n", flush=True)
            return
        if action is None:
            print(f"\nStep 20: {content.strip() or '(no answer; generation ended in reasoning)'}\n",
                  flush=True)
            return
        result = agent.run_action(workspace, action, enabled)
        print(f"  {action.name}({json.dumps(action.arguments, ensure_ascii=False)})\n"
              f"  {result[:800]}{' ...' if len(result) > 800 else ''}", flush=True)
        append_trace(transcript, "tool", turn=turn, name=action.name,
                     arguments=action.arguments, result=result)
        messages.append({"role": "user", "content": agent.observation_text(
            action.name, action.arguments, result, turn, cfg.episode.max_turns)})
    print("Step 20: Reached the 10-turn cap without an ANSWER.", flush=True)
    append_trace(transcript, "cap", turns=cfg.episode.max_turns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="existing copy under runs/interactive/")
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="leg2-final-step20")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--readonly", action="store_true")
    parser.add_argument("--ask", help="one prompt, then exit; omit for interactive chat")
    args = parser.parse_args()
    workspace_path = args.workspace.resolve(strict=True)
    if not workspace_path.is_dir():
        parser.error("workspace must be a directory")
    try:
        workspace_path.relative_to((ROOT / "runs/interactive").resolve())
    except ValueError:
        parser.error("workspace must be an isolated copy under runs/interactive/")

    cfg = Config()
    cfg.serve.base, cfg.serve.served_model = args.base, args.model
    cfg.serve.temperature = args.temperature
    models = requests.get(f"{args.base.rstrip('/')}/v1/models", timeout=10)
    models.raise_for_status()
    matches = [model for model in models.json()["data"] if model["id"] == args.model]
    if len(matches) != 1 or not str(matches[0].get("root", "")).endswith("/step-20/adapter"):
        parser.error("served model is absent or does not point to the step-20 adapter")

    enabled = ["list_dir", "read_file", "search"]
    if not args.readonly:
        enabled += ["edit_file", "write_file"]
    workspace = Workspace(workspace_path, Runner("docker"))
    transcript = ROOT / "runs/interactive" / f"{workspace_path.name}.chat.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(cfg.serve.model_name, local_files_only=True)
    prompt = system_prompt(enabled, cfg.episode.max_turns).replace(
        "and you verify by running the test suite.",
        "and you verify claims by reading the files. This is a copy of a real project.")
    messages = [{"role": "system", "content": prompt}]
    print(f"Step-20 workspace: {workspace_path}\nModel: {args.model} via {args.base}\n"
          f"Tools: {', '.join(enabled)}\nTranscript: {transcript}", flush=True)
    if args.ask:
        chat_turn(cfg, tokenizer, workspace, enabled, messages, transcript, args.ask)
        return
    print("Type a question, /reset for a fresh chat, or /quit to exit. Files persist.", flush=True)
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            messages = [{"role": "system", "content": prompt}]
            print("Chat history cleared; workspace files retained.")
            continue
        if user:
            chat_turn(cfg, tokenizer, workspace, enabled, messages, transcript, user)


if __name__ == "__main__":
    main()
