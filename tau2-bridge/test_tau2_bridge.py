"""Stub tests for run_tau2 (no vLLM, no user-sim API): a scripted agent drives
tau2's mock domain through the real AgentGymEnv with a scripted user.

  .venv\\Scripts\\python.exe -X utf8 test_tau2_bridge.py
"""

from __future__ import annotations

import os
import sys

if not sys.flags.utf8_mode:
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

import json  # noqa: E402

import run_tau2 as R  # noqa: E402
from tau2.data_model.message import UserMessage  # noqa: E402
from tau2.gym.gym_agent import AgentGymEnv  # noqa: E402
from tau2.user.user_simulator import UserSimulator  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    print(("PASS " if cond else "FAIL ") + msg)
    FAILS += not cond


class ScriptedUser(UserSimulator):
    def __init__(self, lines):
        super().__init__(instructions="scripted", llm="stub")
        self.lines = list(lines)

    def _generate_next_message(self, message, state):
        return UserMessage(role="user", content=self.lines.pop(0) if self.lines else "###STOP###")


class StubEnv(AgentGymEnv):
    def __init__(self, *a, user_lines=(), **k):
        self._lines = user_lines
        super().__init__(*a, **k)

    def _get_user(self):
        return ScriptedUser(self._lines)


def env_factory(lines):
    return lambda d, t: StubEnv(domain=d, task_id=t, max_steps=40, user_lines=lines)


def scripted_chat(replies, seen):
    replies = list(replies)

    def chat(cfg, messages, max_tokens):
        seen.append([dict(m) for m in messages])
        if not replies:
            return {"content": "", "reasoning": "", "finish_reason": "stop"}
        r = replies.pop(0)
        if isinstance(r, dict):
            return r
        return {"content": r, "reasoning": "thinking...", "finish_reason": "stop"}
    return chat


cfg = R.Config()
plen = lambda msgs: sum(len(m["content"]) // 4 + 4 for m in msgs)  # noqa: E731
USER = ["Please create a task called 'Important Meeting' for user_1.", "###STOP###"]
CALL = '<tool_call>\n{"name": "create_task", "arguments": {"user_id": "user_1", "title": "Important Meeting"}}\n</tool_call>'

# 1. wire format round trip: our parser -> canonical JSON -> tau2's parser
from tau2.utils.tools import parse_action_string  # noqa: E402
act = R.T.parse_action(CALL)
msg = parse_action_string(R.T.tool_call_text(act))
check(act.name == "create_task" and msg.tool_calls and msg.tool_calls[0].name == "create_task"
      and msg.tool_calls[0].arguments == {"user_id": "user_1", "title": "Important Meeting"},
      "parse_action -> tool_call_text -> tau2 parse_action_string keeps name+args")
txt = parse_action_string("Sure, I can help with that. What is your user id?")
check(txt.tool_calls is None and txt.content.startswith("Sure"), "plain text stays a user-facing message")
check(R.T.parse_action('{"name": "done", "arguments": {}}').name == "done", "done() parses")

# 2. good episode: tool call, confirm, user stops -> official reward 1.0
seen: list = []
rec = R.run_trial("mock", "create_task_1", 0, cfg, plen,
                  scripted_chat([CALL, "Your task 'Important Meeting' has been created."], seen),
                  env_factory(USER), budget=8192, transcripts=True)
check(rec["reward"] == 1.0 and not rec["missing"] and rec["stop"] == "user_stop",
      f"scripted success -> reward 1.0 user_stop (got {rec['reward']} {rec['stop']})")
roles = [m["role"] for m in rec["messages"]]
check(roles[:3] == ["system", "assistant", "user"], f"opening greeting + user message ingested ({roles[:3]})")
tool_obs = [m for m in rec["messages"] if m["role"] == "user" and m["content"].startswith("<tool_response>")]
check(len(tool_obs) == 1 and "Result of create_task(" in tool_obs[0]["content"],
      "tool result shown as sandbox-gym <tool_response> observation")
check(any(m.get("reasoning_content") == "thinking..." for m in rec["messages"]),
      "reasoning sent back in history (served-loop rule)")
sysm = rec["messages"][0]["content"]
check("<policy>" in sysm and "create_task(user_id, title, description?)" in sysm
      and '{"name": "<tool>", "arguments"' in sysm and "done()" in sysm,
      "system prompt: policy + sandbox-style tool lines + JSON wire format + done()")

# 3. agent never calls the tool -> DB check fails -> reward 0
rec = R.run_trial("mock", "create_task_1", 1, cfg, plen,
                  scripted_chat(["Done! Anything else?"], []), env_factory(USER), budget=8192)
check(rec["reward"] == 0.0 and not rec["missing"], f"no tool call -> reward 0, not missing ({rec['stop']})")

# 4. agent ends via done() after doing the work -> agent_stop, graded
rec = R.run_trial("mock", "create_task_1", 2, cfg, plen,
                  scripted_chat([CALL, "Created it.", '{"name": "done", "arguments": {}}'], []),
                  env_factory([USER[0], "Thanks!"]), budget=8192)
check(rec["stop"] == "agent_stop" and rec["reward"] == 1.0, f"done() -> agent_stop reward 1 ({rec['stop']} {rec['reward']})")

# 5. budget: prompt already over budget -> cap, reward 0, not missing
rec = R.run_trial("mock", "create_task_1", 3, cfg, plen, scripted_chat([CALL], []),
                  env_factory(USER), budget=200)
check(rec["stop"] == "cap" and rec["reward"] == 0.0 and not rec["missing"], f"over budget -> cap fail ({rec['stop']})")

# 6. truncated generation -> fail
rec = R.run_trial("mock", "create_task_1", 4, cfg, plen,
                  scripted_chat([{"content": "", "reasoning": "...", "finish_reason": "length"}], []),
                  env_factory(USER), budget=8192)
check(rec["stop"] == "truncated" and rec["reward"] == 0.0 and not rec["missing"], "finish=length -> truncated fail")

# 7. empty reply (unclosed think) -> fail
rec = R.run_trial("mock", "create_task_1", 5, cfg, plen,
                  scripted_chat([{"content": "  ", "reasoning": "x", "finish_reason": "stop"}], []),
                  env_factory(USER), budget=8192)
check(rec["stop"] == "empty" and not rec["missing"], "empty reply -> empty fail")

# 8. vLLM request error -> missing, never a fail
R.time.sleep = lambda s: None
rec = R.run_trial("mock", "create_task_1", 6, cfg, plen,
                  lambda c, m, n: {"error": "ConnectionError: refused"}, env_factory(USER), budget=8192)
check(rec["missing"] and rec["stop"] == "request-error", "request error -> missing")

# 8b. user simulator API failure -> tau2 user_error -> missing, never a fail
class BrokenUser(ScriptedUser):
    def _generate_next_message(self, message, state):
        raise RuntimeError("sim API down")


class BrokenEnv(StubEnv):
    def _get_user(self):
        return BrokenUser([])


rec = R.run_trial("mock", "create_task_1", 7, cfg, plen, scripted_chat([CALL, "ok"], []),
                  lambda d, t: BrokenEnv(domain=d, task_id=t, max_steps=40), budget=8192)
check(rec["missing"] and rec["reward"] == 0.0, f"sim failure -> missing ({rec['stop']} {rec.get('termination_reason')})")

# 9. pass^k math (tau2 definition) + missing excluded
rows = [{"task_id": "a", "reward": 1.0, "missing": False}, {"task_id": "a", "reward": 0.0, "missing": False},
        {"task_id": "b", "reward": 1.0, "missing": False}, {"task_id": "b", "reward": 1.0, "missing": False},
        {"task_id": "b", "reward": 0.0, "missing": True}]
pk = R.pass_hat_k(rows, 3)
check(pk["pass^1"] == 0.75 and pk["pass^2"] == 0.5 and pk["pass^3"] is None, f"pass^k {pk}")

# 10. compare refuses different sim settings
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
base = {"tag": "x", "sim_model": "deepseek/deepseek-flash", "sim_temperature": 0.0, "judge_llm": None,
        "tau2": "1.0.1@x", "tasks": {"airline": ["1"]}, "budget": 8192, "trials": 1}
d = Path(tempfile.mkdtemp())
(d / "a.json").write_text(json.dumps({"settings": base, "by_domain": {}}))
(d / "b.json").write_text(json.dumps({"settings": {**base, "tag": "y", "sim_temperature": 0.5}, "by_domain": {}}))
try:
    R.compare([str(d / "a.json"), str(d / "b.json")])
    check(False, "compare refuses differing sim settings")
except SystemExit as e:
    check("REFUSED" in str(e), "compare refuses differing sim settings")

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
