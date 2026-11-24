"""Bridge PASS-path parity: grade generator-built fixes (not transcripts) two ways --
local Runner("docker") and a PRIVATE bridge instance (Runner("remote") on a spare port)
-- and require exact agreement.

Per fix_h seed, two synthetic episodes:
  gold    edit_file every source file the bug changed back to the generator's clean
          version (repo_gen_hard: _with_intent(generate_hard(seed))), then ANSWER.
          Expected reward 1.
  broken  leave the bug, edit_file tests/test_core.py into a suite that trivially
          "passes", then ANSWER. Workspace run_tests says PASS; the pristine-test
          grader must say FAIL. Expected reward 0.

The private bridge is spawned here (bridge.py, BRIDGE_PORT=<port>, random token) and
stopped on exit. The live :8765 bridge and its tunnel are never touched.

  python bridge_pass_test.py [--seeds 9001-9005] [--port 8767]
Exit 0 iff bridge==local on every episode AND each episode got its expected reward.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def episodes(seed: int):
    """(label, spec, turns, expected_reward) for gold and broken."""
    import repo_gen_hard as rgh
    from repo_gen import find_spec
    from schema import Action, Turn
    spec = find_spec(seed, "fix_h", f"fixh-{seed}")
    clean = rgh._with_intent(rgh.generate_hard(seed)).files
    changed = [k for k, v in spec["files"].items()
               if not k.startswith("tests/") and clean.get(k) != v]
    assert changed, f"seed {seed}: bugged == clean?"
    ans = Turn(content="ANSWER: fixed", resp_text="ANSWER: fixed")
    # edit_file with the whole file as old_string (write_file only creates new files)
    gold = [Turn(content="", action=Action("edit_file", {
                "path": k, "old_string": spec["files"][k], "new_string": clean[k]}))
            for k in changed] + [ans]
    fake = ("def test_ok():\n    assert True\n\n"
            "if __name__ == '__main__':\n    test_ok()\n    print('ALL TESTS PASSED')\n")
    broken = [Turn(content="", action=Action("edit_file", {
                   "path": "tests/test_core.py", "old_string": spec["files"]["tests/test_core.py"],
                   "new_string": fake})), ans]
    return spec, [("gold", gold, 1.0), ("broken", broken, 0.0)]


def run_one(spec: dict, turns, backend: str, tag: str) -> tuple[float, str]:
    from agent import execute_tool
    from bridge_test import _checked_output
    from config import Config
    from eval import make_task
    from repo_gen import write_files
    from runner import Runner
    from workspace import Workspace
    root = (Config().episode.data_root / "workspaces"
            / f"bridgepass-{tag}-{spec['name']}-{uuid.uuid4().hex[:8]}")
    ws = Workspace(root, Runner(backend))
    ws.reset()
    write_files(spec["files"], ws.root)
    for t in turns:                       # same edits replay_edits applies, but checked
        if t.action:
            res = execute_tool(ws, t.action)
            assert res.startswith("edited "), f"{spec['name']} {t.action.arguments['path']}: {res}"
    reward = float(make_task(spec).reward(turns, ws))
    out = _checked_output(ws.runner.run_tests(ws.root))
    first = re.sub(r", \d+s\)$", ")", out.split("\n")[0])
    return reward, first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9001-9005")
    ap.add_argument("--port", type=int, default=8767)
    a = ap.parse_args()
    if a.port == 8765:
        sys.exit("refusing :8765 (live bridge)")
    if not _free(a.port):
        sys.exit(f"port {a.port} busy")
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(f"local Docker unavailable: {probe.stderr.strip()[:300]}")

    token = uuid.uuid4().hex
    env = {**os.environ, "BRIDGE_PORT": str(a.port), "BRIDGE_TOKEN": token}
    br = subprocess.Popen([sys.executable, str(HERE / "bridge.py")], env=env, cwd=HERE,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if not _free(a.port):
                break
            time.sleep(0.2)
        else:
            sys.exit("private bridge did not come up")
        os.environ["REWARD_BRIDGE"] = f"http://127.0.0.1:{a.port}"
        os.environ["BRIDGE_TOKEN"] = token

        n = match = as_expected = passes = 0
        for seed in _seeds(a.seeds):
            spec, eps = episodes(seed)
            for label, turns, want in eps:
                rl, ol = run_one(spec, turns, "docker", "loc")
                rr, orr = run_one(spec, turns, "remote", "rem")
                same = rl == rr and ol == orr
                n += 1
                match += same
                as_expected += rr == want
                passes += rr == 1.0
                flag = "" if same and rr == want else "  <-- MISMATCH"
                print(f"  fixh-{seed} {label:6} local {rl:.0f} [{ol}] bridge {rr:.0f} [{orr}]"
                      f" want {want:.0f}{flag}")
        print(f"EXECUTION bridge==local-docker {match}/{n} | expected {as_expected}/{n} | "
              f"PASS {passes} FAIL {n - passes} | port {a.port}")
        sys.exit(0 if match == n == as_expected else 1)
    finally:
        br.terminate()
        br.wait(timeout=10)


if __name__ == "__main__":
    main()
