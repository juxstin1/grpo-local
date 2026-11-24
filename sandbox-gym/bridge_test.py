"""Bridge parity test: re-grade existing r3 episodes (eval transcripts) three ways --
the recorded home-Docker verdict, a fresh local Runner("docker") grade, and a grade
through the bridge (Runner("remote")) -- and count exact matches.

  python bridge_test.py <results-slug> [<slug> ...] --n 20
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
from pathlib import Path

from agent import replay_edits
from config import Config
from eval import make_task
from repo_gen import find_spec, write_files
from runner import BridgeError, Runner
from schema import Action, Turn
from workspace import Workspace


def _workspace(prefix: str, tag: str, item: dict, backend: str) -> Workspace:
    # A new name preserves any workspaces left by earlier parity runs.
    root = Config().episode.data_root / "workspaces" / f"{prefix}-{tag}-{item['name']}-{uuid.uuid4().hex[:8]}"
    ws = Workspace(root, Runner(backend))
    ws.reset()
    write_files(find_spec(item["seed"], item["family"], item["name"])["files"], ws.root)
    return ws


def _checked_output(out: str) -> str:
    # Runner._docker represents a Docker daemon failure as a test FAIL string.
    # That must not become a matching execution result on both backends.
    bad = ("permission denied", "docker not found", "docker not available",
           "cannot connect to the docker daemon", "error during connect")
    if any(s in out.lower() for s in bad):
        raise RuntimeError(f"Docker infrastructure failure: {out[:500]}")
    if not re.match(r"^(tests|build) (PASS|FAIL) \(exit \d+", out):
        raise RuntimeError(f"invalid Docker execution result: {out[:500]}")
    return out


def grade(item: dict, tr: dict, backend: str, tag: str) -> float:
    ws = _workspace("bridgetest", tag, item, backend)
    turns = [Turn(content=t["content"], resp_text=t["content"],
                  action=Action(t["action"]["name"], t["action"]["arguments"]) if t["action"] else None,
                  obs=t["obs"]) for t in tr["turns"]]
    replay_edits(ws, turns)
    return float(make_task(item).reward(turns, ws))


def execute(item: dict, tr: dict, backend: str, tag: str) -> str:
    """Code actually executed: run_tests on her end-state workspace (first line) + the
    hidden-test fix grader with the ANSWER requirement off (pristine tests + nonce)."""
    from graders import files_ok_ignoring_answer
    ws = _workspace("bridgeexec", tag, item, backend)
    turns = [Turn(content=t["content"], resp_text=t["content"],
                  action=Action(t["action"]["name"], t["action"]["arguments"]) if t["action"] else None,
                  obs=t["obs"]) for t in tr["turns"]]
    replay_edits(ws, turns)
    out = _checked_output(ws.runner.run_tests(ws.root))
    # The elapsed seconds in Runner's first line are transport-dependent.
    normalized = re.sub(r", \d+s\)(?=\n)", ")", out)
    hidden = files_ok_ignoring_answer(make_task(item), turns, ws)
    return f"{normalized}|hidden={int(hidden)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"local Docker unavailable: {probe.stderr.strip()[:500]}")
    res = Config().episode.data_root / "results"
    for slug in a.slugs:
        hold = json.load(open(Path("data/holdout") / f"smoke_{slug.split('docker-', 1)[1]}.json"))
        items = {it["name"]: it for it in hold["items"]}
        trs = sorted((res / slug / "transcripts").glob("*.json"))[: a.n]
        m_rec = m_local = n = 0
        rows = []
        for p in trs:
            tr = json.loads(p.read_text(encoding="utf-8"))
            it = items[tr["name"]]
            loc = grade(it, tr, "docker", "loc")
            rem = grade(it, tr, "remote", "rem")
            rec = 1.0 if tr["passed"] else 0.0
            n += 1
            m_rec += rem == rec
            m_local += rem == loc
            flag = "" if rem == rec else f"  <-- MISMATCH stop={tr.get('stop')}"
            rows.append(f"  {p.stem:34} recorded {rec:.0f} local {loc} bridge {rem}{flag}")
        print(f"[{slug}] bridge==recorded {m_rec}/{n} | bridge==local-docker {m_local}/{n} | "
              f"passes recorded {sum('recorded 1' in r for r in rows)}")
        print("\n".join(r for r in rows if "recorded 1" in r or "ERROR" in r or "MISMATCH" in r))
        if "edit_" in slug and "editv" not in slug:
            continue                      # ring 2: no code runs anywhere
        ex_m, ex_n, outs = 0, 0, {}
        for p in trs:
            tr = json.loads(p.read_text(encoding="utf-8"))
            it = items[tr["name"]]
            loc, rem = execute(it, tr, "docker", "loc"), execute(it, tr, "remote", "rem")
            ex_n += 1
            ex_m += loc == rem
            outcome = f"{rem.splitlines()[0].split(' (')[0]}|hidden={rem.rsplit('|hidden=', 1)[1]}"
            outs[outcome] = outs.get(outcome, 0) + 1
            if loc != rem:
                print(f"  EXEC MISMATCH {p.stem}: local {loc!r} | bridge {rem!r}")
        print(f"[{slug}] EXECUTION bridge==local-docker {ex_m}/{ex_n} | outcomes {outs}")


if __name__ == "__main__":
    main()
