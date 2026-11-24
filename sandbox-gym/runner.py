"""Execution of build/run_tests. Container-only in production, host fallback for selftest.

Production (`docker`): a throwaway container with no network, only the episode
workspace mounted, CPU/memory/time limits, removed afterwards. This is also the
GRPO code grader -- model-written code never runs as the host user with
network access.

Fallback (`host`): a timed subprocess on the host. Used by selftest.py and for
offline smoke runs only; never for a real eval or training run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

IMG = "python:3.12-slim"     # pre-pulled in production (no network at run time)
NETWORK = "none"             # no network in the container
HOST_TIMEOUT = 60
HOST_TIMEOUT_MSG = "error: tests timed out (60s)"


class BridgeError(RuntimeError):
    """The remote sandbox could not run the command. Never a test result: callers
    must treat the episode as errored (excluded), never as a failed test."""


BRIDGE_TIMEOUT = HOST_TIMEOUT + 90


class Runner:
    def __init__(self, backend: str = "docker"):
        self.backend = backend

    def _run(self, cmd: list[str], workdir: Path, label: str) -> str:
        if self.backend == "docker":
            return self._docker(cmd, workdir, label)
        if self.backend == "remote":
            return self._remote(cmd, workdir, label)
        return self._host(cmd, workdir, label)

    def _remote(self, cmd: list[str], workdir: Path, label: str) -> str:
        """Ship the workdir to the home Docker sandbox (bridge.py) over the tunnel.
        URL/token from REWARD_BRIDGE / BRIDGE_TOKEN. Any transport or sandbox failure
        raises BridgeError -- never returned as a FAIL string."""
        import base64, io, tarfile
        import requests
        url = os.environ.get("REWARD_BRIDGE", "http://127.0.0.1:8765")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(workdir, arcname=".")
        try:
            r = requests.post(f"{url}/exec", timeout=BRIDGE_TIMEOUT, json={
                "cmd": cmd, "label": label, "tar": base64.b64encode(buf.getvalue()).decode()},
                headers={"X-Bridge-Token": os.environ.get("BRIDGE_TOKEN", "")})
        except requests.RequestException as e:
            raise BridgeError(f"bridge unreachable: {e}") from e
        if r.status_code != 200:
            raise BridgeError(f"bridge {r.status_code}: {r.text[:200]}")
        return r.json()["out"]

    # --- build ---------------------------------------------------------------

    def build(self, workdir: Path) -> str:
        return self._run(["python", "-m", "compileall", "-q", "."], workdir, label="build")

    # --- tests ---------------------------------------------------------------

    def run_tests(self, workdir: Path, filter: str | None = None) -> str:
        # The generated suite is self-contained (pytest-style test_* functions +
        # a __main__ runner that calls them all), so the plain runner works
        # without pytest installed -- in the throwaway container too. Swap for a
        # pytest image when the repo generator emits real pytest fixtures.
        return self._run(["python", "tests/test_core.py"], workdir, label="tests")

    # --- backends ------------------------------------------------------------

    def _host(self, cmd: list[str], workdir: Path, label: str) -> str:
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONUTF8")}
        try:
            r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                               timeout=HOST_TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            return HOST_TIMEOUT_MSG
        out = (r.stdout or "").strip() + ("\n" + r.stderr.strip() if r.stderr.strip() else "")
        tail = "\n".join(out.splitlines()[-15:]) or "(no output)"
        ok = r.returncode == 0
        return f"{label} {'PASS' if ok else 'FAIL'} (exit {r.returncode})\n{tail}"

    def _docker(self, cmd: list[str], workdir: Path, label: str) -> str:
        if not shutil.which("docker"):
            return f"error: docker not found; set --runner host for selftests only\n{self._host(cmd, workdir, label)}"
        vol = f"{workdir}:/work"
        args = ["docker", "run", "--rm", "--network", NETWORK, "-v", vol, "-w", "/work"]
        args += ["--memory", "1g", "--cpus", "2", "--pids-limit", "128"]
        args += [IMG] + cmd
        t0 = time.time()
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=HOST_TIMEOUT + 30)
        except subprocess.TimeoutExpired:
            return f"tests timed out in container"
        except FileNotFoundError:
            return f"error: docker not available"
        out = (r.stdout or "").strip() + ("\n" + r.stderr.strip() if r.stderr.strip() else "")
        tail = "\n".join(out.splitlines()[-15:]) or "(no output)"
        ok = r.returncode == 0
        return f"{label} {'PASS' if ok else 'FAIL'} (exit {r.returncode}, {time.time() - t0:.0f}s)\n{tail}"