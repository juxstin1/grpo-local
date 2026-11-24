"""Reward bridge: the home Docker sandbox, served to the pod over an SSH reverse tunnel.

The pod has no Docker, so every code execution of a code-ring episode (her run_tests /
build calls AND the grader's hidden-test run) goes through Runner("remote") -> here:
POST /exec {cmd, label, tar(base64 tar.gz of the workdir)} -> unpacked into a fresh dir,
run by the normal Runner("docker") (no network, 1 GB, 2 CPUs, 128 pids), output returned
verbatim. The grader code itself stays on the pod unchanged, so bridge rewards are the
home grader's rewards by construction.

Sandbox failures (docker missing/unavailable) -> HTTP 503, never a FAIL string.
Run:   python bridge.py            (127.0.0.1:8765; BRIDGE_TOKEN env = shared secret)
Tunnel: ssh -N -R 8765:127.0.0.1:8765 -p <port> root@<pod>
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tarfile
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from runner import Runner

TOKEN = os.environ.get("BRIDGE_TOKEN", "")
ROOT = Path(__file__).resolve().parent / "data" / "bridge"
ROOT.mkdir(parents=True, exist_ok=True)
SLOTS = threading.Semaphore(int(os.environ.get("BRIDGE_SLOTS", "12")))
ALLOWED = {("python", "tests/test_core.py"), ("python", "-m", "compileall", "-q", ".")}
STATS = {"ok": 0, "err": 0}


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    for m in tf.getmembers():
        p = (dest / m.name).resolve()
        if not str(p).startswith(str(dest.resolve())) or m.issym() or m.islnk():
            raise ValueError(f"unsafe member {m.name}")
    tf.extractall(dest)


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, **STATS})

    def do_POST(self):
        if self.path != "/exec" or self.headers.get("X-Bridge-Token", "") != TOKEN:
            return self._send(403, {"error": "forbidden"})
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            cmd = [str(c) for c in req["cmd"]]
            if tuple(cmd) not in ALLOWED:
                return self._send(400, {"error": f"cmd not allowed: {cmd}"})
            work = Path(tempfile.mkdtemp(dir=ROOT))
            try:
                with tarfile.open(fileobj=io.BytesIO(base64.b64decode(req["tar"])), mode="r:gz") as tf:
                    _safe_extract(tf, work)
                with SLOTS:
                    out = Runner("docker")._docker(cmd, work, str(req.get("label", "tests")))
            finally:
                shutil.rmtree(work, ignore_errors=True)
            if out.startswith("error: docker"):
                STATS["err"] += 1
                return self._send(503, {"error": out[:300]})
            STATS["ok"] += 1
            self._send(200, {"out": out})
        except Exception as e:  # noqa: BLE001 -- any failure is a bridge error, never a result
            STATS["err"] += 1
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    port = int(os.environ.get("BRIDGE_PORT", "8765"))
    print(f"[bridge] 127.0.0.1:{port} slots={SLOTS._value} root={ROOT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
