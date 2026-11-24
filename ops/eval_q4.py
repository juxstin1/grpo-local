"""Mission 2 runner: eval the Q4_K_M arms of one size on Bionic's llama-server, arms interleaved per set.

  python ops/eval_q4.py --size 0.8b --np 2 --dry-run          # print every command, touch nothing
  python ops/eval_q4.py --size 0.8b --np 2 --suite core       # real run (GPU) - only with the user's go
  python ops/eval_q4.py --size 4b --np 2 --sets smoke_repoq_t1_9301-9305 --limit 1   # gate 2 smoke

Per set: for each arm, start llama-server on that arm's GGUF (sha256-checked), run eval.py on the set,
stop the server. A stop at any point therefore leaves paired results. After every set a row goes into
runs/pilots/q4-eval-mission2.md. Settings match ops/eval_sft_1008.py (the 1008 training data): Bionic
dialect, 16k budget + compaction + facts notes, per-family turn caps, thinking on, no hint, no nudge.

Stop rules: a request error, a non-zero eval exit, a server that won't come up, a set over
--set-timeout, or a projected size time over 1.5x --budget-hours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GYM = Path(os.environ.get("GRPO_GYM_DIR", ROOT.parent / "grpo-local-wt-filetools" / "sandbox-gym"))
Q4 = ROOT / "q4" / "out"
SERVER = Path(os.environ.get("LLAMA_SERVER",
                             Path.home() / "Documents/workspace/llamacpp-prebuilt/llama-server.exe"))
PORT = 8081
HARNESS = "h3"  # bump when the gym's run-time behavior changes (h2 = multi-call fix 2026-10-09; h3 = Bionic
                # persona prompt + collection turn caps, prompt-checked); slugs carry it so eval.py's resume
                # can never mix results from different harnesses
NATIVE_HARNESS = "n1"  # --native: Bionic native tool mode (gym native.py, byte-checked by test_native_parity.py)
NATIVE_SKIP = {"smoke_docmake_t2_9401-9405"}   # needs make_pdf: not in shipped Bionic (unmerged branch)
PROFILE = "default"   # Bionic's shipped persona = what ~all SFT collection used; every arm gets the same prompt
TRAIN = ROOT.parent / "runpod-sft-1007" / "dataset" / "train.jsonl"   # SFT-1008 data: prompts must byte-match
BASE = f"http://127.0.0.1:{PORT}"
SLOT_CTX = 20480                      # per slot: a 16k compaction window + room for the reply
TABLE = ROOT / "runs" / "pilots" / "q4-eval-mission2.md"
LOGS = ROOT / "runs" / "ops"
sys.path.insert(0, str(HERE))
from eval_sft_1008 import MODELS, SETS  # noqa: E402  same sets and turn caps as the 1008 runner

ARMS = {"0.8b": ("base", "sft"), "4b": ("base", "r3", "r3sft"), "9b": ("base", "sft")}
CORE = ["smoke_repoq_t1_9301-9305", "smoke_repoq_t2_9301-9305", "smoke_file_work_9951-9955",  # fast first
        "smoke_file_work_clarify_one_9951-9955", "smoke_edit_9001-9005", "smoke_svg_make_9701-9705",
        "smoke_web_q_11901-11905", "smoke_docmake_t1_9401-9405", "smoke_docmake_t2_9401-9405",
        "smoke_office_draft_9801-9805"]
LONG = ["smoke_docdigest_t1_9601-9605", "smoke_docdigest_t2_9611-9615", "smoke_docdigest_t3_9621-9625",
        "smoke_gather_t1_9501-9505", "smoke_gather_t2_9501-9505", "smoke_docmake_t3_9401-9405",
        "smoke_docmake_t4_9406-9410", "smoke_pptx_make_9961-9965"]


def log(msg: str) -> None:
    print(f"[q4eval] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --- server -------------------------------------------------------------------------------
def gguf_path(size: str, arm: str) -> Path:
    return Q4 / f"{size}-{arm}.Q4_K_M.gguf"


def check_sha(path: Path) -> str:
    want = (path.parent / (path.name + ".sha256")).read_text(encoding="utf-8").split()[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        raise SystemExit(f"[q4eval] sha256 mismatch for {path.name}: refusing to serve it")
    return want


SAMPLING_FLAGS = ["--top-k", "0", "--min-p", "0"]   # gym harness; --native clears it (Bionic sends its own)


def server_cmd(gguf: Path, np_: int) -> list[str]:
    # Bionic's defaults (llama-console src/config.rs) + eval needs; for the gym harness top-k/min-p off to
    # match the gym's sampling (temperature/top_p only), as in collection and every past eval.
    return [str(SERVER), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(PORT),
            "--jinja", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--fit", "on", "--gpu-layers", "auto", "-np", str(np_), "-c", str(np_ * SLOT_CTX),
            *SAMPLING_FLAGS]


PIDFILE = LOGS / "q4-server.pid"   # our server's pid: a killed runner can't stop its child


def _reap_own_server() -> None:
    """A runner killed from outside (ops.ps1 stop) can leave its llama-server alive. Stop it only
    if the recorded pid is still a llama-server; anything else on the port is refused, not killed."""
    if not PIDFILE.exists():
        return
    pid = PIDFILE.read_text().strip()
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout if os.name == "nt" else ""
    if "llama-server" in out:
        log(f"stopping leftover llama-server pid {pid} from an earlier run")
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
        time.sleep(2)
    PIDFILE.unlink(missing_ok=True)


def start_server(gguf: Path, np_: int, logfile: Path, timeout: int = 600) -> subprocess.Popen:
    _reap_own_server()
    if requests_ok(f"{BASE}/health"):
        raise SystemExit(f"[q4eval] something already answers on {BASE}; stop it first")
    fh = open(logfile, "a", encoding="utf-8")
    fh.write(f"\n=== {time.strftime('%F %T')} {' '.join(server_cmd(gguf, np_))}\n")
    fh.flush()
    proc = subprocess.Popen(server_cmd(gguf, np_), stdout=fh, stderr=subprocess.STDOUT)
    PIDFILE.write_text(str(proc.pid))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"[q4eval] llama-server exited ({proc.returncode}) while loading; see {logfile}")
        if requests_ok(f"{BASE}/health"):
            return proc
        time.sleep(2)
    stop_server(proc)
    raise SystemExit(f"[q4eval] llama-server not healthy after {timeout}s; see {logfile}")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        PIDFILE.unlink(missing_ok=True)
        return
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
    PIDFILE.unlink(missing_ok=True)


def requests_ok(url: str) -> bool:
    try:
        return requests.get(url, timeout=3).status_code == 200
    except requests.RequestException:
        return False


# --- eval ---------------------------------------------------------------------------------
def eval_cmd(size: str, arm: str, set_stem: str, a) -> tuple[str, list[str]]:
    if a.native:   # Bionic native tool mode (gym native.py): Bionic's request, loop, sampling; 8 rounds
        # n1 = as shipped, n1r = reasoning sent back only, n2 = Bionic fix 4e8d675 (round trip + recovery)
        tag = "n2" if a.bionic_fix else NATIVE_HARNESS + ("r" if a.send_reasoning else "")
        if a.max_rounds != 8:
            tag += f"u{a.max_rounds}"            # uncapped study, e.g. n2u48
        slug = f"q4{tag}-{size}-{arm}-{set_stem}" + (f"-lim{a.limit}" if a.limit else "")
        cmd = [sys.executable, "eval.py", "--set", f"data/holdout/{set_stem}.json", "--sealed",
               "--attempts", str(a.attempts), "--slug", slug, "--base", BASE,
               # Bionic sends the GGUF stem as `model`; name it as a user's Qwen file would be
               "--served-model", f"Qwen3.5-{size.upper()}-Q4_K_M", "--model", MODELS[size],
               "--runner", "docker", "--workers", str(a.workers or a.np), "--tool-dialect", "bionic", "--native"]
        cmd += ["--native-max-rounds", str(a.max_rounds)]
        if a.bionic_fix:
            cmd += ["--native-fixed"]
        elif a.send_reasoning:
            cmd += ["--native-send-reasoning"]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        return slug, cmd
    tag = HARNESS + ("nt" if a.no_thinking else "")   # h3nt: own slugs, never resumes into h3
    slug = (f"q4{tag}-{size}-{arm}-{set_stem}" + (f"-lim{a.limit}" if a.limit else "")
            + ("" if a.profile == PROFILE else f"-{a.profile}"))
    cmd = [sys.executable, "eval.py", "--set", f"data/holdout/{set_stem}.json", "--sealed",
           "--attempts", str(a.attempts), "--slug", slug, "--base", BASE, "--served-model", f"q4-{arm}",
           "--model", MODELS[size], "--runner", "docker", "--workers", str(a.workers or a.np),
           "--tool-dialect", "bionic", "--bionic-profile", a.profile,
           "--budget", "16000", "--compact", "--note-style", "facts", "--max-turns", str(SETS[set_stem])]
    if a.profile == PROFILE:   # off-profile runs are deliberate mismatches (prompt-familiarity controls)
        cmd += ["--expect-prompts", str(TRAIN)]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    if a.no_thinking:
        cmd += ["--no-thinking"]
    return slug, cmd


def set_result(slug: str) -> dict:
    graded = passed = errors = cap = resets = 0
    latest: dict = {}   # eval.py resume retries an errored attempt and appends: the LAST line per attempt counts
    for f in (GYM / "data" / "results" / slug / "raw").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r.get("name"), r.get("attempt"))] = r
    for r in latest.values():
            if r.get("error"):
                errors += 1
                continue
            graded += 1
            passed += bool(r.get("passed"))
            cap += r.get("fail_reason") == "turn_cap"
            resets += (r.get("compact") or {}).get("resets", 0)
    return {"graded": graded, "passed": passed, "errors": errors, "turn_cap": cap, "resets": resets}


def table_header(a, arms: tuple, sets: list) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    new = not TABLE.exists()
    with TABLE.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("# Mission 2: Q4_K_M on llama-server (pass/graded; e=request errors, c=turn-cap)\n")
        fh.write(f"\n## {a.size} | harness {('n2' if a.bionic_fix else NATIVE_HARNESS + ('r' if a.send_reasoning else '')) if a.native else HARNESS + ('nt' if a.no_thinking else '')} | profile {('bionic-native' + ('+fix' if a.bionic_fix else '+reasoning' if a.send_reasoning else '')) if a.native else a.profile} | np {a.np} | attempts {a.attempts} | {len(sets)} sets | "
                 f"started {time.strftime('%F %T')}{' | limit ' + str(a.limit) if a.limit else ''}\n\n")
        fh.write("| set | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms) + "\n")


def table_row(set_stem: str, arms: tuple, res: dict) -> None:
    cells = []
    for arm in arms:
        r = res.get(arm)
        cells.append("-" if r is None else f"{r['passed']}/{r['graded']}"
                     + (f" e{r['errors']}" if r["errors"] else "") + (f" c{r['turn_cap']}" if r["turn_cap"] else ""))
    with TABLE.open("a", encoding="utf-8") as fh:
        fh.write(f"| {set_stem} | " + " | ".join(cells) + " |\n")


def table_total(arms: tuple, totals: dict, note: str) -> None:
    cells = [f"**{totals[a]['passed']}/{totals[a]['graded']}**" for a in arms]
    with TABLE.open("a", encoding="utf-8") as fh:
        fh.write("| **pooled** | " + " | ".join(cells) + f" |\n\n{note}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=ARMS)
    ap.add_argument("--arms", default="", help="comma list (default: all arms of the size)")
    ap.add_argument("--suite", choices=("core", "long", "all"), default="core")
    ap.add_argument("--sets", default="", help="comma list of set stems (overrides --suite)")
    ap.add_argument("--np", type=int, default=2, help="llama-server slots (from gate 1)")
    ap.add_argument("--workers", type=int, default=0, help="parallel episodes (default = --np)")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N items per set (gate 2 smoke)")
    ap.add_argument("--set-timeout", type=int, default=90, help="minutes per (set, arm)")
    ap.add_argument("--budget-hours", type=float, default=3.0)
    ap.add_argument("--native", action="store_true",
                    help="Bionic native tool mode (harness n1): Bionic's request/loop/sampling, 8 rounds; "
                         "skips sets that need tools shipped Bionic lacks")
    ap.add_argument("--max-rounds", type=int, default=8,
                    help="with --native: requests per turn (Bionic ships 8); other values = uncapped study")
    ap.add_argument("--bionic-fix", action="store_true",
                    help="with --native: Bionic branch fix/native-reasoning-roundtrip (harness n2), parity-checked")
    ap.add_argument("--send-reasoning", action="store_true",
                    help="with --native: send reasoning_content back (harness n1r = Bionic with the proposed fix)")
    ap.add_argument("--profile", choices=("default", "legacy", "none"), default=PROFILE,
                    help="system prompt; non-default skips the collection prompt check (slug gets -<profile>)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="gym harness: enable_thinking=false (SFT-1008 data is thinking-off); slugs tagged h3nt")
    ap.add_argument("--dry-run", action="store_true")
    from run_policy import add_model_flag, require_model_opt_in
    add_model_flag(ap)
    ap.add_argument('--legacy-diagnostic', action='store_true',
                    help='Acknowledge that this historical runner uses unqualified legacy graders.')
    a = ap.parse_args()
    if not a.dry_run:
        require_model_opt_in(a.allow_model_run)
        if not a.legacy_diagnostic:
            raise SystemExit('Historical runner: --legacy-diagnostic required. Use grpo_eval.py for qualified development contracts.')
    arms = tuple(x for x in a.arms.split(",") if x) or ARMS[a.size]
    bad = [x for x in arms if x not in ARMS[a.size]]
    if bad:
        raise SystemExit(f"[q4eval] arms for {a.size}: {ARMS[a.size]}")
    sets = [s for s in a.sets.split(",") if s] or {"core": CORE, "long": LONG, "all": CORE + LONG}[a.suite]
    if a.native:
        global SAMPLING_FLAGS
        SAMPLING_FLAGS = []            # Bionic sends DEFAULT_PARAMS per request
        sets = [s for s in sets if s not in NATIVE_SKIP]
    unknown = [s for s in sets if s not in SETS]
    if unknown:
        raise SystemExit(f"[q4eval] unknown sets {unknown}")
    if not SERVER.exists():
        raise SystemExit(f"[q4eval] llama-server not found: {SERVER}")

    shas = {}
    for arm in arms:
        p = gguf_path(a.size, arm)
        if not p.exists():
            raise SystemExit(f"[q4eval] missing {p}; build it with q4/build_q4.py")
        shas[arm] = check_sha(p)
    log(f"{a.size} arms {arms} sha256 ok; {len(sets)} sets x {len(arms)} arms, np {a.np}, attempts {a.attempts}")
    if a.native:
        # request bodies must be byte-identical to Bionic's own TS before any server starts
        if subprocess.run([sys.executable, "test_native_parity.py", *(["--fixed"] if a.bionic_fix else [])],
                          cwd=GYM).returncode != 0:
            raise SystemExit("[q4eval] native parity failed (or Bionic/node missing); not starting")
        log("native parity ok: request bodies byte-identical to Bionic")
    elif a.profile == PROFILE:
        for s in sets:   # every set's prompts must byte-match collection before any server starts
            _, cmd = eval_cmd(a.size, arms[0], s, a)
            if subprocess.run(cmd + ["--prompt-check-only"], cwd=GYM).returncode != 0:
                raise SystemExit(f"[q4eval] prompt check failed on {s}; not starting")
        log(f"prompt check ok on {len(sets)} sets (profile {PROFILE}, {TRAIN.name})")
    else:
        log(f"profile {a.profile}: deliberate prompt mismatch vs collection, prompt check skipped")

    if a.dry_run:
        print("server:", " ".join(server_cmd(gguf_path(a.size, arms[0]), a.np)))
        for s in sets:
            for arm in arms:
                slug, cmd = eval_cmd(a.size, arm, s, a)
                print(f"[{s} / {arm}] (cwd {GYM}) {' '.join(cmd)}")
        return

    table_header(a, arms, sets)
    totals = {arm: {"passed": 0, "graded": 0} for arm in arms}
    t_start, proc, note = time.time(), None, "completed"
    try:
        for i, s in enumerate(sets, 1):
            res = {}
            for arm in arms:
                slug, cmd = eval_cmd(a.size, arm, s, a)
                log(f"[{i}/{len(sets)}] {s} / {arm}: starting server")
                proc = start_server(gguf_path(a.size, arm), a.np, LOGS / f"q4-server-{a.size}-{arm}.log")
                t0 = time.time()
                try:
                    rc = subprocess.run(cmd, cwd=GYM, timeout=a.set_timeout * 60).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                stop_server(proc)
                proc = None
                r = res[arm] = set_result(slug)
                log(f"{s} / {arm}: {r} rc={rc} in {(time.time() - t0) / 60:.1f} min")
                if rc != 0 or r["errors"]:
                    note = f"STOPPED at {s}/{arm}: rc={rc}, request errors={r['errors']}"
                    table_row(s, arms, res)
                    raise SystemExit(f"[q4eval] {note}")
                totals[arm]["passed"] += r["passed"]
                totals[arm]["graded"] += r["graded"]
            table_row(s, arms, res)
            projected = (time.time() - t_start) / i * len(sets) / 3600
            if projected > 1.5 * a.budget_hours:
                note = f"STOPPED after {s}: projected {projected:.1f} h > 1.5 x {a.budget_hours} h budget"
                raise SystemExit(f"[q4eval] {note}")
    finally:
        stop_server(proc)
        table_total(arms, totals, f"{note}; {(time.time() - t_start) / 60:.0f} min; "
                                  f"gguf sha256 {', '.join(f'{k} {v[:12]}' for k, v in shas.items())}")
    log(f"done: {totals}")


if __name__ == "__main__":
    main()
