"""SFT-1008 eval: each served arm on the sealed per-family sets, with collection's harness settings.

Usage (from grpo-local, via ops.ps1):
  python ops/eval_sft_1008.py <size> <arm> [<arm> ...] [--attempts N] [--limit N] [--sets a,b]
  e.g.  python ops/eval_sft_1008.py 4b r3 r3sft
Arms are LoRA names on the vLLM at --base; "base" means the plain model (no adapter).
Settings mirror the teacher collection the SFT data came from: Bionic dialect, 16k budget with
compaction and facts notes, per-family turn caps, no hint, no reset nudge (the 1008 data has none).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GYM = ROOT.parent / "grpo-local-wt-filetools" / "sandbox-gym"
MODELS = {"4b": "Qwen/Qwen3.5-4B", "9b": "Qwen/Qwen3.5-9B", "0.8b": "Qwen/Qwen3.5-0.8B"}
# set file stem -> max turns (same caps as the collection farms)
SETS = {
    "smoke_docdigest_t1_9601-9605": 72, "smoke_docdigest_t2_9611-9615": 72, "smoke_docdigest_t3_9621-9625": 72,
    "smoke_gather_t1_9501-9505": 48, "smoke_gather_t2_9501-9505": 48,
    "smoke_docmake_t1_9401-9405": 40, "smoke_docmake_t2_9401-9405": 40,
    "smoke_docmake_t3_9401-9405": 40, "smoke_docmake_t4_9406-9410": 40,
    "smoke_office_draft_9801-9805": 48,
    "smoke_repoq_t1_9301-9305": 24, "smoke_repoq_t2_9301-9305": 24,
    "smoke_file_work_9951-9955": 32, "smoke_file_work_clarify_one_9951-9955": 32,
    "smoke_edit_9001-9005": 32, "smoke_svg_make_9701-9705": 32,
    "smoke_web_q_11901-11905": 20, "smoke_pptx_make_9961-9965": 48,  # = collection (train.jsonl)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=MODELS)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N items per set (smoke)")
    ap.add_argument("--sets", default="", help="comma-separated set stems (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    sets = [s for s in a.sets.split(",") if s] or list(SETS)
    failed = []
    for arm in a.arms:
        for s in sets:
            slug = f"sft1008-{a.size}-{arm}-{s}" + (f"-lim{a.limit}" if a.limit else "")
            cmd = [sys.executable, "eval.py", "--set", f"data/holdout/{s}.json", "--sealed",
                   "--attempts", str(a.attempts), "--slug", slug, "--base", a.base,
                   "--model", MODELS[a.size], "--runner", "docker", "--workers", str(a.workers),
                   "--tool-dialect", "bionic", "--budget", "16000", "--compact", "--note-style", "facts",
                   "--max-turns", str(SETS[s])]
            if arm != "base":
                cmd += ["--served-model", arm]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            print(f"[eval1008] {a.size} {arm} {s}", flush=True)
            rc = subprocess.run(cmd, cwd=GYM, check=False).returncode
            if rc:
                failed.append((arm, s, rc))
                print(f"[eval1008] {arm} {s}: exit {rc}", flush=True)
    print(f"[eval1008] done; failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
