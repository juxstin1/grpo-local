"""Unified CLI: .\\cli.py <seal|eval|collect|train|selftest> [flags]

One entrypoint, one vocabulary. Examples (PowerShell, from this folder):

  .\\cli.py selftest
  .\\cli.py seal --family find --seeds 9001
  .\\cli.py eval --family find --attempts 4 --slug r3-find --served-model r3
  .\\cli.py collect --family find --seeds 2000-2019 --name find-r3-s1 --served-model r3
  .\\cli.py train --family find --seeds 1000-1009 --steps 50 --group-size 6
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="sandbox-gym harness")
    ap.add_argument("cmd", choices=["seal", "eval", "collect", "train", "selftest"])
    args, rest = ap.parse_known_args()

    if args.cmd == "seal":
        import seal
        ap2 = argparse.ArgumentParser()
        ap2.add_argument("--family", default="find")
        ap2.add_argument("--seeds", required=True)
        ap2.add_argument("--out", default=str(HERE / "data" / "holdout" / "gym_v1.json"))
        a = ap2.parse_args(rest)
        seeds = []
        for part in a.seeds.split(","):
            lo, _, hi = part.partition("-")
            seeds += list(range(int(lo), int(hi or lo) + 1))
        seal.build(seeds, a.family, Path(a.out))
        return

    if args.cmd == "selftest":
        import selftest
        selftest.main()
        return

    sys.argv = [str(HERE / args.cmd)] + rest
    runpy.run_path(str(HERE / f"{args.cmd}.py"), run_name="__main__")


if __name__ == "__main__":
    main()