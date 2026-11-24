"""Sealed handoff eval split.

Holdout tasks come from world seeds that are never used for freezing/collection
(split by seed = by world, so no fact or person leaks across). The manifest
stores the exact task specs, a canary and a SHA-256; common.make_tasks refuses
these seeds unless allow_holdout=True (eval runs only).

  ..\\.venv\\Scripts\\python.exe seal.py            # writes data/holdout/handoff_v1.json
  ..\\.venv\\Scripts\\python.exe seal.py --verify

If the holdout leaks (collected from, trained on, transcripts published), it
is dead: pick new seeds and reseal, never reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import common as C

CANARY = "HANDOFF-HOLDOUT-SEALED-2026-09-29"
PATH = C.DATA / "holdout" / "handoff_v1.json"
PLAN = {7: 3, 8: 2}          # 5 items: few good, verified -- never scale this up


def _hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, indent=1, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def build() -> None:
    if PATH.exists():
        sys.exit(f"{PATH} exists -- a sealed holdout is never rebuilt in place")
    seed = C.HOLDOUT_SEEDS[0]
    items = []
    for h, n in PLAN.items():
        items += C.make_tasks(seed, (h,), per_hop=n, allow_holdout=True)
    body = {"schema": "handoff_v1", "canary": CANARY, "seeds": list(C.HOLDOUT_SEEDS),
            "plan": {str(k): v for k, v in PLAN.items()}, "items": items}
    body["sha256"] = _hash(body)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False),
                    encoding="utf-8")
    PATH.with_suffix(".sha256").write_text(body["sha256"], encoding="utf-8")
    try:
        PATH.chmod(0o444)
    except OSError:
        pass
    print(f"sealed {PATH} ({len(items)} items, seeds {C.HOLDOUT_SEEDS}, "
          f"sha256 {body['sha256'][:16]}...)")


def verify() -> tuple[bool, dict]:
    m = json.loads(PATH.read_text(encoding="utf-8"))
    body = {k: v for k, v in m.items() if k != "sha256"}
    ok = m.get("canary") == CANARY and _hash(body) == m.get("sha256") \
        and set(m.get("seeds", [])) == set(C.HOLDOUT_SEEDS)
    return ok, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        ok, m = verify()
        print(("OK" if ok else "FAILED"), f"{len(m.get('items', []))} items")
        sys.exit(0 if ok else 1)
    build()
