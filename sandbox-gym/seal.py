"""Sealed holdout for the gym.

Eval items are built from *held-out seeds* -- seeds never generated for any
training pool (split by seed, never by string). Sealing writes the item specs,
a canary, a SHA-256 of the whole file and a sidecar hash; the file is then made
read-only. The eval runner verifies the seal before scoring and, when sealed,
records pass/fail only -- never transcripts, never prompts.

If the holdout leaks it is dead: rebuild it from fresh seeds, don't reuse it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import repo_gen

CANARY = "GYM-HOLDOUT-SEALED-2026-09-29"


def build(seeds: list[int], family: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for s in seeds:
        for spec in repo_gen.make_specs(s, family):
            items.append({
                "name": spec["name"], "family": spec["family"], "seed": s,
                "prompt": spec["prompt"], "ground_truth": spec["ground_truth"],
                "tools": spec["tools"], "checks": spec.get("checks", []),
            })
    manifest = {"schema": "gym_v1", "family": family, "canary": CANARY,
                "seeds": seeds, "items": items}
    text = json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False)
    manifest["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    body = json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False)
    out.write_text(body, encoding="utf-8")
    (out.with_suffix(".sha256")).write_text(manifest["sha256"], encoding="utf-8")
    try:  # read-only on Windows is best-effort; the hash is the real guard
        out.chmod(0o444)
    except OSError:
        pass
    print(f"sealed {out} ({len(items)} items, sha256 {manifest['sha256'][:16]}...)")
    return out


def verify(path: Path) -> tuple[bool, dict]:
    """(ok, manifest). Refuses sealed scoring if the file drifted."""
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, {"error": str(e)}
    body = {k: v for k, v in m.items() if k != "sha256"}
    h = hashlib.sha256(json.dumps(body, indent=1, sort_keys=True,
                                  ensure_ascii=False).encode("utf-8")).hexdigest()
    ok = m.get("canary") == CANARY and h == m.get("sha256")
    return ok, m