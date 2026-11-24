"""Write a reviewable leg-3 patch without older local STOP_AFTER/bridge edits."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/ops"
BASE = OUT / "leg3-trainer-pod-baseline.py"
CLEAN = OUT / "leg3-trainer-clean.py"
PATCH = OUT / "leg3-only.patch"


def diff(old: str, new: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True),
                                        new.splitlines(keepends=True),
                                        fromfile=before, tofile=after))


def main() -> None:
    clean = CLEAN.read_text(encoding="utf-8")
    for older in ("STOP_AFTER", "stop_requested(", "consume_stop(", "BridgeError"):
        if older in clean:
            raise ValueError(f"older local change leaked into clean trainer: {older}")
    ast.parse(clean)
    before = BASE.read_text(encoding="utf-8")
    chunks = ["diff --git a/sandbox-gym/train_grpo.py b/sandbox-gym/train_grpo.py\n",
              diff(before, clean, "a/sandbox-gym/train_grpo.py",
                   "b/sandbox-gym/train_grpo.py")]
    for name in ("grpo_sampling.py", "grpo_mixed.py", "test_grpo_sampling.py", "test_grpo_mixed.py"):
        path = ROOT / "sandbox-gym" / name
        text = path.read_text(encoding="utf-8")
        ast.parse(text)
        chunks += [f"diff --git a/sandbox-gym/{name} b/sandbox-gym/{name}\n",
                   "new file mode 100644\n",
                   diff("", text, "/dev/null", f"b/sandbox-gym/{name}")]
    name = "build-leg3-attestation.py"
    text = (ROOT / "ops" / name).read_text(encoding="utf-8")
    ast.parse(text)
    chunks += [f"diff --git a/ops/{name} b/ops/{name}\n", "new file mode 100644\n",
               diff("", text, "/dev/null", f"b/ops/{name}")]
    PATCH.write_text("".join(chunks), encoding="utf-8", newline="\n")
    print(f"PATCH_OK {PATCH} bytes={PATCH.stat().st_size} files=6")


if __name__ == "__main__":
    main()
