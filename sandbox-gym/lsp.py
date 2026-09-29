"""Code-intelligence seam (definition / references / diagnostics).

These are a *hint*, never the grade -- the reward stays terminal (edit end-state
+ tests). The Python-first rings (find/edit/fix) don't enable LSP at all; it
rolls in with the C++/JUCE repos, where `definition`/`references` are what make
multi-hop cheap and `diagnostics` gives ~1s feedback before a full build.

Python implementation is a best-effort stopgap until clangd arrives:
  - definition/references: token-aware ripgrep scan (exact word boundaries)
  - diagnostics: py_compile syntax check only (Python is dynamic; tests carry
    the real grade). Swap the backend for pyright when one is pinned.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SYMBOL_RE = re.compile(r"\b(?P<sym>\w+)\b")


class LspClient:
    """Interface. Implementations must be deterministic and fast (no indexing)."""

    def definition(self, workspace: Path, symbol: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def references(self, workspace: Path, symbol: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def diagnostics(self, workspace: Path, path: str) -> str:  # pragma: no cover
        raise NotImplementedError


class PythonLsp(LspClient):
    """Word-boundary rg scan + py_compile syntax check. Hint-quality only."""

    def _rg(self, workspace: Path, symbol: str) -> list[tuple[str, int]]:
        pat = rf"\b{re.escape(symbol)}\b"
        r = subprocess.run(
            ["rg", "-n", "--no-heading", pat, str(workspace), "-g", "*.py"],
            capture_output=True, text=True, timeout=30,
        )
        hits = []
        if r.returncode in (0, 1):
            for ln in r.stdout.splitlines():
                f = ln.split(":", 1)[0]
                try:
                    i = int(ln.split(":", 1)[1].split(":", 1)[0])
                except ValueError:
                    i = 0
                hits.append((Path(f).name, i))
        return hits

    def definition(self, workspace: Path, symbol: str) -> str:
        # crude heuristic: the hit on a line that defines it (def / class / = ).
        sym = re.escape(symbol)  # her symbol is data, never regex syntax
        for f, line in self._rg(workspace, symbol):
            text = (workspace / f).read_text(encoding="utf-8").split("\n")[line - 1] \
                if (workspace / f).exists() else ""
            if re.search(rf"(def {sym}\b|class {sym}\b|\b{sym}\s*=)", text):
                return f"{f}:{line}"
        return f"definition not found for {symbol}"

    def references(self, workspace: Path, symbol: str) -> str:
        hits = self._rg(workspace, symbol)
        if not hits:
            return f"no references to {symbol}"
        return "\n".join(f"{f}:{line}" for f, line in hits[:20]) or \
            f"no references to {symbol}"

    def diagnostics(self, workspace: Path, path: str) -> str:
        # Path jail (same rule as Workspace.resolve): nothing outside the workspace.
        root = Path(os.path.realpath(workspace))
        f = Path(os.path.realpath(root / path))
        try:
            f.relative_to(root)
        except ValueError:
            return f"error: path escapes the workspace: {path}"
        if not f.is_file():
            return f"error: no such file: {path}"
        # In-process syntax check: no subprocess, and no .pyc written anywhere.
        try:
            compile(f.read_text(encoding="utf-8"), path, "exec")
        except SyntaxError as e:
            return f"{path}:{e.lineno}: SyntaxError: {e.msg}"
        except (UnicodeDecodeError, ValueError) as e:
            return f"{path}: {type(e).__name__}: {e}"
        return f"{path}: clean (syntax ok)"


def make_lsp() -> LspClient:
    return PythonLsp()