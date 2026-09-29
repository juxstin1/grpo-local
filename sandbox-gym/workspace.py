"""The sandboxed workspace: path jail + the ten tools behind it.

Every episode gets its own scratch directory under data/workspaces/<task>/,
created from a repo seed (repo_gen.py). The path jail refuses anything that
resolves outside the workspace root -- no symlinks, no `..` escapes. Tool
outputs are capped (tools.py constants) so one huge read/search can't blow the
token budget.

Execution tools (build/run_tests) go through runner.py: a throwaway Docker
container with no network by default; `host` is a selftest-only fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import tools as T
from runner import Runner

MAX_TOOL_OUT = 14_000  # hard ceiling on any single tool result string


class JailError(RuntimeError):
    pass


class Workspace:
    """A path-jailed working directory. `runner` owns build/run_tests."""

    def __init__(self, root: Path, runner: Runner):
        self.root = Path(root).resolve()
        self.runner = runner
        self.builds_left = -1  # -1 = unlimited (set per episode by the caller)

    # --- jail ----------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        """Absolute path inside the workspace, or JailError."""
        p = Path(str(rel))
        if p.is_absolute():
            p = Path(os.path.abspath(str(p)))
        else:
            p = (self.root / p)
        p = Path(os.path.realpath(p))  # follow symlinks so links can't escape
        try:
            p.relative_to(self.root)
        except ValueError:
            raise JailError(f"path escapes the workspace: {rel}")
        return p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    # --- file tools ----------------------------------------------------------

    def list_dir(self, path: str) -> str:
        d = self.resolve(path)
        if not d.is_dir():
            return f"error: not a directory: {path}"
        entries = sorted(os.listdir(d))
        shown, rest = entries[: T.LIST_DIR_CAP], len(entries) - T.LIST_DIR_CAP
        lines = [("dir " if (d / e).is_dir() else "file") + f"  {e}" for e in shown]
        if rest > 0:
            lines.append(f"... and {rest} more entries")
        return self._cap("\n".join(lines) or "(empty)")

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        try:
            lines = f.read_text(encoding="utf-8").split("\n")
        except UnicodeDecodeError:
            return f"error: {path} is not UTF-8 text"
        start = (offset or 1) - 1
        if start < 0 or start >= len(lines):
            return f"error: offset {offset} out of range (file has {len(lines)} lines)"
        end = len(lines) if limit is None else start + limit
        if limit is None and end - start > T.READ_LINE_CAP:
            end = start + T.READ_LINE_CAP
        chunk = lines[start:end]
        header = f"{path}:{start + 1}-{start + len(chunk)} of {len(lines)} lines"
        body = "\n".join(f"{i + start + 1:>5}: {ln}" for i, ln in enumerate(chunk))
        if start + len(chunk) < len(lines):
            body += f"\n... {len(lines) - start - len(chunk)} more lines (use offset/limit)"
        return self._cap(header + "\n" + body)

    def search(self, pattern: str, path: str | None = None, glob: str | None = None) -> str:
        d = self.resolve(path or ".")
        if not d.is_dir():
            return f"error: not a directory: {path}"
        rel = d.relative_to(self.root) if d != self.root else Path(".")
        args = ["rg", "-n", "--no-heading", "--color", "never"]
        if glob:
            args += ["-g", glob]
        # -e + `--`: her pattern/path can never be parsed as ripgrep flags
        # (e.g. `--pre=<cmd>` would run a host command on every file).
        args += ["-e", pattern, "--", str(rel)]
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=30,
                               cwd=self.root)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return self._search_py(d, pattern, glob)
        if r.returncode == 1:
            return f"no matches for {pattern}"
        if r.returncode not in (0, 1):
            return f"search error: {(r.stderr or r.stdout).strip()[:200]}"
        hits = [ln for ln in r.stdout.splitlines() if ln.strip()]
        shown, rest = hits[: T.SEARCH_CAP], len(hits) - T.SEARCH_CAP
        body = "\n".join(shown)
        if rest > 0:
            body += f"\n... {rest} more matches"
        return self._cap(body or f"no matches for {pattern}")

    def _search_py(self, d: Path, pattern: str, glob: str | None) -> str:
        """ripgrep-less fallback. Case-sensitive substring scan (not regex)."""
        import fnmatch
        hits = []
        for root, _, files in os.walk(d):
            for fn in files:
                if glob and not fnmatch.fnmatch(fn, glob):
                    continue
                p = Path(root) / fn
                try:
                    for i, ln in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
                        if pattern in ln:
                            hits.append(f"{self._rel(p)}:{i}: {ln}")
                except Exception:
                    continue
                if len(hits) >= T.SEARCH_CAP:
                    break
            if len(hits) >= T.SEARCH_CAP:
                break
        return self._cap("\n".join(hits) or f"no matches for {pattern}")

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"error: {path} is not UTF-8 text"
        n = text.count(old_string)
        if n == 0:
            return "error: 0 matches for old_string (check exact text, whitespace, case)"
        if n > 1:
            return f"error: {n} matches for old_string; add more surrounding context"
        f.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"edited {path}: replaced 1 occurrence"

    def write_file(self, path: str, content: str) -> str:
        f = self.resolve(path)
        if f.exists():
            return f"error: {path} already exists (write_file creates new files only)"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} chars)"

    # --- execution -----------------------------------------------------------

    def build(self) -> str:
        if self.builds_left == 0:
            return "error: build budget exhausted for this episode"
        if self.builds_left > 0:
            self.builds_left -= 1
        return self.runner.build(self.root)

    def run_tests(self, filter: str | None = None) -> str:
        if self.builds_left == 0:
            return "error: run_tests budget exhausted for this episode"
        if self.builds_left > 0:
            self.builds_left -= 1
        return self.runner.run_tests(self.root, filter)

    # --- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cap(s: str) -> str:
        if len(s) > MAX_TOOL_OUT:
            return s[:MAX_TOOL_OUT] + f"\n... [{len(s) - MAX_TOOL_OUT} chars truncated]"
        return s