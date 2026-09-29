"""Offline self-tests: no torch, no GPU, no network, no model.

Exercises everything that doesn't need a served model: the path jail, the tool
semantics (exact-once edit guard, output caps), the seeded repo generator
(determinism + find-and-fix guard via the real test suite), the mechanical
graders, tolerant action parsing, and the token-stream invariants with a stub
tokenizer.

Run:  .\\cli.py selftest     (or: python selftest.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import encoding  # noqa: E402
import repo_gen  # noqa: E402
import tools as T  # noqa: E402
from agent import execute_tool  # noqa: E402
from graders import grade_edit, grade_find, grade_fix  # noqa: E402
from runner import Runner  # noqa: E402
from schema import Task, Turn  # noqa: E402
from workspace import Workspace  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok  {name}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")


# --- path jail --------------------------------------------------------------

def test_jail() -> None:
    print("[path jail]")
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td), Runner("host"))
        (ws.root / "a.txt").write_text("hello", encoding="utf-8")
        (ws.root / "sub").mkdir()
        (ws.root / "sub" / "b.txt").write_text("world", encoding="utf-8")
        check("read inside", "hello" in ws.read_file("a.txt"))
        check("read nested", "world" in ws.read_file("sub/b.txt"))
        for bad in ("..", "../a.txt", str(Path(td).parent / "outside.txt"),
                    "a.txt/../../x", str(ws.root / "..")):
            try:
                ws.resolve(bad)
                check(f"jail blocks {bad}", False)
            except Exception:
                check(f"jail blocks {bad}", True)
        check("edit exact-once", ws.edit_file("a.txt", "hello", "hi").startswith("edited"))
        check("edit 0 matches",
              "0 matches" in ws.edit_file("a.txt", "zzz", "x"))
        ws.write_file("c.txt", "one\ntwo\nthree\nfour\nfive")
        check("write refuses overwrite",
              "already exists" in ws.write_file("c.txt", "x"))


# --- tools / parse ----------------------------------------------------------

def test_parse() -> None:
    print("[tool parsing]")
    cases = [
        ('<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>',
         "read_file", "a.txt"),
        ('prefix {"name":"search","arguments":{"pattern":"foo"}} suffix',
         "search", "foo"),
        ('<tool_call>{"name":"run_tests"}</tool_call>', "run_tests", None),
    ]
    for text, name, arg in cases:
        a = T.parse_action(text)
        check(f"parse {name}", a is not None and a.name == name and
              (arg is None or arg in str(a.arguments)))
    check("no action on prose", T.parse_action("I think the answer is 42.") is None)
    sp = T.describe_tools(["list_dir", "read_file", "search"])
    check("system prompt lists tools",
          all(x in sp for x in ("list_dir", "read_file", "search", "ANSWER:")))


# --- repo generator ---------------------------------------------------------

def test_repo() -> None:
    print("[repo generator]")
    p1, p2 = repo_gen.generate_project(42), repo_gen.generate_project(42)
    p3 = repo_gen.generate_project(43)
    check("deterministic per seed", p1.files == p2.files)
    check("differs across seeds", p1.files != p3.files)
    check("has tests", "tests/test_core.py" in p1.files)
    check("tests pass for base", repo_gen.tests_pass(p1.files))
    for i, bug in enumerate(repo_gen.BUGS):
        b = repo_gen.inject_bug(p1, i)
        check(f"guard {bug}: base passes", repo_gen.tests_pass(p1.files))
        check(f"guard {bug}: bugged fails",
              not repo_gen.tests_pass(b.files), f"bug={b.meta.get('bug')}")
    specs = repo_gen.make_specs(42, "find")
    check("7 find specs", len(specs) == 7)
    check("find specs carry files",
          all("files" in s and "core.py" in s["files"] for s in specs))


# --- graders ----------------------------------------------------------------

def test_graders() -> None:
    print("[graders]")
    spec = repo_gen.find_spec(42, "find", f"find-config-TAX_RATE-42")
    assert "files" in spec, f"find spec missing files: {list(spec.keys())}"
    gt = spec["ground_truth"]

    def turns(text: str) -> list[Turn]:
        return [Turn(content=text, resp_text=text)]

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td), Runner("host"))
        ws.reset()
        repo_gen.write_files(spec["files"], ws.root)
        ok, _ = grade_find(turns(f"ANSWER: {gt}"), ws, gt)
        check("find correct", ok)
        ok2, _ = grade_find(turns(f"ANSWER: 99"), ws, gt)
        check("find wrong", not ok2)
        ok3, _ = grade_find(turns("no answer"), ws, gt)
        check("find no-answer", not ok3)

        # edit grading on the end state
        spec_e = repo_gen.find_spec(42, "edit", f"edit-tax-42")
        ws2 = Workspace(Path(td) / "e2", Runner("host"))
        ws2.reset()
        repo_gen.write_files(spec_e["files"], ws2.root)
        (ws2.root / "config.py").write_text("TAX_RATE = 0.08\n", encoding="utf-8")
        ok_e, _ = grade_edit(turns("ANSWER: done"), ws2,
                             [{"op": "contains", "file": "config.py",
                               "needle": "TAX_RATE = 0.08"}])
        check("edit end-state ok", ok_e)
        ok_e2, _ = grade_edit(turns("ANSWER: done"), ws2,
                              [{"op": "absent", "file": "config.py",
                                "needle": "TAX_RATE"}])
        check("edit absent check", not ok_e2)


def test_find_numbers() -> None:
    print("[find grader: numbers by value]")
    t = [Turn(content="ANSWER: 42.5", resp_text="ANSWER: 42.5")]
    check("42.5 == 42.50", grade_find(t, None, "42.50")[0])
    check("40.0 == 40", grade_find([Turn(content="ANSWER: 40.0", resp_text="ANSWER: 40.0")], None, "40")[0])
    check("42.4 != 42.50", not grade_find([Turn(content="ANSWER: 42.4", resp_text="ANSWER: 42.4")], None, "42.50")[0])


def test_search_injection() -> None:
    print("[search: pattern can't become a ripgrep flag]")
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "inj", Runner("host"))
        ws.reset()
        (ws.root / "a.py").write_text("x = 1\n", encoding="utf-8")
        marker = ws.root.parent / "PWNED"
        # If the pattern were parsed as a flag, --pre would run this command per file.
        out = ws.search(f"--pre=python -c \"open(r'{marker}','w')\"")
        check("--pre pattern did not execute", not marker.exists(), out[:80])
        check("dash pattern searched literally", "a.py" not in out or "--pre" in out)


def test_lsp_jail() -> None:
    print("[lsp diagnostics: path jail, no .pyc written]")
    from lsp import PythonLsp
    with tempfile.TemporaryDirectory() as td:
        ws_root = Path(td) / "ws"
        ws_root.mkdir()
        (ws_root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (ws_root / "bad.py").write_text("def f(:\n", encoding="utf-8")
        outside = Path(td) / "secret.py"
        outside.write_text("TOKEN = 'x'\n", encoding="utf-8")
        lsp = PythonLsp()
        check("escape refused", "escapes the workspace" in lsp.diagnostics(ws_root, "../secret.py"))
        check("clean file ok", "clean" in lsp.diagnostics(ws_root, "ok.py"))
        check("syntax error reported", "SyntaxError" in lsp.diagnostics(ws_root, "bad.py"))
        check("no __pycache__ written", not any(p.name == "__pycache__" for p in Path(td).rglob("*")))


def test_fix_grading() -> None:
    print("[fix grading (host runner)]")
    spec = repo_gen.find_spec(44, "fix", f"fix-core-44")
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "fix", Runner("host"))
        ws.reset()
        repo_gen.write_files(spec["files"], ws.root)
        from graders import pristine_tests
        tests = pristine_tests(44)
        ok_bugged, _ = grade_fix([Turn(content="ANSWER: x", resp_text="ANSWER: x")], ws, tests)
        check("bugged fails", not ok_bugged)
        # Cheat 1: gut the visible tests instead of fixing the bug -> must still fail.
        (ws.root / "tests" / "test_core.py").write_text(
            "print('ALL TESTS PASSED')\n", encoding="utf-8")
        ok_cheat, _ = grade_fix([Turn(content="ANSWER: x", resp_text="ANSWER: x")], ws, tests)
        check("editing the visible tests does not pass", not ok_cheat)
        # Cheat 2: make core.py print the success line and exit 0 on import.
        (ws.root / "core.py").write_text(
            "import os\nprint('ALL TESTS PASSED')\nos._exit(0)\n", encoding="utf-8")
        ok_cheat2, _ = grade_fix([Turn(content="ANSWER: x", resp_text="ANSWER: x")], ws, tests)
        check("exit-0-early core.py does not pass", not ok_cheat2)
        from repo_gen import generate_project
        clean = generate_project(44).files["core.py"]
        (ws.root / "core.py").write_text(clean, encoding="utf-8")
        ok_fixed, _ = grade_fix([Turn(content="ANSWER: done", resp_text="ANSWER: done")], ws, tests)
        check("fixed passes (even with gutted visible tests)", ok_fixed)


# --- encoding invariants (stub tokenizer) -----------------------------------

class StubTok:
    bos_token_id = 1
    eos_token_id = 2

    def convert_tokens_to_ids(self, t: str) -> int:
        return {"<|im_start|>": 5, "<|im_end|>": 6}.get(t, 99)

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in s]


def test_encoding() -> None:
    print("[encoding invariants]")
    tok = StubTok()
    obs_ids = encoding.encode_observation(tok, "Result of read_file: hi", True)
    check("observation wraps im_start/im_end (after the template's newline)",
          obs_ids[0] == ord("\n") and obs_ids[1] == 5 and 6 in obs_ids)
    body = "".join(chr(c) if c < 256 else "?" for c in obs_ids)
    check("observation embeds <tool_response>", "<tool_response>" in body)
    check("response_section split",
          encoding.response_section("reasoning\n</think>\n\nACTION", True) == "ACTION")
    check("response_section still thinking -> empty",
          encoding.response_section("reasoning, no close yet", True) == "")
    check("response_section non-thinking passthrough",
          encoding.response_section("ACTION", False) == "ACTION")
    # render_chat must equal incremental encode_prompt for the first turn
    class MinTok(StubTok):
        def apply_chat_template(self, msgs, **kw):
            return [5, 7, 8, 6]
    mt = MinTok()
    check("render_chat returns ids",
          encoding.render_chat(mt, [{"role": "system", "content": "s"},
                                    {"role": "user", "content": "u"}], True) == [5, 7, 8, 6])


def main() -> None:
    test_jail()
    test_parse()
    test_repo()
    test_graders()
    test_find_numbers()
    test_search_injection()
    test_lsp_jail()
    test_fix_grading()
    test_encoding()
    print()
    if FAILED:
        print(f"SELFTEST FAILED: {len(FAILED)}: {FAILED}")
        sys.exit(1)
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()