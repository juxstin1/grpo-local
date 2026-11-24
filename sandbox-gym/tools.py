"""The frozen tool set (twelve tools, four groups) + tolerant action parsing.

Group 1 -- Files (path-jailed, safe):
  list_dir(path)
  read_file(path, offset?, limit?)      line ranges, so she can't dump a huge file
  search(pattern, path?, glob?)         ripgrep, output capped
  edit_file(path, old_string, new_string)  exact match, must occur exactly once
  write_file(path, content)             new files only

Group 2 -- Code intelligence (LSP, read-only; hint, never the grade):
  definition(symbol)
  references(symbol)
  diagnostics(path)

Group 3 -- Execution (container only; capped):
  build()
  run_tests(filter?)

Group 4 -- Web (frozen corpus; web_tools.py):
  web_search(objective, searchQueries)  Bionic's tool, verbatim schema and result format
  fetch_url(urls, objective?, offset?)  Bionic's tool, objective-trimmed page text; offset pages raw text

Names and argument formats copied from OpenCode wherever they overlap
(edit_file's old_string/new_string, search's pattern/glob). Every episode ends
with a plain `ANSWER:` line, no tool needed.

Canonical wire format is Qwen-native JSON -- `{"name": ..., "arguments": {...}}`
inside (or not) a <tool_call> block. Parsing is tolerant so other models' bare
JSON works in evals too.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from schema import Action

TOOLS = {
    "list_dir": {
        "args": [("path", True, "directory to list")],
        "help": "List files and directories under path.",
    },
    "read_file": {
        "args": [("path", True, "file to read"),
                 ("offset", False, "1-based first line"),
                 ("limit", False, "max lines to return")],
        "help": "Read a file, capped to a line range.",
    },
    "search": {
        "args": [("pattern", True, "regex to search for"),
                 ("path", False, "directory to search (default: workspace root)"),
                 ("glob", False, "file glob filter, e.g. '*.py'")],
        "help": "Find lines matching pattern (ripgrep), output capped.",
    },
    "edit_file": {
        "args": [("path", True, "file to edit"),
                 ("old_string", True, "exact text to replace"),
                 ("new_string", True, "replacement text")],
        "help": "Replace old_string with new_string. old_string must match exactly once.",
    },
    "write_file": {
        "args": [("path", True, "new file to create"),
                 ("content", True, "full file content")],
        "help": "Create a new file (refuses to overwrite an existing one).",
    },
    "definition": {
        "args": [("symbol", True, "identifier to locate")],
        "help": "Exact definition location of a symbol.",
    },
    "references": {
        "args": [("symbol", True, "identifier to find usages of")],
        "help": "The real usages of a symbol.",
    },
    "diagnostics": {
        "args": [("path", True, "file to check")],
        "help": "Static diagnostics for a file. A hint; the grade is tests.",
    },
    "build": {
        "args": [],
        "help": "Compile/byte-compile the workspace. Capped per episode.",
    },
    "run_tests": {
        "args": [("filter", False, "optional test name filter")],
        "help": "Run the test suite. Capped per episode.",
    },
    # Group 4 -- Web (frozen corpus). Names, descriptions and schemas verbatim from the
    # Bionic rebuild's websearch MCP (docs/bionic-web-tools-spec.md; web_tools.py).
    "checkpoint": {   # compaction only (config Episode.compact); handled in agent.run_episode
        "args": [("note", True, "handoff note with the sections Goal, Done, Current state, "
                                "Hypothesis, Next step, Don't repeat"),
                 ("plan", False, "your plan, pinned into every later restart")],
        "help": "Save a handoff note and restart with a fresh context: task, pinned plan, "
                "your note and the workspace files.",
    },
    "web_search": {
        "args": [("objective", True, "Natural-language description of the web research goal, "
                  "including source or freshness guidance and broader context from the task."),
                 ("searchQueries", True, "Concise keyword search queries, 3-6 words each. At "
                  "least one query is required; provide 2-3 for best results. Maximum 5 "
                  "queries, 200 characters per query.")],
        "help": "Search the PUBLIC INTERNET for external, real-world information: current "
                "events, news, documentation, public facts. Returns result snippets; follow up "
                "with fetch_url to read full pages. Do NOT use this for anything about the user "
                "or this conversation (their preferences, saved notes) — the memory tools own "
                "those.",
        "schema": {"type": "object", "properties": {
            "objective": {"type": "string", "description": "Natural-language description of the "
                          "web research goal, including source or freshness guidance and broader "
                          "context from the task."},
            "searchQueries": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                              "maxItems": 5, "description": "Concise keyword search queries, 3-6 "
                              "words each. At least one query is required; provide 2-3 for best "
                              "results. Maximum 5 queries, 200 characters per query."}},
            "required": ["objective", "searchQueries"]},
    },
    "fetch_url": {
        "args": [("urls", True, "1-3 full http(s) URLs to fetch"),
                 ("objective", False, "Natural-language description of what to pull from the "
                  "pages. Keep it specific so the excerpt stays focused."),
                 ("offset", False, "Character position to start reading from (default 0). With "
                  "an offset you get the raw page text in order, not trimmed to the objective; "
                  "copy the offset=N from a '[truncated — …]' line to read on.")],
        "help": "Fetch 1-3 web pages and return their readable article text as markdown, trimmed "
                "to the parts relevant to your objective. Use this to READ pages — e.g. after "
                "web_search hands you links — so you answer from real content instead of "
                "snippets. To read a long page in order, pass offset (0 for the start); a cut "
                "page ends with a line giving the next offset.",
        "schema": {"type": "object", "properties": {
            "urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
                     "description": "1-3 full http(s) URLs to fetch"},
            "objective": {"type": "string", "description": "Natural-language description of what "
                          "to pull from the pages. Keep it specific so the excerpt stays "
                          "focused."},
            "offset": {"type": "integer", "minimum": 0, "description": "Character position to "
                       "start reading from (default 0). With an offset you get the raw page text "
                       "in order, not trimmed to the objective; copy the offset=N from a "
                       "'[truncated — …]' line to read on."}},
            "required": ["urls"]},
    },
    # Group 5 -- Documents (doc_make.py / doc_tools.py). Bionic has no PDF tool (only
    # office_document for DOCX/XLSX/PPTX), so this is a minimal reportlab-backed tool.
    "make_pdf": {
        "args": [("path", True, "output path ending in .pdf, inside the workspace"),
                 ("markdown", True, "document body as Markdown (#/## headings, paragraphs, "
                  "- bullets, | pipe | tables |, **bold**)")],
        "help": "Typeset Markdown into a PDF file at path (overwrites).",
    },
}

# Tool groups, for the system prompt and for the risk split (see config.py).
FILES = ("list_dir", "read_file", "search", "edit_file", "write_file")
LSP = ("definition", "references", "diagnostics")
EXEC = ("build", "run_tests")
WEB = ("web_search", "fetch_url")

# Output caps, so one huge read/search can't blow the token budget.
LIST_DIR_CAP = 60          # entries
READ_LINE_CAP = 200        # lines per read_file call
READ_CHAR_CAP = 12_000     # chars per read_file call
SEARCH_CAP = 40            # matches per search call
SEARCH_CHAR_CAP = 8_000    # chars of search output
DIAG_CAP = 25              # diagnostics per file

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def _extract_objects(text: str) -> list[dict]:
    """Every balanced JSON object in text, in order (tolerant to prose around it)."""
    out: list[dict] = []
    for m in TOOL_CALL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
    i = (text or "").find("{")
    while i != -1:
        depth, j = 0, i
        try:
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 0:
                obj = json.loads(text[i : j + 1])
                if isinstance(obj, dict):
                    out.append(obj)
        except Exception:
            pass
        i = text.find("{", i + 1)
    return out


def parse_action(text: str) -> Optional[Action]:
    """First tool call in a response section, or None."""
    for obj in _extract_objects(text or ""):
        name = obj.get("name") or obj.get("tool")
        if not isinstance(name, str):
            continue
        if name not in TOOLS and not isinstance(obj.get("arguments"), dict):
            continue          # a stray JSON object, not a call
        args = obj.get("arguments") or obj.get("args")
        if not isinstance(args, dict):
            args = {k: v for k, v in obj.items() if k not in ("name", "tool")}
        return Action(name=name, arguments=args)
    return None


def unavailable(action: Action, enabled: list[str]) -> Optional[str]:
    """Error observation for a call to a tool this task doesn't have (unknown or
    outside the ring), or None. The episode continues -- like a real CLI."""
    if action.name in enabled:
        return None
    return (f"error: no tool named {action.name!r} in this task. "
            f"Available: {', '.join(enabled)}")


def describe_tools(enabled: list[str]) -> str:
    """System-prompt block listing the enabled tools, one line each."""
    lines = ["You have access to these tools:"]
    for name in enabled:
        spec = TOOLS[name]
        args = ", ".join(f"{a}?" if not req else a for a, req, _ in spec["args"])
        lines.append(f"  {name}({args}) -- {spec['help']}")
        if "schema" in spec:   # Bionic tools: their arg descriptions are the coaching text
            props = spec["schema"]["properties"]
            for a, _, _ in spec["args"]:
                p = props[a]
                kind = (f"array of {p['minItems']}-{p['maxItems']} strings"
                        if p["type"] == "array" else p["type"])
                lines.append(f"      {a} ({kind}): {p['description']}")
    lines.append(
        "Call exactly one tool per turn as a JSON block:\n"
        '  {"name": "<tool>", "arguments": {"<arg>": value}}\n'
        "A result line follows each call. When you know the answer, or the requested "
        "change is done, end with one line:\n"
        "  ANSWER: <your answer, or a one-line summary of the change>\n"
        "No tool is needed for the final ANSWER line."
    )
    return "\n".join(lines)


def tool_call_text(action: Action) -> str:
    return json.dumps({"name": action.name, "arguments": action.arguments})