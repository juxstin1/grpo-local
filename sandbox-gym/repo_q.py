"""Family repo_q: context-gathering questions over the user's REAL plugin repo.

Source: the JUCE monorepo C:\\Users\\juxxs\\Music\\vst (github juxstin1/vst-plugins),
read-only. Each task's workspace is a snapshot taken with `git archive` at a pinned
commit (never the working tree, which is dirty), so answers stay fixed while the
repo moves on. Snapshot = engine/vstkit (minus third_party) + the plugins of ONE
split side, code files only (.h/.hpp/.cpp/.c/CMakeLists.txt), size-capped, secret
names refused.

Ground truth is parsed from the snapshot at generation time (regex over the C++,
`git log` for history) -- no LLM. Grader: exact match on her final ANSWER line,
normalized; numbers by value; comma lists as sets; `a::b::C` compares as `C`.

Split rule (root = plugin directory):
  holdout plugins  HOLDOUT_PLUGINS, snapshot at PIN, seeds HOLDOUT_SEEDS (9301-9305,
                   seed i -> holdout plugin i). Their workspace contains no train plugin.
  train plugins    every other plugin; a train seed picks a train plugin AND one of
                   that plugin's own commits (history reachable from PIN), so train
                   snapshots vary over time. Their workspace contains no holdout plugin.
  lineage          entities on a holdout question (plugin, member, class names) never
                   appear on a train question: train specs touching a holdout entity
                   (e.g. a shared engine class) are dropped (`_holdout_entities`).

Tiers (one item per tier per seed):
  t1  1-hop lookup: parameter default / range max, which plugins use engine class C,
      which file defines class C, which classes C inherits from.   ~2 tool calls
  t2  2-3 hops: processor member -> its type -> defining file [-> base classes]
      (memberfile / memberbase); member -> type -> plugins using that type
      (memberusers); older copy vs current file -> added parameters (history: what
      changed between two commits).                                 ~2-4 tool calls
Turn caps: the scripted oracle needs <= 2 tool calls on t1 and <= 4 on t2 (+1 answer
turn), so t1 fits cap 6 and t2 fits the default cap 10 with room to explore.

Tools: list_dir, read_file, search (read-only). `lookup_items(seed)` returns
(question, ground_truth, files) per item for reuse by other families.
"""

from __future__ import annotations

import io
import os
import random
import re
import subprocess
import tarfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

FAMILY = "repo_q"
TOOLS = ["list_dir", "read_file", "search"]
REPO = Path(os.environ.get("REPOQ_REPO", r"C:\Users\juxxs\Music\vst"))
PIN = "c5265e41c4b93d73642a721a597d1f56fc4da983"      # HEAD of sync/2026-09-26-backlog (pushed)
ALL_PLUGINS = ("Colossus", "Corona", "Ember", "Firstlight", "Ironworks", "JustBend",
               "Kronos", "Lumen", "Lyra", "Nadir", "Olander", "Polaris", "Redshift",
               "Umbra", "Vela")
HOLDOUT_PLUGINS = ("Corona", "Ember", "Nadir", "Polaris", "Redshift")
TRAIN_PLUGINS = tuple(p for p in ALL_PLUGINS if p not in HOLDOUT_PLUGINS)
HOLDOUT_SEEDS = tuple(range(9301, 9306))
RESERVED = range(9300, 9400)                    # never train seeds

CODE_SUFFIX = {".h", ".hpp", ".cpp", ".c", ".cmake"}
SECRET_RE = re.compile(r"(^|/)\.env|\.pem$|\.key$|\.pfx$|id_rsa|id_ed25519|secret|credential|token",
                       re.I)
MAX_FILE = 200_000                               # bytes; bigger files are dropped
MAX_TOTAL = 3_000_000                            # bytes per workspace
GUESSABLE = {"Bypass", "Mix"}                    # defaults anyone could guess


def is_holdout_seed(seed: int) -> bool:
    return seed in RESERVED


# --- snapshot -----------------------------------------------------------------

def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          check=True).stdout


def _keep(path: str) -> bool:
    p = PurePosixPath(path)
    if SECRET_RE.search(path) or "third_party" in p.parts:
        return False
    if p.parts[0] == "plugins" and len(p.parts) > 2 and p.parts[2] in ("resources", "docs", "proof"):
        return False
    return p.suffix.lower() in CODE_SUFFIX or p.name == "CMakeLists.txt"


@lru_cache(maxsize=64)
def snapshot(commit: str, side: str) -> tuple[tuple[str, str], ...]:
    """(path, text) pairs: engine/vstkit + this side's plugins at `commit`."""
    plugins = HOLDOUT_PLUGINS if side == "holdout" else TRAIN_PLUGINS
    specs = ["engine/vstkit"] + [f"plugins/{p}" for p in plugins]
    present = set(_git("ls-tree", "--name-only", commit, "plugins/").decode().split())
    specs = [s for s in specs if s == "engine/vstkit" or s in present]
    raw = _git("archive", "--format=tar", commit, "--", *specs)
    out, total = [], 0
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not _keep(m.name) or m.size > MAX_FILE:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            try:
                text = fh.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
            total += m.size
            out.append((m.name, text.replace("\r\n", "\n")))
    if total > MAX_TOTAL:
        raise AssertionError(f"snapshot {commit[:7]}/{side} is {total} bytes > cap")
    return tuple(sorted(out))


@lru_cache(maxsize=32)
def plugin_commits(plugin: str) -> tuple[str, ...]:
    """Commits (full sha, newest first, reachable from PIN) that touched the plugin's code."""
    out = _git("log", "--format=%H", PIN, "--", f"plugins/{plugin}").decode().split()
    return tuple(out)


@lru_cache(maxsize=64)
def file_at(commit: str, path: str) -> str | None:
    try:
        return _git("show", f"{commit}:{path}").decode("utf-8").replace("\r\n", "\n")
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        return None


# --- extraction (parsing, no LLM) ----------------------------------------------

NUM = r"(-?\d+(?:\.\d*)?)f?"
PARAM_HEAD = re.compile(
    r"make_unique\s*<\s*(?:juce::)?AudioParameter(Float|Bool|Choice|Int)\s*>\s*\(\s*"
    r"(?:juce::)?ParameterID\s*\{\s*[^{}\n]+?\s*,\s*\d+\s*\}\s*,\s*\"([^\"\n]+)\"\s*,\s*")
PARAM_TAIL = {
    "Float": re.compile(r"(?:juce::)?NormalisableRange\s*<\s*float\s*>\s*\(\s*" + NUM + r"\s*,\s*"
                        + NUM + r"\s*(?:,[^()]*)?\)\s*,\s*" + NUM + r"\s*[,)]"),
    "Int": re.compile(NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*[,)]"),
    "Bool": re.compile(r"(true|false)\s*[,)]"),
    "Choice": re.compile(r"(?:juce::)?StringArray\s*\{([^{}]*)\}\s*,\s*(\d+)\s*[,)]"),
}
# a data member line: [std::vector|unique_ptr|array<] ns::Type[<...>] [, N>] name[, name2] {=;[
MEMBER_RE = re.compile(
    r"(?m)^[ \t]*(?:std::(?:vector|unique_ptr|array)[ \t]*<[ \t]*)?(?:\w+::)*(\w+)[ \t]*"
    r"(?:<[^;\n]*?>)?[ \t]*(?:,[ \t]*\w+[ \t]*)?>?[ \t]+(\w+(?:[ \t]*,[ \t]*\w+)*)[ \t]*[{=;\[]")
CLASS_RE = re.compile(
    r"(?m)^[ \t]*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+(\w+)\s*(?:final\s*)?"
    r"(?::\s*([^{;()]*?))?\s*\{")


def strip_comments(text: str) -> str:
    """Blank out // line comments and /* */ blocks, keeping line numbers."""
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"(?m)//.*$", "", text)


def fmt_num(s: str) -> str:
    x = float(s)
    return str(int(x)) if x == int(x) else repr(x)


def parse_params(text: str) -> list[dict]:
    """Every parameter whose display name and default are literals."""
    code = strip_comments(text)
    out = []
    for m in PARAM_HEAD.finditer(code):
        kind, name = m.group(1), m.group(2)
        t = PARAM_TAIL[kind].match(code, m.end())
        if not t:
            continue
        line = code.count("\n", 0, m.start(2)) + 1
        p = {"name": name, "kind": kind, "line": line}
        if kind in ("Float", "Int"):
            p.update(lo=fmt_num(t.group(1)), hi=fmt_num(t.group(2)), default=fmt_num(t.group(3)))
        elif kind == "Bool":
            p["default"] = t.group(1)
        else:
            items = re.findall(r'"([^"]*)"', t.group(1))
            idx = int(t.group(2))
            if idx >= len(items):
                continue
            p["default"] = items[idx]
        out.append(p)
    return out


def parse_bases(s: str | None) -> list[str]:
    if not s:
        return []
    out = []
    for b in s.split(","):
        b = re.sub(r"\b(public|private|protected|virtual)\b", "", b).strip()
        if b:
            out.append(re.sub(r"\s+", "", b))
    return out


def _plugin_of(path: str) -> str | None:
    p = PurePosixPath(path).parts
    return p[1] if p[0] == "plugins" and len(p) > 1 else None


@lru_cache(maxsize=64)
def facts(commit: str, side: str) -> dict:
    files = dict(snapshot(commit, side))
    params: dict[str, list[dict]] = {}
    classes: dict[str, list[dict]] = {}
    for path, text in files.items():
        pl = _plugin_of(path)
        if pl and path.endswith(".cpp"):
            for p in parse_params(text):
                params.setdefault(pl, []).append({**p, "file": path})
        code = strip_comments(text)
        for m in CLASS_RE.finditer(code):
            classes.setdefault(m.group(1), []).append(
                {"file": path, "line": code.count("\n", 0, m.start(1)) + 1,
                 "bases": parse_bases(m.group(2))})
    for pl, ps in params.items():                        # ambiguous names are not questions
        names = [p["name"] for p in ps]
        params[pl] = [p for p in ps if names.count(p["name"]) == 1]
    unique = {c: d[0] for c, d in classes.items() if len(d) == 1}
    members: dict[str, list[dict]] = {}
    for path, text in files.items():
        pl = _plugin_of(path)
        if not pl or PurePosixPath(path).name != "PluginProcessor.h":
            continue
        code = strip_comments(text)
        for m in MEMBER_RE.finditer(code):
            c, names = m.group(1), [n.strip() for n in m.group(2).split(",")]
            if c not in unique or c in ("PluginProcessor", "PluginEditor") or len(c) < 4:
                continue
            for mem in names:
                if len(re.findall(rf"\b{mem}\b\s*[,{{=;\[]", code)) != 1:
                    continue
                members.setdefault(pl, []).append(
                    {"member": mem, "type": c, "file": path,
                     "line": code.count("\n", 0, m.start(1)) + 1})
    return {"files": files, "params": params, "classes": classes, "unique": unique,
            "members": members}


def users_of(f: dict, cls: str) -> tuple[set[str], int]:
    """(plugins whose code names cls, total matching lines across plugins/)."""
    users, hits = set(), 0
    for path, text in f["files"].items():
        pl = _plugin_of(path)
        if not pl:
            continue
        n = sum(1 for ln in text.split("\n") if re.search(rf"\b{cls}\b", ln))
        if n:
            users.add(pl)
            hits += n
    return users, hits


# --- questions --------------------------------------------------------------------

def _ans(fmt: str) -> str:
    return f"\nEnd with one line: ANSWER: {fmt}"


def t1_candidates(f: dict, plugin: str) -> list[dict]:
    out = []
    for p in f["params"].get(plugin, []):
        if p["name"] in GUESSABLE:
            continue
        unit = {"Bool": "true or false", "Choice": "the choice text exactly as written in the code"
                }.get(p["kind"], "the number only")
        out.append({"kind": "default", "gt": p["default"], "entities": [plugin, p["name"]],
                    "plugin": plugin, "param": p["name"],
                    "q": f'In the {plugin} plugin, what is the default value of the parameter '
                         f'whose display name is "{p["name"]}"?' + _ans(f"<{unit}>")})
        if p["kind"] in ("Float", "Int"):
            out.append({"kind": "max", "gt": p["hi"], "entities": [plugin, p["name"]],
                        "plugin": plugin, "param": p["name"],
                        "q": f'In the {plugin} plugin, what is the maximum of the range of the '
                             f'parameter whose display name is "{p["name"]}"?' + _ans("<number>")})
    for c, d in sorted(f["unique"].items()):
        if _plugin_of(d["file"]) != plugin or c in ("PluginProcessor", "PluginEditor") or len(c) < 5:
            continue
        out.append({"kind": "deffile", "gt": d["file"], "entities": [plugin, c], "plugin": plugin,
                    "cls": c,
                    "q": f"Which file in this workspace defines the class or struct `{c}` "
                         f"(used by the {plugin} plugin)?"
                         + _ans("<path relative to the workspace root, e.g. plugins/X/source/Y.h>")})
        if d["bases"]:
            out.append({"kind": "base", "gt": ", ".join(d["bases"]), "entities": [plugin, c],
                        "plugin": plugin, "cls": c,
                        "q": f"In the {plugin} plugin, which class(es) does `{c}` directly "
                             "inherit from?" + _ans("<base class, or a comma-separated list>")})
    engine = [c for c, d in f["unique"].items() if d["file"].startswith("engine/")]
    for c in sorted(engine):
        users, hits = users_of(f, c)
        if plugin in users and len(users) >= 2 and hits <= 30 and len(c) >= 5:
            out.append({"kind": "users", "gt": ", ".join(sorted(users)), "entities": [c] + sorted(users),
                        "plugin": plugin, "cls": c,
                        "q": f"Which plugins under plugins/ use the vstkit class `{c}` "
                             "(mention it anywhere in their code files)?"
                             + _ans("<comma-separated plugin directory names>")})
    return out


def _param_names_same_line(text: str) -> list[str] | None:
    """Display names, provided every ParameterID line carries its name (cheap to search)."""
    names = []
    for ln in strip_comments(text).split("\n"):
        if "ParameterID" in ln:
            m = re.search(r"ParameterID\s*\{[^{}]*\}\s*,\s*\"([^\"]+)\"", ln)
            if not m:
                return None
            names.append(m.group(1))
    return names if len(set(names)) == len(names) else None


def t2_candidates(f: dict, plugin: str, commit: str) -> list[dict]:
    out = []
    for m in f["members"].get(plugin, []):
        d = f["unique"][m["type"]]
        out.append({"kind": "memberfile", "gt": d["file"], "plugin": plugin,
                    "entities": [plugin, m["member"], m["type"]], "member": m["member"],
                    "q": f"The {plugin} plugin's PluginProcessor class has a data member named "
                         f"`{m['member']}`. Which file defines that member's type (the class "
                         "it holds, ignoring std:: wrappers)?"
                         + _ans("<path relative to the workspace root>")})
        if d["bases"]:
            out.append({"kind": "memberbase", "gt": ", ".join(d["bases"]), "plugin": plugin,
                        "entities": [plugin, m["member"], m["type"]], "member": m["member"],
                        "q": f"The {plugin} plugin's PluginProcessor class has a data member named "
                             f"`{m['member']}`. Which class(es) does that member's type (the "
                             "class it holds, ignoring std:: wrappers) directly inherit from?" + _ans("<base class, or a comma-separated list>")})
        if d["file"].startswith("engine/"):          # 3 hops: member -> type -> its users
            users, hits = users_of(f, m["type"])
            if len(users) >= 2 and hits <= 30:
                out.append({"kind": "memberusers", "gt": ", ".join(sorted(users)), "plugin": plugin,
                            "entities": [plugin, m["member"], m["type"]] + sorted(users),
                            "member": m["member"],
                            "q": f"The {plugin} plugin's PluginProcessor class has a data member named "
                                 f"`{m['member']}`. Which plugins under plugins/ use that member's type "
                                 "(the class it holds, ignoring std:: wrappers; mentioned anywhere in "
                                 "their code files)?" + _ans("<comma-separated plugin directory names>")})
    # history: previous commit that changed the processor, diffed by parameter names
    for path in sorted(f["files"]):
        if _plugin_of(path) != plugin or not path.endswith("PluginProcessor.cpp"):
            continue
        new = f["files"][path]
        nn = _param_names_same_line(new)
        if nn is None or not _has_parent(commit):
            continue
        older = _git("log", "--format=%H", f"{commit}~1", "--", path).decode().split()
        prev, old, added = None, None, []
        for c in older:                       # newest older version with a different set
            old = file_at(c, path)
            on = _param_names_same_line(old) if old else None
            if on is None:
                break
            if set(on) != set(nn):
                prev, added = c, sorted(set(nn) - set(on))
                break
        if not prev or not 1 <= len(added) <= 4:
            continue
        hist = f"history/{prev[:7]}/{path}"
        out.append({"kind": "history", "gt": ", ".join(added), "plugin": plugin,
                    "entities": [plugin] + added, "hist": hist, "hist_text": old, "path": path,
                    "q": f"`{hist}` is an older copy of `{path}`. Which parameters (by display "
                         "name) does the current file add that the older copy does not have?"
                         + _ans("<comma-separated display names>")})
    return out


@lru_cache(maxsize=256)
def _has_parent(commit: str) -> bool:
    return len(_git("rev-list", "--parents", "-n", "1", commit).decode().split()) > 1


# --- specs ------------------------------------------------------------------------

PREAMBLE = ("This workspace is a read-only snapshot of a real JUCE audio-plugin repo: "
            "plugins/<Name>/ holds each plugin, engine/vstkit/ the shared library. "
            "Answer from the code: look things up with the tools, don't guess.\n\nQuestion: ")
# Holdout items are hand-picked (few good): varied answer types, non-guessable from names.
HOLDOUT_PICK = {
    9301: {1: ("default", "Side Hi"), 2: ("memberfile", "engine")},
    9302: {1: ("max", "Drive"), 2: ("memberfile", "shapers")},
    9303: {1: ("default", "Focus"), 2: ("memberfile", "generators")},
    9304: {1: ("users", "StationaryMaterialDial"), 2: ("history", "PluginProcessor.cpp")},
    9305: {1: ("base", "RaceLookAndFeel"), 2: ("memberfile", "wow")},
}


def _key(c: dict) -> str:
    return c.get("param") or c.get("cls") or c.get("member") or c["path"].rsplit("/", 1)[-1]


def _root(seed: int) -> tuple[str, str, str]:
    """(side, plugin, commit) for a seed."""
    if seed in HOLDOUT_SEEDS:
        return "holdout", HOLDOUT_PLUGINS[seed - HOLDOUT_SEEDS[0]], PIN
    if seed in RESERVED:
        raise ValueError(f"seed {seed} is reserved for repo_q holdouts")
    rng = random.Random(seed * 31 + 7)
    plugin = rng.choice(TRAIN_PLUGINS)
    commits = plugin_commits(plugin)
    return "train", plugin, (rng.choice(commits) if commits else PIN)


def _pick(cands: list[dict], seed: int, tier: int, holdout: bool) -> dict | None:
    if not cands:
        return None
    if holdout:
        kind, key = HOLDOUT_PICK[seed][tier]
        pool = [c for c in cands if c["kind"] == kind and _key(c) == key]
        if len(pool) != 1:
            raise AssertionError(f"holdout {seed} t{tier}: {kind} {key} not derivable -- repo drift?")
        return pool[0]
    rng = random.Random(seed * 101 + tier)
    kinds = sorted({c["kind"] for c in cands})
    k = rng.choice(kinds)
    return rng.choice([c for c in cands if c["kind"] == k])


@lru_cache(maxsize=1)
def _holdout_entities() -> frozenset[str]:
    ents = set(HOLDOUT_PLUGINS)
    for s in HOLDOUT_SEEDS:
        for sp in _raw_specs(s):
            ents |= set(sp["entities"])
    return frozenset(ents)


@lru_cache(maxsize=512)
def _raw_specs(seed: int) -> tuple[dict, ...]:
    side, plugin, commit = _root(seed)
    f = facts(commit, side)
    holdout = side == "holdout"
    banned = frozenset() if holdout else _holdout_entities()
    out = []
    for tier, cands in ((1, t1_candidates(f, plugin)), (2, t2_candidates(f, plugin, commit))):
        cands = [c for c in cands if not (set(c["entities"]) & banned)]
        c = _pick(cands, seed, tier, holdout)
        if c:
            out.append({**c, "tier": tier, "seed": seed, "side": side, "commit": commit})
    return tuple(out)


def specs(seed: int) -> list[dict]:
    out = []
    for c in _raw_specs(seed):
        side, commit = c["side"], c["commit"]
        files = dict(snapshot(commit, side))
        if c["kind"] == "history":
            files[c["hist"]] = c["hist_text"]
        out.append({"family": FAMILY, "seed": seed, "files": files, "tools": list(TOOLS),
                    "name": f"repoq-t{c['tier']}-{c['kind']}-{seed}",
                    "tier": c["tier"], "kind": c["kind"], "commit": commit,
                    "plugin": c["plugin"], "entities": list(c["entities"]),
                    "oracle": {k: c[k] for k in ("member", "param", "cls", "hist", "path")
                               if k in c},
                    "ground_truth": c["gt"], "prompt": PREAMBLE + c["q"]})
    return out


def lookup_items(seed: int) -> list[tuple[str, str, dict[str, str]]]:
    """(question, ground_truth, workspace files) per item -- the lookup half that other
    families (e.g. gather_produce) reuse. The question is the full prompt; files map a
    workspace-relative posix path to its text. Reserved seeds 9300-9399 other than
    HOLDOUT_SEEDS raise (same rule as specs); train callers must avoid HOLDOUT_SEEDS."""
    return [(sp["prompt"], sp["ground_truth"], sp["files"]) for sp in specs(seed)]


# --- grader -----------------------------------------------------------------------

def _canon(tok: str) -> str:
    t = tok.strip().strip("`'\"").strip().replace("\\", "/")
    if t.startswith("./"):
        t = t[2:]
    if "::" in t and "/" not in t:
        t = t.split("::")[-1]
    return t.lower().rstrip(".")


def _num(s: str) -> float | None:
    try:
        return float(s.strip().rstrip("f"))
    except ValueError:
        return None


def grade(answer: str | None, ground_truth: str) -> tuple[bool, str]:
    """Normalized exact match; comma lists as sets; numbers by value."""
    if answer is None:
        return False, "no-answer-line"
    if "," in ground_truth:
        want = {_canon(x) for x in ground_truth.split(",")}
        got = {_canon(x) for x in answer.split(",") if x.strip()}
        return got == want, f"pred={answer[:80]}"
    a, g = _num(answer), _num(ground_truth)
    if a is not None and g is not None:
        return abs(a - g) < 1e-9, f"pred={answer[:80]}"
    return _canon(answer) == _canon(ground_truth), f"pred={answer[:80]}"


# --- oracle: answer from the workspace through the agent's own tools ---------------

def _hits(out: str) -> list[tuple[str, int, str]]:
    res = []
    for ln in out.split("\n"):
        m = re.match(r"^(.+?):(\d+):(.*)$", ln)
        if m:
            res.append((m.group(1).replace("\\", "/"), int(m.group(2)), m.group(3)))
    return res


def _body(read_out: str) -> str:
    return "\n".join(re.sub(r"^\s*\d+: ", "", ln) for ln in read_out.split("\n")[1:])


def oracle(spec: dict, ws) -> tuple[str, int, int]:
    """(answer, tool calls, observation chars) using ws.search / ws.read_file only."""
    calls, chars = 0, 0

    def call(fn, *a, **k):
        nonlocal calls, chars
        out = fn(*a, **k)
        calls += 1
        chars += len(out)
        return out

    o, P, kind = spec["oracle"], spec["plugin"], spec["kind"]

    def def_of(cls: str) -> tuple[str, list[str]]:
        hs = [h for h in _hits(call(ws.search, rf"^\s*(class|struct)\s+{cls}\b"))
              if "{" in h[2] or not h[2].rstrip().endswith(";")]       # skip forward decls
        (path, line, _), = hs
        text = _body(call(ws.read_file, path, offset=line, limit=6))
        m = re.match(rf"\s*(?:class|struct)\s+{cls}\s*(?:final\s*)?(?::\s*([^{{;()]*?))?\s*\{{", text)
        return path, parse_bases(m.group(1) if m else None)

    if kind in ("default", "max"):
        hs = [h for h in _hits(call(ws.search, f'"{o["param"]}"', f"plugins/{P}"))
              if "ParameterID" in h[2]]
        (path, line, _), = hs
        text = _body(call(ws.read_file, path, offset=max(1, line - 1), limit=6))
        i = text.find("make_unique")
        p, = parse_params(text[i:])[:1]
        return (p["default"] if kind == "default" else p["hi"]), calls, chars
    if kind == "deffile":
        return def_of(o["cls"])[0], calls, chars
    if kind == "base":
        return ", ".join(def_of(o["cls"])[1]), calls, chars
    if kind == "users":
        users = sorted({h[0].split("/")[1] for h in _hits(call(ws.search, rf"\b{o['cls']}\b", "plugins"))})
        return ", ".join(users), calls, chars
    if kind in ("memberfile", "memberbase", "memberusers"):
        mem = o["member"]
        decls = [d for h in _hits(call(ws.search, rf"\b{mem}\b\s*[,{{=;\[]", f"plugins/{P}"))
                 if h[0].endswith("PluginProcessor.h") for d in [MEMBER_RE.match(h[2])] if d]
        decl, = decls                                    # comment lines don't parse as decls
        typ = decl.group(1)
        if kind == "memberusers":
            hits = _hits(call(ws.search, rf"\b{typ}\b", "plugins"))
            return ", ".join(sorted({h[0].split("/")[1] for h in hits})), calls, chars
        path, bases = def_of(typ)
        return (path if kind == "memberfile" else ", ".join(bases)), calls, chars
    if kind == "history":
        def names(d: str, glob: str) -> set[str]:
            return {m.group(1) for h in _hits(call(ws.search, "ParameterID", d, glob))
                    for m in [re.search(r"ParameterID\s*\{[^{}]*\}\s*,\s*\"([^\"]+)\"", h[2])] if m}
        new = names(str(PurePosixPath(o["path"]).parent), "PluginProcessor.cpp")
        old = names(str(PurePosixPath(o["hist"]).parent), "PluginProcessor.cpp")
        return ", ".join(sorted(new - old)), calls, chars
    raise ValueError(kind)
