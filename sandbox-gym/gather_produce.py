"""Family gather_produce: look several things up, then write the findings into a file.

Lookup half = a pluggable LookupSource (SOURCES). Today: "repo_q" -- the same
generators that back repo_q.lookup_items (snapshot / facts / t1_candidates /
t2_candidates / _holdout_entities over the user's plugin repo, read-only).
lookup_items(seed) itself yields one item per tier per seed; a gather task needs
2-4 lookups over ONE workspace, so the source draws several candidates from the
same (side, plugin, commit) root instead. A web_q source (frozen corpus) can be
registered later: implement `bundle(seed, tier) -> Bundle` + `holdout_entities()`.

Split rule (kept from repo_q, root = plugin directory):
  holdout  seeds HOLDOUT_SEEDS (9501-9505, reserved 9500-9549): repo_q holdout side
           (Corona/Ember/Nadir/Polaris/Redshift @ repo_q.PIN), seed i -> holdout plugin i.
  train    any other seed: repo_q._root(seed) (a train plugin at one of its commits;
           refuses repo_q's reserved 9300-9399). Train items touching any holdout
           entity (repo_q's or ours) are dropped.

Produce half: she writes FINDINGS.md:

  ## Q1
  Result: <answer>
  Evidence: <workspace path>[, <path>...]

and ends with `ANSWER: done`. The line is `Result:`, not `Answer:`: agent.py ends the
episode on graders.ANSWER_RE (case-insensitive `answer:` ANYWHERE in the reply), which
would fire on a write_file call whose content holds `Answer:` before the file is written.
Grade (binary, mechanical, no LLM):
  file exists; a `## Qi` section per question; exactly one Result line each that
  matches ground truth under TOLERANT normalization (match()); >=1 evidence path
  each, every cited path is a file of the ORIGINAL workspace (not one she wrote)
  whose text supports the answer (item["ev"]); no extra answered sections.
  Pass = all items correct. Partial credit (items fully right / n) is logged to
  data/results/gather_produce.partial.jsonl for analysis only -- never the reward.

Tiers:  t1  2 lookups, 1-hop.          oracle <= 4 tool calls + write + answer = 6
        t2  3-4 lookups incl. 1-2 2-hop (repo_q t2 kinds). oracle <= 12 + 2 = 14
Tools: list_dir, read_file, search, write_file (write_file creates new files only:
one shot at FINDINGS.md).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

import repo_q as R

FAMILY = "gather_produce"
TOOLS = ["list_dir", "read_file", "search", "write_file"]
FINDINGS = "FINDINGS.md"
HOLDOUT_SEEDS = tuple(range(9501, 9506))
RESERVED = range(9500, 9550)                       # never train seeds
# Frozen holdout picks (reviewed: varied kinds, no guessable answers). A mismatch means
# the repo or the picker drifted -- reseal from fresh seeds, never silently rescore.
HOLDOUT_KEYS = {
    (9501, 1): ('default:Side Lo', 'deffile:Control'),
    (9501, 2): ('default:Bass Mono', 'deffile:Assets', 'memberfile:engine', 'users:SvfFilter'),
    (9502, 1): ('users:EclipseDial', 'deffile:Canvas'),
    (9502, 2): ('history:PluginProcessor.cpp', 'users:AsymPoly', 'deffile:BiasStrip',
                'memberfile:toneFilters'),
    (9503, 1): ('default:Focus', 'deffile:AnalysisBank'),
    (9503, 2): ('users:AsymPoly', 'memberfile:splitters', 'default:Character'),
    (9504, 1): ('default:Key', 'max:Retune'),
    (9504, 2): ('memberfile:corrector', 'history:PluginProcessor.cpp', 'default:Retune',
                'users:ScaledLookAndFeel'),
    (9505, 1): ('users:ScaledLookAndFeel', 'base:RaceLookAndFeel'),
    (9505, 2): ('base:RaceSwitch', 'max:TruBass', 'deffile:Preset', 'memberfile:wow'),
}
TIER_N = {1: (2,), 2: (3, 4)}
GUESSABLE_GT = {"0", "1", "true", "false"}         # answers a guess hits too often
PARTIAL_LOG = Path(__file__).parent / "data" / "results" / "gather_produce.partial.jsonl"


# --- lookup sources --------------------------------------------------------------

@dataclass
class Bundle:
    """One task's lookups over one workspace."""
    files: dict[str, str]
    subject: str                                   # what the questions are about
    preamble: str
    items: list[dict] = field(default_factory=list)
    # item: {kind, hops, question, fmt, gt, ev, ev_paths, entities, oracle, key}
    #   ev       grader evidence spec: {"self": [paths], "all": [[alt, ...], ...]} where an
    #            alt is a regex string or {"num": float}; a cited file supports the answer
    #            if it is in "self" or every group has an alternative found in its text.
    #   ev_paths generator-known supporting files (oracle / ideal findings file).


class RepoQSource:
    name = "repo_q"

    def root(self, seed: int) -> tuple[str, str, str]:
        if seed in HOLDOUT_SEEDS:
            return "holdout", R.HOLDOUT_PLUGINS[seed - HOLDOUT_SEEDS[0]], R.PIN
        if seed in RESERVED:
            raise ValueError(f"seed {seed} is reserved for gather_produce holdouts")
        if seed in R.RESERVED:                     # _root would hand back repo_q's holdouts
            raise ValueError(f"seed {seed} is reserved for repo_q holdouts")
        return R._root(seed)

    @lru_cache(maxsize=256)
    def pool(self, seed: int) -> tuple[str, str, str, tuple[dict, ...], tuple[dict, ...]]:
        side, plugin, commit = self.root(seed)
        f = R.facts(commit, side)
        banned = frozenset() if side == "holdout" else (R._holdout_entities()
                                                       | self.holdout_entities())
        ok = lambda c: not (set(c["entities"]) & banned) and c["gt"].lower() not in GUESSABLE_GT
        c1 = [_item(c, f, 1) for c in R.t1_candidates(f, plugin) if ok(c)]
        c2 = [_item(c, f, 2) for c in R.t2_candidates(f, plugin, commit) if ok(c)]
        return side, plugin, commit, tuple(c1), tuple(c2)

    def bundle(self, seed: int, tier: int) -> Bundle | None:
        side, plugin, commit, c1, c2 = self.pool(seed)
        rng = random.Random(seed * 7919 + tier)
        n = rng.choice(TIER_N[tier])
        used = {k for t in range(1, tier) for k in _picked_keys(self, seed, t)}
        avail1 = [c for c in c1 if c["key"] not in used]
        avail2 = [c for c in c2 if c["key"] not in used]
        if tier == 1:
            picks = _spread(rng, avail1, n)
        else:
            if not avail2:
                return None
            k2 = min(len({c["key"] for c in avail2}), 2 if n == 4 else 1)
            two = _spread(rng, avail2, k2)
            keys = {c["key"] for c in two}
            picks = two + _spread(rng, [c for c in avail1 if c["key"] not in keys], n - k2)
        if len(picks) < min(TIER_N[tier]):
            return None
        rng.shuffle(picks)
        if side == "holdout" and tuple(c["key"] for c in picks) != HOLDOUT_KEYS[(seed, tier)]:
            raise AssertionError(f"gather holdout {seed} t{tier} drifted: "
                                 f"{[c['key'] for c in picks]}")
        files = dict(R.snapshot(commit, side))
        for c in picks:
            if c["kind"] == "history":
                files[c["oracle"]["hist"]] = c.pop("hist_text")
            c.pop("hist_text", None)
        return Bundle(files=files, subject=f"the {plugin} plugin", preamble=R.PREAMBLE.split(
            "\n\nQuestion:")[0], items=picks)

    @lru_cache(maxsize=1)
    def holdout_entities(self) -> frozenset[str]:
        ents: set[str] = set()
        for s in HOLDOUT_SEEDS:
            side, plugin, commit = self.root(s)
            f = R.facts(commit, side)
            for c in R.t1_candidates(f, plugin) + R.t2_candidates(f, plugin, commit):
                ents |= set(c["entities"])         # whole holdout pool: picks can't leak
        return frozenset(ents)


def _picked_keys(src, seed: int, tier: int) -> set[str]:
    b = src.bundle(seed, tier)
    return {c["key"] for c in b.items} if b else set()


def _spread(rng: random.Random, cands: list[dict], n: int) -> list[dict]:
    """n candidates with distinct keys, kinds spread round-robin."""
    by_kind: dict[str, list[dict]] = {}
    for c in cands:
        by_kind.setdefault(c["kind"], []).append(c)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    out, keys = [], set()
    while len(out) < n and any(by_kind.values()):
        for k in kinds:
            pool = [c for c in by_kind[k] if c["key"] not in keys]
            by_kind[k] = pool
            if pool and len(out) < n:
                c = dict(rng.choice(pool))
                out.append(c)
                keys.add(c["key"])
    return out


def _ident(cls: str) -> str:
    return re.split(r"<", cls)[0].split("::")[-1]


def _item(c: dict, f: dict, tier: int) -> dict:
    """repo_q candidate -> gather item (question text, format hint, evidence spec)."""
    q, fmt = c["q"].split("\nEnd with one line: ANSWER: ")
    fmt = fmt.strip("<>")
    kind, P = c["kind"], c["plugin"]
    oracle = {k: c[k] for k in ("member", "param", "cls", "hist", "path") if k in c}
    esc = re.escape
    if kind in ("default", "max"):
        p, = [p for p in f["params"][P] if p["name"] == c["param"]]
        val = c["gt"]
        alt = {"num": float(val)} if R._num(val) is not None and p["kind"] in ("Float", "Int") \
            else esc(f'"{val}"' if p["kind"] == "Choice" else val)
        ev, paths = {"self": [], "all": [[esc(f'"{c["param"]}"')], [alt]]}, [p["file"]]
    elif kind in ("deffile", "base"):
        d = f["unique"][c["cls"]]
        grp = [[rf"\b(class|struct)\s+{esc(c['cls'])}\b"]]
        if kind == "base":
            grp += [[rf"\b{esc(_ident(b))}\b"] for b in d["bases"]]
        ev = {"self": [d["file"]] if kind == "deffile" else [], "all": grp}
        paths = [d["file"]]
    elif kind == "users":
        ev, paths = {"self": [], "all": [[rf"\b{esc(c['cls'])}\b"]]}, [f["unique"][c["cls"]]["file"]]
    elif kind in ("memberfile", "memberbase", "memberusers"):
        m, = [m for m in f["members"][P] if m["member"] == c["member"]]
        d = f["unique"][m["type"]]
        typ = rf"\b{esc(m['type'])}\b"
        if kind == "memberfile":
            ev = {"self": [d["file"]], "all": [[rf"\b{esc(m['member'])}\b"], [typ]]}
        elif kind == "memberbase":
            ev = {"self": [], "all": [[typ]] + [[rf"\b{esc(_ident(b))}\b"] for b in d["bases"]]}
        else:
            ev = {"self": [], "all": [[typ]]}
        paths = [m["file"], d["file"]]
    elif kind == "history":
        ev = {"self": [c["path"], c["hist"]], "all": [[esc(f'"{n}"') for n in c["gt"].split(", ")]]}
        paths = [c["path"]]
    else:
        raise ValueError(kind)
    return {"kind": kind, "hops": tier, "question": q.strip(), "fmt": fmt, "gt": c["gt"],
            "ev": ev, "ev_paths": paths, "entities": list(c["entities"]), "oracle": oracle,
            "plugin": P, "key": f"{kind}:{R._key(c)}", "hist_text": c.get("hist_text")}


SOURCES = {"repo_q": RepoQSource()}


def source_for(seed: int) -> str:
    """Which lookup source a seed draws from. Only repo_q today; web_q plugs in here
    (e.g. a seed range of its own) once its corpus is verified."""
    return "repo_q"


# --- specs ------------------------------------------------------------------------

def prompt_for(b: Bundle) -> str:
    n = len(b.items)
    qs = "\n".join(f"Q{i}. {it['question']}\n    (format: {it['fmt']})"
                   for i, it in enumerate(b.items, 1))
    tpl = "\n\n".join(f"## Q{i}\nResult: <your answer>\nEvidence: <workspace-relative path of the "
                      "file(s) that show the answer, comma-separated>" for i in range(1, 3))
    return (f"{b.preamble}\n\nAnswer these {n} questions about {b.subject}, then write your "
            f"findings to {FINDINGS} in the workspace root.\n\n{qs}\n\n"
            f"{FINDINGS} must have one section per question, in this exact structure "
            f"(continue the pattern up to Q{n}):\n\n{tpl}\n\n"
            "Evidence must be files you read in this workspace that contain the answer. "
            "Write `Result:` exactly as shown: the word ANSWER followed by a colon ends the "
            "task at once, even inside the file content. "
            f"write_file creates {FINDINGS} once and cannot overwrite it, so write it when "
            "you have every answer. After writing it, end with one line: ANSWER: done")


@lru_cache(maxsize=256)
def _specs(seed: int) -> tuple[dict, ...]:
    src = SOURCES[source_for(seed)]
    out = []
    for tier in (1, 2):
        b = src.bundle(seed, tier)
        if b is None:
            continue
        items = [{k: it[k] for k in ("kind", "hops", "gt", "ev", "ev_paths", "oracle",
                                     "plugin", "key", "entities")} for it in b.items]
        out.append({"family": FAMILY, "seed": seed, "files": b.files, "tools": list(TOOLS),
                    "name": f"gather-t{tier}-n{len(items)}-{seed}", "tier": tier,
                    "source": src.name,
                    "ground_truth": " | ".join(f"Q{i}={it['gt']}" for i, it in enumerate(items, 1)),
                    "checks": [{"op": "gather", "items": [{k: it[k] for k in ("kind", "gt", "ev")}
                                                          for it in items]}],
                    "items": items, "prompt": prompt_for(b)})
    return tuple(out)


def specs(seed: int) -> list[dict]:
    return [dict(sp) for sp in _specs(seed)]


# --- tolerant answer matching ------------------------------------------------------

THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+)(?![\d,]*\d)")
NUMTOK = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE]-?\d+)?")
TRUE, FALSE = {"true", "yes", "on", "enabled"}, {"false", "no", "off", "disabled"}


def _clean(s: str) -> str:
    s = re.sub(r"[*`]+", "", s or "").strip()
    s = re.sub(r"\s*\((?:line|lines|see)[^)]*\)\s*$", "", s, flags=re.I)  # trailing "(line 12)"
    return s.strip().strip("\"'").strip().rstrip(".").strip()


def _canon(tok: str) -> str:
    t = re.sub(r"^(?:(?:public|private|protected|virtual|class|struct)\s+)+", "", _clean(tok),
               flags=re.I)
    t = t.replace("\\", "/")
    t = re.sub(r"^(?:\./|/)+", "", t)
    t = re.sub(r"^workspace/", "", t, flags=re.I)
    if "/" in t:
        t = re.sub(r":\d+(?:-\d+)?$", "", t)                                # path:line
    t = re.sub(r"\s+", " ", t)
    return R._canon(t)


def _split_list(s: str) -> list[str]:
    s = THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), _clean(s))
    parts = re.split(r"\s*(?:,|;|\n|\||\band\b|&)\s*", s, flags=re.I)
    return [p for p in (x.strip() for x in parts) if p]


def match(answer: str | None, gt: str) -> bool:
    """Tolerant: case, `A, B and C` == `A, B, C`, sets unordered, 7,000 == 7000 == 7000.0f,
    units/markdown/quotes ignored, ns::C == C, path variants. Builds on repo_q.grade."""
    if answer is None or not _clean(answer):
        return False
    if R.grade(answer, gt)[0] or R.grade(_clean(answer), gt)[0]:
        return True
    a = _clean(answer)
    if "," in gt:                                                       # set answers
        return {_canon(x) for x in _split_list(a)} == {_canon(x) for x in gt.split(",")}
    g = R._num(gt)
    if g is not None:                                                   # one number
        nums = {float(x) for x in NUMTOK.findall(THOUSANDS.sub(
            lambda m: m.group(1).replace(",", ""), a.replace("f", " ")))}
        return len(nums) == 1 and abs(nums.pop() - g) < 1e-9
    if gt.lower() in ("true", "false"):
        w = a.lower().split()[0] if a.split() else ""
        return (w in TRUE) if gt.lower() == "true" else (w in FALSE)
    return _canon(a) == _canon(gt)


# --- findings file -----------------------------------------------------------------

HEAD = re.compile(r"(?m)^\s*#{1,6}\s*\**\s*Q\s*(\d+)\b.*$")
FIELD = re.compile(r"(?im)^\s*(?:[-*]\s*)?\**\s*(result|answer|evidence)\s*\**\s*:\s*\**\s*(.*)$")


def parse_findings(text: str) -> dict[int, dict]:
    """{qnum: {"answers": [...], "evidence": [paths]}} -- also catches duplicate sections."""
    heads = list(HEAD.finditer(text))
    out: dict[int, dict] = {}
    for i, h in enumerate(heads):
        body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        sec = out.setdefault(int(h.group(1)), {"answers": [], "evidence": [], "dup": False})
        if sec["answers"] or sec["evidence"]:
            sec["dup"] = True
        lines = body.split("\n")
        for j, ln in enumerate(lines):
            m = FIELD.match(ln)
            if not m:
                continue
            if m.group(1).lower() != "evidence":
                sec["answers"].append(m.group(2).strip())
                continue
            ev = [m.group(2)]
            for nxt in lines[j + 1:]:                                  # bullet list under Evidence:
                if re.match(r"^\s*[-*]\s+\S", nxt) and not FIELD.match(nxt):
                    ev.append(re.sub(r"^\s*[-*]\s+", "", nxt))
                else:
                    break
            for e in ev:
                for p in re.split(r"[,\s]+", _clean(e)):
                    p = _canon_path(p)
                    if p:
                        sec["evidence"].append(p)
    return out


def _canon_path(p: str) -> str:
    p = p.strip().strip("`*\"'()[]<>").replace("\\", "/")
    p = re.sub(r"^(?:\./|/)+", "", p)
    p = re.sub(r"^workspace/", "", p, flags=re.I)
    p = re.sub(r"(?::\d+(?:-\d+)?)+:?$", "", p).rstrip(".:;")
    return p if "." in PurePosixPath(p).name or "/" in p else ""


def supports(text: str, path: str, ev: dict) -> bool:
    if path in ev["self"]:
        return True
    nums = None
    for grp in ev["all"]:
        hit = False
        for alt in grp:
            if isinstance(alt, dict):
                if nums is None:
                    nums = {float(x) for x in NUMTOK.findall(R.strip_comments(text))}
                hit = any(abs(x - alt["num"]) < 1e-9 for x in nums)
            else:
                hit = re.search(alt, R.strip_comments(text)) is not None
            if hit:
                break
        if not hit:
            return False
    return True


def grade(findings_text: str | None, items: list[dict], pristine: dict[str, str]) -> dict:
    """Per-item verdicts. items: [{kind, gt, ev}]; pristine: the ORIGINAL workspace files."""
    n = len(items)
    rep = {"n": n, "ok": False, "correct": 0, "items": [], "reason": ""}
    if findings_text is None:
        rep["reason"] = "no-findings-file"
        return rep
    secs = parse_findings(findings_text)
    extra = sorted(k for k, s in secs.items() if k < 1 or k > n and s["answers"])
    for i, it in enumerate(items, 1):
        s = secs.get(i)
        if s is None:
            v = "missing-heading"
        elif s["dup"] or len(s["answers"]) > 1:
            v = "extra-answer"
        elif not s["answers"]:
            v = "no-answer-line"
        elif not match(s["answers"][0], it["gt"]):
            v = "wrong-answer"
        elif not s["evidence"]:
            v = "no-evidence"
        elif any(p not in pristine for p in s["evidence"]):
            v = "evidence-missing-path"
        elif not all(supports(pristine[p], p, it["ev"]) for p in s["evidence"]):
            v = "evidence-unsupported"
        else:
            v = "ok"
        rep["items"].append(v)
    rep["correct"] = sum(v == "ok" for v in rep["items"])
    if extra:
        rep["reason"] = f"extra-sections {extra}"
    else:
        bad = [f"Q{i}:{v}" for i, v in enumerate(rep["items"], 1) if v != "ok"]
        rep["ok"] = not bad
        rep["reason"] = "all-correct" if not bad else ",".join(bad)
    return rep


# --- reward adapter (graders.reward_for dispatches here) -----------------------------

@lru_cache(maxsize=64)
def _pristine(seed: int, name: str) -> dict[str, str]:
    for sp in _specs(seed):
        if sp["name"] == name:
            return sp["files"]
    raise KeyError(name)


def grade_episode(task, turns, ws) -> dict:
    from graders import last_answer, truncated_final
    items = task.meta["checks"][0]["items"]
    if truncated_final(turns):
        return {"ok": False, "reason": "truncated", "correct": 0, "n": len(items), "items": []}
    if last_answer(turns) is None:
        return {"ok": False, "reason": "no-answer-line", "correct": 0, "n": len(items), "items": []}
    f = ws.root / FINDINGS
    text = f.read_text(encoding="utf-8", errors="replace") if f.is_file() else None
    return grade(text, items, _pristine(task.meta["seed"], task.name))


def reward(task):
    def rw(turns, ws) -> float:
        rep = grade_episode(task, turns, ws)
        _log_partial(task, rep)
        return float(rep["ok"])
    return rw


def _log_partial(task, rep: dict) -> None:
    """Analysis only: items fully right / n + per-item verdict codes (no answers)."""
    path = os.environ.get("GATHER_PARTIAL_LOG", str(PARTIAL_LOG))
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 1), "name": task.name,
                                 "partial": round(rep["correct"] / max(1, rep["n"]), 3),
                                 "correct": rep["correct"], "n": rep["n"],
                                 "items": rep["items"], "reason": rep["reason"][:120]}) + "\n")
    except OSError:
        pass


# --- scripted oracle ----------------------------------------------------------------

def render_findings(answers: list[str], evidence: list[list[str]]) -> str:
    return "# Findings\n\n" + "\n\n".join(
        f"## Q{i}\nResult: {a}\nEvidence: {', '.join(e)}"
        for i, (a, e) in enumerate(zip(answers, evidence), 1)) + "\n"


def oracle(spec: dict, ws) -> tuple[str, int, int]:
    """Answer every lookup through ws.search/read_file (repo_q.oracle), write the findings
    file with ws.write_file. Returns (findings text, tool calls incl. write, obs chars)."""
    answers, calls, chars = [], 0, 0
    for it in spec["items"]:
        sub = {"oracle": it["oracle"], "plugin": it["plugin"], "kind": it["kind"]}
        a, c, ch = R.oracle(sub, ws)
        answers.append(a)
        calls, chars = calls + c, chars + ch
    text = render_findings(answers, [it["ev_paths"] for it in spec["items"]])
    ws.write_file(FINDINGS, text)
    return text, calls + 1, chars
