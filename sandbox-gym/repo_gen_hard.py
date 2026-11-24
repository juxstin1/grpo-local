"""Hard seeded workspaces (families find_h / edit_h / fix_h).

v1 (repo_gen.py) is one fixed 4-file shop project: base Qwen3.5-4B scores 100%
on find/edit, so every GRPO group is all-win and dead. v2 randomizes structure:

  - 5 domains, randomized package / subpackage / module / function names
  - 27-29 files, nested packages, so list_dir is not a map of the answer
  - settings resolved through profiles: defaults.py + an ACTIVE profile that
    overrides some constants + an INACTIVE decoy profile with other values
  - a 4-function call chain across 4 modules (top -> f3 -> f2 -> f1), each
    step applying one effective constant
  - decoys: legacy/ package redefines the chain functions with other math;
    plural-named live functions (apply_fee vs apply_fees) that search also hits
  - fix: the bug sits in a lower module or the settings wiring; the failing
    test only exercises the top function

Expected values are never simulated: the generated project is executed once
(our own code, host subprocess -- same as repo_gen.tests_pass) and its outputs
become the test expectations and the find ground truth.
"""

from __future__ import annotations

import ast
import json
import random
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from repo_gen import Project, fmt, write_files

DOMAINS = {
    "shop": {
        "pkgs": ("storefront", "cartly", "shopkit"), "entity": "Order",
        "consts": (("SERVICE_FEE", "add"), ("TAX_MULT", "mul"), ("ORDER_CAP", "cap"),
                   ("BULK_MIN", "thresh"), ("HANDLING_FEE", "add"), ("PROMO_MULT", "mul")),
        "funcs": ("apply_fee", "scale_tax", "cap_order", "bulk_discount",
                  "adjust_total", "price_line", "finalize_total", "quote_price"),
    },
    "fleet": {
        "pkgs": ("fleetops", "routeline", "haulage"), "entity": "Trip",
        "consts": (("FUEL_SURCHARGE", "add"), ("DISTANCE_MULT", "mul"), ("TRIP_CAP", "cap"),
                   ("LONG_HAUL_MIN", "thresh"), ("TOLL_FEE", "add"), ("NIGHT_MULT", "mul")),
        "funcs": ("add_surcharge", "scale_distance", "cap_trip", "long_haul_rate",
                  "route_cost", "leg_cost", "finalize_fare", "quote_trip"),
    },
    "clinic": {
        "pkgs": ("clinicore", "carepath", "medledger"), "entity": "Visit",
        "consts": (("COPAY", "add"), ("RISK_MULT", "mul"), ("CLAIM_CAP", "cap"),
                   ("SENIOR_MIN", "thresh"), ("LAB_FEE", "add"), ("AFTER_HOURS_MULT", "mul")),
        "funcs": ("add_copay", "apply_risk", "cap_claim", "senior_rate",
                  "visit_cost", "bill_line", "finalize_bill", "quote_visit"),
    },
    "payroll": {
        "pkgs": ("paydesk", "wagebook", "shiftly"), "entity": "Shift",
        "consts": (("SHIFT_BONUS", "add"), ("OVERTIME_MULT", "mul"), ("WAGE_CAP", "cap"),
                   ("TENURE_MIN", "thresh"), ("MEAL_ALLOWANCE", "add"), ("HOLIDAY_MULT", "mul")),
        "funcs": ("add_bonus", "apply_overtime", "cap_wage", "tenure_rate",
                  "shift_pay", "pay_line", "finalize_pay", "quote_shift"),
    },
    "library": {
        "pkgs": ("shelfwise", "loandesk", "stacks"), "entity": "Loan",
        "consts": (("LATE_FEE", "add"), ("DAMAGE_MULT", "mul"), ("FINE_CAP", "cap"),
                   ("GRACE_MIN", "thresh"), ("PROCESSING_FEE", "add"), ("RARE_MULT", "mul")),
        "funcs": ("add_late_fee", "apply_damage", "cap_fine", "grace_rate",
                  "loan_charge", "fine_line", "finalize_fine", "quote_loan"),
    },
}
SETTINGS_NAMES = ("settings", "config", "conf")
PROFILES = (("production", "staging"), ("live", "sandbox"), ("prod", "dev"),
            ("site_eu", "site_us"))
RULES_NAMES = ("rules", "policy", "pricing")
CORE_NAMES = ("core", "engine", "domain")
SERVICES_NAMES = ("services", "api", "handlers")
UTIL_NAMES = ("util", "common", "helpers")
LEGACY_NAMES = ("legacy", "old", "deprecated")
MODULES = ("fees", "rates", "limits", "adjust", "calc", "totals", "quotes", "steps",
           "charges", "base")

BUG_KINDS = ("flip_op", "wrong_const", "wrong_profile")


def _value(rng: random.Random, kind: str):
    if kind == "add":
        return rng.randint(2, 25)
    if kind == "mul":
        return rng.choice((0.5, 0.8, 1.25, 1.5, 2))
    if kind == "cap":
        return rng.randint(150, 400)
    return rng.randint(40, 120)                                   # thresh


def _op_body(kind: str, const: str, var: str) -> str:
    if kind == "add":
        return f"    return round({var} + {const}, 2)\n"
    if kind == "mul":
        return f"    return round({var} * {const}, 2)\n"
    if kind == "cap":
        return f"    return min({var}, {const})\n"
    return (f"    if {var} >= {const}:\n"
            f"        return round({var} * 0.9, 2)\n"
            f"    return {var}\n")


FLIP = {"add": (" + ", " - "), "mul": (" * ", " / "), "cap": ("min(", "max("),
        "thresh": (" >= ", " <= ")}


def _consts_block(values: dict) -> str:
    return "".join(f"{k} = {fmt(v) if isinstance(v, float) else v}\n" for k, v in values.items())


def _filler_funcs(rng: random.Random, n: int) -> str:
    pool = [
        ("_normalize", "def _normalize(value):\n    return float(value or 0)\n"),
        ("_is_positive", "def _is_positive(value):\n    return value is not None and value > 0\n"),
        ("_describe", 'def _describe(value):\n    """Human-readable amount."""\n'
                      '    return f"{value:.2f}"\n'),
        ("_clip_zero", "def _clip_zero(value):\n    return value if value > 0 else 0\n"),
    ]
    rng.shuffle(pool)
    return "".join(body + "\n\n" for _, body in pool[:n])


def _render(seed: int) -> tuple[dict[str, str], dict]:
    rng = random.Random(seed * 7919 + 17)
    dname = sorted(DOMAINS)[seed % len(DOMAINS)]
    D = DOMAINS[dname]
    P = rng.choice(D["pkgs"])
    S = rng.choice(SETTINGS_NAMES)
    active, inactive = rng.choice(PROFILES)
    if rng.random() < 0.5:
        active, inactive = inactive, active
    R, C, SV = rng.choice(RULES_NAMES), rng.choice(CORE_NAMES), rng.choice(SERVICES_NAMES)
    U, L = rng.choice(UTIL_NAMES), rng.choice(LEGACY_NAMES)
    entity = D["entity"]
    ent = entity.lower()

    consts = list(D["consts"])
    rng.shuffle(consts)
    chain_consts = consts[:4]                 # f1..f4
    filler_consts = consts[4:]
    defaults = {k: _value(rng, kind) for k, kind in D["consts"]}
    kind_of = dict(D["consts"])

    # active overrides 2 chain consts + 1 filler; inactive overrides 3 others
    over_chain = rng.sample([k for k, _ in chain_consts], 2)
    active_over = {k: _value(rng, kind_of[k]) for k in over_chain + [filler_consts[0][0]]}
    for k in active_over:                     # override must differ from default
        while active_over[k] == defaults[k]:
            active_over[k] = _value(rng, kind_of[k])
    inactive_over = {}
    for k in rng.sample([k for k, _ in D["consts"]], 3):
        v = _value(rng, kind_of[k])
        while v in (defaults[k], active_over.get(k)):
            v = _value(rng, kind_of[k])
        inactive_over[k] = v

    funcs = list(D["funcs"])
    rng.shuffle(funcs)
    chain = funcs[:4]                         # f1, f2, f3, top
    mods = rng.sample(MODULES, 6)
    locs = [f"{P}/{R}/{mods[0]}.py", f"{P}/{R}/{mods[1]}.py",
            f"{P}/{C}/{mods[2]}.py", f"{P}/{SV}/{mods[3]}.py"]
    dotted = [p[:-3].replace("/", ".") for p in locs]

    files: dict[str, str] = {}
    files["README.md"] = (f"# {P}\n\n{entity} {'costing' if dname != 'library' else 'fines'} "
                          f"service. Constants live in `{P}/{S}/`; the active profile is "
                          f"selected in `{P}/{S}/__init__.py`.\n")
    files[f"{P}/__init__.py"] = f'"""{P} package."""\n'
    files[f"{P}/{S}/__init__.py"] = (
        f'"""Effective settings: defaults, then the active profile on top."""\n'
        f"from .defaults import *  # noqa: F401,F403\n"
        f"from .{active} import *  # noqa: F401,F403\n")
    files[f"{P}/{S}/defaults.py"] = '"""Default values for every setting."""\n' + _consts_block(defaults)
    files[f"{P}/{S}/{active}.py"] = f'"""{active} overrides."""\n' + _consts_block(active_over)
    files[f"{P}/{S}/{inactive}.py"] = f'"""{inactive} overrides."""\n' + _consts_block(inactive_over)
    for sub in (R, C, SV, U, L, "models", "reports"):
        files[f"{P}/{sub}/__init__.py"] = ""

    # the live chain
    for i, (fname, (cname, kind)) in enumerate(zip(chain, chain_consts)):
        path = locs[i]
        head = f"from {P}.{S} import {cname}\n"
        if i > 0:
            head += f"from {dotted[i-1]} import {chain[i-1]}\n"
        body = f"def {fname}(amount):\n"
        var = "amount"
        if i > 0:
            body += f"    value = {chain[i-1]}(amount)\n"
            var = "value"
        body += _op_body(kind, cname, var)
        text = head + "\n\n" + _filler_funcs(rng, rng.randint(0, 3)) + body
        files[path] = files.get(path, "") + text

    # live plural decoys (search for f2 also hits f2+"s"), called by reports
    decoy_mod = f"{P}/{R}/{mods[4]}.py"
    dk, dkind = filler_consts[0]
    ek, ekind = filler_consts[1]
    files[decoy_mod] = (
        f"from {P}.{S} import {dk}, {ek}\n\n\n"
        f"def {chain[1]}s(amounts):\n"
        f'    """Batch variant: applies {dk} to each amount."""\n'
        f"    return [x for x in (_one(a) for a in amounts)]\n\n\n"
        f"def _one(amount):\n" + _op_body(dkind, dk, "amount").replace("\n    ", "\n    ", 1)
        + f"\n\ndef {chain[0]}s(amounts):\n"
        + f"    return [round(a, 2) for a in amounts]\n\n\n"
        f"def extra(amount):\n" + _op_body(ekind, ek, "amount"))
    files[f"{P}/reports/summary.py"] = (
        f"from {P}.{R}.{mods[4]} import {chain[1]}s\n"
        f"from {P}.{U}.text import money\n\n\n"
        f"def summarize(amounts):\n"
        f"    adjusted = {chain[1]}s(amounts)\n"
        f'    return {{"count": len(adjusted), "total": money(sum(adjusted))}}\n')
    files[f"{P}/reports/export.py"] = (
        "import csv\nimport io\n\n\n"
        "def to_csv(rows):\n    buf = io.StringIO()\n"
        "    writer = csv.writer(buf)\n    for row in rows:\n"
        "        writer.writerow(row)\n    return buf.getvalue()\n")

    # legacy decoys: same names as the live chain, different math, never imported
    legacy_path = f"{P}/{L}/{mods[5]}.py"
    lk = chain_consts[1][0]
    files[legacy_path] = (
        f'"""Pre-2024 costing. Kept for reference; not wired into {P}."""\n'
        f"from {P}.{S}.defaults import {lk}\n\n\n"
        + "".join(f"def {f}(amount):\n    return round(amount * 1.1 + {lk}, 2)\n\n\n"
                  for f in (chain[1], chain[2], chain[0])))
    files[f"{P}/{L}/README.md"] = f"Old code paths. Nothing in {P} imports this package.\n"

    # utilities + models filler
    files[f"{P}/{U}/text.py"] = (
        "def money(value):\n    return f\"{value:,.2f}\"\n\n\n"
        "def slug(text):\n    return \"-\".join(text.lower().split())\n")
    files[f"{P}/{U}/ids.py"] = (
        "import itertools\n\n_counter = itertools.count(1)\n\n\n"
        f"def next_id(prefix=\"{ent[:3]}\"):\n    return f\"{{prefix}}-{{next(_counter):05d}}\"\n")
    files[f"{P}/{U}/log.py"] = (
        "import logging\n\nlog = logging.getLogger(__name__)\n\n\n"
        "def debug(msg, *args):\n    log.debug(msg, *args)\n")
    if rng.random() < 0.6:
        files[f"{P}/{U}/mathx.py"] = (
            "def clamp(value, low, high):\n    return max(low, min(value, high))\n\n\n"
            "def pct(value, rate):\n    return round(value * rate / 100, 2)\n")
    files[f"{P}/models/{ent}.py"] = (
        "from dataclasses import dataclass, field\n\n\n"
        f"@dataclass\nclass {entity}:\n    id: str\n    amount: float = 0.0\n"
        "    tags: list = field(default_factory=list)\n")
    if rng.random() < 0.6:
        files[f"{P}/models/{ent}_line.py"] = (
            "from dataclasses import dataclass\n\n\n"
            f"@dataclass\nclass {entity}Line:\n    label: str\n    amount: float\n")

    meta = {
        "domain": dname, "pkg": P, "settings": S, "active": active, "inactive": inactive,
        "chain": chain, "locs": locs, "dotted": dotted,
        "chain_consts": [k for k, _ in chain_consts], "kinds": [k for _, k in chain_consts],
        "filler_consts": [k for k, _ in filler_consts],
        "defaults": defaults, "active_over": active_over, "inactive_over": inactive_over,
        "effective": {**defaults, **active_over},
        "legacy_path": legacy_path, "decoy_mod": decoy_mod,
    }
    return files, meta


def _run_top(files: dict[str, str], meta: dict, inputs: list[int]) -> list | None:
    """Execute the top chain function on inputs; None if it errors."""
    code = (f"import json\nfrom {meta['dotted'][3]} import {meta['chain'][3]} as f\n"
            f"print(json.dumps([f(x) for x in {inputs!r}]))\n")
    with tempfile.TemporaryDirectory(prefix="hgen_") as td:
        write_files(files, Path(td))
        r = subprocess.run([sys.executable, "-c", code], cwd=td, capture_output=True,
                           text=True, timeout=60)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def _tests(meta: dict, inputs: list[int], outputs: list) -> str:
    P, top, mod = meta["pkg"], meta["chain"][3], meta["dotted"][3]
    cases = "".join(
        f"def test_{top}_{i}():\n    assert abs({top}({x}) - {y!r}) < 1e-9, {top}({x})\n"
        for i, (x, y) in enumerate(zip(inputs, outputs)))
    return (
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        f"from {mod} import {top}\n"
        f"from {P}.reports.summary import summarize\n\n\n"
        + cases +
        "def test_summary_count():\n    assert summarize([1, 2, 3])[\"count\"] == 3\n\n\n"
        "def _all():\n"
        "    for name in sorted(n for n in globals() if n.startswith('test_')):\n"
        "        globals()[name]()\n"
        "    print('ALL TESTS PASSED')\n\n\n"
        "if __name__ == '__main__':\n    _all()\n")


@lru_cache(maxsize=256)
def generate_hard(seed: int) -> Project:
    files, meta = _render(seed)
    rng = random.Random(seed * 31 + 5)
    pool = rng.sample(range(5, 400), 9)
    inputs, ask = pool[:8], pool[8]           # `ask` never appears in the tests
    outs = _run_top(files, meta, inputs + [ask])
    if outs is None:
        raise AssertionError(f"hard seed {seed}: generated project does not run")
    files["tests/test_core.py"] = _tests(meta, inputs, outs[:8])
    meta = {**meta, "inputs": inputs, "ask": ask, "ask_out": outs[8]}
    return Project(files=files, meta=meta)


def _line(text: str, needle: str) -> int:
    for i, ln in enumerate(text.split("\n"), 1):
        if needle in ln:
            return i
    raise AssertionError(needle)


# --- effective settings (static, for edit grading; never executes her code) --

def effective_settings(root: Path, pkg: str, settings: str) -> dict | None:
    """Resolve `pkg.settings` the way its __init__ does, by AST only: every
    `from .X import *` in order, each module's literal top-level assignments."""
    sdir = root / pkg / settings
    try:
        init = ast.parse((sdir / "__init__.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    out: dict = {}
    for node in init.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            try:
                mod = ast.parse((sdir / f"{node.module}.py").read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                return None
            for a in mod.body:
                if isinstance(a, ast.Assign) and len(a.targets) == 1 \
                        and isinstance(a.targets[0], ast.Name):
                    try:
                        out[a.targets[0].id] = ast.literal_eval(a.value)
                    except ValueError:
                        return None
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                return None
    return out


# --- bugs --------------------------------------------------------------------

def _apply_bug(project: Project, kind: str, pos: int) -> Project | None:
    m, files = project.meta, dict(project.files)
    if kind == "wrong_profile":
        p = f"{m['pkg']}/{m['settings']}/__init__.py"
        files[p] = files[p].replace(f"from .{m['active']} import", f"from .{m['inactive']} import")
        where = p
    else:
        path, fname = m["locs"][pos], m["chain"][pos]
        text = files[path]
        start = text.index(f"def {fname}(")
        # the bug goes in the CODE only: a planted docstring keeps stating intent
        doc_end = text.find('"""', text.find('"""', start) + 3) + 3 \
            if text.startswith('    """', text.index("\n", start) + 1) else start
        head, body = text[:doc_end], text[doc_end:]
        if kind == "flip_op":
            a, b = FLIP[m["kinds"][pos]]
            if a not in body:
                return None
            body = body.replace(a, b, 1)
        else:                                  # wrong_const: a real, similar constant
            good, bad = m["chain_consts"][pos], m["filler_consts"][1]
            head = head.replace(f"import {good}\n", f"import {bad}\n")
            body = body.replace(good, bad)
        files[path] = head + body
        where = path
    return Project(files=files, meta={**m, "bug": kind, "bug_file": where})


def _tests_pass(files: dict[str, str]) -> bool:
    from repo_gen import tests_pass
    return tests_pass(files)


FIX_GEN = 2   # fix_h v2: intent planted (docstrings + deployed profile in README)

_INTENT = {
    "add": "Add {c} to the amount.",
    "mul": "Scale the amount by {c}.",
    "cap": "Cap the amount at {c}: never return more than {c}.",
    "thresh": "Amounts at or above {c} get 10% off; smaller amounts pass through.",
}


def _with_intent(p: Project) -> Project:
    """Fix workspaces state what the code SHOULD do, so a bug contradicts
    something besides the test asserts (no reverse-engineering from expected
    values): each chain function gets a docstring naming its operation and
    constant, and the README names the deployed settings profile."""
    m, files = p.meta, dict(p.files)
    for i, (fname, cname, kind) in enumerate(zip(m["chain"], m["chain_consts"], m["kinds"])):
        doc = _INTENT[kind].format(c=cname)
        if i > 0:
            doc = f"Takes the result of {m['chain'][i-1]}(). " + doc
        line = f"def {fname}(amount):\n"
        files[m["locs"][i]] = files[m["locs"][i]].replace(
            line, f'{line}    """{doc}"""\n', 1)
    files["README.md"] += (f"\nDeployed profile: `{m['active']}`. `{m['inactive']}` is for "
                           "local experiments only and must never be the active import.\n")
    return Project(files=files, meta=m)


@lru_cache(maxsize=256)
def bugged(seed: int) -> Project:
    """First (kind, position) in a seed-rotated order that the guard accepts:
    original passes, bugged fails. The bug is never in the top function.
    The workspace carries planted intent (_with_intent); the bug edits code only."""
    base = _with_intent(generate_hard(seed))
    if not _tests_pass(base.files):
        raise AssertionError(f"hard seed {seed}: base fails its own tests")
    combos = [(k, p) for k in BUG_KINDS for p in (0, 1, 2)
              if not (k == "wrong_profile" and p)]
    r = seed % len(combos)
    for kind, pos in combos[r:] + combos[:r]:
        b = _apply_bug(base, kind, pos)
        if b is not None and not _tests_pass(b.files):
            return b
    raise AssertionError(f"hard seed {seed}: no bug breaks a test")


# --- task specs --------------------------------------------------------------

FIND_TOOLS = ["list_dir", "read_file", "search"]
EDIT_TOOLS = FIND_TOOLS + ["edit_file", "write_file"]
FIX_TOOLS = EDIT_TOOLS + ["build", "run_tests"]


def _find_specs(seed: int) -> list[dict]:
    p = generate_hard(seed)
    m = p.meta
    P, S, chain, locs = m["pkg"], m["settings"], m["chain"], m["locs"]
    oc = [k for k in m["chain_consts"] if k in m["active_over"]][0]
    f2, f3, top = chain[1], chain[2], chain[3]
    tpl = {"family": "find_h", "seed": seed, "files": p.files, "tools": FIND_TOOLS}
    return [
        {**tpl, "name": f"findh-effective-{seed}", "ground_truth": fmt(m["effective"][oc]),
         "prompt": f"What value does {oc} have at runtime, as seen by code that does "
                   f"`from {P}.{S} import {oc}`? Answer with the number only."},
        {**tpl, "name": f"findh-source-{seed}",
         "ground_truth": f"{P}/{S}/{m['active']}.py",
         "prompt": f"Which file supplies the runtime value of {oc}? Answer with the "
                   "path relative to the project root, e.g. pkg/sub/file.py."},
        {**tpl, "name": f"findh-def-{seed}",
         "ground_truth": f"{locs[1]}:{_line(p.files[locs[1]], f'def {f2}(')}",
         "prompt": f"`{top}` calls `{f2}` indirectly. Where is the `{f2}` that it "
                   "actually uses defined? Answer with path:line relative to the "
                   "project root, e.g. pkg/sub/file.py:12."},
        {**tpl, "name": f"findh-caller-{seed}", "ground_truth": f3,
         "prompt": f"Which function calls `{f2}` (exactly that name, not a similarly "
                   "named one)? Answer with the function name only."},
        {**tpl, "name": f"findh-return-{seed}", "ground_truth": fmt(m["ask_out"]),
         "prompt": f"What does `{top}({m['ask']})` return with the settings as they are "
                   "configured? Answer with the number only."},
    ]


def _edit_specs(seed: int) -> list[dict]:
    p = generate_hard(seed)
    m = p.meta
    P, S, eff = m["pkg"], m["settings"], m["effective"]
    rng = random.Random(seed * 13 + 1)
    kind_of = dict(zip(m["chain_consts"], m["kinds"]))
    oc = [k for k in m["chain_consts"] if k in m["active_over"]][0]
    nc = [k for k in m["chain_consts"] if k not in m["active_over"]][0]

    def new_val(k):
        v = _value(rng, kind_of[k])
        while v in (eff[k], m["defaults"][k], m["inactive_over"].get(k)):
            v = _value(rng, kind_of[k])
        return v

    v1, v2 = new_val(oc), new_val(nc)
    f2 = m["chain"][1]
    new_name = f"{f2}_step"
    others = lambda k: {c: v for c, v in eff.items() if c != k}  # noqa: E731
    tpl = {"family": "edit_h", "seed": seed, "files": p.files, "tools": EDIT_TOOLS}
    return [
        {**tpl, "name": f"edith-effective-{seed}", "ground_truth": f"{oc}={fmt(v1)}",
         "prompt": f"Make {oc} resolve to {fmt(v1)} at runtime (as imported from "
                   f"`{P}.{S}`). No other setting may change its runtime value.",
         "checks": [{"op": "effective", "pkg": P, "settings": S,
                     "want": {oc: v1}, "same": others(oc)}]},
        {**tpl, "name": f"edith-override-{seed}", "ground_truth": f"{nc}={fmt(v2)}",
         "prompt": f"Make {nc} resolve to {fmt(v2)} at runtime by overriding it in the "
                   f"active profile. Do not edit {P}/{S}/defaults.py.",
         "checks": [{"op": "effective", "pkg": P, "settings": S,
                     "want": {nc: v2}, "same": others(nc)},
                    {"op": "unchanged", "file": f"{P}/{S}/defaults.py",
                     "text": p.files[f"{P}/{S}/defaults.py"]}]},
        {**tpl, "name": f"edith-rename-{seed}", "ground_truth": f"{f2} -> {new_name}",
         "prompt": f"Rename the live function `{f2}` (the one the service code actually "
                   f"uses) to `{new_name}` and update every caller. Do not touch the "
                   f"{m['legacy_path'].split('/')[1]}/ package.",
         "checks": [{"op": "contains", "file": m["locs"][1], "needle": f"def {new_name}("},
                    {"op": "absent", "file": m["locs"][1], "needle": f"def {f2}("},
                    {"op": "contains", "file": m["locs"][2], "needle": f"{new_name}("},
                    {"op": "absent", "file": m["locs"][2], "needle": f"{f2}("},
                    {"op": "absent", "file": m["locs"][2], "needle": f"import {f2}\n"},
                    {"op": "unchanged", "file": m["legacy_path"],
                     "text": p.files[m["legacy_path"]]}]},
    ]


def _fix_specs(seed: int) -> list[dict]:
    b = bugged(seed)
    return [{
        "family": "fix_h", "seed": seed, "files": b.files, "tools": FIX_TOOLS,
        "name": f"fixh-{seed}",
        "ground_truth": f"bug {b.meta['bug']} in {b.meta['bug_file']} removed (gen{FIX_GEN})",
        "prompt": "Some tests in tests/test_core.py fail. Find the root cause in the "
                  "source code (not the tests) and fix it so every test passes. Run "
                  "the tests to verify before answering.",
    }]


SPECS = {"find_h": _find_specs, "edit_h": _edit_specs, "fix_h": _fix_specs}


def pristine_tests(seed: int) -> dict[str, str]:
    return {k: v for k, v in generate_hard(seed).files.items() if k.startswith("tests/")}


# --- editv_h: edit_h + a verification suite she can run -----------------------
#
# edit_h's workspace ships the project's normal tests, which a correct constant
# edit makes FAIL (they hard-code outputs under the old settings) -- so she can't
# verify and wanders to the turn cap. editv_h ships an acceptance suite built
# from the REFERENCE solution: it fails before her edit, passes after a correct
# one. Grading is still the static end-state checks (the suite is a tool, never
# the grade -- she can edit it). Separate family so the sealed edit_h holdout's
# workspaces never change.

EDITV_TOOLS = EDIT_TOOLS + ["run_tests"]


def _reference_edit(p: Project, spec: dict) -> dict[str, str]:
    m, files = p.meta, dict(p.files)
    P, S, act = m["pkg"], m["settings"], m["active"]
    prof = f"{P}/{S}/{act}.py"
    name = spec["name"]
    if name.startswith(("edith-effective", "edith-override")):
        k, v = spec["ground_truth"].split("=")
        lines = [ln for ln in files[prof].splitlines() if not ln.startswith(f"{k} =")]
        files[prof] = "\n".join(lines) + f"\n{k} = {v}\n"
    else:
        old, new = spec["ground_truth"].split(" -> ")
        for rel in (m["locs"][1], m["locs"][2]):
            files[rel] = files[rel].replace(f"{old}(", f"{new}(") \
                .replace(f"import {old}\n", f"import {new}\n")
    return files


def _acceptance(p: Project, spec: dict, solved: dict[str, str]) -> str:
    m = p.meta
    P, S = m["pkg"], m["settings"]
    outs = _run_top(solved, m, m["inputs"])
    if outs is None:
        raise AssertionError(f"{spec['name']}: reference solution does not run")
    suite = _tests(m, m["inputs"], outs)
    name = spec["name"]
    if name.startswith(("edith-effective", "edith-override")):
        k, v = spec["ground_truth"].split("=")
        extra = (f"def test_setting_{k.lower()}():\n"
                 f"    from {P}.{S} import {k}\n"
                 f"    assert abs({k} - {v}) < 1e-9, {k}\n\n\n")
    else:
        _, new = spec["ground_truth"].split(" -> ")
        extra = (f"def test_renamed():\n"
                 f"    from {m['dotted'][1]} import {new}  # noqa: F401\n\n\n")
    return suite.replace("def _all():", extra + "def _all():", 1)


@lru_cache(maxsize=256)
def _editv_cached(seed: int) -> tuple:
    p = generate_hard(seed)
    out = []
    for spec in _edit_specs(seed):
        solved = _reference_edit(p, spec)
        suite = _acceptance(p, spec, solved)
        files = {**p.files, "tests/test_core.py": suite}
        solved = {**solved, "tests/test_core.py": suite}
        # guard: unedited workspace fails the suite, the reference passes it
        if _tests_pass(files):
            raise AssertionError(f"{spec['name']}: acceptance suite passes before the edit")
        if not _tests_pass(solved):
            raise AssertionError(f"{spec['name']}: reference solution fails its own suite")
        out.append({**spec, "family": "editv_h", "files": files, "tools": EDITV_TOOLS,
                    "name": spec["name"].replace("edith-", "editvh-"),
                    "prompt": spec["prompt"] + " tests/test_core.py checks the requested "
                              "end state; run it to verify before answering."})
    return tuple(out)


def _editv_specs(seed: int) -> list[dict]:
    return [dict(s) for s in _editv_cached(seed)]


SPECS["editv_h"] = _editv_specs


# --- editv_m: middle ring (editv_h minus filler) ------------------------------
#
# "Never right -> make it easier": base is 100% on the 4-file v1 edit and 0% on
# editv_h (27-29 files, exploration eats the 10 turns). editv_m keeps every part
# of editv_h the task depends on (profiles, 4-module chain, legacy decoy,
# acceptance suite) and drops the filler nothing imports.

_MID_DROP = ("/ids.py", "/log.py", "/mathx.py", "/reports/export.py", "/models/")


def _mid_files(files: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in files.items()
            if not any(d in "/" + k for d in _MID_DROP)
            and not (k.endswith("README.md") and k.count("/") == 2)}   # legacy README


@lru_cache(maxsize=256)
def _editvm_cached(seed: int) -> tuple:
    out = []
    for spec in _editv_cached(seed):
        files = _mid_files(spec["files"])
        if _tests_pass(files):
            raise AssertionError(f"{spec['name']}: mid suite passes before the edit")
        out.append({**spec, "family": "editv_m", "files": files,
                    "name": spec["name"].replace("editvh-", "editvm-")})
    return tuple(out)


def _editvm_specs(seed: int) -> list[dict]:
    return [dict(s) for s in _editvm_cached(seed)]


SPECS["editv_m"] = _editvm_specs
