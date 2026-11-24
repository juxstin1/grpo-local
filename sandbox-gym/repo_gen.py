"""Seeded Python workspace generator (contamination-free by construction).

Each seed builds a small shop-inventory project (config + core + inventory +
tests) with randomized names and constants. Tasks reference *generated facts*
the model can't know in advance, so it must read files -- the Find skill.

Rings:
  find  tools 1-3   where is X defined / what value / who calls what / what returns
  edit  tools 4-5   make a specified change; graded on end state (content checks)
  fix   tools 8-10  injected bug; graded by the test suite (hidden to the agent)

Find-and-fix guard: a bug is only kept if the ORIGINAL project passes every
test AND the bugged project fails at least one -- otherwise "no change" would
pass. Enforced at generation time (runs the suite on both variants).

Split by seed: the eval/holdout uses a fixed seed set; training pools use other
seeds. A root is a seed -- never split a seed across both sides.
"""

from __future__ import annotations

import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# --- canonical formatting ---------------------------------------------------

def fmt(x) -> str:
    """Render a generated value the way the model is expected to answer."""
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".") if x == int(x) else f"{x:.2f}"
    return str(x)


CONFIG_KEYS = ("TAX_RATE", "MAX_ITEMS", "MIN_ORDER", "DISCOUNT_THRESHOLD")
FUNCS = ("apply_discount", "compute_total", "total_for_item", "valid_order")
TAXES = (0.03, 0.05, 0.07, 0.09)
THRESHOLDS = (50.0, 100.0, 200.0)
NAMES = ("Tea", "Coffee", "Juice", "Milk", "Bread", "Honey", "Olives", "Cheese")
PRICES = (3.0, 4.5, 7.5, 9.99, 12.99, 15.0, 20.0)

BUGS = ("flip_discount", "drop_null_check", "flip_min_order", "no_discount")


@dataclass
class Project:
    files: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _line(files: dict[str, str], path: str, needle: str) -> int:
    for i, ln in enumerate(files[path].split("\n"), 1):
        if needle in ln:
            return i
    raise AssertionError(f"{needle} not found in {path}")


def generate_project(seed: int) -> Project:
    """Deterministic project for a seed. Structure is fixed; names/constants vary."""
    rng = random.Random(seed)
    tax = TAXES[seed % len(TAXES)]
    thr = THRESHOLDS[(seed // 3) % len(THRESHOLDS)]
    max_items = 5 + (seed % 16)
    min_order = 1 + (seed % 5)
    items = rng.sample(list(zip(NAMES, PRICES)), 3)
    n1, p1 = items[0]

    config = (
        f"TAX_RATE = {tax}\n"
        f"MAX_ITEMS = {max_items}\n"
        f"DISCOUNT_THRESHOLD = {fmt(thr)}\n"
        f"MIN_ORDER = {min_order}\n"
    )
    inventory = (
        "CATALOG = [\n"
        + "".join(f'    {{"name": "{n}", "price": {fmt(p)}}},\n' for n, p in items)
        + "]\n\n\n"
        "def lookup(name):\n"
        '    for item in CATALOG:\n'
        '        if item["name"] == name:\n'
        '            return item["price"]\n'
        "    return None\n"
    )
    core = (
        "from config import DISCOUNT_THRESHOLD, MAX_ITEMS, MIN_ORDER, TAX_RATE\n"
        "from inventory import lookup\n"
        "\n"
        "\n"
        "def apply_discount(price):\n"
        "    if price >= DISCOUNT_THRESHOLD:\n"
        "        return round(price * 0.85, 2)\n"
        "    return round(price, 2)\n"
        "\n"
        "\n"
        "def compute_total(prices):\n"
        "    subtotal = sum(apply_discount(p) for p in prices)\n"
        "    return round(subtotal * (1 + TAX_RATE), 2)\n"
        "\n"
        "\n"
        "def total_for_item(name, qty):\n"
        "    price = lookup(name)\n"
        "    if price is None:\n"
        "        return 0.0\n"
        "    return round(price * qty, 2)\n"
        "\n"
        "\n"
        "def valid_order(quantity):\n"
        "    return quantity >= MIN_ORDER\n"
    )
    small = min(p for _, p in items)
    big = max(p for _, p in items)
    # exercise the discount path: a price at the threshold gets 15% off
    total_expected = fmt(round(
        sum(apply_discount_real(p, thr) for p in (small, big, thr)) * (1 + tax), 2))
    tests = (
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "from config import DISCOUNT_THRESHOLD, MIN_ORDER, TAX_RATE\n"
        "from core import apply_discount, compute_total, total_for_item, valid_order\n"
        "from inventory import lookup\n"
        "\n"
        "\n"
        f"def test_apply_discount_boundary():\n"
        f"    assert apply_discount(DISCOUNT_THRESHOLD) == round(DISCOUNT_THRESHOLD * 0.85, 2)\n"
        f"def test_apply_discount_small():\n"
        f"    assert apply_discount(1.0) == 1.0\n"
        f"def test_total_for_item_known():\n"
        f'    assert total_for_item("{n1}", 3) == round({p1} * 3, 2)\n'
        f"def test_total_for_item_missing():\n"
        f'    assert total_for_item("NoSuchItem", 2) == 0.0\n'
        f"def test_valid_order_boundary():\n"
        f"    assert valid_order(MIN_ORDER) is True\n"
        f"    assert valid_order(MIN_ORDER - 1) is False\n"
        f"def test_compute_total_known():\n"
        f"    assert compute_total([{fmt(small)}, {fmt(big)}, {fmt(thr)}]) == {total_expected}\n"
        f"\n"
        f"\n"
        f"def _all():\n"
        f"    for name in sorted(n for n in globals() if n.startswith('test_')):\n"
        f"        globals()[name]()\n"
        f"    print('ALL TESTS PASSED')\n"
        f"\n"
        f"if __name__ == '__main__':\n"
        f"    _all()\n"
    )
    files = {
        "config.py": config,
        "inventory.py": inventory,
        "core.py": core,
        "tests/test_core.py": tests,
    }
    meta = {
        "seed": seed,
        "config": {"TAX_RATE": tax, "MAX_ITEMS": max_items,
                   "DISCOUNT_THRESHOLD": thr, "MIN_ORDER": min_order},
        "lines": {f: _line(files, "core.py", f"def {f}") for f in
                  ("apply_discount", "compute_total", "total_for_item", "valid_order")},
        "catalog": items,
    }
    return Project(files=files, meta=meta)


def apply_discount_real(price, threshold) -> float:
    return round(price * 0.85, 2) if price >= threshold else round(price, 2)


def inject_bug(project: Project, seed: int) -> Project:
    """Return a bugged copy. Caller must verify the guard (original passes,
    bugged fails) with `tests_pass` before using it."""
    files = dict(project.files)
    core = files["core.py"]
    bug = BUGS[seed % len(BUGS)]
    if bug == "flip_discount":
        core = core.replace("if price >= DISCOUNT_THRESHOLD:", "if price > DISCOUNT_THRESHOLD:")
    elif bug == "drop_null_check":
        core = core.replace("    if price is None:\n        return 0.0\n", "")
    elif bug == "flip_min_order":
        core = core.replace("return quantity >= MIN_ORDER", "return quantity > MIN_ORDER")
    elif bug == "no_discount":
        core = core.replace("sum(apply_discount(p) for p in prices)",
                            "sum(p for p in prices)")
    files["core.py"] = core
    return Project(files=files, meta={**project.meta, "bug": bug})


def tests_pass(files: dict[str, str]) -> bool:
    """Run the generated suite (plain __main__ runner, no pytest needed)."""
    with tempfile.TemporaryDirectory(prefix="guard_") as td:
        root = Path(td)
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
        try:
            r = subprocess.run([sys.executable, "tests/test_core.py"], cwd=root,
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0 and "ALL TESTS PASSED" in r.stdout


# --- task specs --------------------------------------------------------------

def _find_specs(seed: int) -> list[dict]:
    p = generate_project(seed)
    m = p.meta
    cfg, lines, cat = m["config"], m["lines"], m["catalog"]
    tax, thr = cfg["TAX_RATE"], cfg["DISCOUNT_THRESHOLD"]
    low = int(thr) - 10                      # below threshold: unchanged
    high = int(thr)                          # at threshold: 15% off
    tpl = {
        "family": "find", "seed": seed, "files": p.files,
        "tools": ["list_dir", "read_file", "search"],
    }
    return [
        {**tpl, "name": f"find-config-TAX_RATE-{seed}", "ground_truth": fmt(tax),
         "prompt": "What is the value of the config constant TAX_RATE in this "
                   "project? Answer with the number only."},
        {**tpl, "name": f"find-config-MIN_ORDER-{seed}", "ground_truth": fmt(cfg["MIN_ORDER"]),
         "prompt": "What is the value of the config constant MIN_ORDER? Answer "
                   "with the number only."},
        {**tpl, "name": f"find-def-apply_discount-{seed}",
         "ground_truth": f"core.py:{lines['apply_discount']}",
         "prompt": "Where is the function apply_discount defined? Answer with "
                   "file:line, e.g. core.py:5."},
        {**tpl, "name": f"find-callers-lookup-{seed}",
         "ground_truth": "total_for_item",
         "prompt": "Which function in core.py calls lookup() from inventory.py? "
                   "Answer with the function name only."},
        {**tpl, "name": f"find-return-low-{seed}",
         "ground_truth": fmt(low),
         "prompt": f"What does apply_discount return for a price of {low}? "
                   "Answer with the number only."},
        {**tpl, "name": f"find-return-high-{seed}",
         "ground_truth": fmt(round(high * 0.85, 2)),
         "prompt": f"What does apply_discount return for a price of {high}? "
                   "Answer with the number only."},
        {**tpl, "name": f"find-threshold-{seed}",
         "ground_truth": fmt(thr),
         "prompt": "What is the smallest price at which apply_discount no longer "
                   "returns the input unchanged? Answer with the number only."},
    ]


def _edit_specs(seed: int) -> list[dict]:
    p = generate_project(seed)
    new_tax = [x for x in (0.04, 0.06, 0.08) if x != p.meta["config"]["TAX_RATE"]][seed % 3]
    tpl = {
        "family": "edit", "seed": seed, "files": p.files,
        "tools": ["list_dir", "read_file", "search", "edit_file", "write_file"],
    }
    rename = "apply_discount"
    new_name = "apply_reduction"
    return [
        {**tpl, "name": f"edit-tax-{seed}", "ground_truth": fmt(new_tax),
         "prompt": f"Change TAX_RATE to {fmt(new_tax)}. Leave everything else "
                   "exactly as it is.",
         "checks": [{"op": "contains", "file": "config.py",
                     "needle": f"TAX_RATE = {fmt(new_tax)}"}]},
        {**tpl, "name": f"edit-rename-{seed}",
         "ground_truth": f"{rename} -> {new_name}",
         "prompt": f"Rename the function {rename} to {new_name} everywhere it is "
                   f"defined and called. Leave everything else exactly as it is.",
         "checks": [{"op": "contains", "file": "core.py", "needle": f"def {new_name}"},
                    {"op": "absent", "file": "core.py", "needle": rename}]},
        {**tpl, "name": f"edit-add-tax_after-{seed}", "ground_truth": "tax_after",
         "prompt": "Add a function `tax_after(price)` to core.py that returns "
                   "round(price * (1 + TAX_RATE), 2). Do not change any other "
                   "function.",
         "checks": [{"op": "contains", "file": "core.py", "needle": "def tax_after"}]},
    ]


def _fix_specs(seed: int) -> list[dict]:
    base = generate_project(seed)
    if not tests_pass(base.files):
        raise AssertionError(f"seed {seed}: generated project fails its own tests")
    bugged = inject_bug(base, seed)
    if tests_pass(bugged.files):
        raise AssertionError(f"seed {seed}: injected bug {bugged.meta['bug']} does "
                             "not break any test -- refusing (find-and-fix guard)")
    return [{
        "family": "fix", "seed": seed, "files": bugged.files,
        "tools": ["list_dir", "read_file", "search", "edit_file", "write_file",
                  "build", "run_tests"],
        "name": f"fix-core-{seed}",
        "ground_truth": f"bug {bugged.meta['bug']} removed",
        "prompt": "There is a bug in core.py that makes some tests fail. Find it "
                  "and fix it so every test passes. Run the tests to verify "
                  "before answering.",
    }]


def make_specs(seed: int, family: str) -> list[dict]:
    if family.endswith(("_h", "_m")):         # v2 hard / mid workspaces
        import repo_gen_hard
        return repo_gen_hard.SPECS[family](seed)
    if family == "mhqa_t":                     # tool-fetched multi-hop KB (mhqa_tool.py)
        import mhqa_tool
        return mhqa_tool.specs(seed)
    if family == "repo_q":                     # questions over the real plugin repo (repo_q.py)
        import repo_q
        return repo_q.specs(seed)
    if family == "doc_make":                   # write a md/PDF report from notes + csv (doc_make.py)
        import doc_make
        return doc_make.specs(seed)
    if family == "gather_produce":             # lookups -> FINDINGS.md (gather_produce.py)
        import gather_produce
        return gather_produce.specs(seed)
    if family == "doc_digest":                 # long doc -> SUMMARY.md, keys + decoys (doc_digest.py)
        import doc_digest
        return doc_digest.specs(seed)
    return {"find": _find_specs, "edit": _edit_specs, "fix": _fix_specs}[family](seed)


def find_spec(seed: int, family: str, name: str) -> dict:
    """One spec by name -- used to rebuild an episode's exact workspace."""
    for spec in make_specs(seed, family):
        if spec["name"] == name:
            return spec
    raise KeyError(f"no spec {name} (seed {seed}, {family})")


def write_files(files: dict[str, str], root: Path) -> None:
    """Write a project's files into a workspace directory."""
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")