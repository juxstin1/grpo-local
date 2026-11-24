"""Offline public-suite coverage inventory, not a model-performance certificate.

The manifest maps exact public task roots to reviewed capability declarations.
Two different fixture identities per capability are the minimum. Seed/name-only
copies share a fingerprint and cannot increase coverage; semantic independence
and capability declarations still require human review when adding new tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "suite-coverage-v1"
MIN_INDEPENDENT_ROOTS = 2
CAPABILITIES = {
    "constrained-refactoring": "Rename a symbol and its callers while preserving unrelated behavior.",
    "approved-evidence-selection": "Resolve facts using explicit approval and revision rules.",
    "revision-arithmetic": "Select current records and calculate exact decimal results.",
    "scoped-preservation": "Change only authorized artifacts and preserve protected content.",
    "tool-error-recovery": "Observe a tool failure, recover, and verify the resulting artifact.",
    "long-context": "Retain and use relevant constraints across a deliberately long context.",
    "conflicting-source-injection": "Resolve conflicting source authority and reject injected instructions.",
    "underspecified-clarification": "Ask for necessary missing information before a consequential action.",
    "multistep-stateful-tools": "Complete dependent tool actions whose later steps rely on verified earlier state.",
}


class CoverageError(ValueError):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CoverageError("Duplicate JSON object key")
        result[key] = value
    return result


def _read(path: Path):
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(CoverageError("Nonfinite JSON")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CoverageError(f"Unreadable or invalid coverage input: {path.name}") from exc
    if not isinstance(data, dict):
        raise CoverageError("Coverage inputs must be JSON objects")
    return data, hashlib.sha256(raw).hexdigest()


def _strings(values, label, *, allow_empty=False):
    if (not isinstance(values, list) or (not allow_empty and not values)
            or any(not isinstance(v, str) or not v.strip() for v in values)
            or len(set(values)) != len(values)):
        raise CoverageError(f"Invalid or duplicate {label}")
    return set(values)


def _fixture_fingerprint(task):
    # A changed prompt, name, seed or delivery path alone is not a new fixture.
    # Input and output contents both contribute; instructions are reviewed via
    # the raw suite hash and explicit mapping rather than counted as diversity.
    if not isinstance(task.get("files"), dict) or not task["files"]:
        raise CoverageError("A coverage root needs authored public input files")
    contract = task.get("contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("outputs"), dict) or not contract["outputs"]:
        raise CoverageError("A coverage root needs explicit output contracts")
    # Ignore filenames too: simply renaming one fixture's files is not coverage.
    evidence = {"inputs": sorted(task["files"].values(), key=lambda v: json.dumps(v, sort_keys=True)),
                "outputs": sorted(contract["outputs"].values(), key=lambda v: json.dumps(v, sort_keys=True))}
    raw = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def check_coverage(suite_path: str | Path, manifest_path: str | Path) -> dict:
    suite, suite_sha = _read(Path(suite_path))
    manifest, manifest_sha = _read(Path(manifest_path))
    if manifest.get("schema") != SCHEMA or manifest.get("suite_file") != "suite-v2/tasks.json":
        raise CoverageError("Unsupported suite coverage identity")
    if suite.get("sealed") is not False or not isinstance(suite.get("version"), str):
        raise CoverageError("Coverage is limited to an explicitly public versioned suite")
    if manifest.get("suite_version") != suite["version"] or manifest.get("suite_sha256") != suite_sha:
        raise CoverageError("Suite version/raw SHA256 mismatch; review coverage after task changes")
    if _strings(manifest.get("required_capabilities"), "required capabilities") != set(CAPABILITIES):
        raise CoverageError("Required capability policy cannot be weakened")
    minimum = manifest.get("minimum_independent_roots")
    if type(minimum) is not int or minimum != MIN_INDEPENDENT_ROOTS:
        raise CoverageError("Minimum independent roots must be exactly 2")
    tasks, mappings = suite.get("tasks"), manifest.get("roots")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, dict) for t in tasks):
        raise CoverageError("Suite tasks are missing or invalid")
    if not isinstance(mappings, list) or not mappings or any(not isinstance(m, dict) for m in mappings):
        raise CoverageError("Coverage root mappings are missing or invalid")
    _strings([t.get("root_id") for t in tasks], "suite roots")
    _strings([t.get("name") for t in tasks], "suite task names")
    _strings([m.get("root_id") for m in mappings], "coverage roots")
    task_by_root = {task["root_id"]: task for task in tasks}
    if {m["root_id"] for m in mappings} != set(task_by_root):
        raise CoverageError("Coverage mappings must match every suite root exactly")
    declared, independent = {c: [] for c in CAPABILITIES}, {c: set() for c in CAPABILITIES}
    fingerprints = {}
    for mapping in mappings:
        root_id = mapping["root_id"]
        task = task_by_root[root_id]
        if mapping.get("task_name") != task["name"]:
            raise CoverageError("Coverage root/task name mismatch")
        capabilities = _strings(mapping.get("capabilities"), "root capabilities")
        if not capabilities <= set(CAPABILITIES):
            raise CoverageError("Unknown capability name")
        if not isinstance(mapping.get("evidence"), str) or not mapping["evidence"].strip():
            raise CoverageError("Capability mapping needs a reviewable evidence description")
        fingerprint = _fixture_fingerprint(task)
        fingerprints[root_id] = fingerprint
        for capability in capabilities:
            declared[capability].append(root_id)
            independent[capability].add(fingerprint)
    missing = sorted(c for c, roots in declared.items() if not roots)
    if _strings(manifest.get("missing_capabilities"), "missing capabilities", allow_empty=True) != set(missing):
        raise CoverageError("Declared missing capabilities do not match the actual root mapping")
    counts = {c: len(independent[c]) for c in CAPABILITIES}
    below = {c: {"independent_roots": counts[c], "required": MIN_INDEPENDENT_ROOTS}
             for c in CAPABILITIES if counts[c] < MIN_INDEPENDENT_ROOTS}
    return {"schema": "suite-coverage-report-v1", "status": "development-only", "sealed": False,
            "suite_version": suite["version"], "suite_sha256": suite_sha, "coverage_sha256": manifest_sha,
            "roots": len(tasks), "independent_fixtures": len(set(fingerprints.values())),
            "capabilities": {c: {"description": CAPABILITIES[c], "roots": sorted(declared[c]),
                                 "independent_roots": counts[c]} for c in CAPABILITIES},
            "minimum_independent_roots": MIN_INDEPENDENT_ROOTS, "missing_capabilities": missing,
            "below_minimum": below, "broad_coverage": not below,
            "limitation": "Fixture coverage declarations only; no model performance or private evaluation claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "suite-v2/tasks.json")
    parser.add_argument("--coverage", type=Path, default=ROOT / "suite-v2/coverage.json")
    parser.add_argument("--require-broad", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check_coverage(args.suite, args.coverage)
    except CoverageError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.require_broad and not report["broad_coverage"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
