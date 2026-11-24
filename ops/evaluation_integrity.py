"""Integrity checks for development fixtures; this is not a secrecy boundary.

The operator must provide a trusted manifest from outside the task directory:
{"schema": "evaluation-integrity-v1", "files": {"example.json": "<raw SHA256>"}}.
Its file map covers every directory entry, including any legacy .sha256 sidecars.
There is deliberately no baseline-generation or automatic trust fallback here.
An external manifest only detects changes relative to that manifest. A process
which can replace both inputs can defeat it; confidential final evaluation must
run under a separate evaluator identity with enforced access restrictions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat


CANARY = "GYM-HOLDOUT-SEALED-2026-09-29"
MANIFEST_SCHEMA = "evaluation-integrity-v1"


class IntegrityError(ValueError):
    """The fixture registry is unavailable, ambiguous, or inconsistent."""


class ExternalEvaluatorRequired(IntegrityError):
    """Same-identity evaluation cannot establish confidentiality."""


def validate_export_policy(holdout: bool, transcripts: bool = False,
                           task_level_exports: bool = False) -> None:
    """Refuse the legacy sealed claim before loading tasks or starting a model."""
    if not holdout:
        return
    if transcripts or task_level_exports:
        raise IntegrityError("Sealed evaluation forbids transcripts and task-level exports; "
                             "an external evaluator is required.")
    raise ExternalEvaluatorRequired(
        "Sealed evaluation requires an external evaluator identity with enforced access "
        "restrictions. This developer process cannot establish confidentiality.")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError("Duplicate JSON object key in evaluation registry.")
        result[key] = value
    return result


def _json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              IntegrityError("Nonfinite JSON value in evaluation registry.")))
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, IntegrityError):
            raise
        raise IntegrityError("Invalid JSON in evaluation registry.") from None


def _plain_path(path: Path) -> Path:
    """Reject known link/reparse components; no ACL or race-proof claim is made."""
    # Check components before resolve(), which would otherwise hide symlinks.
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        try:
            info = component.lstat()
        except OSError:
            raise IntegrityError("Evaluation registry path is missing or unreadable.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise IntegrityError("Links and reparse points are forbidden in evaluation registry paths.")
    return absolute.resolve()


def _read(path: Path) -> bytes:
    _plain_path(path)
    if not path.is_file():
        raise IntegrityError("Evaluation registry entries must be ordinary files.")
    try:
        return path.read_bytes()
    except OSError:
        raise IntegrityError("Evaluation registry file is unreadable.") from None


def _name(value) -> bool:
    # Portable basename restriction also rejects Windows ADS, drive letters,
    # trailing dot/space aliases, and case-insensitive duplicates below.
    return (isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value))
            and not value.endswith((".", " ")) and ".." not in value
            and value.split(".")[0].upper() not in
            {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
             *[f"LPT{i}" for i in range(1, 10)]})


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _validate_document(document: dict) -> list[dict]:
    required = {"schema", "family", "canary", "seeds", "items", "sha256"}
    if not isinstance(document, dict) or set(document) != required:
        raise IntegrityError("Invalid gym registry document schema.")
    if (document["schema"] != "gym_v1" or document["canary"] != CANARY
            or not _text(document["family"]) or not _digest(document["sha256"])):
        raise IntegrityError("Invalid gym registry identity or embedded seal.")
    body = {key: value for key, value in document.items() if key != "sha256"}
    canonical = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != document["sha256"]:
        raise IntegrityError("Embedded gym seal mismatch.")
    seeds, items = document["seeds"], document["items"]
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or not isinstance(items, list) or not items):
        raise IntegrityError("Empty or invalid gym seed/item registry.")
    keys = {"family", "name", "seed", "prompt", "ground_truth", "tools", "checks"}
    for item in items:
        if not isinstance(item, dict) or set(item) != keys:
            raise IntegrityError("Invalid gym task schema.")
        if (item["family"] != document["family"] or not _name(item["name"])
                or type(item["seed"]) is not int or item["seed"] not in seeds
                or not _text(item["prompt"]) or not _text(item["ground_truth"])):
            raise IntegrityError("Invalid gym task identity, seed, prompt, or ground truth.")
        if (not isinstance(item["tools"], list) or any(not _text(t) for t in item["tools"])
                or len(set(item["tools"])) != len(item["tools"])
                or not isinstance(item["checks"], list)
                or any(not isinstance(check, dict) or not check for check in item["checks"])):
            raise IntegrityError("Invalid gym task tools or checks.")
    if {item["seed"] for item in items} != set(seeds):
        raise IntegrityError("Incomplete gym seed registry.")
    return items


def load_verified_tasks(directory: str | Path,
                        trusted_manifest: str | Path | None) -> list[tuple[str, dict]]:
    """Load development fixtures using explicitly supplied external trust material.

    No prompt, answer, filename, or item identifier is included in error messages.
    This function does not certify that a manifest's provider is trustworthy.
    """
    if trusted_manifest is None:
        raise IntegrityError("An operator-supplied trusted manifest is required; no automatic baseline exists.")
    root = _plain_path(Path(directory))
    manifest_path = _plain_path(Path(trusted_manifest))
    if not root.is_dir() or manifest_path == root or root in manifest_path.parents:
        raise IntegrityError("Trusted manifest must be outside the task directory.")
    manifest = _json(_read(manifest_path))
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "files"}
            or manifest["schema"] != MANIFEST_SCHEMA or not isinstance(manifest["files"], dict)
            or not manifest["files"]):
        raise IntegrityError("Invalid or empty trusted evaluation manifest.")
    files = manifest["files"]
    if (any(not _name(name) or not _digest(digest) for name, digest in files.items())
            or len({name.casefold() for name in files}) != len(files)):
        raise IntegrityError("Invalid or duplicate trusted manifest filenames/hashes.")
    json_names = {name for name in files if name.endswith(".json")}
    if not json_names or any(name not in json_names and
                            (not name.endswith(".sha256") or name[:-7] + ".json" not in json_names)
                            for name in files):
        raise IntegrityError("Manifest must contain task JSON files and optional matching seal sidecars.")
    try:
        actual = {entry.name for entry in root.iterdir()}
    except OSError:
        raise IntegrityError("Evaluation directory is unreadable.") from None
    if actual != set(files):
        raise IntegrityError("Evaluation inventory mismatch: missing or unregistered entries.")
    raw_files = {}
    for name, digest in files.items():
        raw = _read(root / name)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise IntegrityError("Evaluation file does not match the trusted raw SHA256.")
        raw_files[name] = raw
    result, seen = [], set()
    for name in sorted(json_names):
        document = _json(raw_files[name])
        items = _validate_document(document)
        sidecar = name[:-5] + ".sha256"
        if sidecar in raw_files and raw_files[sidecar].strip() != document["sha256"].encode("ascii"):
            raise IntegrityError("Legacy seal sidecar mismatch.")
        for item in items:
            identity = item["name"].casefold()
            if identity in seen:
                raise IntegrityError("Duplicate task identity in evaluation registry.")
            seen.add(identity)
            result.append((item["family"], item))
    return result


def reserved_seeds(directory: str | Path,
                   trusted_manifest: str | Path | None) -> set[tuple[str, int]]:
    """Fail closed instead of silently omitting corrupt exclusion files."""
    return {(family, task["seed"]) for family, task in load_verified_tasks(directory, trusted_manifest)}
