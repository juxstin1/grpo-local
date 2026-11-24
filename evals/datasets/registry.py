"""Explicit public development registry; metadata integrity, not sealed authority.

Only allowlisted public inputs and sealed *manifests* are read. No private task
bodies are opened. A registry pins raw bytes and is never generated implicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

SCHEMA = "public-dataset-registry-v2"
ENV = "GRPO_DATASET_REGISTRY"
TRAIN_SOURCES = ("evals/data/pools/math.train.jsonl", "evals/data/pools/code.train.jsonl")
OPTIONAL_TRAIN = ("evals/data/pools/mhqa.train.jsonl",)
PUBLIC_SETS = tuple("evals/data/sets/" + name for name in (
    "baseline_v1.json", "baseline_20.json", "post_cutoff_v1.json", "post_cutoff_v1_5pb.json"))
SEALED_MANIFESTS = tuple(f"evals/holdout/sealed/{name}-v1/manifest.json" for name in ("math", "code"))


class RegistryError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def root_key(value) -> str:
    return re.sub(r"[^a-z0-9]+", ":", str(value or "").strip().lower()).strip(":")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError("Duplicate JSON object key")
        result[key] = value
    return result


def decode(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(RegistryError("Nonfinite JSON")))
    except (UnicodeError, ValueError) as exc:
        raise RegistryError("Invalid registry/source JSON") from exc


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RegistryError(f"Missing/unreadable registry input: {path}") from exc


def jsonl(raw: bytes) -> list[dict]:
    rows = [decode(line) for line in raw.split(b"\n") if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise RegistryError("Task rows must be objects")
    return rows


def semantic(row: dict) -> dict:
    return {key: row[key] for key in ("dataset", "prompt", "answer", "tests") if key in row}


def content_hash(row: dict) -> str:
    return digest(canonical(semantic(row)))


def stable_uid(row: dict) -> str:
    lineage = row.get("source_lineage") or [{
        "source": row.get("source"), "original_uid": row.get("original_uid", row.get("uid")),
        "root_id": row.get("root_id"), "lineage": row.get("lineage", []),
    }]
    content = semantic(row)
    if "prompt" not in content:
        content["skel_text"] = row.get("skel_text")
    return f"{row['dataset']}:v2:{digest(canonical({'content': content, 'sources': lineage}))}"


def assert_unique_ids(rows: list[dict], field: str = "uid") -> None:
    seen = set()
    for row in rows:
        uid = row.get(field)
        if not isinstance(uid, str) or not uid or uid in seen:
            raise RegistryError(f"Missing or duplicate {field}; refusing ambiguous task mapping")
        seen.add(uid)


def identities(row: dict) -> set[str]:
    values = [row.get("root_id"), row.get("original_uid", row.get("uid"))]
    for source in row.get("source_lineage", []):
        values.extend([source.get("root_id"), source.get("original_uid")])
        values.extend(source.get("lineage", []))
    values.extend(row.get("lineage", []))
    return {root_key(v) for v in values if root_key(v)}


def _merkle(hashes):
    layer = sorted(hashes)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [digest((layer[i] + layer[i + 1]).encode()) for i in range(0, len(layer), 2)]
    return layer[0] if layer else digest(b"")


def sealed_exclusions(document: dict, name: str | None = None) -> set[str]:
    if not isinstance(document, dict) or (name is not None and document.get("name") != name):
        raise RegistryError("Sealed manifest identity mismatch")
    hashes = document.get("root_hashes")
    if (not isinstance(hashes, dict) or not hashes or document.get("role") != "eval"
            or document.get("root_field") != "root_id"
            or document.get("n_root_tasks") != len(hashes)
            or type(document.get("n_items")) is not int or document["n_items"] < len(hashes)
            or not isinstance(document.get("canary"), str) or not document["canary"]
            or any(not root_key(k) or k != root_key(k) for k in hashes)
            or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in hashes.values())
            or document.get("merkle_root") != _merkle(hashes.values())):
        raise RegistryError("Missing/corrupt sealed exclusion manifest metadata")
    for field in ("holdout_sha256", "roots_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(document.get(field, ""))):
            raise RegistryError("Invalid sealed manifest digest")
    return set(hashes)


def public_exclusions(document: dict) -> tuple[set[str], set[str]]:
    buckets = document.get("items")
    if not isinstance(buckets, dict) or not buckets:
        raise RegistryError("Missing public evaluation items")
    roots, contents = set(), set()
    for bucket, rows in buckets.items():
        if not isinstance(rows, list):
            raise RegistryError("Invalid public evaluation bucket")
        if bucket == "holdout":  # identifier-only legacy bucket: never resolve private tasks
            if any(set(row) != {"id"} for row in rows):
                raise RegistryError("Legacy holdout bucket must contain identifiers only")
            roots.update(root_key(row["id"]) for row in rows)
            continue
        for row in rows:
            if not isinstance(row, dict) or not (row.get("uid") or row.get("id")):
                raise RegistryError("Public evaluation identity missing")
            if row.get("root_id"):
                roots.update(identities(row))
            elif bucket == "lcb" and row.get("question_id"):
                roots.add(root_key("code:lcb:" + str(row["question_id"])))
                if row.get("title"):
                    title = re.sub(r"-(i|ii|iii|iv|v)$", "", re.sub(r"[^a-z0-9]+", "-", row["title"].lower()).strip("-"))
                    roots.add(root_key("code:lc:" + title))
            elif bucket in ("bfcl", "when2call"):
                roots.add(root_key(f"{bucket}:{row.get('item_id') or row.get('uuid') or row['id']}"))
            else:
                raise RegistryError("Public evaluation root missing")
            roots.add(root_key(row.get("uid") or row["id"]))
            if row.get("prompt"):
                contents.add(content_hash({**row, "dataset": row.get("dataset", bucket)}))
    return roots, contents


def allowed_source(relative: str) -> bool:
    return (relative in TRAIN_SOURCES + OPTIONAL_TRAIN + PUBLIC_SETS + SEALED_MANIFESTS
            or bool(re.fullmatch(r"evals/datasets/logs/(?:latest|prepare-[0-9TZ]+)\.json", relative)))


def normalize_rows(rows: list[dict]) -> list[dict]:
    """Deduplicate identical content; retain every contributing source identity."""
    groups = {}
    for row in rows:
        if (row.get("dataset") not in ("math", "code", "mhqa")
                or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("uid", "source", "root_id"))
                or not ("answer" in row or "tests" in row)):
            raise RegistryError("Incomplete public training task")
        groups.setdefault(content_hash(row), []).append(row)
    normalized = []
    for content, copies in sorted(groups.items()):
        copies.sort(key=lambda r: canonical(r))
        row = dict(copies[0])
        lineage = [{"source": r["source"], "original_uid": r.get("original_uid", r["uid"]),
                    "root_id": r["root_id"], "lineage": sorted(set(r.get("lineage", [])))} for r in copies]
        row["source_lineage"] = [decode(raw) for raw in sorted(set(canonical(s) for s in lineage))]
        row["original_uid"] = row.get("original_uid", row["uid"])
        row["content_sha256"] = content
        row["uid"] = stable_uid(row)
        row.pop("id", None)
        row["split"] = "train"
        row["qualification"] = ("unsupported-assert-grader" if row.get("tests", {}).get("kind") == "assert"
                                else "candidate-unqualified")
        normalized.append(row)
    assert_unique_ids(normalized)
    return sorted(normalized, key=lambda r: r["uid"])


def partition_rows(rows: list[dict], source: str, forbidden: set[str], contents: set[str]):
    normalized, kept, quarantine = normalize_rows(rows), [], []
    for row in normalized:
        overlap = sorted(identities(row) & forbidden)
        if overlap or content_hash(row) in contents:
            quarantine.append({"uid": row["uid"], "original_uid": row["original_uid"],
                               "source_file": source, "root_id": row["root_id"],
                               "reason": "evaluation-root-overlap" if overlap else "evaluation-content-overlap",
                               "matched_identities": overlap, "content_sha256": row["content_sha256"]})
        else:
            kept.append(row)
    counts = {"rows": len(kept), "original_rows": len(rows), "exact_duplicates_removed": len(rows) - len(normalized),
              "quarantined_rows": len(quarantine),
              "unsupported_assert_rows": sum(r["qualification"] == "unsupported-assert-grader" for r in kept)}
    return kept, quarantine, counts


class DatasetRegistry:
    def __init__(self, path: Path, document: dict, raw_hash: str):
        self.path, self.document, self.sha256 = path, document, raw_hash
        self.forbidden_roots = set(document["exclusions"]["roots"])
        self.forbidden_contents = set(document["exclusions"]["content_sha256"])

    def validate_training_rows(self, rows: list[dict]) -> None:
        for row in rows:
            if not root_key(row.get("root_id")):
                raise RegistryError("Training row root_id is required")
            if identities(row) & self.forbidden_roots or content_hash(row) in self.forbidden_contents:
                raise RegistryError("Training/evaluation overlap; refusing training data")

    def load_pool(self, path: str | Path) -> list[dict]:
        path = Path(path).resolve()
        matches = [item for item in self.document["pools"] if (self.path.parent / item["path"]).resolve() == path]
        if len(matches) != 1:
            raise RegistryError("Collection requires a registered normalized pool")
        record = matches[0]
        raw = read_bytes(path)
        if digest(raw) != record["sha256"]:
            raise RegistryError("Normalized pool raw SHA256 mismatch")
        rows = jsonl(raw)
        if len(rows) != record["rows"]:
            raise RegistryError("Normalized pool row count mismatch")
        assert_unique_ids(rows)
        for row in rows:
            if (row.get("uid") != stable_uid(row) or row.get("content_sha256") != content_hash(row)
                    or not row.get("source_lineage") or not row.get("original_uid")
                    or row.get("split") != "train"):
                raise RegistryError("Normalized pool identity/content mismatch")
        self.validate_training_rows(rows)
        return rows


def load_registry(path: str | Path | None = None) -> DatasetRegistry:
    path = path or os.environ.get(ENV)
    if not path:
        raise RegistryError(f"Explicit dataset registry required (--registry or {ENV})")
    path = Path(path).resolve()
    raw = read_bytes(path)
    doc = decode(raw)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("status") != "public-development":
        raise RegistryError("Unsupported dataset registry")
    sources = doc.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RegistryError("Dataset registry sources missing")
    if any(not isinstance(s, dict) or not isinstance(s.get("path"), str) for s in sources):
        raise RegistryError("Invalid source identity")
    source_paths = [s["path"] for s in sources]
    required = set(TRAIN_SOURCES + PUBLIC_SETS + SEALED_MANIFESTS)
    if len(source_paths) != len(set(source_paths)) or not required <= set(source_paths):
        raise RegistryError("Missing or duplicate required registry/exclusion sources")
    if not isinstance(doc.get("source_root"), str) or not Path(doc["source_root"]).is_absolute():
        raise RegistryError("Registry source_root must be explicit and absolute")
    root = Path(doc["source_root"]).resolve()
    required.update(p for p in OPTIONAL_TRAIN if (root / p).exists())
    required.update(p.relative_to(root).as_posix() for p in (root / "evals/datasets/logs").glob("prepare-*.json"))
    if (root / "evals/datasets/logs/latest.json").exists():
        required.add("evals/datasets/logs/latest.json")
    if set(source_paths) != required:
        raise RegistryError("Registry does not cover the explicit public source inventory")
    exclusions, contents, public_roots, sealed_roots = set(), set(), set(), set()
    source_data = {}
    provenance = {}
    for source in sources:
        relative = source["path"]
        if not isinstance(relative, str) or not allowed_source(relative):
            raise RegistryError("Unallowlisted registry source")
        data = read_bytes(root / relative)
        source_data[relative] = data
        if digest(data) != source.get("sha256"):
            raise RegistryError(f"Registry source raw SHA256 mismatch: {relative}")
        if relative in PUBLIC_SETS:
            rs, cs = public_exclusions(decode(data))
            public_roots.update(rs)
            contents.update(cs)
        elif relative in SEALED_MANIFESTS:
            sealed_roots.update(sealed_exclusions(decode(data), Path(relative).parent.name))
        elif relative.startswith("evals/datasets/logs/"):
            evidence = decode(data)
            if not isinstance(evidence, dict) or not isinstance(evidence.get("provenance", {}), dict):
                raise RegistryError("Invalid provenance metadata")
            for name, item in evidence.get("provenance", {}).items():
                provenance.setdefault(name, []).append({"metadata_file": relative, "evidence": item})
    if public_roots & sealed_roots:
        raise RegistryError("Public evaluation/sealed metadata root overlap")
    exclusions = public_roots | sealed_roots
    if doc.get("provenance") != provenance:
        raise RegistryError("Provenance does not match pinned metadata")
    expected = {"roots": sorted(exclusions), "content_sha256": sorted(contents)}
    if doc.get("exclusions") != expected:
        raise RegistryError("Registry exclusions missing or changed")
    pools = doc.get("pools")
    if not isinstance(pools, list) or not pools:
        raise RegistryError("Normalized pools missing")
    for pool in pools:
        if not isinstance(pool, dict) or not re.fullmatch(r"pools/(math|code|mhqa)\.train\.jsonl", str(pool.get("path", ""))):
            raise RegistryError("Invalid normalized pool path")
    if len({p["path"] for p in pools}) != len(pools):
        raise RegistryError("Duplicate normalized pool")
    expected_sources = set(TRAIN_SOURCES) | {p for p in OPTIONAL_TRAIN if p in source_data}
    if {p.get("source") for p in pools} != expected_sources or len(pools) != len(expected_sources):
        raise RegistryError("Missing or duplicate normalized source pools")
    result = DatasetRegistry(path, doc, digest(raw))
    expected_quarantine = []
    for pool in pools:
        expected_rows, quarantine, counts = partition_rows(jsonl(source_data[pool["source"]]), pool["source"], exclusions, contents)
        expected_quarantine.extend(quarantine)
        if any(pool.get(key) != value for key, value in counts.items()):
            raise RegistryError("Normalized pool disposition counts mismatch")
        actual = result.load_pool(path.parent / pool["path"])
        if canonical(actual) != canonical(expected_rows):
            raise RegistryError("Normalized pool does not match pinned source content/lineage")
    if doc.get("quarantine") != expected_quarantine:
        raise RegistryError("Quarantine disposition missing or changed")
    return result
