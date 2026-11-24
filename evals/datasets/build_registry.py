"""Offline derivation from fixed public allowlist; originals are never rewritten.

Run: python -B evals/datasets/build_registry.py --root REPO --out NEW_DIRECTORY
Private task bodies and roots.json are not read. Sealed manifest metadata is
an exclusion input only, not proof of independently protected evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

if __package__:
    from . import registry as R
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import registry as R


def build(root: str | Path, output: str | Path) -> Path:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise R.RegistryError("Registry output must be a new exclusive directory")
    paths = list(R.TRAIN_SOURCES + R.PUBLIC_SETS + R.SEALED_MANIFESTS)
    paths += [p for p in R.OPTIONAL_TRAIN if (root / p).exists()]
    logs = root / "evals/datasets/logs"
    paths += [p.relative_to(root).as_posix() for p in sorted(logs.glob("prepare-*.json"))]
    if (logs / "latest.json").exists():
        paths.append("evals/datasets/logs/latest.json")
    sources, raw_by_path, provenance = [], {}, {}
    forbidden, public_content, public_roots, sealed_roots = set(), set(), set(), set()
    for path in sorted(paths):
        if not R.allowed_source(path):
            raise R.RegistryError("Source outside public metadata allowlist")
        raw = R.read_bytes(root / path)
        raw_by_path[path] = raw
        sources.append({"path": path, "sha256": R.digest(raw), "bytes": len(raw)})
        if path in R.PUBLIC_SETS:
            roots, content = R.public_exclusions(R.decode(raw))
            public_roots.update(roots)
            public_content.update(content)
        elif path in R.SEALED_MANIFESTS:
            sealed_roots.update(R.sealed_exclusions(R.decode(raw), Path(path).parent.name))
        elif path.startswith("evals/datasets/logs/"):
            data = R.decode(raw)
            if not isinstance(data, dict) or not isinstance(data.get("provenance", {}), dict):
                raise R.RegistryError("Invalid source provenance document")
            for name, evidence in data.get("provenance", {}).items():
                entry = {"metadata_file": path, "evidence": evidence}
                provenance.setdefault(name, []).append(entry)
    if public_roots & sealed_roots:
        raise R.RegistryError("Public evaluation/sealed metadata root overlap")
    forbidden = public_roots | sealed_roots
    records, pools, quarantine = {}, [], []
    for source in R.TRAIN_SOURCES + R.OPTIONAL_TRAIN:
        if source not in raw_by_path:
            continue
        originals = R.jsonl(raw_by_path[source])
        kept, excluded, counts = R.partition_rows(originals, source, forbidden, public_content)
        quarantine.extend(excluded)
        relative = "pools/" + Path(source).name
        raw = b"".join(R.canonical(row) + b"\n" for row in kept)
        records[relative] = raw
        pools.append({"path": relative, "source": source, "sha256": R.digest(raw), **counts})
    document = {"schema": R.SCHEMA, "status": "public-development", "source_root": str(root),
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scope": "Public input integrity and exclusions; not independent sealed authority or training qualification",
                "sources": sources, "provenance": provenance, "pools": pools,
                "exclusions": {"roots": sorted(forbidden), "content_sha256": sorted(public_content)},
                "quarantine": quarantine}
    # Finish every input check before creating the exclusive destination.
    output.mkdir(parents=True, exist_ok=False)
    (output / "pools").mkdir()
    for name, raw in records.items():
        with (output / name).open("xb") as fh:
            fh.write(raw)
    registry_path = output / "registry.json"
    with registry_path.open("xb") as fh:
        fh.write(R.canonical(document) + b"\n")
    R.load_registry(registry_path)
    return registry_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = build(args.root, args.out)
    registry = R.load_registry(path)
    print(f"registry={path} sha256={registry.sha256} pools={len(registry.document['pools'])} "
          f"quarantined={len(registry.document['quarantine'])}")


if __name__ == "__main__":
    main()
