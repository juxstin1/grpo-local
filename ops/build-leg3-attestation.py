"""Bind completed, read-only pod prescreens to the leg-3 pool before approval.

Run this on the pod only after screen.done exists. It writes one sidecar, never
changes a prescreen result, and refuses to replace a different sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(config_path: Path, root: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["prescreen"]
    if not (output_dir / "screen.done").is_file():
        raise ValueError("prescreens incomplete: screen.done absent")
    report = (output_dir.parent.parent / "REPORT-leg3-diagnostics.md").read_text(encoding="utf-8")
    names = {"find_h": "findh", "edit": "edit", "mhqa": "mhqa"}
    for family, short in names.items():
        source = spec["families"][family]
        line = (f"STATS_OK leg3-screen-step20-{short}.stats.json "
                f"tasks={source['expected_tasks']} attempts={spec['attempts_per_task']} errors=0")
        if line not in report:
            raise ValueError(f"verified screen marker absent: {family}")
    manifest_path = output_dir / "pool-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["adapter_step"] != 20 or
            manifest["attempts"] != spec["attempts_per_task"] or
            manifest["keep_passes"] != spec["keep_pass_counts_inclusive"]):
        raise ValueError("pool manifest differs from pre-registered screen")
    adapter = Path(config["init_adapter"]["pod_path"]) / "adapter_model.safetensors"
    adapter_sha = digest(adapter)
    if adapter_sha != config["init_adapter"]["adapter_model_sha256"]:
        raise ValueError("step-20 adapter hash drift")
    script_sha = digest(Path(spec["diagnostics_script"]))
    if script_sha != spec["diagnostics_script_sha256"]:
        raise ValueError("diagnostic script hash drift")
    for relative, expected in spec["source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"screen source drift: {relative}")
    stats_sha, pool_sha = {}, {}
    for family in names:
        source = spec["families"][family]
        stats_path = Path(source["stats"])
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats["attempts"] != spec["attempts_per_task"] or \
                len(stats["per_task"]) != source["expected_tasks"]:
            raise ValueError(f"{family}: stats size mismatch")
        if any(row["error"] or sum(row[k] for k in ("pass", "fail", "truncated")) !=
               spec["attempts_per_task"] for row in stats["per_task"].values()):
            raise ValueError(f"{family}: incomplete or errored stats")
        if family == "mhqa":
            args = stats["args"]
            if (args["served_model"] != spec["model"] or
                    args["mhqa_seeds"] != source["seeds"] or
                    args["hops"] != ",".join(map(str, source["hops"])) or
                    args["per_hop"] != source["per_hop"] or
                    args["temperature"] != config["training"]["temperature"]):
                raise ValueError("MHQA screen arguments differ from pre-registration")
        if manifest["families"][family]["source_stats"] != str(stats_path):
            raise ValueError(f"{family}: manifest source mismatch")
        pool_path = output_dir / f"pool-{family}.txt"
        ids = pool_path.read_text(encoding="utf-8").splitlines()
        low, high = manifest["keep_passes"]
        eligible = {uid for uid, row in stats["per_task"].items()
                    if low <= row["pass"] <= high}
        if len(ids) != len(set(ids)) or set(ids) != eligible or len(ids) < 16:
            raise ValueError(f"{family}: pool IDs differ from screen")
        stats_sha[family], pool_sha[family] = digest(stats_path), digest(pool_path)
    return {"adapter_sha256": adapter_sha, "served_model": spec["model"],
            "temperature": config["training"]["temperature"],
            "attempts": spec["attempts_per_task"],
            "manifest_sha256": digest(manifest_path),
            "diagnostics_script_sha256": script_sha,
            "source_sha256": spec["source_sha256"],
            "stats_sha256": stats_sha, "pool_sha256": pool_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = build(args.config, args.root, args.output_dir)
    path = args.output_dir / "pool-attestation.json"
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError("different pool attestation already exists")
    else:
        path.write_text(payload, encoding="utf-8")
    print(f"ATTESTATION_OK {path} sha256={digest(path)}")


if __name__ == "__main__":
    main()
