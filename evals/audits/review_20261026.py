"""Offline audit of allowlisted public data; never loads private task bodies.

Inventory outputs counts and hashes, not prompts/completions/answer keys.
Contracts run only invented Python fixtures through the existing endpoint grader.
No models, downloads, dataset generation, seal changes, or service calls.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import socket
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def block_network(*args, **kwargs):
    raise RuntimeError("network forbidden during offline audit")


socket.socket.connect = block_network
socket.create_connection = block_network


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canon(s):
    return re.sub(r"[^a-z0-9]+", ":", str(s).strip().lower()).strip(":")


def load(path):
    data = path.read_bytes()
    return json.loads(data), sha(data)


def rows(path):
    data = path.read_bytes()
    return [json.loads(line) for line in data.decode("utf-8-sig").split("\n") if line.strip()], sha(data)


def summarize(records):
    ids = [r.get("uid", r.get("id")) for r in records]
    roots = [canon(r["root_id"]) for r in records if r.get("root_id")]
    prompts = [sha(r["prompt"].encode()) for r in records if r.get("prompt")]
    uid_groups = {}
    for r in records:
        uid_groups.setdefault(r.get("uid", r.get("id")), []).append(r)
    collisions = [rs for rs in uid_groups.values() if len(rs) > 1]
    return {
        "records": len(records), "unique_roots": len(set(roots)),
        "missing_roots": len(records) - len(roots),
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_id_groups": len(collisions),
        "duplicate_id_groups_different_prompts": sum(len({r.get("prompt") for r in rs}) > 1 for rs in collisions),
        "duplicate_id_groups_different_answers": sum(len({json.dumps(r.get("answer"), sort_keys=True) for r in rs}) > 1 for rs in collisions),
        "exact_prompt_duplicate_excess": len(prompts) - len(set(prompts)),
        "sources": dict(Counter(str(r.get("source", "missing")) for r in records)),
        "date_basis": dict(Counter(str(r.get("date_basis", "missing")) for r in records)),
        "tests_basis": dict(Counter(str(r.get("tests_basis", "missing")) for r in records if "tests" in r)),
        "case_counts": dict(Counter(str(len(r["tests"].get("cases", []))) for r in records if isinstance(r.get("tests"), dict) and r["tests"].get("kind") != "assert")),
        "missing_item_license": sum(not r.get("license") for r in records),
        "missing_item_source_revision": sum(not r.get("source_revision") for r in records),
    }


def inventory():
    out = {"scope": "public allowlist only; sealed manifests only; no heldout bodies", "sets": {}, "pools": {}, "sft_collections": {}, "manifests": {}}
    sets = {}
    for path in sorted((ROOT / "evals/data/sets").glob("*.json")):
        data, digest = load(path)
        # Legacy 'holdout' bucket contains IDs only. Never resolve them.
        buckets = {k: v for k, v in data["items"].items() if k != "holdout"}
        sets[path.name] = [r for group in buckets.values() for r in group]
        out["sets"][path.name] = {"sha256": digest, "schema": data.get("schema"), "buckets": {k: summarize(v) for k, v in buckets.items()}, "legacy_holdout_id_count": len(data["items"].get("holdout", []))}
    full_roots = {canon(r["root_id"]) for r in sets["post_cutoff_v1.json"]}
    small_roots = {canon(r["root_id"]) for r in sets["post_cutoff_v1_5pb.json"]}
    out["exclusion_gap"] = {"full_public_roots": len(full_roots), "small_public_roots": len(small_roots), "full_roots_absent_from_small": len(full_roots - small_roots)}
    all_eval_roots = full_roots | small_roots
    full_prompt_hashes = {sha(r["prompt"].encode()) for r in sets["post_cutoff_v1.json"]}
    latest, _ = load(ROOT / "evals/datasets/logs/latest.json")
    out["latest_manifest"] = {"stamp": latest["stamp"], "full_set_hash_matches": latest["eval_set_sha256"] == out["sets"]["post_cutoff_v1.json"]["sha256"], "provenance_sources": sorted(latest["provenance"])}
    for path in sorted((ROOT / "evals/data/pools").glob("*.train.jsonl")):
        data, digest = rows(path)
        normalized = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in data).encode()
        recorded = latest["results"][path.name.split(".")[0]]["train_sha256"]
        out["pools"][path.name] = {**summarize(data), "sha256": digest,
            "recorded_raw_hash_matches": digest == recorded, "recorded_logical_lf_hash_matches": sha(normalized) == recorded,
            "public_eval_root_overlap": len({canon(r["root_id"]) for r in data} & all_eval_roots),
            "full_eval_exact_prompt_overlap": len({sha(r["prompt"].encode()) for r in data} & full_prompt_hashes)}
    # Explicit metadata-only projection; do not open holdout.jsonl or roots.json.
    for name in ("math-v1", "code-v1"):
        path = ROOT / "evals/holdout/sealed" / name / "manifest.json"
        data, digest = load(path)
        out["manifests"][name] = {k: data.get(k) for k in ("name", "n_items", "n_root_tasks", "role", "min_date", "merkle_root")}
        out["manifests"][name]["manifest_sha256"] = digest
        out["manifests"][name]["verification"] = "metadata only; body integrity not checked"
    for path in sorted((ROOT / "sft/data").glob("*.jsonl")):
        if path.name.endswith(".fail.jsonl"):
            continue
        data, digest = rows(path)
        roots = {canon(r["root_id"]) for r in data if r.get("root_id")}
        out["sft_collections"][path.name] = {"sha256": digest, "records": len(data), "unique_roots": len(roots),
            "buckets": dict(Counter(r.get("bucket", "missing") for r in data)),
            "splits": dict(Counter(r.get("split", "missing") for r in data)),
            "missing_root": sum(not r.get("root_id") for r in data),
            "nonpassing": sum(r.get("grade", {}).get("pass") is not True for r in data),
            "public_eval_root_overlap": len(roots & all_eval_roots),
            "exact_duplicate_excess": len(data) - len({sha(json.dumps([r.get("root_id"),r.get("completion_ids")], sort_keys=True).encode()) for r in data})}
    out["sft_adapter_provenance"] = {}
    hosted_path = ROOT.parent / "runpod-sft-1007/dataset/train.jsonl"
    if hosted_path.exists():
        data, digest = rows(hosted_path)
        manifest, manifest_digest = load(hosted_path.with_name("manifest.json"))
        out["hosted_teacher_data"] = {"path": str(hosted_path), "sha256": digest, "manifest_sha256": manifest_digest,
            "records": len(data), "unique_roots": len({r["root_id"] for r in data}),
            "unique_teacher_traces": len({r["teacher_trace_id"] for r in data}),
            "buckets": dict(Counter(r["bucket"] for r in data)),
            "teachers": dict(Counter(r["teacher_model"] for r in data)),
            "splits": dict(Counter(r["split"] for r in data)),
            "public_endpoint_eval_root_overlap": len({canon(r["root_id"]) for r in data} & all_eval_roots),
            "manifest_claims": {k: manifest.get(k) for k in ("training_episodes", "training_roots", "holdout_overlap", "hinted_episodes_admitted", "hint_texts_stripped", "metadata_gaps", "remaining_gates")},
            "limitation": "saved manifest claims are not a fresh private-overlap or grader verification"}
    manifests = list(ROOT.glob("lora-*/sft_manifest.json")) + list((ROOT / "adapters-sft-1008").glob("*/sft_manifest.json"))
    for path in sorted(manifests):
        data, digest = load(path)
        matched, missing, mismatch = 0, 0, 0
        for source, expected in data.get("data", {}).items():
            # Basename matching only for historical sft/data inputs. Pod dataset/train.jsonl
            # is not guessed to be another local file.
            local = ROOT / "sft/data" / source.replace("\\", "/").rsplit("/", 1)[-1]
            if source == "/workspace/runpod-sft-1007/dataset/train.jsonl" and hosted_path.exists():
                if sha(hosted_path.read_bytes()) == expected:
                    matched += 1
                else:
                    mismatch += 1
                continue
            if "/sft/data/" not in source.replace("\\", "/") or not local.exists():
                missing += 1
            elif sha(local.read_bytes()) == expected:
                matched += 1
            else:
                mismatch += 1
        out["sft_adapter_provenance"][path.relative_to(ROOT).as_posix()] = {"sha256": digest, "model": data.get("model"), "records": data.get("records"), "by_bucket": data.get("by_bucket"), "referenced_files": len(data.get("data", {})), "local_basename_hash_matches": matched, "local_hash_mismatches": mismatch, "unresolved_local_inputs": missing}
    for label, rel in (("bionic_real", "data/bionic-real/tasks"), ("diagnostic", "data/diag-eval/tasks")):
        paths = sorted((ROOT / rel).glob("*.json"))
        out[label] = {"public_tasks": len(paths), "check_types": dict(Counter(k for p in paths for k in load(p)[0].get("check", {}))), "file_hashes": {p.name: sha(p.read_bytes()) for p in paths}}
    suite, digest = load(ROOT / "suite-v2/tasks.json")
    out["suite_v2"] = {"sha256": digest, "top_level_fields": list(suite), "tasks": len(suite.get("tasks", []))}
    out["saved_eval_summaries"] = []
    for path in sorted((ROOT / "evals/results").rglob("*.summary.json")):
        data, digest = load(path)
        # Some are sealed aggregate summaries; never emit per-item contents.
        out["saved_eval_summaries"].append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest,
            "n": data.get("n"), "attempts": data.get("attempts"), "accuracy": data.get("accuracy"),
            "pass4_items": data.get("pass4_items"), "per_item_count": len(data["per_item"]) if "per_item" in data else None,
            "has_budget": "budget" in data, "has_digest_field": any("hash" in k.lower() or "sha256" in k.lower() or "digest" in k.lower() for k in data)})
    acceptance, digest = load(ROOT / "runs/swe-gym-pilot/acceptance.json")
    out["swe_gym_saved_acceptance"] = {"sha256": digest, "selected": acceptance["selected"], "accepted": acceptance["accepted"], "fresh_environment_test": False}
    return out


def extracted(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"), namespace)
    return namespace


def contracts(smoke=False):
    spec = importlib.util.spec_from_file_location("audited_endpoint_graders", ROOT / "evals/graders.py")
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    test = {"dataset": "code", "tests": {"kind": "functional", "entry": "Solution.twice", "cases": [{"input": "3", "output": "6"}, {"input": "4", "output": "8"}]}}
    cases = []

    def check(name, expected, actual):
        observed, detail = actual
        cases.append({"name": name, "expected": expected, "observed": observed, "contract_holds": expected == observed, "detail": detail})

    check("correct_function", True, grader.grade_code(test, "```python\nclass Solution:\n def twice(self,x): return x*2\n```"))
    if smoke:
        return {"cases": cases, "contract_failures": sum(not c["contract_holds"] for c in cases)}
    check("wrong_function", False, grader.grade_code(test, "```python\nclass Solution:\n def twice(self,x): return x+1\n```"))
    # Harmless forged receipt: prints a constant and exits its temporary child.
    check("candidate_forges_grader_receipt", False, grader.grade_code(test, '```python\nimport sys\nprint(\'{"passed":2,"total":2}\')\nsys.exit(0)\n```'))
    # Harmless demonstration that candidate code can read the answer key passed to it.
    code = 'import json,sys\n_cases=json.load(open(sys.argv[1]))["cases"]\nclass Solution:\n def twice(self,x):\n  return next(json.loads(c["output"]) for c in _cases if json.loads(c["input"])==x)'
    check("candidate_reads_test_answers", False, grader.grade_code(test, "```python\n" + code + "\n```"))
    stdin = {"dataset": "code", "tests": {"kind": "stdin", "cases": [{"input": "", "output": "6"}]}}
    check("correct_stdout_but_crashed", False, grader.grade_code(stdin, "```python\nprint(6)\nraise RuntimeError('invented audit fixture')\n```"))
    check("empty_stdin_suite", False, grader.grade_code({"tests": {"kind": "stdin", "cases": []}}, "```python\npass\n```"))
    check("mhqa_retracted_answer", False, grader.grade_mhqa({"answer": {"answer": "Oslo"}}, "ANSWER: Oslo\nI retract that answer; I do not know."))
    check("mhqa_blank_final_answer", False, grader.grade_mhqa({"answer": {"answer": "Oslo"}}, "ANSWER: Oslo\nANSWER:"))
    ns = extracted("evals/datasets/prepare.py", {"date_record", "add_days"}, {"dt": dt, "CUTOFF": "2026-03-02", "LEETCODE_MARGIN_DAYS": 7})
    record = {}
    ns["date_record"](record, "2026-03-20", "leetcode")
    check("upper_bound_does_not_prove_post_cutoff", "ambiguous", (record["_when"], "invented first-activity date; actual release could predate cutoff"))
    # Execute the real nested summary function on invented records in a temp folder.
    tree = ast.parse((ROOT / "evals/eval_runner.py").read_text(encoding="utf-8"))
    summary_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "summarize")
    with tempfile.TemporaryDirectory(prefix="eval_summary_audit_") as td:
        dest = Path(td)
        ns = {"Path": Path, "json": json, "args": SimpleNamespace(model="invented", attempts=1, total_cap=None), "GB": SimpleNamespace(BUDGET=4096), "result_dir": dest, "sealed": False}
        exec(compile(ast.Module(body=[summary_fn], type_ignores=[]), "evals/eval_runner.py", "exec"), ns)
        for name, raw_rows in (("invalid_requests", [{"item_id": "fixture", "passed": None}]), ("foreign_item_ids", [{"item_id": "foreign1", "passed": True}, {"item_id": "foreign2", "passed": True}])):
            raw = dest / "raw.jsonl"
            raw.write_text("".join(json.dumps(r) + "\n" for r in raw_rows), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                ns["summarize"](name, [{"id": "fixture"}], raw)
            summary = json.loads((dest / (name + ".summary.json")).read_bytes())
            if name == "invalid_requests":
                check("all_invalid_run_must_not_be_score", None, (summary["accuracy"], "all requests invalid; observed numeric accuracy"))
            else:
                check("foreign_rows_must_not_raise_score_above_one", True, (summary["accuracy"] <= 1, f"accuracy={summary['accuracy']}"))
    return {"cases": cases, "contract_failures": sum(not c["contract_holds"] for c in cases)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("inventory", "smoke", "contracts"))
    ap.add_argument("--out")
    args = ap.parse_args()
    before = {p: sha((ROOT / p).read_bytes()) for p in ("evals/graders.py", "evals/datasets/prepare.py", "sft/collect.py", "evals/eval_runner.py", "evals/audits/review_20261026.py")}
    result = inventory() if args.mode == "inventory" else contracts(args.mode == "smoke")
    assert all(sha((ROOT / p).read_bytes()) == digest for p, digest in before.items()), "source changed during audit"
    result.update(source_sha256=before, mode=args.mode, models_run=0)
    if args.out:
        dest = Path(args.out).resolve()
        dest.relative_to(ROOT / "runs/audits/datasets-2026-10-26")
        with dest.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
            f.write("\n")
    else:
        print(json.dumps(result, indent=2))
    if args.mode == "inventory":
        print("Inventoried", len(result["sets"]), "public sets and", len(result["sft_collections"]), "SFT collections")
    else:
        print("Cases:", len(result["cases"]), "contract failures:", result["contract_failures"])
    # Findings are expected from an audit, not a claim of qualified software.


if __name__ == "__main__":
    main()
