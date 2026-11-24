"""Invented offline fixtures only: no source datasets, private tasks, or models."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "evals/datasets/registry.py"
spec = importlib.util.spec_from_file_location("dataset_registry_under_test", REGISTRY_FILE)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
# Load builder with its sibling import supplied explicitly; avoid path/module collisions.
builder_tree = ast.parse((ROOT / "evals/datasets/build_registry.py").read_text(encoding="utf-8"))
builder_ns = {"R": R, "Path": Path, "dt": __import__("datetime")}
exec(compile(ast.Module(body=[n for n in builder_tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"],
                        type_ignores=[]), "build_registry.py", "exec"), builder_ns)
build = builder_ns["build"]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(R.canonical(value) + b"\n")


def task(uid="old:one", prompt="Compute invented one", answer="1", root="math:train", source="invented/A"):
    return {"uid": uid, "dataset": "math", "prompt": prompt, "answer": answer,
            "root_id": root, "source": source, "first_published": "2025-01-01", "date_basis": "exact"}


def fixture(root, rows=None):
    rows = rows if rows is not None else [task()]
    for path in R.TRAIN_SOURCES:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        selected = rows if "math." in path else [{"uid": "code:old", "dataset": "code", "source": "invented/code",
            "root_id": "code:train", "prompt": "Echo invented input", "tests": {"kind": "stdin", "cases": [{"input": "a", "output": "a"}]}}]
        target.write_bytes(b"".join(R.canonical(row) + b"\n" for row in selected))
    for index, path in enumerate(R.PUBLIC_SETS):
        row = task(uid=f"eval:{index}", prompt=f"Invented evaluation {index}", root=f"math:eval:{index}")
        row["id"] = row["uid"]
        put(root / path, {"items": {"math": [row]}})
    for index, path in enumerate(R.SEALED_MANIFESTS):
        hashes = {f"sealed:root:{index}": hashlib.sha256(f"invented-{index}".encode()).hexdigest()}
        put(root / path, {"name": Path(path).parent.name, "role": "eval", "root_field": "root_id", "n_root_tasks": 1, "n_items": 1,
            "canary": "INVENTED-NO-PRIVATE-TASKS", "root_hashes": hashes, "merkle_root": R._merkle(hashes.values()),
            "holdout_sha256": "a" * 64, "roots_sha256": "b" * 64})


def prepare_functions(*names):
    tree = ast.parse((ROOT / "evals/datasets/prepare.py").read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"DR": R, "CUTOFF": "2026-03-02", "Path": Path, "hashlib": hashlib, "json": json,
          "collections": __import__("collections"), "time": time, "log": logging.getLogger("fixture")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "prepare.py", "exec"), ns)
    return ns


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        self.out = Path(self.temp.name) / "derived"
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_smoke_roundtrip(self):
        path = build(self.root, self.out)
        registry = R.load_registry(path)
        self.assertEqual(len(registry.load_pool(self.out / "pools/math.train.jsonl")), 1)
        self.assertTrue({f"math:eval:{i}" for i in range(4)} <= registry.forbidden_roots)
        self.assertIn("sealed:root:0", registry.forbidden_roots)

    def test_required_registry(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(R.RegistryError):
            R.load_registry()

    def test_explicit_env_registry(self):
        path = build(self.root, self.out)
        with patch.dict(os.environ, {R.ENV: str(path)}):
            self.assertEqual(R.load_registry().path, path)

    def test_missing_required_manifest_fails_before_output(self):
        (self.root / R.SEALED_MANIFESTS[0]).unlink()
        with self.assertRaises(R.RegistryError):
            build(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_corrupt_manifest_fails(self):
        path = self.root / R.SEALED_MANIFESTS[0]
        document = R.decode(path.read_bytes())
        document["merkle_root"] = "0" * 64
        put(path, document)
        with self.assertRaises(R.RegistryError):
            build(self.root, self.out)

    def test_no_private_body_reads(self):
        original = Path.read_bytes
        def guarded(path):
            self.assertNotIn(path.name, ("holdout.jsonl", "roots.json"))
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            build(self.root, self.out)

    def test_changed_raw_source_rejected(self):
        path = build(self.root, self.out)
        source = self.root / R.TRAIN_SOURCES[0]
        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(R.RegistryError, "raw SHA256"):
            R.load_registry(path)

    def test_removed_exclusions_rejected(self):
        path = build(self.root, self.out)
        doc = R.decode(path.read_bytes())
        doc["exclusions"]["roots"] = []
        put(path, doc)
        with self.assertRaises(R.RegistryError):
            R.load_registry(path)

    def test_removed_source_manifest_rejected(self):
        path = build(self.root, self.out)
        doc = R.decode(path.read_bytes())
        doc["sources"] = [s for s in doc["sources"] if s["path"] != R.SEALED_MANIFESTS[0]]
        put(path, doc)
        with self.assertRaises(R.RegistryError):
            R.load_registry(path)

    def test_quarantine_cannot_be_silently_removed(self):
        fixture(self.root, [task(root="math:eval:0")])
        path = build(self.root, self.out)
        doc = R.decode(path.read_bytes())
        doc["quarantine"] = []
        put(path, doc)
        with self.assertRaisesRegex(R.RegistryError, "Quarantine"):
            R.load_registry(path)

    def test_public_and_sealed_overlap_rejected(self):
        document = {"items": {"math": [{**task(root="sealed:root:0"), "id": "eval"}]}}
        put(self.root / R.PUBLIC_SETS[0], document)
        with self.assertRaisesRegex(R.RegistryError, "Public evaluation/sealed"):
            build(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_provenance_pinned_and_preserved(self):
        put(self.root / "evals/datasets/logs/prepare-20260101T000000Z.json", {"provenance": {"math": {"revision": "A"}}})
        put(self.root / "evals/datasets/logs/latest.json", {"provenance": {"code": {"revision": "B"}}})
        path = build(self.root, self.out)
        registry = R.load_registry(path)
        self.assertEqual(set(registry.document["provenance"]), {"math", "code"})
        doc = R.decode(path.read_bytes())
        doc["provenance"] = {}
        put(path, doc)
        with self.assertRaisesRegex(R.RegistryError, "Provenance"):
            R.load_registry(path)

    def test_identity_collisions_become_distinct(self):
        fixture(self.root, [task(answer="1"), task(answer="2"), task(prompt="Different problem", answer="2")])
        path = build(self.root, self.out)
        rows = R.load_registry(path).load_pool(self.out / "pools/math.train.jsonl")
        self.assertEqual(len({r["uid"] for r in rows}), 3)
        self.assertEqual({r["original_uid"] for r in rows}, {"old:one"})

    def test_exact_copies_deduplicated_with_lineage(self):
        rows = [task(), task(), task(source="invented/B")]
        fixture(self.root, rows)
        path = build(self.root, self.out)
        registry = R.load_registry(path)
        result = registry.load_pool(self.out / "pools/math.train.jsonl")
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["source_lineage"]), 2)
        self.assertEqual(registry.document["pools"][0]["exact_duplicates_removed"], 2)

    def test_input_order_does_not_change_ids(self):
        rows = [task(source="invented/B"), task(), task(answer="2")]
        self.assertEqual(R.normalize_rows(rows), R.normalize_rows(list(reversed(rows))))

    def test_overlap_quarantined_with_receipt(self):
        fixture(self.root, [task(), task(uid="overlap", prompt="Distinct overlap task", root="math:eval:0")])
        path = build(self.root, self.out)
        registry = R.load_registry(path)
        self.assertEqual(len(registry.document["quarantine"]), 1)
        self.assertEqual(registry.document["quarantine"][0]["reason"], "evaluation-root-overlap")
        self.assertEqual(len(registry.load_pool(self.out / "pools/math.train.jsonl")), 1)
        with self.assertRaises(R.RegistryError):
            registry.validate_training_rows([task(root="sealed:root:0")])

    def test_content_overlap_quarantined_even_different_root(self):
        row = task(uid="eval-copy", prompt="Invented evaluation 0", root="new:root")
        fixture(self.root, [row])
        registry = R.load_registry(build(self.root, self.out))
        self.assertEqual(registry.document["quarantine"][0]["reason"], "evaluation-content-overlap")

    def test_legacy_lcb_identity_excluded(self):
        doc = {"items": {"lcb": [{"id": "lcb-001", "question_id": "1234", "title": "Invented Problem II"}]}}
        roots, _ = R.public_exclusions(doc)
        self.assertIn("code:lcb:1234", roots)
        self.assertIn("code:lc:invented:problem", roots)

    def test_original_pool_not_accepted_for_collection(self):
        registry = R.load_registry(build(self.root, self.out))
        with self.assertRaises(R.RegistryError):
            registry.load_pool(self.root / R.TRAIN_SOURCES[0])

    def test_duplicate_ids_refused_before_mapping(self):
        with self.assertRaises(R.RegistryError):
            R.assert_unique_ids([task(), task(answer="2")])

    def test_normalized_tamper_rejected_even_rehashed(self):
        path = build(self.root, self.out)
        pool = self.out / "pools/math.train.jsonl"
        row = R.jsonl(pool.read_bytes())[0]
        row["answer"] = "invented tamper"
        row["uid"] = R.stable_uid(row)
        row["content_sha256"] = R.content_hash(row)
        pool.write_bytes(R.canonical(row) + b"\n")
        doc = R.decode(path.read_bytes())
        doc["pools"][0]["sha256"] = R.digest(pool.read_bytes())
        put(path, doc)
        with self.assertRaises(R.RegistryError):
            R.load_registry(path)

    def test_output_exclusive(self):
        build(self.root, self.out)
        with self.assertRaises(R.RegistryError):
            build(self.root, self.out)

    def test_raw_lf_hash(self):
        write = prepare_functions("write_jsonl")["write_jsonl"]
        path = Path(self.temp.name) / "lf.jsonl"
        sha = write(path, [{"value": "café\nnext"}])
        self.assertNotIn(b"\r\n", path.read_bytes())
        self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_late_upper_bound_not_post(self):
        date_record = prepare_functions("date_record")["date_record"]
        for kind in ("upper", "leetcode"):
            row = {}
            date_record(row, "2026-08-01", kind)
            self.assertEqual(row["_when"], "ambiguous")
        row = {}
        date_record(row, "2026-08-01", "exact")
        self.assertEqual(row["_when"], "post")

    def test_prepare_collision_identity(self):
        assign = prepare_functions("assign_content_ids")["assign_content_ids"]
        rows = [{**task(answer="1"), "_when": "pre"}, {**task(answer="2"), "_when": "pre"}]
        self.assertEqual(len({r["uid"] for r in assign(rows)}), 2)

    def test_partial_provenance_merged(self):
        merge = prepare_functions("merge_provenance")["merge_provenance"]
        self.assertEqual(merge({"MathArena/A": {"revision": "old"}}, {"code/A": {"revision": "new"}}),
                         {"MathArena/A": {"revision": "old"}, "code/A": {"revision": "new"}})

    def test_seal_failure_prevents_pool_publication(self):
        ns = prepare_functions("process", "assign_content_ids", "public", "write_jsonl")
        row = {**task(), "_when": "post", "first_published": "2026-08-01", "emit": True}
        ns.update(LOADERS={"math": lambda: [row]}, SKEL={"math": (5, .7)}, VERSION="fixture",
                  R=SimpleNamespace(cluster=lambda rows, *args: {r["uid"]: r["root_id"] for r in rows}),
                  DROPS=__import__("collections").defaultdict(__import__("collections").Counter), hfrac=lambda _: 0,
                  HOLDOUT_FRAC=.4, HERE=Path(self.temp.name) / "work", SEALED=Path(self.temp.name) / "sealed",
                  POOLS=Path(self.temp.name) / "unpublished", PUBLIC=("uid", "root_id", "prompt", "answer"),
                  sh=lambda action, *args: (1 if action == "verify" else 0, "invented"))
        with self.assertRaisesRegex(SystemExit, "no pools published"):
            ns["process"]("math")
        self.assertFalse(ns["POOLS"].exists())

    def test_sft_entrypoints_no_model_gate_before_import(self):
        for relative in ("sft/collect.py", "sft/train_sft.py"):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            top_model = [n for n in tree.body if isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names)
                         or isinstance(n, ast.ImportFrom) and n.module == "transformers"]
            self.assertEqual(top_model, [])
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            ns = {"argparse": argparse, "os": os, "DR": R}
            exec(compile(ast.Module(body=[main], type_ignores=[]), relative, "exec"), ns)
            argv = [relative, "--name", "invented"] + (["--data", "missing.jsonl"] if "train_sft" in relative else [])
            with patch("sys.argv", argv), self.assertRaisesRegex(SystemExit, "Model execution disabled"):
                ns["main"]()

    def test_sft_registry_required_before_tokenizer(self):
        for relative in ("sft/collect.py", "sft/train_sft.py"):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            ns = {"argparse": argparse, "os": os, "DR": R}
            exec(compile(ast.Module(body=[main], type_ignores=[]), relative, "exec"), ns)
            argv = [relative, "--name", "invented", "--allow-model-run"]
            if "train_sft" in relative:
                argv += ["--data", "missing.jsonl"]
            with patch.dict(os.environ, {}, clear=True), patch("sys.argv", argv), self.assertRaises(R.RegistryError):
                ns["main"]()

    def test_invalid_grading_never_becomes_training_failure(self):
        tree = ast.parse((ROOT / "sft/collect.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        one = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "one")
        class InvalidGrading(Exception):
            pass
        def invalid(*args):
            raise InvalidGrading("invented unavailable executor")
        response = {"prompt_token_ids": [1], "choices": [{"token_ids": [2], "message": {"content": "invented"}}]}
        request = SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)
        ns = {"prompts": {"test": ([], [1])}, "a": SimpleNamespace(served_model="stub", model="stub",
              temperature=0, top_p=1, base="http://127.0.0.1:1", api_key=None), "MAX_TOKENS": 10,
              "requests": SimpleNamespace(post=lambda *args, **kwargs: request),
              "graders": SimpleNamespace(InvalidGrading=InvalidGrading, grade=invalid),
              "tok": SimpleNamespace(decode=lambda *args, **kwargs: "invented")}
        exec(compile(ast.Module(body=[one], type_ignores=[]), "collect.py", "exec"), ns)
        result = ns["one"]({"uid": "test"}, 1)
        self.assertTrue(result["status"].startswith("invalid-grading:"))
        self.assertNotIn("passed", result)
        self.assertNotIn("rec", result)
        self.assertNotIn("fail", result)

    def test_collection_preflight_before_tokenizer(self):
        tree = ast.parse((ROOT / "sft/collect.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        class InvalidGrading(Exception):
            pass
        def reject(records):
            self.assertEqual(records, [task()])
            raise InvalidGrading("unsupported tests")
        registry = SimpleNamespace(validate_training_rows=lambda _: None)
        ns = {"argparse": argparse, "os": os,
              "DR": SimpleNamespace(ENV=R.ENV, load_registry=lambda _: registry, assert_unique_ids=R.assert_unique_ids),
              "load_tasks": lambda _: [task()], "graders": SimpleNamespace(preflight=reject),
              "admission": SimpleNamespace(require_registered_tasks=lambda *args: None, build_task_index=lambda _: {})}
        exec(compile(ast.Module(body=[main], type_ignores=[]), "collect.py", "exec"), ns)
        with patch("sys.argv", ["collect.py", "--name", "invented", "--allow-model-run"]), self.assertRaises(InvalidGrading):
            ns["main"]()


if __name__ == "__main__":
    unittest.main()
