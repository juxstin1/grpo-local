"""Offline authored coverage fixtures; no models, calls, or new suite tasks."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coverage_under_test", ROOT / "ops/check_suite_coverage.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def write(path, value):
    path.write_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def task(number):
    return {"name": f"authored-{number}", "root_id": f"invented/root-{number}", "seed": number,
            "prompt": f"Perform invented fixture {number}", "files": {"input.txt": f"input fixture {number}"},
            "contract": {"outputs": {"result.txt": {"kind": "text_exact", "expected": f"result {number}"}}}}


class SuiteCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.suite_path = Path(self.temp.name) / "tasks.json"
        self.coverage_path = Path(self.temp.name) / "coverage.json"
        self.suite = {"version": "invented-fixtures-v1", "sealed": False, "tasks": [task(1), task(2)]}
        self.manifest = {
            "schema": C.SCHEMA, "suite_file": "suite-v2/tasks.json", "suite_version": self.suite["version"],
            "minimum_independent_roots": 2, "required_capabilities": list(C.CAPABILITIES),
            "roots": [{"root_id": t["root_id"], "task_name": t["name"], "capabilities": list(C.CAPABILITIES),
                       "evidence": "Invented test declaration; not evidence of model capability."} for t in self.suite["tasks"]],
            "missing_capabilities": []}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        write(self.suite_path, self.suite)
        self.manifest["suite_sha256"] = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        write(self.coverage_path, self.manifest)

    def check(self):
        return C.check_coverage(self.suite_path, self.coverage_path)

    def run_cli(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = C.main(["--suite", str(self.suite_path), "--coverage", str(self.coverage_path), *extra])
        return status, json.loads(output.getvalue())

    def test_positive_independent_fixture_inventory(self):
        report = self.check()
        self.assertTrue(report["broad_coverage"])
        self.assertEqual(report["independent_fixtures"], 2)
        self.assertTrue(all(v["independent_roots"] == 2 for v in report["capabilities"].values()))
        self.assertEqual(report["status"], "development-only")
        self.assertFalse(report["sealed"])
        self.assertIn("no model performance", report["limitation"])
        self.assertEqual(self.run_cli("--require-broad")[0], 0)

    def test_actual_five_roots_remain_development_only(self):
        report = C.check_coverage(ROOT / "suite-v2/tasks.json", ROOT / "suite-v2/coverage.json")
        self.assertEqual(report["roots"], 5)
        self.assertEqual(report["independent_fixtures"], 5)
        self.assertFalse(report["broad_coverage"])
        self.assertEqual(report["status"], "development-only")
        self.assertEqual(set(report["missing_capabilities"]), {
            "long-context", "conflicting-source-injection",
            "underspecified-clarification", "multistep-stateful-tools"})
        self.assertEqual(report["capabilities"]["scoped-preservation"]["independent_roots"], 5)
        self.assertEqual(report["capabilities"]["tool-error-recovery"]["independent_roots"], 1)

    def test_partial_inventory_default_passes_strict_fails(self):
        for mapping in self.manifest["roots"]:
            mapping["capabilities"] = ["scoped-preservation"]
        self.manifest["missing_capabilities"] = sorted(set(C.CAPABILITIES) - {"scoped-preservation"})
        self.save()
        self.assertEqual(self.run_cli()[0], 0)
        status, report = self.run_cli("--require-broad")
        self.assertEqual(status, 2)
        self.assertFalse(report["broad_coverage"])
        self.assertEqual(len(report["missing_capabilities"]), 8)

    def test_raw_suite_hash_change_is_invalid(self):
        self.suite_path.write_bytes(self.suite_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(C.CoverageError, "SHA256"):
            self.check()
        self.assertEqual(self.run_cli()[0], 1)

    def test_task_edit_requires_coverage_review(self):
        self.suite["tasks"][0]["prompt"] = "Changed instruction"
        write(self.suite_path, self.suite)
        with self.assertRaises(C.CoverageError):
            self.check()

    def test_wrong_version_is_invalid(self):
        self.manifest["suite_version"] = "wrong"
        self.save()
        with self.assertRaises(C.CoverageError):
            self.check()

    def test_missing_root_mapping_is_invalid(self):
        self.manifest["roots"].pop()
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "every suite root"):
            self.check()

    def test_foreign_root_mapping_is_invalid(self):
        self.manifest["roots"][0]["root_id"] = "foreign/root"
        self.save()
        with self.assertRaises(C.CoverageError):
            self.check()

    def test_duplicate_manifest_roots_are_invalid(self):
        self.manifest["roots"].append(copy.deepcopy(self.manifest["roots"][0]))
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "duplicate coverage roots"):
            self.check()

    def test_duplicate_suite_roots_are_invalid(self):
        self.suite["tasks"].append(copy.deepcopy(self.suite["tasks"][0]))
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "duplicate suite roots"):
            self.check()

    def test_name_only_copies_do_not_add_independence(self):
        self.suite["tasks"][1] = {**copy.deepcopy(self.suite["tasks"][0]), "name": "authored-2",
                                  "root_id": "invented/root-2", "seed": 99, "prompt": "Reworded prompt"}
        self.save()
        report = self.check()
        self.assertEqual(report["roots"], 2)
        self.assertEqual(report["independent_fixtures"], 1)
        self.assertFalse(report["broad_coverage"])
        self.assertTrue(all(v["independent_roots"] == 1 for v in report["below_minimum"].values()))

    def test_filename_only_copies_do_not_add_independence(self):
        first = copy.deepcopy(self.suite["tasks"][0])
        self.suite["tasks"][1]["files"] = {"renamed.txt": first["files"]["input.txt"]}
        self.suite["tasks"][1]["contract"] = {"outputs": {"other.txt": first["contract"]["outputs"]["result.txt"]}}
        self.save()
        self.assertEqual(self.check()["independent_fixtures"], 1)

    def test_task_name_mapping_tamper_is_invalid(self):
        self.manifest["roots"][0]["task_name"] = "wrong-task"
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "name mismatch"):
            self.check()

    def test_unknown_capability_is_invalid(self):
        self.manifest["roots"][0]["capabilities"].append("unreviewed-capability")
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "Unknown capability"):
            self.check()

    def test_duplicate_capability_is_invalid(self):
        self.manifest["roots"][0]["capabilities"].append("scoped-preservation")
        self.save()
        with self.assertRaises(C.CoverageError):
            self.check()

    def test_threshold_cannot_be_weakened(self):
        self.manifest["minimum_independent_roots"] = 1
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "exactly 2"):
            self.check()

    def test_required_capabilities_cannot_be_removed(self):
        self.manifest["required_capabilities"].remove("long-context")
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "cannot be weakened"):
            self.check()

    def test_missing_declarations_must_be_accurate(self):
        self.manifest["missing_capabilities"] = ["long-context"]
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "Declared missing"):
            self.check()

    def test_missing_files_and_corrupt_json_are_invalid(self):
        self.coverage_path.unlink()
        with self.assertRaises(C.CoverageError):
            self.check()
        self.coverage_path.write_bytes(b"{broken")
        with self.assertRaises(C.CoverageError):
            self.check()

    def test_duplicate_json_keys_are_invalid(self):
        self.coverage_path.write_bytes(b'{"schema":"suite-coverage-v1","schema":"suite-coverage-v1"}')
        with self.assertRaises(C.CoverageError):
            self.check()

    def test_private_suite_claim_is_invalid(self):
        self.suite["sealed"] = True
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "explicitly public"):
            self.check()

    def test_evidence_description_required(self):
        self.manifest["roots"][0]["evidence"] = ""
        self.save()
        with self.assertRaisesRegex(C.CoverageError, "evidence description"):
            self.check()


if __name__ == "__main__":
    unittest.main()
