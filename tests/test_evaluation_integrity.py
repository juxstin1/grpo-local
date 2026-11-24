"""Only invented fixtures; no production holdout data or model imports."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
from evaluation_integrity import (CANARY, MANIFEST_SCHEMA, ExternalEvaluatorRequired,
                                  IntegrityError, load_verified_tasks, reserved_seeds,
                                  validate_export_policy)


class EvaluationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = self.root / "fixtures"
        self.tasks.mkdir()
        self.manifest = self.root / "operator-manifest.json"
        self.document = {"schema": "gym_v1", "family": "invented", "canary": CANARY,
                         "seeds": [123], "items": [{"name": "invented-task", "family": "invented",
                         "seed": 123, "prompt": "Invented prompt", "ground_truth": "alpha",
                         "tools": ["read_file"], "checks": []}]}
        self.write_document(self.document)
        self.trust_fixture()

    def write_document(self, document, name="invented.json"):
        body = {key: value for key, value in document.items() if key != "sha256"}
        canonical = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
        sealed = {**body, "sha256": hashlib.sha256(canonical).hexdigest()}
        (self.tasks / name).write_text(json.dumps(sealed), encoding="utf-8")
        return sealed

    def trust_fixture(self):
        # Test fixture author represents the operator; production has no such helper.
        registry = {"schema": MANIFEST_SCHEMA, "files": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in self.tasks.iterdir()}}
        self.manifest.write_text(json.dumps(registry), encoding="utf-8")

    def load(self):
        return load_verified_tasks(self.tasks, self.manifest)

    def test_valid_fixture_and_reserved_seed(self):
        self.assertEqual(self.load()[0][1]["ground_truth"], "alpha")
        self.assertEqual(reserved_seeds(self.tasks, self.manifest), {("invented", 123)})

    def test_replacing_embedded_hash_cannot_replace_external_trust(self):
        changed = copy.deepcopy(self.document)
        changed["items"][0]["ground_truth"] = "beta"
        self.write_document(changed)
        with self.assertRaisesRegex(IntegrityError, "trusted raw SHA256"):
            self.load()

    def test_missing_and_extra_files_fail_closed(self):
        (self.tasks / "invented.json").rename(self.tasks / "extra.json")
        with self.assertRaisesRegex(IntegrityError, "inventory mismatch"):
            self.load()

    def test_extra_file_alone_fails(self):
        (self.tasks / "ignored.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(IntegrityError, "inventory mismatch"):
            self.load()

    def test_missing_file_alone_fails(self):
        (self.tasks / "invented.json").unlink()
        with self.assertRaisesRegex(IntegrityError, "inventory mismatch"):
            self.load()

    def test_no_automatic_trust(self):
        with self.assertRaisesRegex(IntegrityError, "operator-supplied"):
            load_verified_tasks(self.tasks, None)
        with self.assertRaises(IntegrityError):
            load_verified_tasks(self.tasks, self.root / "missing.json")

    def test_empty_invalid_and_ambiguous_manifest(self):
        for raw in ['{}', '{"schema":"evaluation-integrity-v1","files":{}}',
                    '{"schema":"evaluation-integrity-v1","files":{},"files":{}}',
                    'not json', '[]']:
            with self.subTest(raw=raw):
                self.manifest.write_text(raw, encoding="utf-8")
                with self.assertRaises(IntegrityError):
                    self.load()

    def test_manifest_inside_task_directory_rejected(self):
        inside = self.tasks / "manifest.json"
        inside.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(IntegrityError, "outside"):
            load_verified_tasks(self.tasks, inside)

    def test_path_traversal_and_alias_names_rejected(self):
        for name in ["../outside.json", "sub/task.json", "sub\\task.json", "C:task.json", "NUL.json"]:
            with self.subTest(name=name):
                self.manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA,
                                                    "files": {name: "0" * 64}}), encoding="utf-8")
                with self.assertRaises(IntegrityError):
                    self.load()

    def test_case_insensitive_filename_aliases_rejected(self):
        self.manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA,
                                            "files": {"task.json": "0" * 64,
                                                      "TASK.json": "0" * 64}}), encoding="utf-8")
        with self.assertRaisesRegex(IntegrityError, "duplicate trusted"):
            self.load()

    def test_reparse_and_symbolic_link_attributes_rejected(self):
        # Mock OS attributes rather than requesting Windows symlink privileges.
        for info in [SimpleNamespace(st_mode=0o040755, st_file_attributes=0x400),
                     SimpleNamespace(st_mode=0o120777, st_file_attributes=0)]:
            with self.subTest(info=info), mock.patch.object(Path, "lstat", return_value=info):
                with self.assertRaisesRegex(IntegrityError, "Links and reparse points"):
                    self.load()

    def test_errors_do_not_echo_private_values(self):
        changed = copy.deepcopy(self.document)
        changed["items"][0]["name"] = "private-task/secret"
        changed["items"][0]["ground_truth"] = "private-answer"
        self.write_document(changed)
        self.trust_fixture()
        with self.assertRaises(IntegrityError) as caught:
            self.load()
        self.assertNotIn("private", str(caught.exception))

    def test_nonfinite_json_rejected(self):
        self.manifest.write_text('{"schema":"evaluation-integrity-v1","files":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(IntegrityError, "Nonfinite"):
            self.load()

    def test_invalid_document_schema_rejected_even_when_raw_hash_is_trusted(self):
        mutations = [("canary", "wrong"), ("items", []), ("seeds", []),
                     ("seeds", [123, 456]), ("seeds", [True]), ("seeds", [123, 123])]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.document)
                changed[key] = value
                self.write_document(changed)
                self.trust_fixture()
                with self.assertRaises(IntegrityError):
                    self.load()

    def test_required_item_fields_and_types(self):
        changes = [("seed", True), ("family", "other"), ("name", "../escape"),
                   ("prompt", ""), ("ground_truth", None), ("tools", [1]),
                   ("tools", ["read_file", "read_file"]), ("checks", "not a list"),
                   ("checks", [{}])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.document)
                changed["items"][0][key] = value
                self.write_document(changed)
                self.trust_fixture()
                with self.assertRaises(IntegrityError):
                    self.load()
        del changed["items"][0]["checks"]
        self.write_document(changed)
        self.trust_fixture()
        with self.assertRaises(IntegrityError):
            self.load()

    def test_duplicate_tasks_rejected(self):
        self.write_document(self.document, "second.json")
        self.trust_fixture()
        with self.assertRaisesRegex(IntegrityError, "Duplicate task"):
            self.load()

    def test_duplicate_document_keys_rejected(self):
        path = self.tasks / "invented.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"schema":', '"schema":"gym_v1","schema":'),
                        encoding="utf-8")
        self.trust_fixture()
        with self.assertRaisesRegex(IntegrityError, "Duplicate JSON"):
            self.load()

    def test_bad_embedded_hash_rejected(self):
        sealed = self.write_document(self.document)
        sealed["sha256"] = "0" * 64
        (self.tasks / "invented.json").write_text(json.dumps(sealed), encoding="utf-8")
        self.trust_fixture()
        with self.assertRaisesRegex(IntegrityError, "Embedded gym seal mismatch"):
            self.load()

    def test_sidecar_integrity_and_agreement(self):
        sealed = self.write_document(self.document)
        sidecar = self.tasks / "invented.sha256"
        sidecar.write_text(sealed["sha256"], encoding="utf-8")
        self.trust_fixture()
        self.assertEqual(len(self.load()), 1)
        sidecar.write_text("0" * 64, encoding="utf-8")
        self.trust_fixture()
        with self.assertRaisesRegex(IntegrityError, "sidecar mismatch"):
            self.load()

    def test_sealed_policy_rejects_all_in_process_execution(self):
        validate_export_policy(False, transcripts=True, task_level_exports=True)
        with self.assertRaises(ExternalEvaluatorRequired):
            validate_export_policy(True)
        for kwargs in [{"transcripts": True}, {"task_level_exports": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(IntegrityError, "forbids"):
                validate_export_policy(True, **kwargs)


if __name__ == "__main__":
    unittest.main()
