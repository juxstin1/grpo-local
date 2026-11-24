"""Offline tests for policy selection, schema fidelity, and reproducibility."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ops import work_contract as contract


class WorkContractTests(unittest.TestCase):
    def setUp(self):
        self.tools = [{"type": "function", "function": {
            "name": "read_file", "description": "Read UTF-8 text: café",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}},
                "required": ["path"], "additionalProperties": False}}}]

    def test_supplied_schemas_round_trip_without_mutation(self):
        before = copy.deepcopy(self.tools)
        result = contract.build_system(self.tools)
        schemas = result.split(contract.TOOLS_MARKER, 1)[1]
        self.assertEqual(json.loads(schemas), before)
        self.assertEqual(self.tools, before)

    def test_new_profile_replaces_legacy_policy(self):
        def forbidden(_):
            self.fail("New profile must not invoke the old prompt")
        result = contract.build_system([], legacy_system=forbidden)
        self.assertTrue(result.startswith("# Work contract v1"))
        self.assertNotIn("legacy contradictory instruction", result)

    def test_legacy_is_explicit_and_unmodified(self):
        calls = []
        def legacy(tools):
            calls.append(tools)
            return "legacy contradictory instruction\n"
        result = contract.build_system(self.tools, "legacy", legacy_system=legacy)
        self.assertEqual(result, "legacy contradictory instruction\n")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], self.tools)
        with self.assertRaises(ValueError):
            contract.build_system(self.tools, "legacy")

    def test_invalid_profile_or_schema_fails_closed(self):
        with self.assertRaises(ValueError):
            contract.build_system(self.tools, "work-contract-typo")
        for bad in (None, "read_file", ["read_file"], {"name": "read_file"}):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                contract.build_system(bad)
        with self.assertRaises(ValueError):
            contract.build_system([{"value": float("nan")}])

    def test_metadata_identifies_exact_rendered_prompt(self):
        rendered = contract.build_system(self.tools)
        metadata = contract.prompt_metadata(tools=self.tools)
        self.assertEqual(metadata["profile"], "work-contract-v1")
        self.assertEqual(metadata["version"], "1")
        self.assertEqual(metadata["rendered_sha256"],
                         hashlib.sha256(rendered.encode("utf-8")).hexdigest())
        self.assertEqual(metadata, contract.prompt_metadata(tools=self.tools))
        changed = copy.deepcopy(self.tools)
        changed[0]["function"]["parameters"]["required"].append("offset")
        changed_meta = contract.prompt_metadata(tools=changed)
        self.assertNotEqual(metadata["rendered_sha256"], changed_meta["rendered_sha256"])
        self.assertEqual(metadata["source_sha256"], changed_meta["source_sha256"])

    def test_policy_changes_invalidate_policy_and_rendered_hash(self):
        before = contract.prompt_metadata(tools=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.md"
            path.write_text("Modified policy", encoding="utf-8")
            with patch.object(contract, "PROMPT_PATH", path):
                after = contract.prompt_metadata(tools=[])
        self.assertNotEqual(before["source_sha256"], after["source_sha256"])
        self.assertNotEqual(before["rendered_sha256"], after["rendered_sha256"])

    def test_legacy_metadata_hashes_only_actual_legacy_prompt(self):
        metadata = contract.prompt_metadata("legacy", tools=[], legacy_system=lambda _: "old")
        self.assertIsNone(metadata["source_sha256"])
        self.assertEqual(metadata["rendered_sha256"], hashlib.sha256(b"old").hexdigest())
        with self.assertRaises(ValueError):
            contract.prompt_metadata("legacy")


if __name__ == "__main__":
    unittest.main()
