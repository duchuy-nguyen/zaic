import json
from pathlib import Path
import tempfile
import unittest

from lifemate.runtime.configuration import load_config, strict_json
from lifemate.runtime.contracts import RuntimeConfig
from lifemate.runtime.manifest import build_manifest, provenance_template, validate_provenance
from lifemate.runtime.submission_check import check_submission, expected_ids, InputError


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "candidate.jsonl"

    def write(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return self.path

    def row(self, identifier="a", calls=None, state_hash="from-environment"):
        return {"id": identifier, "tool_calls": calls or [], "state_hash": state_hash}

    def codes(self, report):
        return {e["code"] for e in report["errors"]}

    def test_valid_empty_and_multiple_calls_not_subject_to_model_turn_limit(self):
        calls = [{"name": "demo.lookup", "arguments": {"query": "x"}}] * 3
        self.write([self.row("b", calls), self.row("a")])
        before = self.path.read_bytes()
        report = check_submission(self.path, {"a", "b"})
        self.assertTrue(report["valid"])
        self.assertEqual(report["checks"], "structural_only")
        self.assertIn("state_hash_authenticity", report["not_checked"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_duplicate_and_unknown_ids(self):
        self.write([self.row(), self.row(), self.row("unknown")])
        report = check_submission(self.path, {"a", "b"})
        self.assertFalse(report["valid"])
        self.assertTrue({"duplicate_id", "unknown_id", "missing_ids"} <= self.codes(report))

    def test_bad_schema_and_hash(self):
        invalid = [self.row(state_hash=""), self.row(calls=[{"name": "x", "arguments": []}]),
                   {"id": "a", "tool_calls": [], "state_hash": "x", "extra": 1},
                   {"id": 7, "tool_calls": {}, "state_hash": None}, []]
        self.write(invalid)
        codes = self.codes(check_submission(self.path, {"a"}))
        self.assertTrue({"invalid_state_hash", "invalid_tool_call", "invalid_row_fields",
                         "invalid_id", "tool_calls_must_be_list", "row_must_be_object"} <= codes)

    def test_malformed_blank_duplicate_keys_and_nonfinite_json(self):
        for line in ('{bad}\n', '\n', '{"id":"a","id":"b"}\n',
                     '{"id":"a","tool_calls":[],"state_hash":NaN}\n'):
            self.path.write_text(line)
            self.assertIn("invalid_json", self.codes(check_submission(self.path, {"a"})))

    def test_invalid_utf8_and_bom(self):
        for data in (b'\xff', b'\xef\xbb\xbf{}\n'):
            self.path.write_bytes(data)
            self.assertFalse(check_submission(self.path, {"a"})["valid"])

    def test_cli_exit_codes_and_no_dataset_content_in_report(self):
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[2]
        expected = self.root / "ids.jsonl"
        expected.write_text('{"id":"a","context":{"secret":"never-print-this"}}\n')
        command = [sys.executable, str(root / "scripts/ops/check_submission.py"),
                   str(self.path), "--expected-ids", str(expected)]
        self.write([self.row()])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(json.loads(result.stdout)["valid"])
        self.assertNotIn("never-print-this", result.stdout)
        self.write([self.row("unknown")])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        expected.write_text("bad-json")
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
        config_command = [sys.executable, str(root / "scripts/ops/check_config.py"),
                          str(root / "configs/ops/runtime.example.json")]
        self.assertEqual(subprocess.run(config_command, capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(config_command + ["--provenance", str(expected)],
                                        capture_output=True).returncode, 2)

    def test_json_numeric_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"number":1e400}')

    def test_size_boundary(self):
        self.write([self.row()])
        size = self.path.stat().st_size
        self.assertTrue(check_submission(self.path, {"a"}, size)["valid"])
        self.assertIn("file_size_exceeded", self.codes(check_submission(self.path, {"a"}, size - 1)))

    def test_expected_id_loader_ignores_other_fields_and_rejects_duplicates(self):
        self.write([{"id": "a", "context": {"private": "do-not-log"}}, {"id": "b"}])
        self.assertEqual(expected_ids(self.path), {"a", "b"})
        for rows in ([], [{"id": "a"}, {"id": "a"}], [{"id": ""}], [{"id": 1}]):
            self.write(rows)
            with self.assertRaises(InputError):
                expected_ids(self.path)

    def test_empty_submission_is_missing_ids(self):
        self.write([])
        self.assertIn("missing_ids", self.codes(check_submission(self.path, {"a"})))

    def test_config_requires_exact_fields_and_strict_types(self):
        from dataclasses import asdict
        good = asdict(RuntimeConfig())
        self.path.write_text(json.dumps(good))
        self.assertEqual(load_config(self.path), RuntimeConfig())
        for changed in ({**good, "unknown": 1}, {"max_steps": 1},
                        {**good, "max_repairs": True}, {**good, "sample_deadline_seconds": 0}):
            self.path.write_text(json.dumps(changed))
            with self.assertRaises((ValueError, TypeError)):
                load_config(self.path)
        self.path.write_text('{"max_steps":1,"max_steps":2}')
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_manifest_lists_missing_provenance_without_claiming_readiness(self):
        manifest = build_manifest(self.root, RuntimeConfig(), mode="development_mock_only")
        self.assertFalse(manifest["provenance_complete"])
        self.assertFalse(manifest["official_submission"])
        self.assertIn("model.base_revision", manifest["missing_provenance"])
        self.assertIsNone(manifest["provenance"]["model"]["base_revision"])
        self.assertEqual(manifest["config"]["max_repairs"], 2)
        self.assertEqual(manifest["manifest_version"], 1)

    def test_provenance_rejects_unknown_secrets_bad_hash_context_and_types(self):
        for section, key, value in (("model", "api_key", "private"),
                                    ("model", "prompt_sha256", "bad"),
                                    ("inference", "context_limit", 16385),
                                    ("inference", "seed", True),
                                    ("model", "base_revision", "https://user:secret@server")):
            metadata = provenance_template()
            metadata[section][key] = value
            with self.assertRaises(ValueError):
                validate_provenance(metadata)

    def test_manifest_includes_valid_supplied_identifiers(self):
        metadata = provenance_template()
        metadata["data"]["split_manifest_sha256"] = "a" * 64
        metadata["inference"]["seed"] = 0
        metadata["inference"]["context_limit"] = 16384
        result = build_manifest(self.root, RuntimeConfig(), mode="development_mock_only", provenance=metadata)
        self.assertEqual(result["provenance"]["inference"]["seed"], 0)
        self.assertNotIn("inference.seed", result["missing_provenance"])


if __name__ == "__main__":
    unittest.main()
