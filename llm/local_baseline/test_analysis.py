"""Check the paired/group bootstrap used to interpret model differences."""
import unittest
from pathlib import Path
import tempfile
import json
from unittest.mock import patch

from .analyze_experiments import paired, load


def run(values):
    return {"manifest": {"ids": list(values)},
            "scores": {key: {"score_spec_local": value} for key, value in values.items()}}


class PairedAnalysisTests(unittest.TestCase):
    def test_zero_difference_has_zero_interval(self):
        data = run({"a": 20, "b": 70})
        result = paired(data, data, {"a": "g1", "b": "g2"}, repeats=1000)
        self.assertEqual(result["delta_right_minus_left"], 0)
        self.assertEqual(result["ci95_group_bootstrap"], [0, 0])
        self.assertEqual(result["ties"], 2)

    def test_grouping_does_not_treat_duplicate_conversations_as_independent(self):
        left = run({"a": 0, "b": 0, "c": 0})
        right = run({"a": 100, "b": 100, "c": 0})
        result = paired(left, right, {"a": "same", "b": "same", "c": "other"}, repeats=1000)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["delta_right_minus_left"], 200 / 3)
        self.assertEqual(result["ci95_group_bootstrap"], [0, 100])
        self.assertEqual((result["right_wins"], result["right_losses"], result["ties"]), (2, 0, 1))

    def test_internal_call_cap_is_not_an_extra_official_zero_penalty(self):
        root = Path(tempfile.mkdtemp(prefix="lifemate-score-policy-"))
        (root / "run_manifest.json").write_text(json.dumps({"status": "completed", "ids": ["a"]}))
        (root / "metrics.json").write_text(json.dumps({"score_spec_local": 0, "failed_samples_counted_zero": 1}))
        (root / "scores.jsonl").write_text(json.dumps({"id": "a", "status": "call_budget_exhausted", "elapsed_s": 1,
                                                      "score_spec_local": 0, "failure_override": True}) + "\n")
        prediction = {"id": "a", "tool_calls": [{"name": "read", "arguments": {}}], "state_hash": "correct"}
        (root / "submission.jsonl").write_text(json.dumps(prediction) + "\n")
        refs = root / "refs.json"
        refs.write_text(json.dumps({"references": {"a": prediction}}))
        with patch("local_baseline.analyze_experiments.REFERENCE_PATH", refs), patch("local_baseline.analyze_experiments.catalog", return_value={}):
            result = load(root)
        self.assertEqual(result["metrics"]["score_spec_local"], 100)
        self.assertEqual(result["original_metrics"]["score_spec_local"], 0)
        self.assertFalse(result["scores"]["a"]["failure_override"])


if __name__ == "__main__":
    unittest.main()
