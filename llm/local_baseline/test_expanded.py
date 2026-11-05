"""Protect fixed sample selection and source controls for expanded evaluation."""
import copy
import json
from pathlib import Path
import unittest

from .expanded_selection import PATH, validate_selection
from .expanded_compare import OLD_GEMMA, expected_expanded_source, merged
from .data import prepare
from .full_selection import expected_full_source


class ExpandedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = json.loads(PATH.read_text(encoding="utf-8"))
        cls.split = prepare()

    def test_selection_is_full_and_rejects_repeated_ids(self):
        self.assertEqual(len(validate_selection(self.selection, self.split)), 600)
        altered = copy.deepcopy(self.selection)
        altered["ids"][1] = altered["ids"][0]
        with self.assertRaises(ValueError):
            validate_selection(altered, self.split)

    def test_seen_smoke_is_rejected(self):
        altered = copy.deepcopy(self.selection)
        altered["ids"][0] = self.split["smoke_train_fit"][0]
        with self.assertRaisesRegex(ValueError, "Previously seen"):
            validate_selection(altered, self.split)

    def test_only_expected_runner_changes(self):
        old = (OLD_GEMMA / "code_snapshot/experiment.py").read_text(encoding="utf-8")
        current = (Path(__file__).parent / "experiment.py").read_text(encoding="utf-8")
        self.assertEqual(expected_full_source(expected_expanded_source(old)), current)

    def test_old_new_overlap_cannot_inflate_sample_count(self):
        run = {"scores": {"same": {"score_spec_local": 1}}}
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            merged(run, run)


if __name__ == "__main__":
    unittest.main()
