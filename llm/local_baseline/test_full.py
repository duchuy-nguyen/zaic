"""Protect full-set completeness and unchanged inference policy."""
import copy
from pathlib import Path
import unittest
from .data import prepare
from .full_selection import select, expected_full_source
from .full_compare import OLD_4B


class FullEvaluationTests(unittest.TestCase):
    def test_complement_exactly_covers_all_train(self):
        split = prepare()
        remaining = set(select("train-extra", split))
        all_train = set(select("all-train", split))
        old = set(split["val_main"])
        self.assertEqual(len(remaining),1236)
        self.assertEqual(len(all_train),1545)
        self.assertFalse(remaining & old)
        self.assertEqual(remaining | old,all_train)

    def test_missing_or_duplicated_split_members_are_rejected(self):
        split = copy.deepcopy(prepare())
        split["train_fit"].pop()
        with self.assertRaises(ValueError):
            select("all-train",split)

    def test_runner_has_only_selection_changes(self):
        before = (OLD_4B / "code_snapshot/experiment.py").read_text(encoding="utf-8")
        after = (Path(__file__).parent / "experiment.py").read_text(encoding="utf-8")
        self.assertEqual(expected_full_source(before),after)


if __name__ == '__main__':
    unittest.main()
