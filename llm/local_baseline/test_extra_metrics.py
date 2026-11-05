"""Meaningful checks for ASR edit rates and normalization."""
import unittest
from .extra_metrics import edit_distance, transcript_errors


class TranscriptMetricsTests(unittest.TestCase):
    def test_insert_delete_substitute(self):
        self.assertEqual(edit_distance(['a', 'b'], ['a', 'c', 'd']), 2)
        self.assertEqual(edit_distance([], ['a', 'b']), 2)
        self.assertEqual(edit_distance(['a'], []), 1)

    def test_nfc_case_punctuation_and_vietnamese_accents(self):
        same = transcript_errors('Bật đèn!', 'bật đèn')
        self.assertEqual(same['word_edits'], 0)
        self.assertEqual(same['character_edits'], 0)
        different = transcript_errors('bật đèn', 'bat den')
        self.assertEqual(different['word_edits'], 2)
        self.assertGreater(different['character_edits'], 0)


if __name__ == '__main__':
    unittest.main()
