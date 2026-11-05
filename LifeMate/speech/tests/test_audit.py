import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts' / 'audit_audio.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_missing_corrupt_silence_clipping_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(audit.inspect_audio(root / 'missing.wav')['flags'], ['missing'])
            broken = root / 'broken.wav'
            broken.write_bytes(b'invalid')
            self.assertIn('bad_header', audit.inspect_audio(broken)['flags'])
            for name, value, flag in [('silent', 0, 'mostly_silent'), ('clipped', -32768, 'clipped')]:
                path = root / f'{name}.wav'
                with wave.open(str(path), 'wb') as wav:
                    wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    wav.writeframes(struct.pack('<h', value) * 1600)
                result = audit.inspect_audio(path)
                self.assertIn(flag, result['flags'])
                self.assertIn('too_short', result['flags'])
                self.assertEqual(result['duration_s'], 0.1)
                path.write_bytes(path.read_bytes()[:-2])
                self.assertIn('bad_header', audit.inspect_audio(path)['flags'])

    def test_history_indices_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(id='one', audio='audio/current.wav', history=[
                dict(role='user', audio='audio/first.wav'), dict(role='assistant', text='ok'),
                dict(role='user', audio='audio/last.wav')])
            (root / 'train.jsonl').write_text(json.dumps(row), encoding='utf-8')
            refs, ids = audit.references(root, 'train')
            self.assertEqual([r['turn_index'] for r in refs], [None, 0, 2])
            self.assertEqual(len(refs), 3)
            self.assertEqual(ids, {'one'})
            with self.assertRaises(ValueError):
                audit.relative_audio('../escape.wav', root)

    def test_denominators_and_incomplete_sample(self):
        rows = [dict(sample_id='a', duration_s=1, referenced=True, flags=[]),
                dict(sample_id='a', duration_s=None, referenced=True, flags=['missing'])]
        result = audit.summarize(rows)
        self.assertEqual(result['file_count'], 2)
        self.assertEqual(result['missing_count'], 1)
        self.assertEqual(result['incomplete_sample_count'], 1)
        self.assertIsNone(result['sample_total_duration_s']['p50'])


if __name__ == '__main__':
    unittest.main()
