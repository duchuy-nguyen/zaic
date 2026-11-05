import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unicodedata
import wave
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lifemate_speech.normalize import NORMALIZATION_VERSION, normalize
from lifemate_speech.cache import TranscriptCache, cache_key
from lifemate_speech.io_utils import ensure_work_path, load_audio_refs, speech_work_dir, wav_info
from lifemate_speech.asr import PhoWhisperASR, REVISION, model_weight_provenance

spec = importlib.util.spec_from_file_location('transcribe', ROOT/'scripts'/'transcribe.py')
transcribe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe)


def wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0' * int(seconds*16000))


class FakeProvider:
    """Unit-test fixture ONLY, never used by CLI or real smoke runs."""
    calls = []
    model = True
    model_load_s = 0.0
    effective_generation_config = None
    hardware = {'fixture': True}

    def __init__(self, **kwargs):
        pass

    def transcribe_batch(self, paths):
        self.calls.append(paths)
        return [dict(text_raw='  nội dung thử  ', latency_s=1/len(paths), audio_duration_s=wav_info(p)['audio_duration_s'],
                     batch_size=len(paths), batch_wall_time_s=1.0, error=None) for p in paths]


class SpeechTests(unittest.TestCase):
    def test_work_root_override_and_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected=Path(tmp).resolve()
            with patch.dict('os.environ',{'LIFEMATE_SPEECH_WORK_DIR':str(expected)}):
                self.assertEqual(speech_work_dir(),expected)
                self.assertEqual(ensure_work_path(expected/'run'),expected/'run')
                with self.assertRaises(ValueError):
                    ensure_work_path(expected.parent/'outside_speech_work')

    def test_normalization_preserves_content(self):
        text = 'Chuyển hai trăm nghìn, 08:30, ngày 5/11, Nguyễn Ánh, Wi-Fi!'
        self.assertEqual(normalize(' \n' + unicodedata.normalize('NFD', text) + '\t '), text)
        self.assertEqual(normalize('một\t\n triệu'), 'một triệu')
        self.assertEqual(normalize(''), '')
        with self.assertRaises(TypeError):
            normalize(None)

    def test_cache_key_all_components_and_order(self):
        original = cache_key('hash', 'rev', {'a': 1, 'b': 2})
        self.assertEqual(original, cache_key('hash', 'rev', {'b': 2, 'a': 1}))
        for args in [('other', 'rev', {'a': 1, 'b': 2}, NORMALIZATION_VERSION),
                     ('hash', 'rev2', {'a': 1, 'b': 2}, NORMALIZATION_VERSION),
                     ('hash', 'rev', {'a': 2, 'b': 2}, NORMALIZATION_VERSION),
                     ('hash', 'rev', {'a': 1, 'b': 2}, 'future-version')]:
            self.assertNotEqual(original, cache_key(*args))

    def test_loader_no_gold_and_history_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'train').mkdir()
            sample = dict(id='one', audio='audio/a.wav', audio_text='SECRET', label={'secret': True}, history=[
                dict(role='user', audio='audio/b.wav', audio_text='SECRET'), dict(role='assistant', text='IGNORE'),
                dict(role='user', audio='audio/c.wav', audio_text='SECRET')])
            (root/'train'/'train.jsonl').write_text(json.dumps(sample), encoding='utf-8')
            refs = load_audio_refs(root, 'train', 1)
            self.assertEqual([r['turn_index'] for r in refs], [None, 0, 2])
            self.assertNotIn('SECRET', str(refs))
            self.assertEqual(len(load_audio_refs(root,'train',sample_ids=['one'])),3)
            with self.assertRaises(ValueError):
                load_audio_refs(root,'train',sample_ids=['unknown'])
            sample['audio'] = '../escape.wav'
            (root/'train'/'train.jsonl').write_text(json.dumps(sample), encoding='utf-8')
            self.assertIsNotNone(load_audio_refs(root, 'train')[0]['input_error'])

    def test_long_audio_missing_corrupt_rejected_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav(root/'long.wav', 30.01)
            (root/'bad.wav').write_bytes(b'bad')
            provider = PhoWhisperASR()
            results = provider.transcribe_batch([root/'long.wav', root/'missing.wav', root/'bad.wav'])
            self.assertEqual(len(results), 3)
            self.assertIn('AudioTooLong', results[0]['error'])
            self.assertTrue(all(r['error'] for r in results))
            self.assertIsNone(provider.model)

    def test_cli_cache_resume_denominator_and_sorted_batches(self):
        work = speech_work_dir()
        work.mkdir(parents=True,exist_ok=True)
        FakeProvider.calls = []
        with tempfile.TemporaryDirectory(prefix='unit_test_', dir=work) as tmp:
            root = Path(tmp)
            data = root/'data'
            (data/'train').mkdir(parents=True)
            wav(data/'train'/'audio'/'long.wav', .2)
            wav(data/'train'/'audio'/'short.wav', .1)
            sample = dict(id='one', audio='audio/long.wav', audio_text='GOLD MUST NOT APPEAR', history=[
                dict(role='user', audio='audio/short.wav', audio_text='GOLD'),
                dict(role='assistant', text='x'), dict(role='user', audio='audio/missing.wav')])
            (data/'train'/'train.jsonl').write_text(json.dumps(sample), encoding='utf-8')
            args = argparse.Namespace(data_root=data, split='train', out=root/'cold', device='cpu', batch_size=2,
                                      cache_dir=root/'cache', model_cache_dir=root/'models', limit=1,
                                      resume=False, no_cache=False, local_files_only=True)
            cold = transcribe.execute(args, FakeProvider)
            self.assertEqual(cold['recorded_audio_count'], 3)
            self.assertEqual(cold['success_count'], 2)
            self.assertEqual(cold['error_count'], 1)
            self.assertEqual([p.name for p in FakeProvider.calls[0]], ['short.wav', 'long.wav'])
            args.out = root/'warm'
            warm = transcribe.execute(args, FakeProvider)
            self.assertEqual(warm['from_cache_count'], 2)
            self.assertFalse(warm['model_initialized'])
            self.assertEqual(len(FakeProvider.calls), 1)
            args.resume = True
            resumed = transcribe.execute(args, FakeProvider)
            self.assertEqual(resumed['resumed_count'], 2)
            self.assertEqual(resumed['recorded_audio_count'], 3)
            args.resume, args.no_cache, args.out = False, True, root/'no_cache'
            fresh = transcribe.execute(args, FakeProvider)
            self.assertEqual(fresh['from_cache_count'], 0)
            self.assertEqual(len(FakeProvider.calls), 2)
            cache = TranscriptCache(args.cache_dir)
            with self.assertRaises(ValueError):
                cache.put('bad', {'error': 'failure'})

    def test_resume_torn_final_line_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows.jsonl'
            row = dict(sample_id='one', kind='current', turn_index=None)
            path.write_text(json.dumps(row)+'\n{"sample_id":', encoding='utf-8')
            self.assertEqual(transcribe.read_resume(path), [row])
            path.write_text('{broken\n'+json.dumps(row)+'\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                transcribe.read_resume(path)

    def test_provider_batch_mapping_and_eos_with_structured_output(self):
        import torch
        class Features(dict):
            def to(self, device):
                return self
        class Processor:
            def __call__(self, audio, **kwargs):
                self.sizes = [len(a) for a in audio]
                return Features(input_features=torch.zeros((len(audio), 1)))
            def batch_decode(self, ids, **kwargs):
                return [str(n) for n in self.sizes]
        class Model:
            generation_config = SimpleNamespace(eos_token_id=9)
            def generate(self, **kwargs):
                return SimpleNamespace(sequences=torch.tensor([[1, 2, 9]] * len(kwargs['input_features'])))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav(root/'long.wav', .2)
            wav(root/'short.wav', .1)
            provider = PhoWhisperASR(batch_size=2)
            self.assertTrue(provider.config['return_dict_in_generate'])
            provider.model, provider.processor = Model(), Processor()
            rows = provider.transcribe_batch([root/'long.wav', root/'short.wav'])
            self.assertEqual([r['text_raw'] for r in rows], ['3200', '1600'])
            self.assertTrue(all(r['error'] is None for r in rows))
            self.assertEqual(sum(r['latency_s'] for r in rows), rows[0]['batch_wall_time_s'])

    def test_model_weight_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(model_weight_provenance(root)['verified'])
            path = root/'models--vinai--PhoWhisper-small'/'snapshots'/REVISION/'pytorch_model.bin'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'tampered unit fixture')
            with self.assertRaises(ValueError):
                model_weight_provenance(root)


if __name__ == '__main__':
    unittest.main()
