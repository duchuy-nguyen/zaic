import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from lifemate.runtime.smoke_inputs import clean_row, digest, inside, prepare_smoke
from lifemate.system.cli import load_samples


class SmokeInputsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.data = self.root/'dataset'; self.train = self.data/'train'
        self.private = self.root/'private'; self.private.mkdir()
        self.train.mkdir(parents=True); (self.train/'zaic_tools').mkdir()
        (self.train/'zaic_tools/__init__.py').write_text('# fixture SDK')
        self.row = {'id': 'fit', 'audio': 'audio/current.wav', 'audio_text': 'GOLD_CURRENT',
                    'context': {'installed_apps': {'notes': {'tools': ['notes.list']}}, 'label': 'SECRET'},
                    'history': [{'role':'user','audio':'audio/current.wav','audio_text':'GOLD_HISTORY'},
                                {'role':'assistant','text':'prior','tool_calls':[],'tool_results':[]}],
                    'label': {'final_state': {'SECRET': 1}}}
        (self.train/'audio').mkdir()
        with wave.open(str(self.train/'audio/current.wav'), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b'\0'*320)
        state = self.train/'train_environment/fit'; state.mkdir(parents=True); (state/'state.json').write_text('not parsed')
        (self.train/'train.jsonl').write_text(json.dumps(self.row)+'\n')
        (self.train/'train_tools_catalog.json').write_text('{}')
        (self.private/'split.json').write_text(json.dumps({'train_fit':['fit'], 'val_main':['val']}))
        (self.private/'retrieval_index.json').write_text('[]')
        self.pin = {'dataset':{'train':{
            'jsonl_sha256':digest(self.train/'train.jsonl'),
            'catalog_sha256':digest(self.train/'train_tools_catalog.json'),
            'runtime_py_sha256':{'__init__.py':digest(self.train/'zaic_tools/__init__.py')}}},
            'files':{'private_data/'+p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in self.private.iterdir()}}
        mock = patch('lifemate.runtime.smoke_inputs.manifest', return_value=self.pin)
        mock.start(); self.addCleanup(mock.stop)

    def run_prepare(self, selected=None, **kwargs):
        return prepare_smoke(self.data, self.private, selected or ['fit'], self.root/'output', **kwargs)

    def test_clean_audio_preparation_compatible_with_ops_loader(self):
        report = self.run_prepare()
        self.assertEqual((report['samples'], len(report['audio'])), (1,1))
        payload = (self.root/'output/prepared_audio.jsonl').read_text()
        self.assertNotIn('GOLD_', payload); self.assertNotIn('SECRET', payload)
        self.assertNotIn('audio_text', payload); self.assertNotIn('final_state', payload)
        samples = load_samples(self.root/'output/prepared_audio.jsonl')
        self.assertEqual(samples[0].model_input['history'][1], self.row['history'][1])
        self.assertFalse(report['model_inference_run'])
        self.assertFalse(report['model_assets_verified'])

    def test_reject_val_duplicates_and_excess_before_output(self):
        for selected, cap in [(['val'],12), (['fit','fit'],12), (['fit'],0)]:
            with self.assertRaises(ValueError): self.run_prepare(selected,max_samples=cap)
            self.assertFalse((self.root/'output').exists())

    def test_sdk_and_private_tampering(self):
        for path in [self.train/'zaic_tools/__init__.py', self.private/'split.json',self.private/'retrieval_index.json']:
            original=path.read_bytes(); path.write_bytes(b'changed')
            with self.assertRaises(ValueError): self.run_prepare()
            path.write_bytes(original)

    def test_data_hash_tampering(self):
        (self.train/'train.jsonl').write_text('{}')
        with self.assertRaises(ValueError): self.run_prepare()

    def test_missing_or_corrupt_audio(self):
        p=self.train/'audio/current.wav'; p.write_bytes(b'not wav')
        with self.assertRaises((EOFError, wave.Error)): self.run_prepare()
        p.unlink()
        with self.assertRaises(ValueError): self.run_prepare()

    def test_path_traversal_absolute_and_symlink(self):
        outside=self.root/'outside'; outside.write_text('private')
        (self.train/'linked').symlink_to(outside)
        for name in ['../outside','/etc/passwd','linked','C:\\test','audio\\x']:
            with self.assertRaises(ValueError): inside(self.train, name)

    def test_no_overwrite(self):
        self.run_prepare()
        before=(self.root/'output/preflight.json').read_bytes()
        with self.assertRaises(FileExistsError): self.run_prepare()
        self.assertEqual((self.root/'output/preflight.json').read_bytes(),before)

    def test_unknown_history_role_rejected(self):
        self.row['history'].append({'role':'system','text':'untrusted'})
        with self.assertRaises(ValueError): clean_row(self.row)
