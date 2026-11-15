"""Regression: a failed startup must not poison subsequent sample log paths."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from baseline50 import run

def test_failed_startup_then_next_sample():
    with TemporaryDirectory() as temp:
        root=Path(temp); package=root/'baseline50';package.mkdir()
        data=root/'data';data.mkdir();assets=root/'assets'
        (assets/'private_data').mkdir(parents=True)
        (assets/'private_data/split.json').write_text(json.dumps({'train_fit':['fit_1']}))
        (root/'RUNTIME.json').write_text(json.dumps({'placement':'serial','ASR_batch_size':2}))
        rows=[dict(id=f'train_{i}',audio=f'{i}.wav',history=[],context={}) for i in range(3)]
        config=dict(assets_dir=assets,dataset_dir=data,output_dir=root/'out',llama_server=root/'server.exe')
        class Model:
            def to(self,device):pass
        class ASR:
            def __init__(self,*args,**kw):
                self.model=Model();self.torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
            def warmup(self,*args):pass
            def transcribe(self,paths,deadline):return {},{}
        class Env:
            def __init__(self,*args,**kw):pass
            def get_state_hash(self):return 'hash'
            def get_tool_calls(self):return []
        paths=[]
        class Backend:
            def __init__(self,weights,executable,log_dir,**kw):self.directory=log_dir
            def __enter__(self):
                self.directory.mkdir(parents=True,exist_ok=True)
                with (self.directory/'server.log').open('xb') as f:f.write(b'kept')
                paths.append(self.directory)
                if len(paths)==1:raise TimeoutError('simulated first startup timeout')
                return self
            def __exit__(self,*args):pass
            def public_command(self):return ['redacted-test-server']
        def infer(row,*args,**kw):
            assert row['id']=='train_1'
            (root/'PAUSE_REQUESTED').write_text('test boundary')
            return dict(id=row['id'],tool_calls=[],state_hash='hash',trace=[],status='finished',elapsed_s=0)
        fake_sdk=SimpleNamespace(ToolEnv=Env)
        with patch.object(run,'__file__',str(package/'run.py')), patch.object(run,'HERE',root), patch.object(run,'verify'), patch.object(run,'verify_preservation'), \
             patch.object(run.sys,'argv',['test','val','--config','dummy.json']), \
             patch.object(run,'load_config',return_value=config), \
             patch.object(run,'verify_assets',return_value={'release':'test','qwen_file':'qwen.gguf','asr_vocab_sha256':'x','asr_parameters':1543304960,'qwen_text_parameters':4205751296,'merged_parameters':5749056256}), \
             patch.object(run,'load_inputs',return_value=(rows,{})), \
             patch.object(run,'sha',return_value='test-hash'), \
             patch.object(run,'SpeechASR',ASR),patch.object(run,'LocalServer',Backend), \
             patch.object(run,'audio_view',side_effect=lambda row,mapping:row), \
             patch.object(run.actor,'bind_package'),patch.object(run.actor,'infer',side_effect=infer), \
             patch.object(run.importlib,'import_module',return_value=fake_sdk):
            run.main()
        assert len(paths)==2 and paths[0]!=paths[1]
        output=root/'out/val'
        records=[json.loads(p.read_text()) for p in sorted((output/'records').glob('*.json'))]
        assert [r['status'] for r in records]==['timeout','finished']
        assert json.loads((output/'run_manifest.json').read_text())['completed_samples']==2
        assert all((p/'server.log').read_bytes()==b'kept' for p in paths)
        print('PASS: startup failure retained; next sample uses unique log and completes; both records preserved')

if __name__=='__main__':test_failed_startup_then_next_sample()

import unittest
class StartupRegressionTests(unittest.TestCase):
    def test_startup_timeout_does_not_poison_next_log(self):test_failed_startup_then_next_sample()
