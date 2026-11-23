import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from lifemate.runtime.container_io import split_root,input_rows,timed_submission,check_timed_output

REPO=Path(__file__).resolve().parents[3]


class ContainerIOTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.data=self.root/'data';self.data.mkdir()
        self.directory=self.data/'private_test';self.directory.mkdir()
        (self.directory/'audio').mkdir();(self.directory/'audio/a.wav').write_bytes(b'fixture')
        (self.directory/'private_test_tools_catalog.json').write_text('{}')
        self.row={'id':'a','audio':'audio/a.wav','context':{'installed_apps':{}},'history':[]}
        self.write(self.row)

    def write(self,row):
        (self.directory/'private_test.jsonl').write_text(json.dumps(row)+'\n')

    def test_auto_split_and_clean_input(self):
        root,split,d=split_root(self.data)
        self.assertEqual((root,split,d),(self.data.resolve(),'private_test',self.directory.resolve()))
        rows=list(input_rows(d,split))
        self.assertEqual(rows[0]['latest_audio'],'audio/a.wav')
        self.assertNotIn('audio_text',json.dumps(rows))

    def test_test_gold_fields_rejected(self):
        self.row['label']={};self.write(self.row)
        with self.assertRaises(ValueError):list(input_rows(self.directory,'private_test'))
        del self.row['label'];self.row['history']=[{'role':'user','audio':'audio/a.wav','audio_text':'gold'}];self.write(self.row)
        with self.assertRaises(ValueError):list(input_rows(self.directory,'private_test'))

    def test_traversal_and_duplicate_ids(self):
        self.row['audio']='../secret';self.write(self.row)
        with self.assertRaises(ValueError):list(input_rows(self.directory,'private_test'))
        self.row['audio']='audio/a.wav';self.write(self.row)
        p=self.directory/'private_test.jsonl';p.write_text(p.read_text()*2)
        with self.assertRaises(ValueError):list(input_rows(self.directory,'private_test'))

    def test_ambiguous_and_missing_split(self):
        with self.assertRaises(ValueError):split_root(self.data,'not_present')
        d=self.data/'other';d.mkdir();(d/'other.jsonl').write_text('{}');(d/'other_tools_catalog.json').write_text('{}')
        with self.assertRaises(ValueError):split_root(self.data)
        self.assertEqual(split_root(self.data,'private_test')[1],'private_test')

    def test_time_and_coverage(self):
        base={'id':'a','tool_calls':[],'state_hash':'fixture_hash'}
        for bad in [True,-1,2.5]:
            with self.assertRaises(ValueError):timed_submission(base,bad)
        p=self.root/'submission.jsonl';p.write_text(json.dumps(timed_submission(base,42))+'\n')
        self.assertTrue(check_timed_output(p,['a'])['valid'])
        with self.assertRaises(ValueError):check_timed_output(p,['b'])
        p.write_text(json.dumps(base)+'\n')
        with self.assertRaises(ValueError):check_timed_output(p,['a'])

    def test_preserve_failed_calls_in_timed_output(self):
        base={'id':'a','tool_calls':[{'name':'bad.tool','arguments':{}}],'state_hash':'fixture_hash'}
        self.assertEqual(timed_submission(base,1)['tool_calls'],base['tool_calls'])

    def test_predict_and_notebook_share_writer_and_no_overwrite(self):
        spec=importlib.util.spec_from_file_location('ops_container_predict',REPO/'ops/packaging/predict.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=self.root/'result';result.mkdir()
        def infer(row,started=None):
            return {'id':row['id'],'tool_calls':[],'state_hash':'fixture_hash','time':7},{'failure':False}
        predictor=SimpleNamespace(result=result,directory=self.directory,split='private_test',infer=infer)
        module.write_predictions(predictor)
        module.write_predictions(predictor,'jupyter_submission.jsonl')
        self.assertEqual((result/'submission.jsonl').read_bytes(),(result/'jupyter_submission.jsonl').read_bytes())
        with self.assertRaises(FileExistsError):module.write_predictions(predictor)
        notebook=json.loads((REPO/'ops/packaging/predict_notebook.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
        self.assertIn('jupyter_submission.jsonl',json.dumps(notebook))

    def test_context_guard_and_runtime_checksum_rejection(self):
        def load(name):
            spec=importlib.util.spec_from_file_location('packaging_'+name,REPO/'ops/packaging'/(name+'.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
        builder=load('prepare_context')
        with self.assertRaises(ValueError):
            builder.prepare(self.root/'assets',self.root/'runtime',self.root/'training',REPO/'private_context')
        with self.assertRaises(ValueError):
            builder.prepare(self.root/'assets',self.root/'runtime',self.root/'training',self.root/'context')
        archive=self.root/'bad-runtime.tar.gz';archive.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):load('stage_linux_runtime').stage(archive,self.root/'runtime')
        self.assertFalse((self.root/'runtime').exists())
