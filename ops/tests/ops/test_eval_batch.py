from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'llm/baseline_50_2'))
from lifemate.runtime.batch import atomic,run_resumable
from lifemate.runtime.contracts import Sample
from lifemate.runtime.val_evaluation import evaluate_val
from lifemate.adapters.baseline50_partitions import load_partition
from lifemate.runtime.smoke_inputs import digest


class BatchTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)/'run';self.calls=[];self.created=0;self.closed=0
        self.samples=[Sample('a',{'latest_audio':'a.wav'}),Sample('b',{'latest_audio':'b.wav'})]
        self.fail_id=None

    @contextmanager
    def factory(self):
        self.created+=1
        def run(sample):
            self.calls.append(sample.sample_id)
            if sample.sample_id==self.fail_id:raise RuntimeError('uncommitted session')
            return SimpleNamespace(submission={'id':sample.sample_id,'tool_calls':[],'state_hash':'real_fixture_hash'},
                diagnostics={'id':sample.sample_id,'status':'finished','failure':False,'elapsed_seconds':.1})
        try:yield SimpleNamespace(run=run)
        finally:self.closed+=1

    def execute(self,**kwargs):
        return run_resumable(self.factory,self.samples,['a','b'],self.root,{'fixture':1},**kwargs)

    def test_full_batch_and_completed_resume_never_loads_model(self):
        state=self.execute();self.assertEqual(state['completed'],2)
        self.assertEqual((self.created,self.closed),(1,1))
        self.execute(resume=True)
        self.assertEqual(self.calls,['a','b']);self.assertEqual(self.created,1)

    def test_unknown_session_blocks_resume_without_reset(self):
        self.fail_id='b'
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.closed,1)
        with self.assertRaisesRegex(ValueError,'Uncommitted'):self.execute(resume=True)
        self.assertEqual(self.calls,['a','b'])

    def test_committed_marker_is_cleaned_and_not_reexecuted(self):
        self.execute();atomic(self.root/'inflight.json',{'ordinal':1,'id':'b'})
        self.execute(resume=True)
        self.assertEqual(self.calls,['a','b']);self.assertFalse((self.root/'inflight.json').exists())

    def test_safe_prefix_continues_only_next_sample(self):
        original=atomic
        def stop(path,value):
            if path.name=='run_manifest.json' and value.get('completed')==1:
                raise KeyboardInterrupt()
            original(path,value)
        with patch('lifemate.runtime.batch.atomic',side_effect=stop):
            with self.assertRaises(KeyboardInterrupt):self.execute()
        self.execute(resume=True)
        self.assertEqual(self.calls,['a','b']);self.assertEqual(self.created,2)

    def test_changed_input_or_configuration_rejected_before_factory(self):
        self.execute();self.samples[0]=Sample('a',{'latest_audio':'changed.wav'})
        with self.assertRaises(ValueError):self.execute(resume=True)
        self.assertEqual(self.created,1)

    def test_corrupt_checkpoint_rejected(self):
        self.execute();p=self.root/'records/000000.json';row=json.loads(p.read_text())
        row['submission']['state_hash']='tampered';p.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,'Corrupt'):self.execute(resume=True)

    def test_missing_prefix_rejected(self):
        self.execute();(self.root/'records/000000.json').unlink()
        with self.assertRaisesRegex(ValueError,'prefix'):self.execute(resume=True)

    def test_coverage_no_overwrite_and_missing_resume(self):
        with self.assertRaises(ValueError):
            run_resumable(self.factory,self.samples,['a'],self.root,{})
        with self.assertRaises(FileNotFoundError):self.execute(resume=True)
        self.execute()
        with self.assertRaises(FileExistsError):self.execute()


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.refs={'a':{'tool_calls':[],'state_hash':'initial'},
                   'b':{'tool_calls':[{'name':'demo.write','arguments':{'value':1}}],'state_hash':'final'}}
        self.rows=[{'id':sid,**value,'status':'finished','elapsed_s':1} for sid,value in self.refs.items()]

    def test_gold_and_missing_denominator(self):
        self.assertEqual(evaluate_val(self.rows,self.refs,{},['a','b'])['score_100'],100)
        report=evaluate_val(self.rows[:1],self.refs,{},['a','b'])
        self.assertEqual((report['score_100'],report['missing'],report['complete']),(50,1,False))

    def test_timeout_and_over_budget_zero_even_empty_gold(self):
        self.rows[0]['status']='timeout';self.rows[1]['elapsed_s']=31
        report=evaluate_val(self.rows,self.refs,{},['a','b'])
        self.assertEqual((report['score_100'],report['failed_samples'],report['over_30s']),(0,2,1))

    def test_ops_diagnostics_and_call_budget_keep_formula(self):
        diags=[{'id':sid,'status':'call_budget_exhausted','elapsed_seconds':2,'failure':False} for sid in self.refs]
        predictions=[{k:v for k,v in row.items() if k not in ('status','elapsed_s')} for row in self.rows]
        self.assertEqual(evaluate_val(predictions,self.refs,{},['a','b'],diags)['score_100'],100)

    def test_duplicate_unknown_and_invalid_latency(self):
        with self.assertRaises(ValueError):evaluate_val(self.rows*2,self.refs,{},['a','b'])
        with self.assertRaises(ValueError):evaluate_val(self.rows,self.refs,{},['a'])
        self.rows[0]['elapsed_s']=float('nan')
        with self.assertRaises(ValueError):evaluate_val(self.rows,self.refs,{},['a','b'])


class PartitionTests(unittest.TestCase):
    def test_sanitized_inputs_path_check_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);private=root/'private_data';private.mkdir()
            (private/'split.json').write_text('{}');(private/'retrieval_index.json').write_text('[]')
            pin={'files':{'private_data/'+p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in private.iterdir()}}
            split=root/'train';split.mkdir();(split/'a.wav').write_bytes(b'fixture')
            row={'id':'a','audio':'a.wav','audio_text':'gold','label':{},'context':{'installed_apps':{}},
                 'history':[{'role':'user','audio':'a.wav','audio_text':'gold'}]}
            with patch('lifemate.adapters.baseline50_partitions.manifest',return_value=pin), \
                 patch('baseline50.io.load_inputs',return_value=([row],{})):
                samples,_=load_partition(root,private,'val')
                self.assertNotIn('gold',json.dumps(samples[0].model_input))
                row['audio']='../escape.wav'
                with self.assertRaises(ValueError):load_partition(root,private,'val')
            with self.assertRaises(ValueError):load_partition(root,private,'train')
