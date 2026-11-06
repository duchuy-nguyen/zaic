"""Offline correctness tests; no BTC environments, downloads or pretrained models."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import wave
import zipfile
import tarfile
import io

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'local_baseline')]
import core
import worker


CATALOG = {
    'fixture.write': {'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'},
        'enabled': {'type': 'boolean'}, 'items': {'type': 'array'}, 'text': {'type': 'string'}}, 'required': ['value']}},
    'sys.ask_missing_info': {'parameters': {'type': 'object', 'properties': {
        'tool_name': {'type': 'string'}, 'params': {'type': 'array'}}}}}
CALL = {'name': 'fixture.write', 'arguments': {'value': 1}}
XML = '<tool_call>\n<function=fixture.write>\n<parameter=value>\n1\n</parameter>\n</function>\n</tool_call>'


def row(sid='a'):
    return {'id': sid, 'audio': 'audio/'+sid+'.wav', 'audio_text': 'fixture request '+sid,
        'context': {'current_time': '2026-11-05T10:00:00+07:00',
            'installed_apps': {'fixture': {'tools': ['fixture.write'], 'state': {'secret': 'DO_NOT_LEAK'}}}},
        'history': [], 'label': {'tool_calls': [CALL], 'tool_results': [{'ok': True, 'done': True}],
                               'final_state': {'private': 'GOLD_STATE'}}}


class FakeEnv:
    def __init__(self, raise_after_call=False, rejected=False):
        self.calls = []
        self.raise_after_call, self.rejected = raise_after_call, rejected
    def validate(self, name, arguments):
        return {'ok': not self.rejected, 'error': 'missing value' if self.rejected else None}
    def call(self, name, arguments):
        self.calls.append({'name': name, 'arguments': deepcopy(arguments)})
        if self.raise_after_call: raise OSError('fixture unknown side effect')
        return {'ok': True, 'done': True}
    def get_tool_calls(self): return deepcopy(self.calls)
    def get_state_hash(self): return 'h'


def scripted(*outputs):
    iterator = iter(outputs)
    return lambda *args: next(iterator)


class ProtocolTests(unittest.TestCase):
    def test_model_and_timeout_messages_are_preserved(self):
        for exception, status in ((RuntimeError('fixture kernel failure'), 'model_error'),
                                  (TimeoutError('fixture prefill deadline'), 'timeout')):
            def generate(*args): raise exception
            result = core.run_agent(row(), CATALOG, FakeEnv(), generate)
            self.assertEqual(result['status'], status)
            self.assertIn(str(exception), result['error'])
            self.assertEqual(result['tool_calls'], [])

    def test_native_xml_and_json_calls(self):
        self.assertEqual(core.parse_action(XML, CATALOG), CALL)
        self.assertEqual(core.parse_action('<tool_call>'+json.dumps(CALL)+'</tool_call>', CATALOG), CALL)

    def test_parameter_types_and_strings_preserved(self):
        text = '<tool_call><function=fixture.write><parameter=value>1</parameter><parameter=enabled>True</parameter><parameter=items>[1,"a"]</parameter><parameter=text>12 triệu</parameter></function></tool_call>'
        self.assertEqual(core.parse_action(text, CATALOG)['arguments'], {'value': 1, 'enabled': True, 'items': [1, 'a'], 'text': '12 triệu'})

    def test_malformed_multicall_duplicate_nonfinite_not_stop(self):
        invalid = [XML+XML, XML[:-12], '', '{"name":"fixture.write"}',
            '<function=fixture.write></function>', XML.replace('1\n</parameter>', 'NaN\n</parameter>'),
            XML.replace('</function>', '<parameter=value>2</parameter></function>')]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                core.parse_action(value, CATALOG)
        with self.assertRaises(ValueError): core.strict_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError): core.strict_json('{"x":1e400}')

    def test_thinking_stripped_and_eos_stop_representation(self):
        self.assertEqual(core.parse_action('<think>fixture</think>'+XML, CATALOG), CALL)
        self.assertIsNone(core.parse_action('Đã xong.', CATALOG))

    def test_input_gold_state_and_history_replay_boundary(self):
        sample = row()
        sample['history'] = [{'role': 'assistant', 'text': 'past', 'tool_calls': [CALL]}]
        view = core.clean_view(sample)
        serialized = json.dumps(view)
        self.assertNotIn('GOLD_STATE', serialized)
        self.assertNotIn('DO_NOT_LEAK', serialized)
        env = FakeEnv()
        result = core.run_agent(sample, CATALOG, env, scripted('Đã xong.'))
        self.assertEqual(result['tool_calls'], [])

    def test_execution_and_successful_stop(self):
        env = FakeEnv()
        result = core.run_agent(row(), CATALOG, env, scripted(XML, 'Đã xong.'))
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['tool_calls'], [CALL])
        self.assertTrue(result['result_alignment_known'])

    def test_schema_and_permissions_reject_before_call(self):
        env = FakeEnv(rejected=True)
        result = core.run_agent(row(), CATALOG, env, scripted(XML), max_repairs=0)
        self.assertEqual(result['status'], 'structured_output_error')
        self.assertEqual(env.calls, [])
        sample = row(); sample['context']['installed_apps'] = {}
        result = core.run_agent(sample, CATALOG, FakeEnv(), scripted(XML), max_repairs=0)
        self.assertEqual(result['tool_calls'], [])

    def test_unknown_call_keeps_authoritative_log(self):
        env = FakeEnv(raise_after_call=True)
        result = core.run_agent(row(), CATALOG, env, scripted(XML))
        self.assertEqual(result['status'], 'execution_unknown')
        self.assertEqual(result['tool_calls'], [CALL])
        self.assertFalse(result['result_alignment_known'])

    def test_call_budget_is_not_failure_override(self):
        result = core.run_agent(row(), CATALOG, FakeEnv(), scripted(XML, XML), max_calls=1)
        self.assertEqual(result['status'], 'call_budget')
        self.assertFalse(core.failed_status(result['status']))
        self.assertTrue(core.failed_status('execution_unknown'))

    def test_ask_stops_once(self):
        ask = '<tool_call><function=sys.ask_missing_info><parameter=tool_name>fixture.write</parameter><parameter=params>["value"]</parameter></function></tool_call>'
        result = core.run_agent(row(), CATALOG, FakeEnv(), scripted(ask))
        self.assertEqual(result['status'], 'ask')
        self.assertEqual(len(result['tool_calls']), 1)


class DataTests(unittest.TestCase):
    def _teacher_reuse_fixture(self):
        cfg,source,target=self._asr_reuse_fixture()
        cfg.update(teacher_id='fixture/teacher',teacher_revision='fixed',teacher_prequantized=False,
                   teacher_min_score=100,minimum_teacher_accepted=1,reuse_teacher_from=str(source))
        cfg['limits']['teacher_samples']=None
        previous=json.loads((source/'config.json').read_text())
        previous.update(teacher_id=cfg['teacher_id'],teacher_revision='fixed',teacher_prequantized=False)
        core.write_json(source/'config.json',previous)
        data=Path(cfg['data_root']);(data/'train').mkdir(parents=True)
        core.append_jsonl(data/'train/train.jsonl',row('a'))
        core.write_json(data/'train/train_tools_catalog.json',{'apps':[{'name':'fixture','tools':[
            {'name':name,**spec} for name,spec in CATALOG.items()]}]})
        reference={'id':'a','tool_calls':[CALL],'state_hash':'h'}
        for root in (source,target):core.append_jsonl(root/'references.jsonl',reference)
        record={'id':'a','status':'finished','tool_calls':[CALL],'state_hash':'h','accepted':True,
                'result_alignment_known':True,'results':[{'ok':True}],'trace':[],'elapsed_s':1.}
        core.append_jsonl(source/'teacher_rollouts.jsonl',record)
        core.write_json(source/'teacher_metrics.json',{'tasks':1,'accepted':1})
        artifacts={name:core.digest(source/name) for name in ('teacher_rollouts.jsonl','teacher_metrics.json')}
        core.write_json(source/'stage_teacher.json',{'status':'completed','identity':core.fingerprint(previous),'artifacts':artifacts})
        return cfg,source,target

    def test_teacher_reuse_revalidates_calls_and_imports_without_execution(self):
        cfg,source,target=self._teacher_reuse_fixture()
        with patch.object(worker,'runtime',side_effect=AssertionError('No ToolEnv during import')), \
             patch.object(worker,'load_llm',side_effect=AssertionError('No model during import')):
            worker.reuse_teacher(cfg)
        self.assertTrue(json.loads((target/'teacher_reuse.json').read_text())['enabled'])
        self.assertEqual(json.loads((target/'stage_teacher.json').read_text())['origin'],'verified_teacher_data_import')
        self.assertEqual(core.digest(source/'teacher_rollouts.jsonl'),core.digest(target/'teacher_rollouts.jsonl'))

    def test_teacher_reuse_refuses_state_mismatch_even_if_artifact_hash_updated(self):
        cfg,source,target=self._teacher_reuse_fixture()
        record=core.read_jsonl(source/'teacher_rollouts.jsonl')[0];record['state_hash']='incorrect'
        (source/'teacher_rollouts.jsonl').write_text(json.dumps(record)+'\n')
        marker=json.loads((source/'stage_teacher.json').read_text())
        marker['artifacts']['teacher_rollouts.jsonl']=core.digest(source/'teacher_rollouts.jsonl')
        core.write_json(source/'stage_teacher.json',marker)
        with self.assertRaisesRegex(ValueError,'revalidation changed acceptance'):worker.reuse_teacher(cfg)
        self.assertFalse((target/'teacher_rollouts.jsonl').exists())

    def _asr_reuse_fixture(self):
        source = self.root / 'old-run'; target = self.root / 'new-run'
        source.mkdir(); target.mkdir()
        cfg = dict(run_dir=str(target), data_root=str(self.root / 'data'), mode='pilot', seed=1,
            deterministic=True, val_fraction=.2, expected_train_sha256='fixture', asr_id='fixture/asr',
            asr_revision='fixed', asr_lora_r=16, asr_grad_accum=4, asr_lr=1e-4, asr_epochs=2,
            asr_max_new_tokens=256, asr_eval_seconds=60, save_steps=5,
            limits={'fit_samples': 64, 'val_samples': 8, 'asr_max_steps': 20},
            reuse_asr_from=str(source), teacher_id='fixture/new-teacher', bundle_sha256='new')
        bundle = {'core.py': Path(core.__file__).read_text(encoding='utf-8'),
                  'worker.py': Path(worker.__file__).read_text(encoding='utf-8'),
                  'scoring.py': (HERE.parents[1] / 'local_baseline/scoring.py').read_text(encoding='utf-8')}
        (source / 'source').mkdir()
        for name, value in bundle.items(): (source / 'source' / name).write_text(value, encoding='utf-8')
        previous = {**cfg, 'run_dir': str(source), 'teacher_id': 'fixture/old-teacher',
            'bundle_sha256': core.hashlib.sha256(json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        core.write_json(source / 'config.json', previous)
        for root in (source, target):
            core.write_json(root / 'split.json', {'fit_ids': ['a'], 'val_ids': ['b']})
            core.write_json(root / 'provenance.json', {'data': {'train': 'fixture'}, 'packages': {'torch': 'fixed'}, 'cuda': '12.6'})
        artifacts = {'asr_base': ['asr_base_metrics.json'],
                     'asr_train': ['asr_training.json', 'asr_adapter/adapter_model.safetensors'],
                     'asr_tuned': ['asr_tuned_metrics.json', 'selected_asr.json']}
        for stage, names in artifacts.items():
            for name in names: core.write_json(source / name, {'fixture': name})
            core.write_json(source / ('stage_' + stage + '.json'), {'status': 'completed',
                'identity': core.fingerprint(previous), 'artifacts': {n: core.digest(source / n) for n in names}})
        return cfg, source, target

    def test_asr_reuse_verifies_and_preserves_source_provenance(self):
        cfg, source, target = self._asr_reuse_fixture()
        worker.reuse_asr(cfg)
        receipt = json.loads((target / 'asr_reuse.json').read_text())
        self.assertTrue(receipt['enabled'])
        self.assertEqual(receipt['source_run'], str(source.resolve()))
        marker = json.loads((target / 'stage_asr_train.json').read_text())
        self.assertEqual(marker['origin'], 'verified_asr_import')
        self.assertEqual(marker['identity'], core.fingerprint(cfg))
        self.assertEqual(core.digest(source / 'asr_adapter/adapter_model.safetensors'),
                         core.digest(target / 'asr_adapter/adapter_model.safetensors'))

    def test_asr_reuse_refuses_changed_split_or_artifacts_before_copying(self):
        cfg, source, target = self._asr_reuse_fixture()
        split = (target / 'split.json').read_bytes()
        core.write_json(target / 'split.json', {'fit_ids': ['b'], 'val_ids': ['a']})
        with self.assertRaisesRegex(ValueError, 'split differs'): worker.reuse_asr(cfg)
        self.assertFalse((target / 'asr_adapter').exists())
        (target / 'split.json').write_bytes(split)
        (source / 'asr_adapter/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'): worker.reuse_asr(cfg)
        self.assertFalse((target / 'asr_adapter').exists())

    def test_teacher_diagnostics_explain_rejections_without_changing_gate(self):
        records = [{'status': 'timeout', 'accepted': False, 'elapsed_s': 181.,
            'error': 'TimeoutError: fixture deadline', 'result_alignment_known': True,
            'trace': [], 'local_score': {'state_match': False, 'score_spec_local': 0}},
            {'status': 'finished', 'accepted': False, 'elapsed_s': 20., 'error': None,
             'result_alignment_known': True, 'trace': [{'repair_error': 'fixture invalid JSON'}],
             'local_score': {'state_match': True, 'score_spec_local': 80}}]
        result = worker.teacher_diagnostics(records, 100)
        self.assertEqual(result['statuses'], {'timeout': 1, 'finished': 1})
        self.assertEqual(result['rejection_reasons_nonexclusive'],
                         {'state_mismatch': 1, 'score_below_threshold': 2, 'failed_status': 1})
        self.assertEqual(result['state_matches'], 1)
        self.assertEqual(result['score_max'], 80)
        self.assertEqual(result['repair_errors'], {'fixture invalid JSON': 1})

    def test_teacher_chunked_prefill_is_explicit_and_student_uses_last_logits(self):
        self.assertEqual(worker.generation_memory_options({'teacher_prefill_chunk_size': 256}, teacher=True),
                         {'logits_to_keep': 1, 'prefill_chunk_size': 256})
        self.assertEqual(worker.generation_memory_options({}, teacher=False), {'logits_to_keep': 1})
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                worker.generation_memory_options({'teacher_prefill_chunk_size': invalid}, teacher=True)

    def test_teacher_oom_commits_first_record_and_stops_without_retry(self):
        data = self.root / 'oom-kit'; (data / 'train').mkdir(parents=True)
        for sid in ('a', 'b'):
            core.append_jsonl(data / 'train/train.jsonl', row(sid))
            core.append_jsonl(self.root / 'references.jsonl', {'id': sid, 'tool_calls': [CALL], 'state_hash': 'h'})
        cat = {'apps': [{'name': 'fixture', 'tools': [{'name': k, **v} for k,v in CATALOG.items()]}]}
        core.write_json(data / 'train/train_tools_catalog.json', cat)
        core.write_json(self.root / 'split.json', {'fit_ids': ['a','b'], 'val_ids': []})
        cfg = {'run_dir': str(self.root), 'data_root': str(data),
            'limits': {'fit_samples': None, 'val_samples': None, 'teacher_samples': None},
            'teacher_seconds': 5, 'max_steps': 4, 'max_calls': 2,
            'teacher_min_score': 100, 'minimum_teacher_accepted': 1}
        sessions = []
        def factory(*args, **kwargs): sessions.append(args); return FakeEnv()
        OutOfMemoryError = type('OutOfMemoryError', (RuntimeError,), {})
        def generate(*args): raise OutOfMemoryError('fixture allocation failed')
        with patch.object(worker, 'runtime', return_value=factory), \
             patch.object(worker, 'load_llm', return_value=(SimpleNamespace(eval=lambda: None), None)), \
             patch.object(worker, 'teacher_memory_smoke'), \
             patch.object(worker, 'generator', return_value=generate):
            with self.assertRaisesRegex(RuntimeError, 'stopped on CUDA OOM'):
                worker.teacher(cfg)
        saved = core.read_jsonl(self.root / 'teacher_rollouts.jsonl')
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['status'], 'model_error')
        self.assertIn('fixture allocation failed', saved[0]['error'])
        self.assertFalse((self.root / 'teacher_rollouts.inflight.json').exists())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_wave(self, relative, byte):
        path = self.root / 'train' / relative; path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(bytes([byte, 0])*100)

    def test_split_shared_audio_never_crosses_fit_val(self):
        rows = [row(str(i)) for i in range(8)]
        for i, r in enumerate(rows): self.make_wave(r['audio'], i+1)
        rows[1]['history'] = [{'role': 'user', 'audio': rows[0]['audio'], 'audio_text': rows[0]['audio_text']}]
        split = core.make_split(rows, self.root)
        self.assertEqual(split['groups']['0'], split['groups']['1'])
        self.assertFalse(set(split['fit_ids']) & set(split['val_ids']))
        fit = core.utterances(rows, split['fit_ids'], split)
        val = core.utterances(rows, split['val_ids'], split)
        self.assertFalse({x['audio_sha256'] for x in fit} & {x['audio_sha256'] for x in val})
        self.assertEqual(split, core.make_split(rows, self.root))

    def test_zip_and_audio_traversal_rejected(self):
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as stream: stream.writestr('../escape', 'x')
        with self.assertRaises(ValueError): core.extract_archive(archive, self.root / 'out')
        with self.assertRaises(ValueError): core.audio_path(self.root, '../outside.wav')

    def test_resume_tar_links_and_traversal_rejected(self):
        archive = self.root / 'resume.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            item = tarfile.TarInfo('run/config.json'); item.size = 2
            stream.addfile(item, io.BytesIO(b'{}'))
        self.assertTrue((core.restore_archive(archive, self.root / 'safe', 'run') / 'config.json').is_file())
        with tarfile.open(archive, 'w:gz') as stream:
            item = tarfile.TarInfo('run/../../escape'); item.size = 1
            stream.addfile(item, io.BytesIO(b'x'))
        with self.assertRaises(ValueError): core.restore_archive(archive, self.root / 'bad', 'run')

    def test_gold_next_action_stop_and_ask_examples(self):
        sample = row()
        examples = core.supervised_examples(sample, [], [CALL], [{'ok': True}], 'gold')
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]['target']['tool_calls'][0]['function']['name'], 'fixture.write')
        self.assertEqual(examples[1]['messages'][-1]['role'], 'tool')
        ask = {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'fixture.write', 'params': ['value']}}
        self.assertEqual(len(core.supervised_examples(sample, [], [ask], [{'ok': True}], 'gold')), 1)
        self.assertEqual(len(core.supervised_examples(sample, [], [], [], 'gold')), 1)

    def test_token_mask_and_no_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                return 'prefix' if add_generation_prompt else 'prefixtarget'
            def encode(self, text, **kwargs): return list(text.encode())
        record = {'messages': [], 'tools': [], 'target': {}}
        tokens, length = core.training_tokens(Tokenizer(), record, 100)
        self.assertEqual(tokens['labels'][:6], [-100]*6)
        self.assertEqual(tokens['labels'][6:], list(b'target'))
        self.assertEqual(core.training_tokens(Tokenizer(), record, 2), (None, length))

    def test_inflight_prevents_hidden_reset_and_prefix_is_checked(self):
        core.write_json(self.root / 'teacher.inflight.json', {'id': 'a'})
        with self.assertRaisesRegex(RuntimeError, 'Uncommitted'):
            worker.guarded_records(self.root, 'teacher', ['a'])
        core.append_jsonl(self.root / 'teacher.jsonl', {'id': 'a'})
        _, active, records = worker.guarded_records(self.root, 'teacher', ['a', 'b'])
        self.assertFalse(active.exists()); self.assertEqual(len(records), 1)
        with self.assertRaises(ValueError): worker.guarded_records(self.root, 'teacher', ['b', 'a'])

    def test_asr_error_counts_as_deletions(self):
        metrics = core.asr_metrics([{'reference': 'Một hai', 'hypothesis': '', 'error': 'x'}])
        self.assertEqual(metrics['wer'], 1.)
        self.assertEqual(metrics['errors'], 1)

    def test_session_budget_stops_before_starting_another_sample(self):
        with patch.dict(__import__('os').environ, {'LIFEMATE_SESSION_DEADLINE': '1'}):
            with self.assertRaises(worker.SessionBudget): worker.budget_guard()

    def test_completed_stage_integrity_and_config_guard(self):
        cfg = {'run_dir': str(self.root)}
        core.write_json(self.root / 'config.json', cfg)
        (self.root / 'artifact.json').write_text('{}')
        marker = {'identity': core.fingerprint(cfg), 'status': 'completed',
                  'artifacts': {'artifact.json': core.digest(self.root / 'artifact.json')}}
        core.write_json(self.root / 'stage_prepare.json', marker)
        with patch.object(sys, 'argv', ['worker.py', 'prepare', '--config', str(self.root / 'config.json')]):
            worker.main()
            (self.root / 'artifact.json').write_text('{"changed":1}')
            with self.assertRaisesRegex(ValueError, 'artifact'): worker.main()

    def test_teacher_rollout_factory_once_and_gold_dedup_provenance(self):
        sample = row()
        data = self.root / 'dataset'; (data / 'train').mkdir(parents=True)
        (data / 'train/train.jsonl').write_text(json.dumps(sample)+'\n', encoding='utf-8')
        cat = {'apps': [{'tools': [{'name': n, **v} for n,v in CATALOG.items()]}]}
        core.write_json(data / 'train/train_tools_catalog.json', cat)
        core.write_json(self.root / 'split.json', {'fit_ids': ['a'], 'val_ids': ['b']})
        core.append_jsonl(self.root / 'references.jsonl', {'id': 'a', 'tool_calls': [CALL], 'state_hash': 'h'})
        cfg = {'run_dir': str(self.root), 'data_root': str(data),
            'limits': {'fit_samples': None, 'val_samples': None, 'teacher_samples': None},
            'teacher_seconds': 5, 'max_steps': 4, 'max_calls': 2, 'teacher_min_score': 100,
            'minimum_teacher_accepted': 1}
        calls = []
        def factory(*args, **kwargs): calls.append(args); return FakeEnv()
        with patch.object(worker, 'runtime', return_value=factory), patch.object(worker, 'load_llm', return_value=(SimpleNamespace(eval=lambda: None), None)), patch.object(worker, 'teacher_memory_smoke'), patch.object(worker, 'generator', return_value=scripted(XML, 'Đã xong.')):
            worker.teacher(cfg)
        self.assertEqual(len(calls), 1)
        self.assertTrue(core.read_jsonl(self.root / 'teacher_rollouts.jsonl')[0]['accepted'])
        with patch.object(worker, 'load_llm', side_effect=AssertionError('Completed rollouts must not reload model')), \
             patch.object(worker, 'runtime', side_effect=AssertionError('Completed rollouts must not replay ToolEnv')):
            worker.teacher(cfg)
            with self.assertRaisesRegex(RuntimeError, '1/1 accepted, required=2'):
                worker.teacher({**cfg, 'minimum_teacher_accepted': 2})
        self.assertEqual(len(core.read_jsonl(self.root / 'teacher_rollouts.jsonl')), 1)
        worker.make_examples(cfg)
        report = json.loads((self.root / 'examples_report.json').read_text())
        self.assertEqual(report['teacher_only_unique_examples'], 0)
        self.assertEqual(report['unique_with_teacher_provenance'], 2)


class SelectiveLossTests(unittest.TestCase):
    def test_generation_end_ids_include_checkpoint_and_chat_eos(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
        tokenizer = SimpleNamespace(eos_token_id=248046)
        self.assertEqual(worker.generation_end_ids(model, tokenizer), [248044, 248046])
        model.generation_config.eos_token_id = [248044, 248046]
        self.assertEqual(worker.generation_end_ids(model, tokenizer), [248044, 248046])

    def test_generator_stops_at_chat_eos_before_next_turn_and_executes_one_call(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        class Batch(dict):
            @property
            def input_ids(self): return self['input_ids']
        class Tokenizer:
            eos_token_id, pad_token_id = 248046, 248044
            def apply_chat_template(self, *args, **kwargs):
                self.kwargs = kwargs
                return 'fixture'
            def __call__(self, *args, **kwargs):
                return Batch(input_ids=torch.tensor([[3,4]]), attention_mask=torch.ones(1,2,dtype=torch.long))
            def decode(self, ids, **kwargs):
                return ''.join({7: XML, 8: 'Đã xong.', 9: '<think>wrong next turn',
                                248046: '<|im_end|>', 248044: '<|endoftext|>'}[value] for value in ids)
        class Model:
            generation_config = SimpleNamespace(eos_token_id=248044)
            calls = 0
            def get_input_embeddings(self): return SimpleNamespace(weight=torch.zeros(1))
            def generate(self, input_ids, eos_token_id=None, **kwargs):
                self.calls += 1
                allowed = eos_token_id or [self.generation_config.eos_token_id]
                stream = [7 if self.calls == 1 else 8, 248046, 9, 248044]
                emitted = []
                for token in stream:
                    emitted.append(token)
                    if token in allowed: break
                return torch.cat((input_ids, torch.tensor([emitted])), dim=1)
        model, tokenizer = Model(), Tokenizer()
        cfg = {'inference_context': 64, 'max_new_tokens': 20}
        with patch.object(worker, 'stopping', return_value=[]):
            result = core.run_agent(row(), CATALOG, FakeEnv(), worker.generator(model, tokenizer, cfg))
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['tool_calls'], [CALL])
        self.assertEqual(model.calls, 2)
        self.assertFalse(tokenizer.kwargs['enable_thinking'])

    def test_asr_and_student_training_use_v5_warmup_steps_ratio(self):
        cfg = {'mode': 'pilot', 'seed': 1, 'save_steps': 5,
               'asr_grad_accum': 4, 'student_grad_accum': 8,
               'asr_lr': 1e-4, 'student_lr': 1e-4,
               'asr_epochs': 2, 'student_epochs': 2,
               'limits': {'asr_max_steps': 20, 'student_max_steps': 20}}
        def arguments(*, warmup_steps, **kwargs):
            self.assertNotIn('warmup_ratio', kwargs)
            return SimpleNamespace(warmup_steps=warmup_steps, **kwargs)
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(TrainingArguments=arguments)}):
            for asr in (True, False):
                args = worker.trainer_args(cfg, Path('fixture-checkpoints'), asr=asr)
                self.assertEqual(args.warmup_steps, .05)
                self.assertEqual(args.gradient_accumulation_steps, 4 if asr else 8)

    def test_loss_and_gradients_match_full_causal_cross_entropy(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable; core tests remain dependency-free')
        torch.manual_seed(1)
        class Tiny(torch.nn.Module):
            class Backbone(torch.nn.Module):
                def __init__(self): super().__init__(); self.embed = torch.nn.Embedding(12,4)
                def forward(self, input_ids, **kwargs):
                    return SimpleNamespace(last_hidden_state=self.embed(input_ids))
            def __init__(self):
                super().__init__(); self.model = self.Backbone(); self.lm_head = torch.nn.Linear(4,12)
            def forward(self, input_ids, attention_mask, logits_to_keep=0, **kwargs):
                hidden = self.model(input_ids=input_ids).last_hidden_state
                if not isinstance(logits_to_keep, int): hidden = hidden.index_select(1, logits_to_keep)
                return SimpleNamespace(logits=self.lm_head(hidden))
        a = Tiny(); b = deepcopy(a)
        inputs = {'input_ids': torch.tensor([[1,2,3,4,5]]), 'attention_mask': torch.ones(1,5,dtype=torch.long),
                  'labels': torch.tensor([[-100,-100,-100,4,5]])}
        loss, _ = worker.selective_loss(a, inputs, target_chunk=1); loss.backward()
        logits = b(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        full = torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,12), inputs['labels'][:,1:].reshape(-1), ignore_index=-100)
        full.backward()
        self.assertTrue(torch.allclose(loss, full))
        for x,y in zip(a.parameters(),b.parameters()): self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-7))


if __name__ == '__main__': unittest.main()
