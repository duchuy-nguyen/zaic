from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'llm/baseline_50_2'))
sys.path.insert(0, str(REPO / 'data/synthetic'))
from baseline50 import actor
from baseline50.asr import audio_paths, audio_view
from zaic_tools import ToolEnv
from lifemate.adapters.baseline50 import Baseline50ASR, Baseline50Policy, Baseline50System
from lifemate.runtime.contracts import Sample, RuntimeConfig
from lifemate.system.cli import main, run_batch


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value


def response(name=None, arguments=None, *, content='', call_id='native_1', finish=None):
    message = {'role': 'assistant', 'content': content}
    if name:
        message['tool_calls'] = [{'id': call_id, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments or {}, ensure_ascii=False)}}]
    return {'choices': [{'finish_reason': finish or ('tool_calls' if name else 'stop'), 'message': message}]}


class Transport:
    def __init__(self, outputs, clock=None, late=False):
        self.outputs, self.inputs = deepcopy(outputs), []
        self.clock, self.late = clock, late
    def generate(self, messages, tools, deadline, *, max_tokens=512):
        self.inputs.append((deepcopy(messages), deepcopy(tools), deadline, max_tokens))
        output = self.outputs.pop(0)
        if isinstance(output, Exception): raise output
        if self.late: self.clock.value = deadline
        return output, 17


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.private = self.root / 'private'
        self.private.mkdir()
        self.examples = [{'source_id': 'fit_example', 'group_id': 'example_group',
            'request': 'Gửi ghi chú hai trăm nghìn', 'context': {}, 'history': [],
            'next_action': {'tool_call': {'name': 'notes.list', 'arguments': {}}}}]
        (self.private / 'retrieval_index.json').write_text(json.dumps(self.examples))
        self.split = {'train_fit': ['sample_a', 'sample_b', 'fit_example'], 'val_main': ['val_only'],
                      'group_ids': {'sample_a': 'sample_group', 'sample_b': 'other_group', 'fit_example': 'example_group', 'val_only': 'val_group'}}
        (self.private / 'split.json').write_text(json.dumps(self.split))
        self.clock = Clock()
        self.policy = Baseline50Policy(self.private, clock=self.clock)
        catalog = json.loads((REPO / 'data/synthetic/synthetic_tools_catalog.json').read_text())
        self.catalog = {t['name']: t for a in catalog['apps'] for t in a['tools']}
        self.sample = Sample('sample_a', {'latest_text': 'Gửi ghi chú hai trăm nghìn',
            'context': {'current_time': '2026-11-09T10:00:00+07:00',
                        'installed_apps': {'notes': {'tools': ['notes.list', 'notes.append']}}},
            'history': [{'role': 'user', 'text': 'Lúc 7 giờ tối nhắc tôi.'},
                        {'role': 'assistant', 'text': 'Đã ghi nhận.', 'tool_calls': [], 'tool_results': []}]})
        directory = self.root / 'synthetic'
        directory.mkdir()
        (directory / 'synthetic_tools_catalog.json').write_text(json.dumps(catalog))
        for sid in ['sample_a', 'sample_b']:
            d = directory / 'synthetic_environment' / sid
            d.mkdir(parents=True)
            (d / 'state.json').write_text(json.dumps({
                '_meta': {'current_time': '2026-11-09T10:00:00+07:00', 'tools': ['notes.list', 'notes.append']},
                'notes': {'notes': [{'id': 'fixture_note', 'title': 'Test', 'content': 'old'}]}}))
        self.sessions, self.factory_inputs = [], []

    def session(self, sid='sample_a'):
        return ToolEnv('synthetic', sid, root=self.root)

    def factory(self, sample):
        self.factory_inputs.append(sample)
        env = self.session(sample.sample_id)
        self.sessions.append(env)
        return env

    def system(self, transport, **kwargs):
        return Baseline50System(catalog=self.catalog, environment_factory=self.factory,
            transport=transport, policy=self.policy, clock=self.clock, **kwargs)

    def baseline(self, outputs, *, seconds=30, max_calls=8, max_repairs=2):
        self.clock.value = 0
        backend = Transport(outputs)
        env = self.session()
        view = {'id': self.sample.sample_id, **deepcopy(self.sample.model_input)}
        with patch.object(actor, '_RETRIEVER', actor.TfidfExamples(deepcopy(self.examples))), \
             patch.object(actor, '_GROUPS', deepcopy(self.split['group_ids'])), \
             patch.object(actor, 'time', SimpleNamespace(monotonic=self.clock)):
            result = actor.infer(view, self.catalog, env, backend, seconds=seconds,
                                 max_calls=max_calls, max_repairs=max_repairs)
        return result, backend

    def assert_parity(self, outputs, *, max_calls=8, max_repairs=2):
        expected, original = self.baseline(outputs, max_calls=max_calls, max_repairs=max_repairs)
        self.clock.value = 0
        bridged = Transport(outputs)
        result = self.system(bridged, runtime=RuntimeConfig(max_executed_calls=max_calls, max_repairs=max_repairs)).run(self.sample)
        self.assertEqual(result.submission['tool_calls'], expected['tool_calls'])
        self.assertEqual(result.submission['state_hash'], expected['state_hash'])
        self.assertEqual(result.diagnostics['status'], expected['status'])
        self.assertEqual(result.diagnostics['repairs_used'], expected['repairs'])
        # Includes exact JSON content, original native call IDs and repair text.
        self.assertEqual(bridged.inputs, original.inputs)
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(self.factory_inputs[0].model_input, {})
        return result

    def test_prompt_hints_retrieval_schema_and_lookup_update_parity(self):
        self.assert_parity([response('notes.list', {}),
            response('notes.append', {'note_id': 'fixture_note', 'content': 'Tiếng Việt'}, content='Đã tìm', call_id='real_id_2'), response()])

    def test_finish_and_ask_parity(self):
        self.assert_parity([response()])
        self.sessions.clear(); self.factory_inputs.clear()
        self.assert_parity([response('sys.ask_missing_info', {'tool_name': 'notes.append', 'params': ['content']})])

    def test_parse_and_schema_repair_message_parity(self):
        self.assert_parity([response(finish='length'), response('notes.append', {'content': 'missing ID'}),
                           response('notes.list', {}), response()])

    def test_repair_budget_and_multi_call_parity(self):
        multiple = response('notes.list', {})
        multiple['choices'][0]['message']['tool_calls'] *= 2
        self.assert_parity([multiple, multiple, multiple])

    def test_tool_done_false_and_call_budget_parity(self):
        self.assert_parity([response('notes.append', {'note_id': 'not_present', 'content': 'x'}), response()])
        self.sessions.clear(); self.factory_inputs.clear()
        result = self.assert_parity([response('notes.list', {})], max_calls=1)
        self.assertEqual(result.diagnostics['status'], 'call_budget_exhausted')

    def test_policies_are_isolated_without_rebinding_original_actor(self):
        before = actor._RETRIEVER
        other = self.root / 'other'; other.mkdir()
        (other / 'split.json').write_text(json.dumps(self.split))
        (other / 'retrieval_index.json').write_text('[]')
        second = Baseline50Policy(other, clock=self.clock)
        view = {'id': self.sample.sample_id, **deepcopy(self.sample.model_input)}
        self.assertNotEqual(self.policy.initial_messages(view), second.initial_messages(view))
        self.assertIs(actor._RETRIEVER, before)

    def test_retrieval_bank_cannot_use_val_sources(self):
        self.examples[0]['source_id'] = 'val_only'
        (self.private / 'retrieval_index.json').write_text(json.dumps(self.examples))
        with self.assertRaises(ValueError): Baseline50Policy(self.private)

    def test_native_parser_parity_for_valid_and_malformed_responses(self):
        cases=[response(), response('notes.list', {}), response(finish='length'),
               {'choices':[]}, response(finish='tool_calls')]
        for value in cases:
            try:
                expected=actor.parse_decision(deepcopy(value))
            except Exception as error:
                with self.assertRaises(type(error)):
                    self.policy.parse_decision(deepcopy(value))
            else:
                self.assertEqual(self.policy.parse_decision(deepcopy(value)),expected)

    def test_bank_same_source_and_same_group_exclusions_match_baseline(self):
        self.examples[0]['source_id'] = 'sample_a'
        self.examples[0]['group_id'] = 'sample_group'
        (self.private / 'retrieval_index.json').write_text(json.dumps(self.examples))
        self.policy = Baseline50Policy(self.private, clock=self.clock)
        self.assert_parity([response()])
        view = {'id': self.sample.sample_id, **deepcopy(self.sample.model_input)}
        body = json.loads(self.policy.initial_messages(view)[1]['content'])
        self.assertNotIn('independent_training_example_do_not_execute_or_copy_ids', body)
        self.sessions.clear(); self.factory_inputs.clear()
        self.examples[0]['source_id'] = 'fit_example'
        self.split['group_ids']['fit_example'] = 'sample_group'
        (self.private / 'retrieval_index.json').write_text(json.dumps(self.examples))
        (self.private / 'split.json').write_text(json.dumps(self.split))
        self.policy = Baseline50Policy(self.private, clock=self.clock)
        self.assert_parity([response()])

    def test_invalid_missing_groups_or_overlapping_split_rejected(self):
        self.split['group_ids'].pop('sample_b')
        (self.private / 'split.json').write_text(json.dumps(self.split))
        with self.assertRaises(ValueError): Baseline50Policy(self.private)
        self.split['group_ids']['sample_b'] = 'other'
        self.split['val_main'].append('sample_a')
        (self.private / 'split.json').write_text(json.dumps(self.split))
        with self.assertRaises(ValueError): Baseline50Policy(self.private)

    def test_asr_batch_order_view_parity_and_remaining_budget(self):
        clock = self.clock
        class ASR:
            def __init__(self): self.inputs=[]
            def transcribe(self, paths, deadline):
                self.inputs.append((paths, deadline)); clock.value += 6
                return {p: '  nội dung  ' + p for p in paths}, {'clips': len(paths)}
        asr = ASR()
        raw = {'id': 'sample_a', 'audio': 'audio/current.wav', 'context': self.sample.model_input['context'],
               'history': [{'role': 'user', 'audio': 'audio/history.wav'}, {'role': 'assistant', 'text': 'done'}]}
        audio_sample = Sample('sample_a', {'latest_audio': raw['audio'], 'context': raw['context'], 'history': raw['history'],
                                          'audio_text': 'GOLD_SECRET', 'label': {'secret': True}})
        prepared = Baseline50ASR(asr, clock=clock).prepare(audio_sample, deadline=30)
        mapping = {p: '  nội dung  ' + p for p in audio_paths(raw)}
        self.assertEqual({'id': prepared.sample_id, **prepared.model_input}, audio_view(raw, mapping))
        self.assertEqual(asr.inputs, [(['audio/current.wav', 'audio/history.wav'], 30)])
        clock.value = 0; asr.inputs.clear()
        backend = Transport([response()])
        result = self.system(backend, asr=asr).run(audio_sample)
        self.assertEqual(backend.inputs[0][2], 30)
        self.assertNotIn('GOLD_SECRET', str(backend.inputs))
        self.assertEqual(result.diagnostics['elapsed_seconds'], 6)

    def test_asr_missing_or_late_never_uses_gold_or_calls_model(self):
        clock = self.clock
        class ASR:
            def __init__(self, late): self.late=late
            def transcribe(self, paths, deadline):
                if self.late: clock.value = deadline
                return {}, {}
        sample = Sample('sample_a', {'latest_audio': 'audio/a.wav', 'audio_text': 'GOLD_SECRET',
                                    'context': self.sample.model_input['context'], 'history': []})
        for late in [False, True]:
            clock.value = 0
            transport = Transport([])
            result = self.system(transport, asr=ASR(late)).run(sample)
            self.assertTrue(result.diagnostics['failure'])
            self.assertEqual(result.submission['tool_calls'], [])
            self.assertEqual(transport.inputs, [])

    def test_prepared_text_does_not_invoke_asr_and_nonfit_id_rejected_before_factory(self):
        class ASR:
            def transcribe(self, *args): raise AssertionError('Text input must not call ASR')
        system = self.system(Transport([response()]), asr=ASR(), allowed_ids=self.policy.fit_ids)
        result = system.run(self.sample)
        self.assertFalse(result.diagnostics['failure'])
        with self.assertRaises(ValueError): system.run(Sample('val_only', {}))
        self.assertEqual(len(self.sessions), 1)

    def test_single_audio_protocol_and_permitted_tools_guard(self):
        clock=self.clock
        class ASR:
            def transcribe(self, paths, deadline):
                self.inputs=(paths,deadline)
                return {paths[0]:'  xin   chào  '},{}
        asr=ASR()
        self.assertEqual(Baseline50ASR(asr,clock=clock).transcribe('audio/a.wav',remaining_seconds=5),'xin chào')
        self.assertEqual(asr.inputs,(['audio/a.wav'],5))
        sample=deepcopy(self.sample)
        sample.model_input['context']['installed_apps']['notes']['tools']=['notes.list']
        backend=Transport([response('notes.append',{'note_id':'fixture_note','content':'not_allowed'})]*3)
        result=self.system(backend).run(sample)
        self.assertEqual(result.diagnostics['status'],'structured_output_error')
        self.assertEqual(result.submission['tool_calls'],[])

    def test_late_model_response_executes_nothing_and_error_text_not_logged(self):
        backend = Transport([response('notes.list', {})], clock=self.clock, late=True)
        trace=[]
        result = self.system(backend, trace=trace.append).run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'timeout')
        self.assertEqual(result.submission['tool_calls'], [])
        self.clock.value=0
        result = self.system(Transport([RuntimeError('SECRET_EXCEPTION')]), trace=trace.append).run(self.sample)
        self.assertTrue(result.diagnostics['failure'])
        self.assertNotIn('SECRET_EXCEPTION', json.dumps(trace))

    def test_unknown_execution_keeps_session_calls_and_never_retries(self):
        original = self.session()
        class Uncertain:
            validate = original.validate
            get_tool_calls = original.get_tool_calls
            get_state_hash = original.get_state_hash
            def call(self, name, arguments):
                original.call(name, arguments)
                raise RuntimeError('SECRET_AFTER_WRITE')
        system = self.system(Transport([response('notes.append', {'note_id':'fixture_note','content':'new'})]))
        system.environment_factory = lambda sample: Uncertain()
        result = system.run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'execution_error')
        self.assertEqual(len(result.submission['tool_calls']), 1)
        self.assertEqual(len(system.transport.inputs), 1)

    def test_ops_batch_and_factory_context_cleanup_on_success_and_failure(self):
        module=ModuleType('baseline50_fixture_factory'); lifecycle=[]
        @contextmanager
        def factory():
            lifecycle.append('enter')
            try: yield self.system(Transport([response()]))
            finally: lifecycle.append('exit')
        module.build=factory
        input_file=self.root/'input.jsonl'
        input_file.write_text(json.dumps({'id':self.sample.sample_id,**self.sample.model_input})+'\n')
        with patch.dict(sys.modules, {'baseline50_fixture_factory':module}):
            result=main(['--factory','baseline50_fixture_factory:build','--input',str(input_file),
                         '--output-dir',str(self.root/'run')])
            self.assertEqual(result,0)
            self.assertEqual(lifecycle,['enter','exit'])
            with self.assertRaises(ValueError):
                main(['--factory','baseline50_fixture_factory:build','--input',str(input_file),
                      '--output-dir',str(self.root/'too_big'), '--max-samples','0'])
            self.assertEqual(lifecycle,['enter','exit'])
            with self.assertRaises(FileExistsError):
                main(['--factory','baseline50_fixture_factory:build','--input',str(input_file),
                      '--output-dir',str(self.root/'run')])
            self.assertEqual(lifecycle,['enter','exit'])
            with patch('lifemate.system.cli.run_batch', side_effect=RuntimeError('write failed')):
                with self.assertRaises(RuntimeError):
                    main(['--factory','baseline50_fixture_factory:build','--input',str(input_file),
                          '--output-dir',str(self.root/'failed')])
            self.assertEqual(lifecycle,['enter','exit','enter','exit'])


if __name__ == '__main__': unittest.main()
