"""Committed ToolEnv + LLM parser integration, fictional state/model fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'llm'))
sys.path.insert(0, str(REPO / 'data/synthetic'))
from zaic_tools import ToolEnv
from lifemate.adapters.native_llm import NativeLLMBackend
from lifemate.adapters.toolenv import ToolEnvAdapter, ToolEnvFactory
from lifemate.dev.mocks import ScriptedModel
from lifemate.runtime.contracts import Calls, ProposedCall, Sample, StopReason
from lifemate.runtime.runner import Runner


def response(name=None, arguments=None):
    message = {'role': 'assistant', 'content': ''}
    if name:
        message['tool_calls'] = [{'id': 'native_1', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments or {})}}]
    return {'choices': [{'finish_reason': 'tool_calls' if name else 'stop', 'message': message}]}


class Transport:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def generate(self, messages, tools, deadline, *, max_tokens):
        self.requests.append(deepcopy((messages, tools, deadline, max_tokens)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, 42


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        catalog = json.loads((REPO / 'data/synthetic/synthetic_tools_catalog.json').read_text())
        self.catalog = {t['name']: t for a in catalog['apps'] for t in a['tools']}
        split = self.root / 'synthetic'
        split.mkdir()
        (split / 'synthetic_tools_catalog.json').write_text(json.dumps(catalog))
        state = {'_meta': {'current_time': '2026-11-05T10:00:00+07:00',
                          'tools': ['notes.list', 'notes.append']},
                 'notes': {'notes': [{'id': 'fixture_note', 'title': 'Test', 'content': 'old'}]}}
        for sid in ['fixture_a', 'fixture_b']:
            directory = split / 'synthetic_environment' / sid
            directory.mkdir(parents=True)
            (directory / 'state.json').write_text(json.dumps(state))
        self.environments = []
        factory = ToolEnvFactory(ToolEnv, root=self.root, split='synthetic')
        def capture(sample):
            env = factory(sample)
            self.environments.append(env)
            return env
        self.factory = capture
        self.sample = Sample('fixture_a', {
            'latest_text': 'Thêm nội dung vào ghi chú', 'history': [],
            'label': 'LABEL_SECRET', 'state': 'STATE_SECRET',
            'context': {'current_time': '2026-11-05T10:00:00+07:00',
                        'installed_apps': {'notes': {'tools': ['notes.list', 'notes.append'],
                                                     'state': 'APP_STATE_SECRET'}}}})

    def run_native(self, *responses):
        transport = Transport(responses)
        result = Runner(self.factory).run(self.sample, NativeLLMBackend(transport, self.catalog))
        return result, transport

    def test_lookup_update_real_runtime_parser_and_sample_isolation(self):
        result, transport = self.run_native(response('notes.list'),
            response('notes.append', {'note_id': 'fixture_note', 'content': 'new'}), response())
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(len(self.environments), 1)
        self.assertEqual(len(result.executed_calls), 2)
        first_output = result.executed_calls[0].execution.observation.data
        self.assertIn('fixture_note', json.dumps(transport.requests[1][0]))
        self.assertNotIn('SECRET', json.dumps(transport.requests))
        output = self.environments[0].session_output()
        self.assertEqual(output['tool_calls'], [
            {'name': 'notes.list', 'arguments': {}},
            {'name': 'notes.append', 'arguments': {'note_id': 'fixture_note', 'content': 'new'}}])
        fresh = self.factory(Sample('fixture_b'))
        # Runtime canonicalization may omit free-text content from its hash.
        # Inspect state only inside this fixture test, never in an agent input.
        self.assertEqual(self.environments[0]._session.get_state()['notes']['notes'][0]['content'], 'old\nnew')
        self.assertEqual(fresh._session.get_state()['notes']['notes'][0]['content'], 'old')
        self.assertEqual(output['state_hash'], self.environments[0]._session.get_state_hash())
        self.assertEqual(fresh.session_output()['tool_calls'], [])
        self.assertEqual(first_output['notes'][0]['content'], 'old')
        self.assertEqual(transport.requests[0][3], 512)

    def test_schema_repair_does_not_execute_rejected_proposal(self):
        result, transport = self.run_native(response('notes.append', {'content': 'bad'}),
                                           response('notes.list'), response())
        self.assertEqual(result.repairs_used, 1)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual(self.environments[0].session_output()['tool_calls'][0]['name'], 'notes.list')
        self.assertIn('validation_error', json.dumps(transport.requests[1][0]))

    def test_parse_error_repair_is_not_finish(self):
        result, transport = self.run_native({'choices': []}, response())
        self.assertEqual(result.stop_reason, StopReason.FINISH)
        self.assertEqual(result.repairs_used, 1)
        self.assertEqual(len(result.executed_calls), 0)
        self.assertIn('parse_error', json.dumps(transport.requests[1][0]))

    def test_ask_stops_after_execution_without_inventing_done(self):
        result, transport = self.run_native(response('sys.ask_missing_info',
            {'tool_name': 'notes.append', 'params': ['note_id']}), response('notes.list'))
        self.assertEqual(result.stop_reason, StopReason.ASK)
        self.assertEqual(len(transport.requests), 1)
        self.assertFalse(result.last_observation.done)
        self.assertNotIn('done', result.last_observation.data)
        self.assertTrue(result.last_observation.data['waiting_for_user'])
        self.assertEqual(len(self.environments[0].session_output()['tool_calls']), 1)

    def test_done_false_keeps_executed_call(self):
        result, _ = self.run_native(response('notes.append', {'note_id': 'absent', 'content': 'new'}), response())
        self.assertFalse(result.last_observation.done)
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual(len(self.environments[0].session_output()['tool_calls']), 1)

    def test_transport_error_not_finish_or_execution(self):
        result, _ = self.run_native(TimeoutError('private transport details'))
        self.assertEqual(result.stop_reason, StopReason.MODEL_ERROR)
        self.assertEqual(self.environments[0].session_output()['tool_calls'], [])

    def test_multiple_calls_rejected_before_execution(self):
        item = response('notes.list')
        item['choices'][0]['message']['tool_calls'] *= 2
        result, _ = self.run_native(item, response())
        self.assertEqual(result.repairs_used, 1)
        self.assertEqual(self.environments[0].session_output()['tool_calls'], [])

    def test_unknown_execution_not_retried(self):
        class BrokenSession:
            def validate(self, *args):
                return {'ok': True}
            def call(self, *args):
                raise OSError('may already have executed')
        result = Runner(lambda sample: ToolEnvAdapter(BrokenSession())).run(self.sample,
            ScriptedModel([Calls((ProposedCall('notes.list', {}),)), Calls(())]))
        self.assertEqual(result.stop_reason, StopReason.EXECUTION_UNKNOWN)
        self.assertEqual(len(result.uncertain_calls), 1)

    def test_missing_catalog_and_unsafe_sample_id_rejected(self):
        with self.assertRaises(FileNotFoundError):
            ToolEnvFactory(ToolEnv, root=self.root, split='train')
        factory = ToolEnvFactory(ToolEnv, root=self.root, split='synthetic')
        with self.assertRaises(ValueError):
            factory(Sample('../fixture_a'))

    def test_history_is_prompt_only_and_never_replayed(self):
        self.sample.model_input['history'] = [{'role': 'assistant', 'text': '',
            'tool_calls': [{'name': 'notes.append', 'arguments': {'note_id': 'fixture_note', 'content': 'PAST'}}]}]
        result, transport = self.run_native(response('notes.list'), response())
        self.assertEqual(len(result.executed_calls), 1)
        self.assertEqual([c['name'] for c in self.environments[0].session_output()['tool_calls']], ['notes.list'])
        self.assertEqual(result.last_observation.data['notes'][0]['content'], 'old')
        self.assertIn('history_already_executed', transport.requests[0][0][1]['content'])

    def test_validation_side_effect_free_and_tool_error_still_executed(self):
        env = self.factory(self.sample)
        before = env.session_output()
        self.assertFalse(env.validate(ProposedCall('notes.append', {'content': 'bad'})).valid)
        self.assertEqual(env.session_output(), before)
        execution = env.execute(ProposedCall('unknown.tool', {}))
        self.assertTrue(execution.executed)
        self.assertFalse(execution.observation.ok)
        self.assertEqual(len(env.session_output()['tool_calls']), 1)

    def test_system_scaffold_with_committed_runtime_and_grounding_repair(self):
        from lifemate.system import LifeMateSystem, FactSource, GroundingRule, ToolSelector
        transport = Transport([
            response('notes.append', {'note_id': 'invented', 'content': 'bad'}),
            response('notes.list'),
            response('notes.append', {'note_id': 'fixture_note', 'content': 'new'}), response()])
        system = LifeMateSystem(catalog=self.catalog, environment_factory=self.factory,
            transport=transport, selector=ToolSelector(1, {'notes.append': ['notes.list']}),
            grounding_rules=(GroundingRule('notes.append', 'note_id',
                (FactSource('notes.list', ('notes', '*', 'id')),)),))
        result = system.run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'finish')
        self.assertEqual(result.diagnostics['repairs_used'], 1)
        self.assertEqual(len(self.environments), 1)
        actual = self.environments[0].session_output()
        self.assertEqual(result.submission, {'id': self.sample.sample_id, **actual})
        self.assertEqual([c['name'] for c in actual['tool_calls']], ['notes.list', 'notes.append'])
        self.assertNotIn('SECRET', json.dumps(transport.requests))
