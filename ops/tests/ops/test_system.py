"""System contracts: fictional fixtures only, no BTC state/model downloads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'llm'))

from lifemate.adapters.toolenv import ToolEnvAdapter
from lifemate.dev.mocks import FakeClock
from lifemate.dev.system_demo import FixtureSession, FixtureTransport, catalog
from lifemate.runtime.contracts import CallRecord, Execution, Observation, ProposedCall, RuntimeConfig, Sample
from lifemate.system import LifeMateSystem, FactSource, GroundingRule, ToolSelector, ExampleBank
from lifemate.system.cli import load_samples, run_batch
from lifemate.system.context import prepare_sample
from lifemate.system.grounding import evidence, validate_grounding


def native(name=None, args=None):
    message = {'role': 'assistant', 'content': 'Đã xong.'}
    if name:
        message['tool_calls'] = [{'id': 'test_call', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args or {})}}]
    return {'choices': [{'finish_reason': 'tool_calls' if name else 'stop', 'message': message}]}


class ScriptedTransport:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def generate(self, messages, tools, deadline, *, max_tokens):
        self.requests.append(deepcopy((messages, tools)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, 0


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.sample = Sample('fixture_a', {'latest_text': 'Thêm nội dung ghi chú', 'history': [],
            'context': {'installed_apps': {'fixture': {'tools': ['fixture.list', 'fixture.append']}}}})
        self.sessions, self.events = [], []
        self.rule = GroundingRule('fixture.append', 'note_id',
            (FactSource('fixture.list', ('notes', '*', 'id')),))

    def factory(self, sample):
        self.assertEqual(sample.model_input, {})
        session = FixtureSession()
        self.sessions.append(session)
        return ToolEnvAdapter(session)

    def system(self, transport=None, **kwargs):
        return LifeMateSystem(catalog=catalog(), environment_factory=self.factory,
            transport=transport or FixtureTransport(), trace=self.events.append, **kwargs)

    def observation(self, name='fixture.list', data=None, ok=True):
        return CallRecord(ProposedCall(name, {}), Execution(True, Observation(ok, False,
            data if data is not None else {'ok': True, 'notes': [{'id': 'fixture_note'}]})))

    def test_lookup_ground_write_export_and_samples_isolated(self):
        system = self.system(grounding_rules=(self.rule,))
        result = system.run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'finish')
        self.assertFalse(result.diagnostics['failure'])
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(result.submission['tool_calls'], self.sessions[0].get_tool_calls())
        self.assertEqual(result.submission['state_hash'], self.sessions[0].get_state_hash())
        self.assertEqual([c['name'] for c in result.submission['tool_calls']], ['fixture.list', 'fixture.append'])
        system.run(Sample('fixture_b', deepcopy(self.sample.model_input)))
        self.assertEqual(len(self.sessions), 2)
        self.assertEqual([s.content for s in self.sessions], ['old\nnew', 'old\nnew'])

    def test_ungrounded_call_repair_then_lookup_not_executed_twice(self):
        transport = ScriptedTransport(native('fixture.append', {'note_id': 'invented', 'content': 'bad'}),
            native('fixture.list'), native('fixture.append', {'note_id': 'fixture_note', 'content': 'new'}), native())
        result = self.system(transport, grounding_rules=(self.rule,), selector=ToolSelector(1)).run(self.sample)
        self.assertEqual(result.diagnostics['repairs_used'], 1)
        self.assertEqual(len(result.submission['tool_calls']), 2)
        self.assertNotIn('bad', self.sessions[0].content)
        self.assertTrue(any(e.get('repair_fallback') for e in self.events if e['event'] == 'retrieval'))
        self.assertIn('ungrounded_argument', json.dumps(transport.requests[1]))

    def test_uninstalled_tool_blocked_even_if_model_ignores_pruning(self):
        self.sample.model_input['context']['installed_apps']['fixture']['tools'] = ['fixture.list']
        transport = ScriptedTransport(native('fixture.append', {'note_id': 'fixture_note', 'content': 'bad'}), native())
        result = self.system(transport).run(self.sample)
        self.assertEqual(result.submission['tool_calls'], [])
        self.assertEqual(result.diagnostics['repairs_used'], 1)

    def test_ask_stops_and_oos_can_stop_without_reads(self):
        transport = ScriptedTransport(native('sys.ask_missing_info', {'tool_name': 'fixture.append', 'params': ['content']}))
        result = self.system(transport).run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'ask')
        self.assertEqual(len(result.submission['tool_calls']), 1)
        result = self.system(ScriptedTransport(native())).run(self.sample)
        self.assertEqual(result.submission['tool_calls'], [])
        self.assertFalse(result.diagnostics['failure'])

    def test_model_error_retains_already_executed_call(self):
        result = self.system(ScriptedTransport(native('fixture.list'), OSError('private error'))).run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'model_error')
        self.assertTrue(result.diagnostics['failure'])
        self.assertEqual(len(result.submission['tool_calls']), 1)

    def test_unknown_execution_exports_same_env_without_retry(self):
        class UncertainSession(FixtureSession):
            def call(self, name, args):
                super().call(name, args)
                raise OSError('after side effect')
        session = UncertainSession()
        system = LifeMateSystem(catalog=catalog(), environment_factory=lambda _: ToolEnvAdapter(session),
            transport=ScriptedTransport(native('fixture.list'), native()))
        result = system.run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'execution_unknown')
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(result.submission['tool_calls'], session.get_tool_calls())

    def test_failed_tool_calls_are_exported_and_returned_to_model(self):
        class FailedSession(FixtureSession):
            def call(self, name, args):
                super().call(name, args)
                return {'ok': False, 'error': 'fixture_failure'}
        transport = ScriptedTransport(native('fixture.list'), native())
        session = FailedSession()
        system = LifeMateSystem(catalog=catalog(), environment_factory=lambda _: ToolEnvAdapter(session), transport=transport)
        result = system.run(self.sample)
        self.assertEqual(len(result.submission['tool_calls']), 1)
        self.assertIn('fixture_failure', json.dumps(transport.requests[1]))

    def test_preparation_strips_gold_and_history_not_replayed(self):
        self.sample.model_input['label'] = 'GOLD_SECRET'
        self.sample.model_input['audio_text'] = 'TRANSCRIPT_SECRET'
        self.sample.model_input['state'] = 'STATE_SECRET'
        self.sample.model_input['context']['installed_apps']['fixture']['state'] = 'APP_SECRET'
        self.sample.model_input['history'] = [{'role': 'assistant', 'text': 'past',
            'label': 'HISTORY_SECRET', 'tool_calls': [{'name': 'fixture.append', 'arguments': {}}]}]
        transport = ScriptedTransport(native())
        result = self.system(transport).run(self.sample)
        self.assertNotIn('SECRET', json.dumps(transport.requests))
        self.assertIn('history_already_executed', json.dumps(transport.requests))
        self.assertEqual(result.submission['tool_calls'], [])

    def test_audio_latest_and_history_without_gold_fallback(self):
        seen = []
        class ASR:
            def transcribe(self, audio, *, remaining_seconds):
                seen.append((audio, remaining_seconds))
                return 'ASR ' + audio
        value = deepcopy(self.sample.model_input)
        del value['latest_text']
        value.update(latest_audio='current.wav', audio_text='GOLD_SECRET',
                     history=[{'role': 'user', 'audio': 'past.wav', 'audio_text': 'PAST_SECRET'}])
        prepared = prepare_sample(Sample('a', value), transcriber=ASR())
        self.assertEqual([a for a, _ in seen], ['past.wav', 'current.wav'])
        self.assertEqual(prepared.model_input['latest_text'], 'ASR current.wav')
        self.assertNotIn('SECRET', json.dumps(prepared.model_input))

    def test_asr_failure_outputs_real_empty_hash_without_model(self):
        class ASR:
            def transcribe(self, *args, **kwargs):
                raise OSError('ASR failed')
        value = deepcopy(self.sample.model_input)
        del value['latest_text']
        value['latest_audio'] = 'sample.wav'
        transport = ScriptedTransport()
        result = self.system(transport, transcriber=ASR()).run(Sample('a', value))
        self.assertEqual(result.diagnostics['status'], 'preparation_error')
        self.assertEqual(result.submission['state_hash'], self.sessions[0].get_state_hash())
        self.assertEqual(transport.requests, [])

    def test_audio_deadline_is_shared_with_agent(self):
        clock = FakeClock()
        class ASR:
            def transcribe(self, audio, **kwargs):
                clock.advance(31)
                return 'text'
        value = deepcopy(self.sample.model_input)
        del value['latest_text']
        value['latest_audio'] = 'sample.wav'
        transport = ScriptedTransport()
        result = self.system(transport, transcriber=ASR(), clock=clock).run(Sample('a', value))
        self.assertEqual(result.diagnostics['status'], 'deadline')
        self.assertEqual(transport.requests, [])
        self.assertEqual(len(self.sessions), 1)

    def test_typed_evidence_failed_results_and_write_invalidation(self):
        rule = GroundingRule('fixture.append', 'note_id', self.rule.sources, ('fixture.append',))
        self.assertTrue(validate_grounding(self.sample, [self.observation()],
            ProposedCall('fixture.append', {'note_id': 'fixture_note'}), (rule,)).valid)
        stale = [self.observation(), self.observation('fixture.append', {'ok': True})]
        self.assertFalse(evidence(self.sample, stale, rule))
        stale.append(self.observation())
        self.assertTrue(evidence(self.sample, stale, rule))
        self.assertFalse(evidence(self.sample, [self.observation(ok=False)], rule))
        self.assertFalse(evidence(self.sample, [self.observation(data={'ok': True, 'done': False,
            'notes': [{'id': 'fixture_note'}]})], rule))
        numeric = self.observation(data={'notes': [{'id': 1}]})
        self.assertFalse(validate_grounding(self.sample, [numeric],
            ProposedCall('fixture.append', {'note_id': True}), (rule,)).valid)

    def test_context_binding_does_not_accept_arbitrary_text(self):
        self.sample.model_input['context']['user_profile'] = {'note_id': 'trusted'}
        rule = GroundingRule('fixture.append', 'note_id', (FactSource(None, ('context', 'user_profile', 'note_id')),))
        self.assertEqual(evidence(self.sample, (), rule)[0]['value'], 'trusted')
        self.assertFalse(validate_grounding(self.sample, (), ProposedCall('fixture.append', {'note_id': 'invented'}), (rule,)).valid)

    def test_missing_catalog_and_invalid_grounding_configuration(self):
        with self.assertRaises(ValueError):
            self.system(grounding_rules=(GroundingRule('fixture.append', 'typo', self.rule.sources),))
        self.sample.model_input['context']['installed_apps']['fixture']['tools'].append('absent.tool')
        result = self.system(ScriptedTransport()).run(self.sample)
        self.assertEqual(result.diagnostics['status'], 'preparation_error')
        self.assertEqual(result.submission['tool_calls'], [])

    def test_batch_manifest_coverage_no_overwrite_and_no_redaction_of_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            manifest = run_batch(self.system(), [self.sample], root)
            self.assertEqual(manifest['status'], 'completed')
            self.assertTrue(json.loads((root / 'submission_check.json').read_text())['valid'])
            self.assertEqual(json.loads((root / 'submission.jsonl').read_text())['tool_calls'], self.sessions[0].calls)
            self.assertIn('catalog_hash', manifest['configuration'])
            self.assertTrue((root / 'traces.jsonl').is_file())
            with self.assertRaises(FileExistsError):
                run_batch(self.system(), [self.sample], root)

    def test_batch_initialization_failure_has_no_invented_submission(self):
        def broken(_):
            raise OSError('official environment missing')
        system = LifeMateSystem(catalog=catalog(), environment_factory=broken, transport=FixtureTransport())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            with self.assertRaises(OSError):
                run_batch(system, [self.sample], root)
            self.assertEqual(json.loads((root / 'run_manifest.json').read_text())['status'], 'failed')
            self.assertEqual((root / 'submission.jsonl').read_text(), '')

    def test_prepared_loader_rejects_gold_duplicates_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            row = {'id': self.sample.sample_id, **self.sample.model_input}
            for text in ('', json.dumps({**row, 'label': {}}), json.dumps(row) + '\n' + json.dumps(row)):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_samples(path)
            path.write_text(json.dumps(row), encoding='utf-8')
            self.assertEqual(load_samples(path), [self.sample])


class RetrievalTests(unittest.TestCase):
    def tools(self):
        return [{'type': 'function', 'function': {'name': n, 'description': desc,
                 'parameters': {'properties': {}}}} for n, desc in [
            ('notes.append', 'append note content'), ('notes.list', 'lookup identifiers'),
            ('ride.book', 'book car'), ('sys.ask_missing_info', 'ask missing input')]]

    def test_full_default_unknown_query_and_repair_fallback(self):
        tools = self.tools()
        self.assertEqual(ToolSelector().select('note', tools), tools)
        self.assertEqual(ToolSelector(1).select('zz', tools), tools)
        self.assertEqual(ToolSelector(1).select('note', tools, repair=True), tools)

    def test_dependency_cycles_determinism_and_permissions(self):
        selector = ToolSelector(1, {'notes.append': ['notes.list', 'hidden.tool'], 'notes.list': ['notes.append']})
        selected = selector.select('append note content', self.tools())
        names = [t['function']['name'] for t in selected]
        self.assertEqual(names, ['notes.append', 'notes.list', 'sys.ask_missing_info'])
        self.assertEqual(selected, selector.select('append note content', list(reversed(self.tools()))))

    def test_examples_fit_only_schema_permissions_and_copy_isolation(self):
        example = {'id': 'fit_a', 'utterance': 'append note', 'guidance': 'Lookup ID, then write.', 'tools': ['notes.list']}
        with self.assertRaises(ValueError):
            ExampleBank([example], fit_ids={'other'}, split_hash='frozen', top_k=1)
        with self.assertRaises(ValueError):
            ExampleBank([example, example], fit_ids={'fit_a'}, split_hash='frozen')
        bank = ExampleBank([example], fit_ids={'fit_a'}, split_hash='frozen', top_k=1)
        self.assertEqual(bank.retrieve('append note', set()), [])
        result = bank.retrieve('append note', {'notes.list'})
        self.assertEqual(len(result), 1)
        result[0]['guidance'] = 'changed'
        self.assertEqual(bank.examples[0]['guidance'], example['guidance'])


if __name__ == '__main__':
    unittest.main()
