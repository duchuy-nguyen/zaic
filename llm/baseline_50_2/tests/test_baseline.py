import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import zipfile

from baseline50 import actor
from baseline50.io import safe_row, resume_records, write_json


class Backend:
    def __init__(self, messages):
        self.queue = messages; self.inputs = []

    def generate(self, messages, tools, deadline):
        self.inputs.append(copy.deepcopy(messages))
        message = self.queue.pop(0)
        return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
                'usage': {'prompt_tokens': 10}}, 10


def call(name, args, i='c1'):
    return {'content': '', 'tool_calls': [{'id': i, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


class Environment:
    def __init__(self, valid=True): self.calls = []; self.valid = valid
    def validate(self, name, arguments): return {'ok': self.valid, 'error': None if self.valid else 'synthetic schema error'}
    def call(self, name, arguments):
        self.calls.append({'name': name, 'arguments': arguments})
        return {'ok': True, 'contacts': [{'id': 'contact_test'}]}
    def get_tool_calls(self): return copy.deepcopy(self.calls)
    def get_state_hash(self): return 'a' * 64


class Tests(unittest.TestCase):
    def setUp(self):
        actor._RETRIEVER = actor.TfidfExamples([]); actor._GROUPS = {}
        self.view = {'id': 'synthetic', 'latest_text': 'Tìm liên hệ rồi gửi tin.', 'context': {
            'installed_apps': {'contacts': {'tools': ['contacts.search', 'contacts.send']}}}, 'history': []}
        self.tools = {n: {'parameters': {'type': 'object', 'properties': {}}, 'description': n}
                      for n in ('contacts.search', 'contacts.send', 'sys.ask_missing_info')}

    def test_observation_returned_before_next_decision(self):
        backend = Backend([call('contacts.search', {'query': 'test'}),
                           call('contacts.send', {'contact_id': 'contact_test', 'text': 'hello'}, 'c2'), {'content': ''}])
        env = Environment(); r = actor.infer(self.view, self.tools, env, backend)
        self.assertEqual(r['status'], 'finished'); self.assertEqual(len(r['tool_calls']), 2)
        self.assertEqual(backend.inputs[1][-1]['role'], 'tool')
        self.assertIn('contact_test', backend.inputs[1][-1]['content'])

    def test_ask_stops_without_inventing_user_response(self):
        backend = Backend([call('sys.ask_missing_info', {'tool_name': 'contacts.send', 'params': ['contact_id']})])
        r = actor.infer(self.view, self.tools, Environment(), backend)
        self.assertEqual(r['status'], 'asked_missing_info'); self.assertEqual(len(backend.inputs), 1)

    def test_bad_schema_never_executed_and_two_repair_limit(self):
        backend = Backend([call('contacts.send', {}) for _ in range(3)])
        r = actor.infer(self.view, self.tools, Environment(False), backend)
        self.assertEqual(r['status'], 'structured_output_error'); self.assertEqual(r['repairs'], 2)
        self.assertEqual(r['tool_calls'], [])

    def test_multiple_proposals_not_silently_executed(self):
        message = call('contacts.search', {}); message['tool_calls'] *= 2
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            actor.parse_decision({'choices': [{'finish_reason': 'tool_calls', 'message': message}]})

    def test_timeout_executes_nothing(self):
        r = actor.infer(self.view, self.tools, Environment(), Backend([]), seconds=-1)
        self.assertEqual(r['status'], 'timeout'); self.assertEqual(r['tool_calls'], [])

    def test_history_user_gold_stripped_assistant_preserved(self):
        row = {'id': 'x', 'audio': 'x.wav', 'context': {}, 'audio_text': 'SECRET_GOLD', 'label': {},
               'history': [{'role': 'user', 'audio': 'h.wav', 'audio_text': 'SECRET_GOLD'},
                           {'role': 'assistant', 'text': 'done', 'tool_results': [{'id': 'id1'}]}]}
        safe = safe_row(row)
        self.assertNotIn('SECRET_GOLD', str(safe)); self.assertNotIn('label', safe)
        self.assertEqual(safe['history'][1], row['history'][1])

    def test_uncommitted_inflight_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); write_json(root / 'inflight.json', {'id': 'a'})
            with self.assertRaisesRegex(ValueError, 'Uncommitted'):
                resume_records(root, ['a'])
            self.assertTrue((root / 'inflight.json').exists())

    def test_committed_inflight_does_not_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'records').mkdir()
            write_json(root / 'records/a.json', {'id': 'a'}); write_json(root / 'inflight.json', {'id': 'a'})
            self.assertEqual(set(resume_records(root, ['a', 'b'])), {'a'})
            self.assertFalse((root / 'inflight.json').exists())

    def test_unpack_traversal_rejected(self):
        p = Path(__file__).parents[1] / 'tools/unpack_assets.py'
        spec = importlib.util.spec_from_file_location('unpack_test', p); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'test.zip'
            with zipfile.ZipFile(archive, 'x') as z:
                z.writestr('../escape', 'x')
                z.writestr('ARCHIVE_MANIFEST.json', json.dumps({'files': {'../escape': {'bytes': 1, 'sha256': 'x'}}}))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                module.unpack(archive, root / 'assets')
            self.assertFalse((root / 'escape').exists())


if __name__ == '__main__': unittest.main()
