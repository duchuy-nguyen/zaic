"""Entirely fictional model/app fixture; NOT an official solution or benchmark."""
from copy import deepcopy
import hashlib
import json

from lifemate.adapters.toolenv import ToolEnvAdapter
from lifemate.system import LifeMateSystem, FactSource, GroundingRule, ToolSelector


def catalog():
    def spec(name, description, properties, required=(), kind='read'):
        return {'name': name, 'kind': kind, 'description': description,
                'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}
    return {t['name']: t for t in [
        spec('fixture.list', 'Tìm ghi chú và ID', {}),
        spec('fixture.append', 'Thêm nội dung ghi chú',
             {'note_id': {'type': 'string'}, 'content': {'type': 'string'}},
             ('note_id', 'content'), 'write'),
        spec('sys.ask_missing_info', 'Hỏi thông tin thiếu',
             {'tool_name': {'type': 'string'}, 'params': {'type': 'array', 'items': {'type': 'string'}}},
             ('tool_name', 'params'), 'meta')]}


class FixtureSession:
    def __init__(self):
        self.calls, self.content = [], 'old'

    def validate(self, name, args):
        schemas = catalog()
        return {'ok': name in schemas and all(k in args for k in schemas[name]['parameters']['required'])}

    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': deepcopy(args)})
        if name == 'fixture.list':
            return {'ok': True, 'notes': [{'id': 'fixture_note', 'content': self.content}]}
        if name == 'fixture.append':
            self.content += '\n' + args['content']
            return {'ok': True, 'done': True}
        return {'ok': True}

    def get_tool_calls(self):
        return deepcopy(self.calls)

    def get_state_hash(self):
        # Fixture hash only. Production delegates this to official ToolEnv.
        return hashlib.sha256(self.content.encode()).hexdigest()


class FixtureTransport:
    def generate(self, messages, tools, deadline, *, max_tokens):
        observed = sum(m['role'] == 'tool' for m in messages)
        call = [('fixture.list', {}), ('fixture.append', {'note_id': 'fixture_note', 'content': 'new'})]
        message = {'role': 'assistant', 'content': 'Đã xong.'}
        if observed < len(call):
            name, args = call[observed]
            message['tool_calls'] = [{'id': 'fixture_call', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]
        return {'choices': [{'finish_reason': 'tool_calls' if observed < len(call) else 'stop',
                             'message': message}]}, 0


def build_system():
    return LifeMateSystem(catalog=catalog(),
        environment_factory=lambda _: ToolEnvAdapter(FixtureSession()), transport=FixtureTransport(),
        selector=ToolSelector(top_k=1, dependencies={'fixture.append': ['fixture.list']}),
        grounding_rules=(GroundingRule('fixture.append', 'note_id',
            (FactSource('fixture.list', ('notes', '*', 'id')),)),),
        provenance={'mode': 'fictional_fixture', 'model': 'scripted_no_weights',
                    'environment': 'FixtureSession_not_official_ToolEnv'})
