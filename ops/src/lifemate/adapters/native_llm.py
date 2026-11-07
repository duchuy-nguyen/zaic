"""Reuse the LLM lane's prompt/parser and LocalServer.generate interface.

The transport is injected: this adapter never starts/downloads a model or calls
ToolEnv. Make llm/ importable explicitly in the integration entrypoint.
"""
from copy import deepcopy
import json
import time

from lifemate.runtime.contracts import Calls, ModelError, ModelRequest, ProposedCall


def prepared_view(sample):
    """Allowlist prepared transcript/context/history; no dataset/state loader."""
    value = sample.model_input
    context = value['context']
    installed = {name: {'tools': deepcopy(app['tools'])}
                 for name, app in context['installed_apps'].items()}
    clean_context = {k: deepcopy(context[k]) for k in
                     ('current_time', 'timezone', 'location', 'current_location', 'user_profile') if k in context}
    clean_context['installed_apps'] = installed
    history = [{k: deepcopy(turn[k]) for k in
                ('role', 'text', 'tool_calls', 'tool_results') if k in turn}
               for turn in value['history']]
    return {'id': sample.sample_id, 'context': clean_context,
            'history': history, 'latest_text': value['latest_text']}


class NativeLLMBackend:
    def __init__(self, transport, catalog: dict, *, clock=time.monotonic, augment_prompt=None):
        from local_baseline.agent import allowed_schemas, initial_messages, parse_decision
        self.transport, self.catalog, self.clock = transport, deepcopy(catalog), clock
        self.allowed_schemas, self.initial_messages = allowed_schemas, initial_messages
        self.parse_decision = parse_decision
        # Optional system-layer hook. It cannot execute tools or change the
        # environment allowlist; execution authorization lives at that boundary.
        self.augment_prompt = augment_prompt

    def respond(self, request: ModelRequest):
        deadline = self.clock() + request.remaining_seconds
        try:
            view = prepared_view(request.sample)
            messages = self.initial_messages(view)
            tools = self.allowed_schemas(view, self.catalog)
            for index, record in enumerate(request.observations, 1):
                if record.execution.executed is not True:
                    raise ValueError('Unexecuted observations require an explicit protocol')
                call_id = f'ops_call_{index:03d}'
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
                    'id': call_id, 'type': 'function', 'function': {
                        'name': record.proposal.name,
                        'arguments': json.dumps(record.proposal.arguments, ensure_ascii=False)}}]})
                messages.append({'role': 'tool', 'tool_call_id': call_id,
                                 'content': json.dumps(record.execution.observation.data, ensure_ascii=False)})
            if request.repair:
                # Repair policy text lives here at the LLM boundary, not in Runner.
                feedback = {'kind': request.repair.kind}
                if request.repair.proposal:
                    feedback['proposal'] = {'name': request.repair.proposal.name,
                                            'arguments': request.repair.proposal.arguments}
                if request.repair.validation:
                    feedback['validation_code'] = request.repair.validation.code
                messages.append({'role': 'user', 'content':
                    'Đề xuất trước chưa được thực thi. Hãy sửa định dạng/schema, trả tối đa một tool call '
                    'hợp lệ hoặc kết thúc khi đã hoàn tất. Chi tiết: ' + json.dumps(feedback, ensure_ascii=False)})
            if self.augment_prompt is not None:
                messages, tools = self.augment_prompt(request, messages, tools)
            if self.clock() >= deadline:
                return ModelError('model_error')
            response, _usage = self.transport.generate(messages, tools, deadline,
                                                       max_tokens=request.output_tokens)
        except Exception:
            return ModelError('model_error')
        try:
            kind, proposal = self.parse_decision(response)
            if kind == 'finish':
                return Calls(())
            return Calls((ProposedCall(proposal['name'], proposal['arguments']),))
        except (ValueError, TypeError, KeyError, AttributeError):
            return ModelError('parse_error')
