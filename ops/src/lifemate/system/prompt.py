"""Prompt-only augmentation. Tool execution and authorization stay outside."""
from copy import deepcopy
import json

from .context import permitted_names
from .grounding import evidence


class PromptAugmenter:
    def __init__(self, catalog, selector, rules=(), examples=None, trace=lambda e: None):
        self.catalog, self.selector, self.rules = catalog, selector, tuple(rules)
        self.examples, self.trace = examples, trace

    def __call__(self, request, messages, tools):
        query = request.sample.model_input['latest_text']
        query += ' ' + json.dumps(request.sample.model_input['history'], ensure_ascii=False)
        if request.observations:
            query += ' ' + json.dumps(request.observations[-1].execution.observation.data, ensure_ascii=False)
        selected = self.selector.select(query, tools, repair=request.repair is not None)
        allowed = permitted_names(request.sample, self.catalog)
        if any(t['function']['name'] not in allowed for t in selected):
            raise ValueError('Retriever attempted to widen permissions')
        demonstrations = self.examples.retrieve(query, allowed) if self.examples else []
        facts = [{'tool': r.tool, 'argument': r.argument,
                  'evidence': evidence(request.sample, request.observations, r)} for r in self.rules]
        body = {'grounding_evidence': facts,
                'demonstrations_not_current_actions': demonstrations}
        result = deepcopy(messages)
        if facts or demonstrations:
            result.insert(1, {'role': 'user', 'content': json.dumps(body, ensure_ascii=False, allow_nan=False)})
        self.trace({'event': 'retrieval', 'sample_id': request.sample.sample_id,
            'step': request.step, 'permitted_count': len(allowed),
            'selected_tools': [t['function']['name'] for t in selected],
            'example_ids': [e['id'] for e in demonstrations],
            'example_split_hash': self.examples.split_hash if self.examples else None,
            'repair_fallback': request.repair is not None})
        return result, selected
