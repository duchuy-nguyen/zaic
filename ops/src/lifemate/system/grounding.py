"""Explicit typed evidence bindings; no inferred x-ref syntax or hidden IDs."""
from dataclasses import dataclass
from copy import deepcopy

from lifemate.runtime.contracts import CallRecord, Validation


def values_at(value, path):
    if not path:
        return [value]
    key, *tail = path
    if key == '*':
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        return [v for child in children for v in values_at(child, tail)]
    if isinstance(value, dict) and key in value:
        return values_at(value[key], tail)
    return []


@dataclass(frozen=True)
class FactSource:
    # None reads from sanitized model_input (e.g. context.user_profile IDs).
    # A tool name reads only successful actual observations from that tool.
    tool: str | None
    path: tuple[str, ...]


@dataclass(frozen=True)
class GroundingRule:
    tool: str
    argument: str
    sources: tuple[FactSource, ...]
    # Explicit writes invalidate older evidence, even if side effects are unclear.
    invalidate_on: tuple[str, ...] = ()


def evidence(sample, observations, rule):
    last_invalidation = max((i for i, r in enumerate(observations)
        if r.proposal.name in rule.invalidate_on and r.execution.executed is not False), default=-1)
    facts = []
    for source in rule.sources:
        if source.tool is None:
            candidates = [(sample.model_input, 'context', -1)] if last_invalidation < 0 else []
        else:
            candidates = [(r.execution.observation.data, source.tool, i)
                for i, r in enumerate(observations) if i > last_invalidation
                and r.proposal.name == source.tool and r.execution.executed is True
                and r.execution.observation.ok and r.execution.observation.data.get('done') is not False]
        for body, origin, index in candidates:
            for value in values_at(body, source.path):
                if value is not None and type(value) in (str, int, float, bool):
                    facts.append({'value': deepcopy(value), 'source': origin,
                                  'path': list(source.path), 'observation_index': index})
    return facts


def validate_grounding(sample, observations, call, rules):
    for rule in rules:
        if call.name == rule.tool and rule.argument in call.arguments:
            candidate = call.arguments[rule.argument]
            facts = evidence(sample, observations, rule)
            if not any(type(candidate) is type(f['value']) and candidate == f['value'] for f in facts):
                return Validation(False, code='ungrounded_argument:' + rule.argument)
    return Validation(True, is_ask=call.name == 'sys.ask_missing_info')


class GuardedEnvironment:
    """Authorization/grounding before the existing ToolEnv validation boundary."""
    def __init__(self, environment, sample, permitted, rules):
        self.environment, self.sample = environment, sample
        self.permitted, self.rules, self.observations = set(permitted), tuple(rules), []

    def validate(self, call):
        if call.name not in self.permitted:
            return Validation(False, code='tool_not_installed')
        grounded = validate_grounding(self.sample, self.observations, call, self.rules)
        return self.environment.validate(call) if grounded.valid else grounded

    def execute(self, call):
        result = self.environment.execute(call)
        self.observations.append(CallRecord(deepcopy(call), deepcopy(result)))
        return result

    def session_output(self):
        return self.environment.session_output()
