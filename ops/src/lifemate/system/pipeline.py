"""ASR → augmented native agent → one real environment → authoritative export."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

from lifemate.adapters.native_llm import NativeLLMBackend
from lifemate.runtime.contracts import RuntimeConfig, StopReason
from lifemate.runtime.runner import Runner
from .context import prepare_sample, permitted_names
from .grounding import GuardedEnvironment
from .prompt import PromptAugmenter
from .retrieval import ToolSelector


@dataclass(frozen=True)
class SystemResult:
    submission: dict
    diagnostics: dict


class LifeMateSystem:
    def __init__(self, *, catalog, environment_factory, transport,
                 runtime=RuntimeConfig(), transcriber=None, selector=None,
                 grounding_rules=(), examples=None, trace=lambda e: None,
                 clock=time.monotonic, provenance=None):
        self.catalog, self.environment_factory, self.transport = deepcopy(catalog), environment_factory, transport
        self.runtime, self.transcriber, self.clock = runtime, transcriber, clock
        self.selector = selector if selector is not None else ToolSelector()
        self.rules, self.examples, self.trace = tuple(grounding_rules), examples, trace
        self.provenance = deepcopy(provenance or {'status': 'unverified_scaffold'})
        for rule in self.rules:
            if rule.tool not in self.catalog or rule.argument not in self.catalog[rule.tool]['parameters'].get('properties', {}):
                raise ValueError('Grounding binding not in supplied catalog')
            if not rule.sources:
                raise ValueError('Grounding binding requires evidence sources')
            if any(s.tool is not None and s.tool not in self.catalog for s in rule.sources):
                raise ValueError('Unknown grounding source tool')
            if any(n not in self.catalog for n in rule.invalidate_on):
                raise ValueError('Unknown invalidation tool')

    def manifest(self):
        def fingerprint(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        package = Path(__file__).resolve().parents[1]
        source_files = sorted((package / 'system').glob('*.py')) + [
            package / 'runtime' / n for n in ('runner.py', 'contracts.py', 'deadlines.py')]
        source_files += [package / 'adapters' / n for n in ('native_llm.py', 'toolenv.py')]
        source_hashes = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source_files}
        return {'runtime': asdict(self.runtime), 'catalog_hash': fingerprint(self.catalog),
            'source_hashes': source_hashes,
            'retrieval': {'top_k': self.selector.top_k,
                          'dependencies': deepcopy(self.selector.dependencies)},
            'grounding_rules': [asdict(r) for r in self.rules],
            'examples': {'top_k': self.examples.top_k, 'split_hash': self.examples.split_hash,
                         'corpus_hash': fingerprint(self.examples.examples)} if self.examples else None,
            'provenance': deepcopy(self.provenance),
            'limitations': ['no_official_score', 'no_hard_preemption',
                           'no_target_gpu_certification', 'caller_supplies_pinned_models_and_runtime']}

    def run(self, sample):
        if not isinstance(sample.sample_id, str) or not sample.sample_id.strip():
            raise ValueError('Sample ID required')
        started = self.clock()
        remaining = lambda: self.runtime.sample_deadline_seconds - (self.clock() - started)
        # Factory receives ID only. Labels/audio/context are not a state loader API.
        environment = self.environment_factory(replace(sample, model_input={}))
        status, failure, repairs = 'preparation_error', True, 0
        try:
            prepared = prepare_sample(sample, transcriber=self.transcriber, remaining=remaining)
            allowed = permitted_names(prepared, self.catalog)
        except Exception as exc:
            self.trace({'event': 'preparation_failure', 'sample_id': sample.sample_id,
                        'error_type': type(exc).__name__})
        else:
            if remaining() > 0:
                guarded = GuardedEnvironment(environment, prepared, allowed, self.rules)
                augmenter = PromptAugmenter(self.catalog, self.selector, self.rules, self.examples, self.trace)
                backend = NativeLLMBackend(self.transport, self.catalog, clock=self.clock, augment_prompt=augmenter)
                result = Runner(lambda _: guarded,
                    replace(self.runtime, sample_deadline_seconds=remaining()),
                    clock=self.clock, trace=self.trace).run(prepared, backend)
                status, failure, repairs = result.stop_reason.value, result.failure, result.repairs_used
            else:
                status = StopReason.DEADLINE.value
        # Export failures propagate: never fabricate a hash or replay an env.
        output = environment.session_output()
        if (set(output) != {'tool_calls', 'state_hash'} or type(output['tool_calls']) is not list
                or not isinstance(output['state_hash'], str) or not output['state_hash']):
            raise ValueError('Invalid authoritative environment output')
        elapsed = self.clock() - started
        if elapsed > self.runtime.sample_deadline_seconds:
            status, failure = StopReason.DEADLINE.value, True
        return SystemResult({'id': sample.sample_id, **deepcopy(output)},
            {'id': sample.sample_id, 'status': status, 'failure': failure,
             'elapsed_seconds': elapsed, 'repairs_used': repairs,
             'executed_call_count': len(output['tool_calls']),
             'provenance': deepcopy(self.provenance)})
