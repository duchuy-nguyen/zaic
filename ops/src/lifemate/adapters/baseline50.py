"""Frozen baseline50 actor and batched ASR at the Ops prepared-input boundary.

No model imports/downloads at module import. Caller supplies an already running
transport, ASR instance and the pinned raw SDK session factory.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

from lifemate.runtime.configuration import strict_json
from lifemate.runtime.contracts import RuntimeConfig
from lifemate.system.context import prepare_sample, permitted_names
from lifemate.system.pipeline import LifeMateSystem, SystemResult
from lifemate.adapters.native_llm import prepared_view


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scoped(function, namespace):
    """Same frozen function code, isolated package globals; no global rebinding."""
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


class Baseline50Policy:
    def __init__(self, private_data, *, clock=time.monotonic):
        from baseline50 import actor
        self.private_data = Path(private_data).resolve()
        index_path = self.private_data / 'retrieval_index.json'
        split_path = self.private_data / 'split.json'
        examples = strict_json(index_path.read_text(encoding='utf-8'))
        split = strict_json(split_path.read_text(encoding='utf-8'))
        fit, val, groups = split['train_fit'], split['val_main'], split['group_ids']
        if (type(fit) is not list or type(val) is not list or type(groups) is not dict
                or any(not isinstance(s, str) or not s for s in fit + val)
                or len(set(fit)) != len(fit) or len(set(val)) != len(val)
                or set(fit) & set(val) or not set(fit + val) <= groups.keys()
                or type(examples) is not list):
            raise ValueError('Invalid frozen baseline split/index')
        for example in examples:
            if (type(example) is not dict or example.get('source_id') not in set(fit)
                    or example.get('group_id') != groups.get(example.get('source_id'))):
                raise ValueError('Example source/group outside frozen TRAIN-fit')
        self.fit_ids, self.val_ids = frozenset(fit), frozenset(val)
        # The original actor functions use module globals. Give each bridge its
        # own namespace, while keeping their exact code objects/defaults.
        namespace = dict(vars(actor))
        namespace.update(_RETRIEVER=actor.TfidfExamples(deepcopy(examples)),
                         _GROUPS=deepcopy(groups), time=SimpleNamespace(monotonic=clock))
        namespace['initial_messages'] = _scoped(actor.initial_messages, namespace)
        self.infer = _scoped(actor.infer, namespace)
        self.initial_messages = namespace['initial_messages']
        self.allowed_schemas = actor.allowed_schemas
        self.parse_decision = actor.parse_decision
        self.identity = {'split_sha256': _sha(split_path), 'retrieval_index_sha256': _sha(index_path),
                         'baseline_source': {p.name: _sha(p) for p in sorted(Path(actor.__file__).parent.glob('*.py'))},
                         'assets_verified_by_bridge': False}


class Baseline50ASR:
    """Keep current-first/history-user order and one batch call per sample."""
    def __init__(self, asr, *, clock=time.monotonic):
        self.asr, self.clock = asr, clock

    def _batch(self, paths, deadline):
        if self.clock() >= deadline:
            raise TimeoutError('ASR deadline')
        mapping, details = self.asr.transcribe(list(paths), deadline)
        if self.clock() >= deadline:
            raise TimeoutError('ASR returned after deadline')
        if (type(mapping) is not dict or set(mapping) != set(paths) or type(details) is not dict
                or any(not isinstance(t, str) or not t.strip() for t in mapping.values())):
            raise ValueError('ASR missing/invalid transcript coverage')
        return mapping

    def transcribe(self, audio, *, remaining_seconds):
        from baseline50.asr import normalize_transcript
        if not isinstance(audio, str) or not audio:
            raise ValueError('Invalid audio reference')
        mapping = self._batch([audio], self.clock() + remaining_seconds)
        return normalize_transcript(mapping[audio])

    def prepare(self, sample, *, deadline):
        from baseline50.asr import normalize_transcript
        value = sample.model_input
        paths = []
        if 'latest_text' not in value:
            paths.append(value['latest_audio'])
        for turn in value.get('history', []):
            if turn.get('role') == 'user' and 'text' not in turn:
                paths.append(turn['audio'])
        if any(not isinstance(p, str) or not p for p in paths):
            raise ValueError('Invalid audio reference')
        mapping = self._batch(paths, deadline) if paths else {}

        class Transcripts:
            def transcribe(self, audio, *, remaining_seconds):
                return normalize_transcript(mapping[audio])

        return prepare_sample(sample, transcriber=Transcripts(),
                              remaining=lambda: deadline - self.clock())


class Baseline50System(LifeMateSystem):
    """Ops batch interface with the frozen actor, without scaffold augmentations.

    environment_factory receives an ID-only Sample and returns a raw SDK session
    (validate/call/get_tool_calls/get_state_hash). It must create exactly one.
    """
    def __init__(self, *, catalog, environment_factory, transport, policy,
                 asr=None, runtime=RuntimeConfig(), clock=time.monotonic,
                 trace=lambda event: None, provenance=None, allowed_ids=None):
        self.catalog = deepcopy(catalog)
        self.environment_factory, self.transport, self.policy = environment_factory, transport, policy
        self.runtime, self.clock, self.trace = runtime, clock, trace
        if getattr(transport, 'context', 16384) != 16384:
            raise ValueError('Frozen baseline requires context 16384')
        self.asr = Baseline50ASR(asr, clock=clock) if asr is not None else None
        self.allowed_ids = frozenset(allowed_ids) if allowed_ids is not None else None
        self.provenance = deepcopy(provenance or {'mode': 'unverified_baseline50_bridge'})
        if runtime.output_tokens_per_step != 512:
            raise ValueError('Frozen baseline actor uses 512 output tokens')
        if policy.infer.__globals__['time'].monotonic is not clock:
            raise ValueError('Policy and system must share the same monotonic clock')

    def manifest(self):
        from dataclasses import asdict
        return {'engine': 'frozen_baseline50_actor', 'runtime': asdict(self.runtime),
                'policy': deepcopy(self.policy.identity), 'provenance': deepcopy(self.provenance),
                'bridge_sha256': _sha(__file__),
                'allowed_id_count': len(self.allowed_ids) if self.allowed_ids is not None else None,
                'asr': 'baseline50_current_then_user_history_batch' if self.asr else 'prepared_text',
                'runtime_settings_not_used_by_frozen_actor': ['max_steps'],
                'deadline_scope': 'environment_creation_ASR_actor_and_session_export; excludes startup_and_trace_artifact_writing',
                'scaffold_retrieval_grounding': False,
                'limitations': ['fixture_parity_is_not_live_model_parity', 'no_hard_preemption',
                               'caller_owns_models_transport_and_pinned_SDK',
                               'no_automatic_equivalence_of_DATA_308_and_baseline_309_splits']}

    def run(self, sample):
        from dataclasses import replace
        if not isinstance(sample.sample_id, str) or not sample.sample_id.strip():
            raise ValueError('Sample ID required')
        if self.allowed_ids is not None and sample.sample_id not in self.allowed_ids:
            raise ValueError('Sample outside configured smoke ID allowlist')
        started = self.clock()
        deadline = started + self.runtime.sample_deadline_seconds
        env = self.environment_factory(replace(sample, model_input={}))
        status, repairs, trace_rows = 'preparation_error', 0, []
        try:
            prepared = (self.asr.prepare(sample, deadline=deadline) if self.asr else
                        prepare_sample(sample, remaining=lambda: deadline - self.clock()))
            if self.clock() >= deadline:
                raise TimeoutError('Preparation exhausted deadline')
            view = prepared_view(prepared)
            allowed = permitted_names(prepared, self.catalog)

            class InstalledSession:
                # Preserve the SDK error verbatim for frozen repair-message parity.
                # Extra restriction matters only if prepared context and SDK disagree.
                def validate(self, name, arguments):
                    validation = env.validate(name, arguments)
                    if validation.get('ok') and name not in allowed:
                        return {'ok': False, 'error': 'Tool not permitted by prepared input'}
                    return validation
                def call(self, name, arguments):
                    if name not in allowed:
                        raise ValueError('Tool not permitted by prepared input')
                    return env.call(name, arguments)
                def get_tool_calls(self): return env.get_tool_calls()
                def get_state_hash(self): return env.get_state_hash()

            result = self.policy.infer(view, self.catalog, InstalledSession(), self.transport,
                seconds=deadline - self.clock(), max_calls=self.runtime.max_executed_calls,
                max_repairs=self.runtime.max_repairs)
            status, repairs, trace_rows = result['status'], result['repairs'], result['trace']
        except Exception as exc:
            status = 'timeout' if isinstance(exc, TimeoutError) else 'preparation_error'
            self.trace({'event': 'baseline50_preparation_failure', 'sample_id': sample.sample_id,
                        'error_type': type(exc).__name__})
        # Never reopen a session on failure. Export errors propagate to Ops CLI.
        calls, state_hash = env.get_tool_calls(), env.get_state_hash()
        if type(calls) is not list or not isinstance(state_hash, str) or not state_hash:
            raise ValueError('Invalid authoritative baseline session output')
        elapsed = self.clock() - started
        if elapsed > self.runtime.sample_deadline_seconds:
            status = 'timeout'
        for entry in trace_rows:
            # Omit raw response, model input and arbitrary exception text.
            self.trace({'event': 'baseline50_step', 'sample_id': sample.sample_id,
                        **{k: deepcopy(entry[k]) for k in ('step', 'decision', 'proposal', 'executed', 'observation') if k in entry},
                        'proposal_rejected': 'proposal_error' in entry})
        # Frozen baseline treats reaching the call cap as a reported completion;
        # keep the status visible, without interpreting it as task success.
        failure = status not in ('finished', 'asked_missing_info', 'call_budget_exhausted')
        return SystemResult({'id': sample.sample_id, 'tool_calls': deepcopy(calls), 'state_hash': state_hash},
            {'id': sample.sample_id, 'status': status, 'failure': failure,
             'elapsed_seconds': elapsed, 'repairs_used': repairs,
             'executed_call_count': len(calls), 'provenance': deepcopy(self.provenance)})
