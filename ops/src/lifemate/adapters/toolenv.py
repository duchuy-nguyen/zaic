"""Bridge for the inspected zaic_tools Session API. Does not read state files.

The caller supplies ToolEnv from one pinned runtime per process. The committed
synthetic runtime is not proof of official train/public split compatibility.
"""
from copy import deepcopy
from pathlib import Path
from typing import Callable

from lifemate.runtime.contracts import Execution, Observation, ProposedCall, Sample, Validation


class ToolEnvAdapter:
    def __init__(self, session):
        self._session = session

    def validate(self, call: ProposedCall) -> Validation:
        result = self._session.validate(call.name, deepcopy(call.arguments))
        if not isinstance(result, dict) or type(result.get('ok')) is not bool:
            raise TypeError('Invalid ToolEnv validation response')
        return Validation(result['ok'], is_ask=call.name == 'sys.ask_missing_info',
                          code='accepted' if result['ok'] else 'toolenv_schema_error')

    def execute(self, call: ProposedCall) -> Execution:
        # A completed call() is an execution even when ok=false or done=false.
        # Any exception/invalid response propagates: Runner marks unknown and stops.
        result = deepcopy(self._session.call(call.name, deepcopy(call.arguments)))
        if (not isinstance(result, dict) or type(result.get('ok')) is not bool
                or ('done' in result and type(result['done']) is not bool)):
            raise TypeError('Invalid ToolEnv observation')
        return Execution(True, Observation(result['ok'], result.get('done', False), data=result))

    def session_output(self) -> dict:
        """Calls/hash from this same session; no task-success or submission claim."""
        return deepcopy({'tool_calls': self._session.get_tool_calls(),
                         'state_hash': self._session.get_state_hash()})


class ToolEnvFactory:
    def __init__(self, toolenv: Callable, *, root: Path, split: str):
        if split not in ('train', 'public_test', 'synthetic'):
            raise ValueError('Unsupported split; confirm the runtime contract first')
        self.toolenv, self.root, self.split = toolenv, Path(root).resolve(), split
        if not (self.root / split / f'{split}_tools_catalog.json').is_file():
            raise FileNotFoundError('A catalog for the selected split is required')

    def __call__(self, sample: Sample) -> ToolEnvAdapter:
        sid = sample.sample_id
        if not sid or sid in ('.', '..') or '/' in sid or '\\' in sid:
            raise ValueError('Invalid sample ID')
        # Constructor alone loads the environment; Ops never parses state.json.
        return ToolEnvAdapter(self.toolenv(self.split, sid, root=self.root))
