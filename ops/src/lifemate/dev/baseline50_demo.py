"""Offline fixture: frozen baseline50 policy/actor through the Ops batch CLI."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

from lifemate.adapters.baseline50 import Baseline50Policy, Baseline50System
from lifemate.dev.system_demo import catalog, FixtureSession, FixtureTransport


class Transport(FixtureTransport):
    def generate(self, messages, tools, deadline, *, max_tokens=512):
        return super().generate(messages, tools, deadline, max_tokens=max_tokens)


@contextmanager
def build_system():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'retrieval_index.json').write_text('[]', encoding='utf-8')
        (root / 'split.json').write_text(json.dumps({
            'train_fit': ['fixture_001', 'fixture_002'], 'val_main': [],
            'group_ids': {'fixture_001': 'a', 'fixture_002': 'b'}}), encoding='utf-8')
        policy = Baseline50Policy(root)
        yield Baseline50System(catalog=catalog(), environment_factory=lambda _: FixtureSession(),
            transport=Transport(), policy=policy, allowed_ids=policy.fit_ids,
            provenance={'mode': 'baseline50_fictional_fixture', 'model_inference': False,
                        'environment': 'FixtureSession_not_official_ToolEnv',
                        'score_claimed': False, 'assets_verified': False})
