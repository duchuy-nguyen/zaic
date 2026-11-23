import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from lifemate.runtime.pilot_audit import MODES, audit_archive


class PilotAuditTests(unittest.TestCase):
    def fixture(self):
        row = dict(id='val1', status='timeout', elapsed_s=31,
                   tool_calls=[dict(name='demo.find', arguments={})], results=[dict(ok=False)],
                   result_alignment_known=True, trace=[dict(repair_error='private raw text')],
                   local_score=dict(score_spec_local=0, tool_f1=.5, state_match=False))
        metric = dict(ids=['val1'], samples=1, statuses={'timeout': 1}, score_spec_local=0,
                      tool_f1=.5, state_match=0, p50_s=31, p95_s=31, max_s=31)
        files = {'export_manifest.json': dict(artifacts={'adapter.bin': hashlib.sha256(b'weights').hexdigest()}),
                 'adapter.bin': b'weights', 'config.json': dict(agent_seconds=30),
                 'split.json': dict(fit_ids=['fit1'], val_ids=['val1'], groups={'fit1':'g1', 'val1':'g2'})}
        for mode in MODES:
            files[mode+'.jsonl'] = json.dumps(row)+'\n'
            files[mode+'_metrics.json'] = copy.deepcopy(metric)
        return files

    def run_audit(self, files, extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pilot.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                for name, value in files.items():
                    data = value if isinstance(value, bytes) else (value if isinstance(value, str) else json.dumps(value)).encode()
                    member = tarfile.TarInfo('pilot/'+name); member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
                if extra is not None:
                    archive.addfile(extra, io.BytesIO(b''))
            return audit_archive(path)

    def test_valid_diagnostics_keep_failure_and_do_not_export_raw_trace(self):
        report = self.run_audit(self.fixture())
        self.assertTrue(report['structurally_valid'])
        self.assertFalse(report['deployment_ready'])
        mode = report['modes']['agent_audio']
        self.assertEqual((mode['timeout_rate'], mode['over_budget_count'], mode['failed_tool_results'], mode['repair_events']), (1, 1, 1, 1))
        self.assertNotIn('private raw text', json.dumps(report))

    def test_corrupt_weights(self):
        files = self.fixture(); files['adapter.bin'] = b'changed'
        self.assertIn('artifact_checksum_mismatch:adapter.bin', self.run_audit(files)['errors'])

    def test_missing_weights(self):
        files = self.fixture(); del files['adapter.bin']
        self.assertIn('missing_artifact:adapter.bin', self.run_audit(files)['errors'])

    def test_group_leakage(self):
        files = self.fixture(); files['split.json']['groups']['val1'] = 'g1'
        self.assertIn('fit_val_group_overlap', self.run_audit(files)['errors'])

    def test_metric_tampering(self):
        files = self.fixture(); files['agent_audio_metrics.json']['score_spec_local'] = 100
        self.assertIn('agent_audio:metric_mismatch:score_spec_local', self.run_audit(files)['errors'])

    def test_wrong_split_and_mode_coverage(self):
        files = self.fixture(); row = json.loads(files['agent_audio.jsonl']); row['id'] = 'fit1'
        files['agent_audio.jsonl'] = json.dumps(row)
        errors = self.run_audit(files)['errors']
        self.assertIn('agent_audio:outside_val', errors)
        self.assertIn('evaluation_modes_id_mismatch', errors)

    def test_duplicate_and_nan_rows_rejected(self):
        for change in ('duplicate', 'nan'):
            files = self.fixture()
            if change == 'duplicate': files['agent_audio.jsonl'] *= 2
            else:
                row = json.loads(files['agent_audio.jsonl']); row['elapsed_s'] = float('nan')
                files['agent_audio.jsonl'] = json.dumps(row)
            with self.assertRaises(ValueError): self.run_audit(files)

    def test_archive_paths_links_and_duplicates_rejected(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('pilot/link', tarfile.SYMTYPE),
                           ('pilot/config.json', tarfile.REGTYPE)]:
            member = tarfile.TarInfo(name); member.type = kind; member.linkname = '/tmp/elsewhere'
            with self.assertRaises(ValueError): self.run_audit(self.fixture(), member)

    def test_result_alignment_and_timeout_score(self):
        files = self.fixture(); row = json.loads(files['agent_audio.jsonl'])
        row['results'] = []; row['local_score']['score_spec_local'] = 50
        files['agent_audio.jsonl'] = json.dumps(row)
        errors = self.run_audit(files)['errors']
        self.assertIn('agent_audio:call_result_length_mismatch', errors)
        self.assertIn('agent_audio:nonzero_timeout_score', errors)
