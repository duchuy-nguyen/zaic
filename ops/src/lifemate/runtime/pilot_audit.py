"""Read-only pilot artifact audit. Never imports archived code or loads checkpoints."""
import hashlib
import json
import math
from collections import Counter
from pathlib import PurePosixPath
import statistics
import tarfile


MODES = ('agent_base_text', 'agent_tuned_text', 'agent_audio')


def load_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('nonfinite JSON')

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def number(value, low=0, high=float('inf')):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('invalid numeric field')
    return value


def ids(values):
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError('invalid ID list')
    if len(set(values)) != len(values):
        raise ValueError('duplicate IDs')
    return set(values)


def audit_archive(path):
    errors = []
    summaries = {}
    with tarfile.open(path, 'r:gz') as archive:
        members = {}
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts or '..' in parts or PurePosixPath(member.name).is_absolute():
                raise ValueError('unsafe archive path')
            if member.isdir():
                continue
            if not member.isfile() or member.name in members:
                raise ValueError('link, special file or duplicate archive member')
            members[member.name] = member
        manifests = [n for n in members if n.endswith('/export_manifest.json')]
        if len(manifests) != 1:
            raise ValueError('expected exactly one pilot export manifest')
        root = manifests[0].rsplit('/', 1)[0] + '/'

        def read(name):
            member = members[root + name]
            if member.size > 16 * 1024 * 1024:
                raise ValueError('metadata size limit exceeded')
            with archive.extractfile(member) as stream:
                return stream.read().decode('utf-8')

        def obj(name):
            return load_json(read(name))

        manifest = obj('export_manifest.json')
        artifacts = manifest['artifacts']
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError('empty artifact manifest')
        verified = 0
        for name, expected in artifacts.items():
            parts = PurePosixPath(name).parts
            if not parts or '..' in parts or PurePosixPath(name).is_absolute():
                raise ValueError('unsafe artifact path')
            member = members.get(root + name)
            if member is None:
                errors.append('missing_artifact:' + name)
                continue
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                errors.append('artifact_checksum_mismatch:' + name)
            else:
                verified += 1
        split = obj('split.json')
        fit, val = ids(split['fit_ids']), ids(split['val_ids'])
        if fit & val:
            errors.append('fit_val_id_overlap')
        groups = split['groups']
        if not isinstance(groups, dict) or not (fit | val) <= groups.keys():
            raise ValueError('missing split groups')
        if {groups[x] for x in fit} & {groups[x] for x in val}:
            errors.append('fit_val_group_overlap')
        budget = number(obj('config.json')['agent_seconds'], low=0.001)
        mode_ids = []
        for mode in MODES:
            rows = [load_json(line) for line in read(mode + '.jsonl').splitlines() if line.strip()]
            observed = ids([row['id'] for row in rows])
            mode_ids.append(observed)
            if not observed <= val:
                errors.append(mode + ':outside_val')
            latencies, scores, f1s, matches = [], [], [], []
            statuses, repair_types = Counter(), Counter()
            failures, unknown_alignment = 0, 0
            for row in rows:
                if row['status'] not in ('finished', 'ask', 'call_budget', 'step_budget', 'timeout',
                                         'structured_output_error', 'model_error', 'execution_unknown'):
                    raise ValueError('invalid status')
                statuses[row['status']] += 1
                latencies.append(number(row['elapsed_s']))
                calls, results = row['tool_calls'], row['results']
                if not isinstance(calls, list) or not isinstance(results, list):
                    raise ValueError('invalid calls/results')
                for call in calls:
                    if not isinstance(call['name'], str) or not isinstance(call['arguments'], dict):
                        raise ValueError('invalid tool call')
                aligned = row['result_alignment_known']
                if type(aligned) is not bool:
                    raise ValueError('invalid alignment flag')
                unknown_alignment += not aligned
                if aligned and len(calls) != len(results):
                    errors.append(mode + ':call_result_length_mismatch')
                failures += sum(isinstance(result, dict) and result.get('ok') is False for result in results)
                if not isinstance(row['trace'], list) or any(not isinstance(x, dict) for x in row['trace']):
                    raise ValueError('invalid trace')
                for event in row['trace']:
                    if 'repair_error' in event:
                        # Counts only: do not copy raw model/error text into the report.
                        repair_types['repair_events'] += 1
                score = row['local_score']
                scores.append(number(score['score_spec_local'], high=100))
                f1s.append(number(score['tool_f1'], high=1))
                if type(score['state_match']) is not bool:
                    raise ValueError('invalid state_match')
                matches.append(score['state_match'])
                if row['status'] == 'timeout' and scores[-1] != 0:
                    errors.append(mode + ':nonzero_timeout_score')
            ordered = sorted(latencies)
            summary = dict(samples=len(rows), statuses=dict(statuses),
                           timeout_rate=statuses['timeout']/len(rows),
                           over_budget_count=sum(x > budget for x in latencies),
                           p50_s=statistics.median(latencies),
                           p95_s=ordered[math.ceil(.95*len(rows))-1], max_s=max(latencies),
                           score_spec_local=statistics.mean(scores), tool_f1=statistics.mean(f1s),
                           state_match=statistics.mean(matches), failed_tool_results=failures,
                           unknown_alignment_count=unknown_alignment, repair_events=repair_types['repair_events'])
            reported = obj(mode + '_metrics.json')
            if ids(reported['ids']) != observed:
                errors.append(mode + ':metric_ids_mismatch')
            for key in ('samples', 'statuses', 'score_spec_local', 'tool_f1', 'state_match', 'p50_s', 'p95_s', 'max_s'):
                actual, expected = summary[key], reported[key]
                equal = (math.isclose(number(expected), actual, rel_tol=1e-9, abs_tol=1e-9)
                         if type(actual) in (int, float) else actual == expected)
                if not equal:
                    errors.append(mode + ':metric_mismatch:' + key)
            summaries[mode] = summary
        if any(value != mode_ids[0] for value in mode_ids[1:]):
            errors.append('evaluation_modes_id_mismatch')
    return dict(structurally_valid=not errors, errors=sorted(set(errors)),
                artifact_count=len(artifacts), verified_artifact_count=verified,
                fit_count=len(fit), val_count=len(val), agent_budget_s=budget, modes=summaries,
                deployment_ready=False,
                scope='Offline integrity and rollout diagnostics; scores aggregated from supplied local_score, not independently scored against official gold.',
                limitations=['No model inference, tensor validation or official replay.',
                             'ID/group split check does not establish semantic/audio leakage absence.',
                             'Archive manifest is self-supplied; matching hashes do not establish trusted provenance.'])
