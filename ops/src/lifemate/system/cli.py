"""Prepared JSONL entrypoint. Plugins supply pinned ASR/model/ToolEnv adapters."""
import argparse
from contextlib import ExitStack
import importlib
import json
from pathlib import Path

from lifemate.runtime.configuration import strict_json
from lifemate.runtime.contracts import Sample
from lifemate.runtime.submission_check import check_submission
from lifemate.runtime.tracing import JsonlWriter, safe_json
from .pipeline import LifeMateSystem


def load_samples(path):
    samples, seen = [], set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = strict_json(line)
        if (type(row) is not dict or set(row) - {
                'id', 'context', 'history', 'latest_text', 'latest_audio'}
                or not isinstance(row.get('id'), str) or not row['id'].strip()
                or row['id'] in seen or 'context' not in row):
            raise ValueError('Expected unique prepared inference rows; raw BTC/gold schema is not accepted')
        seen.add(row['id'])
        samples.append(Sample(row['id'], {k: v for k, v in row.items() if k != 'id'}))
    if not samples:
        raise ValueError('Empty inference input')
    return samples


def run_batch(system, samples, output_dir):
    samples = list(samples)
    ids = [s.sample_id for s in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Nonempty unique sample IDs required')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'sample_ids': ids, 'configuration': system.manifest(),
                'completed': 0, 'failures': 0}

    def save_manifest():
        (root / 'run_manifest.json').write_text(json.dumps(safe_json(manifest),
            ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    save_manifest()
    previous_trace = system.trace
    try:
        with JsonlWriter(root / 'traces.jsonl') as trace, \
             JsonlWriter(root / 'diagnostics.jsonl') as diagnostics, \
             (root / 'submission.jsonl').open('x', encoding='utf-8') as output:
            system.trace = lambda e: (previous_trace(e), trace(e))
            for sample in samples:
                result = system.run(sample)
                # No redaction/reconstruction of executed calls in submission.
                output.write(json.dumps(result.submission, ensure_ascii=False, allow_nan=False) + '\n')
                output.flush()
                diagnostics(result.diagnostics)
                manifest['completed'] += 1
                manifest['failures'] += int(result.diagnostics['failure'])
                save_manifest()
        report = check_submission(root / 'submission.jsonl', set(ids))
        (root / 'submission_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if not report['valid']:
            raise ValueError('Submission structural validation failed')
        manifest['status'] = 'completed_with_failures' if manifest['failures'] else 'completed'
    except Exception as exc:
        manifest['status'], manifest['error_type'] = 'failed', type(exc).__name__
        raise
    finally:
        system.trace = previous_trace
        save_manifest()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory', required=True, help='Trusted module:factory returning LifeMateSystem')
    parser.add_argument('--input', required=True, type=Path, help='Prepared text/audio JSONL, not raw BTC rows')
    parser.add_argument('--output-dir', required=True, type=Path, help='New private directory; never overwritten')
    parser.add_argument('--max-samples', type=int, help='Optional smoke cap; reject larger input before loading models')
    args = parser.parse_args(argv)
    module, name = args.factory.split(':', 1)
    samples = load_samples(args.input)
    if args.max_samples is not None and (args.max_samples <= 0 or len(samples) > args.max_samples):
        raise ValueError('Input exceeds requested smoke sample cap')
    if args.output_dir.exists():
        raise FileExistsError('Output directory already exists')
    with ExitStack() as stack:
        system = getattr(importlib.import_module(module), name)()
        if hasattr(system, '__enter__') and hasattr(system, '__exit__'):
            system = stack.enter_context(system)
        if not isinstance(system, LifeMateSystem):
            raise TypeError('Factory must return LifeMateSystem or a context manager yielding one')
        manifest = run_batch(system, samples, args.output_dir)
    print(json.dumps({k: manifest[k] for k in ('status', 'completed', 'failures')}))
    return 1 if manifest['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
