"""Durable prefix journal: never automatically retry an uncommitted session."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .configuration import strict_json
from .submission_check import check_submission
from .tracing import safe_json


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(encoded(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(path):
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_resumable(factory, samples, expected_ids, output_dir, identity, *, resume=False, clock=time.monotonic):
    samples = list(samples); ids = [s.sample_id for s in samples]
    if not ids or len(ids) != len(set(ids)) or ids != list(expected_ids):
        raise ValueError('Input must exactly cover independent ordered partition IDs')
    frozen = {'ids': ids, 'input_sha256': hashlib.sha256(encoded([
        {'id':s.sample_id, 'input':s.model_input} for s in samples]).encode()).hexdigest(),
        'identity': identity}
    root = Path(output_dir)
    if root.exists() and not resume:
        raise FileExistsError('Existing run requires explicit resume')
    if not root.exists() and resume:
        raise FileNotFoundError('Cannot resume absent run')
    root.mkdir(parents=True, exist_ok=resume)
    with lock(root/'run.lock'):
        marker = root/'run_manifest.json'
        if marker.exists():
            state = strict_json(marker.read_text(encoding='utf-8'))
            if state['configuration'] != frozen:
                raise ValueError('Code/assets/input/config changed; preserve existing run')
        elif resume:
            raise ValueError('Missing run manifest; cannot safely resume')
        else:
            state = {'configuration':frozen, 'status':'preparing', 'completed':0}
            atomic(marker,state)
        records_dir = root/'records'; records_dir.mkdir(exist_ok=True)
        paths = sorted(records_dir.glob('*.json'))
        records = {}
        for path in paths:
            record = strict_json(path.read_text(encoding='utf-8'))
            position = record['ordinal']
            if type(position) is not int or not 0 <= position < len(ids) or path.name != f'{position:06d}.json':
                raise ValueError('Invalid checkpoint ordinal')
            if record['submission']['id'] != ids[position]:
                raise ValueError('Checkpoint ID mismatch')
            if record.get('record_sha256') != hashlib.sha256(encoded({k:v for k,v in record.items() if k!='record_sha256'}).encode()).hexdigest():
                raise ValueError('Corrupt checkpoint')
            records[position]=record
        if set(records) != set(range(len(records))):
            raise ValueError('Checkpoint must be a committed prefix')
        inflight = root/'inflight.json'
        if inflight.exists():
            pending = strict_json(inflight.read_text(encoding='utf-8'))
            ordinal = pending['ordinal']
            if type(ordinal) is not int or ordinal not in records or pending['id'] != ids[ordinal]:
                raise ValueError('Uncommitted sample: preserve artifacts; automatic retry is forbidden')
            inflight.unlink()  # committed before crash: never execute it again
        if len(records) < len(samples):
            try:
                with factory() as system:
                    for ordinal in range(len(records),len(samples)):
                        sample = samples[ordinal]
                        atomic(inflight,{'id':sample.sample_id,'ordinal':ordinal})
                        started=clock()
                        result=system.run(sample)
                        if result.submission['id'] != sample.sample_id:
                            raise ValueError('System returned wrong ID')
                        record={'ordinal':ordinal, 'submission':result.submission,
                                'diagnostics':safe_json(result.diagnostics),
                                'ops_run_wall_s':clock()-started}
                        record['record_sha256']=hashlib.sha256(encoded(record).encode()).hexdigest()
                        atomic(records_dir/f'{ordinal:06d}.json',record)
                        records[ordinal]=record; inflight.unlink()
                        state.update(status='running',completed=len(records)); atomic(marker,state)
            except BaseException as error:
                state.update(status='interrupted',error_type=type(error).__name__,completed=len(records))
                atomic(marker,state)
                raise
        # Finalization can be resumed without loading any model or opening SDK sessions.
        for filename,key in [('submission.jsonl','submission'),('diagnostics.jsonl','diagnostics')]:
            payload=''.join(encoded(records[i][key])+'\n' for i in range(len(ids)))
            target=root/filename
            if target.exists():
                if target.read_text(encoding='utf-8') != payload:
                    raise ValueError('Preserve conflicting final output')
            else:
                temporary=root/(filename+'.'+uuid.uuid4().hex+'.tmp')
                with temporary.open('x',encoding='utf-8') as stream:
                    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary,target)
        report=check_submission(root/'submission.jsonl',set(ids))
        atomic(root/'submission_check.json',report)
        if not report['valid']:
            raise ValueError('Invalid submission export')
        state.update(status='completed',completed=len(records),
            failures=sum(bool(r['diagnostics']['failure']) for r in records.values()),
            timing_scope='ops_run_wall_s wraps system.run including ASR/session/export; excludes startup and journal/output disk writes; not BTC timing certification')
        atomic(marker,state)
        return state
