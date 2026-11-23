"""BTC Docker I/O contract (guide received 2026-11-11), no model imports."""
import json
from pathlib import Path

from .configuration import strict_json
from .smoke_inputs import clean_row, inside


def split_root(data_root, split=None):
    root=Path(data_root).resolve()
    names=[p.name for p in root.iterdir() if p.is_dir() and
           (p/(p.name+'.jsonl')).is_file() and (p/(p.name+'_tools_catalog.json')).is_file()]
    if split is None:
        if len(names)!=1:raise ValueError('Expected exactly one mounted split; specify split for local smoke')
        split=names[0]
    if split not in names or '/' in split or '\\' in split or split in ('.','..'):
        raise ValueError('Invalid mounted split')
    directory=(root/split).resolve()
    if not directory.is_relative_to(root):raise ValueError('Split escapes data root')
    return root,split,directory


def input_rows(directory,split):
    seen=set()
    with (directory/(split+'.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            row=strict_json(line)
            sid=row['id']
            if (not isinstance(sid,str) or not sid or sid in seen
                    or (split!='train' and set(row)!={'id','audio','context','history'})):
                raise ValueError('Invalid/duplicate ID or unexpected test fields')
            seen.add(sid)
            if split!='train' and any('audio_text' in t for t in row['history'] if t['role']=='user'):
                raise ValueError('Gold transcript in test input')
            clean=clean_row(row)
            for ref in [clean['latest_audio']]+[t['audio'] for t in clean['history'] if t['role']=='user']:
                inside(directory,ref)
            yield clean
    if not seen:raise ValueError('Empty mounted dataset')


def timed_submission(submission,milliseconds):
    if type(milliseconds) is not int or milliseconds<0:
        raise ValueError('time must be nonnegative integer milliseconds')
    if set(submission)!={'id','tool_calls','state_hash'}:
        raise ValueError('Expected authoritative three-field session output')
    if not isinstance(submission['id'],str) or not submission['id'] or not isinstance(submission['state_hash'],str) or not submission['state_hash']:
        raise ValueError('Missing ID/hash')
    if not isinstance(submission['tool_calls'],list):raise ValueError('Invalid call log')
    for call in submission['tool_calls']:
        if set(call)!={'name','arguments'} or not isinstance(call['name'],str) or not call['name'] or not isinstance(call['arguments'],dict):
            raise ValueError('Invalid call entry')
    return {**submission,'time':milliseconds}


def check_timed_output(path,expected_ids):
    expected=list(expected_ids)
    if not expected or len(expected)!=len(set(expected)):raise ValueError('Invalid independent expected IDs')
    actual=[]
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row=strict_json(line)
            if set(row)!={'id','tool_calls','state_hash','time'}:raise ValueError('Docker output requires time')
            timed_submission({k:row[k] for k in ('id','tool_calls','state_hash')},row['time'])
            actual.append(row['id'])
    if actual!=expected:raise ValueError('Output ID coverage/order mismatch')
    return {'valid':True,'rows':len(actual),'scope':'Docker structure and coverage, not model quality'}
