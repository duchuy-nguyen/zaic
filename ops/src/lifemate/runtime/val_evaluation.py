"""Evaluate committed predictions separately from inference using frozen references."""
from collections import Counter
import math
import statistics

SUCCESS = {'finished','asked_missing_info','call_budget_exhausted'}


def unique(rows):
    indexed={}
    for row in rows:
        sid=row['id']
        if not isinstance(sid,str) or not sid or sid in indexed:
            raise ValueError('Invalid or duplicate evaluation ID')
        indexed[sid]=row
    return indexed


def evaluate_val(predictions,references,tools,expected_ids,diagnostics=None):
    from baseline50.scoring import score
    ids=list(expected_ids)
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=references.keys():
        raise ValueError('Invalid reference coverage')
    predicted=unique(predictions);diags=unique(diagnostics or [])
    if set(predicted)-set(ids) or set(diags)-set(ids):
        raise ValueError('Unknown evaluation IDs')
    scores={};durations=[];statuses=Counter();missing=0;failed=0;over=0
    for sid in ids:
        row=predicted.get(sid)
        if row is None:
            missing+=1;failed+=1;statuses['missing']+=1
            scores[sid]=score({'tool_calls':[],'state_hash':''},references[sid],tools,failed=True)
            continue
        if not isinstance(row['tool_calls'],list) or not isinstance(row['state_hash'],str) or not row['state_hash']:
            raise ValueError('Invalid prediction fields')
        metadata=diags.get(sid,row)
        status=metadata['status'];duration=metadata.get('elapsed_seconds',metadata.get('elapsed_s'))
        if (not isinstance(status,str) or type(duration) not in (int,float)
                or not math.isfinite(duration) or duration<0):
            raise ValueError('Execution status and finite latency required')
        if 'failure' in metadata and type(metadata['failure']) is not bool:
            raise ValueError('Invalid orchestration failure flag')
        is_failed=status not in SUCCESS or duration>30 or metadata.get('failure',False)
        failed+=bool(is_failed);over+=duration>30;statuses[status]+=1;durations.append(duration)
        scores[sid]=score(row,references[sid],tools,failed=is_failed)
    ordered=sorted(durations)
    return {'scorer':'local-spec-v1','scope':'Frozen baseline50 VAL; local approximation, not leaderboard; historical development split.',
            'n':len(ids),'predicted':len(predicted),'missing':missing,'complete':not missing,
            'failed_samples':failed,'over_30s':over,'statuses':dict(statuses),
            'score_100':statistics.mean(v['score_spec_local'] for v in scores.values()),
            'tool_f1_percent':100*statistics.mean(v['tool_f1'] for v in scores.values()),
            'state_match_percent':100*statistics.mean(v['state_match'] for v in scores.values()),
            'p50_s':statistics.median(durations) if durations else None,
            'p95_s':ordered[math.ceil(.95*len(ordered))-1] if ordered else None,
            'max_s':max(durations) if durations else None,'latency_samples':len(durations),
            'timing_scope':'Supplied execution latency; excludes model startup and may exclude output I/O. Not BTC end-to-end certification.',
            'model_inference_run_by_evaluator':False,'scores':scores}
