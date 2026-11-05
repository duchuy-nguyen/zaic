"""TRAIN-only ASR evaluation; identity join (sample_id, kind, turn_index).

Default denominator is ALL train references. For smoke explicitly request
--scope transcript_samples: all current/history of selected samples remain in
denominator, including missing/error hypotheses. No test data is evaluated.
"""
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lifemate_speech.io_utils import atomic_json, audio_sha256, ensure_work_path, ref_key
from lifemate_speech.metrics import aggregate, percentiles, score_utterance
from lifemate_speech.entities import (CATEGORIES, ENTITY_RULES_VERSION, MONEY_RELAXED_VERSION,
    RELATIVE_TIME_RULES_VERSION, TIME_RULES_VERSION, RULES, RELATIVE_PHRASES, entity_comparison,
    extract_entities, parser_uncertainties, relaxed_money_comparison, relaxed_time_comparison, relative_time_comparison)
from lifemate_speech.normalize import NORMALIZATION_VERSION, normalize


def load_references(data_root):
    refs = {}
    with (data_root/'train'/'train.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            for index, turn in [(None,row)]+[(i,t) for i,t in enumerate(row.get('history',[])) if t.get('role')=='user']:
                if not isinstance(turn.get('audio_text'),str):
                    raise ValueError(f'Missing gold text: {row["id"]}, turn {index}')
                ref = dict(sample_id=row['id'], kind='current' if index is None else 'history', turn_index=index,
                           audio_path=turn.get('audio'), reference=turn['audio_text'])
                key = ref_key(ref)
                if key in refs:
                    raise ValueError(f'Duplicate reference identity: {key}')
                refs[key] = ref
    return refs


def write_jsonl(path, rows):
    with path.open('w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def status_summary(values):
    counts = Counter(v['status'] for v in values)
    present = sum(v['status']!='not_present' for v in values)
    return dict(sentences_with_entity=present, exact_match_count=counts['correct'],
                exact_match_rate=counts['correct']/present if present else None,
                incorrect_count=counts['incorrect'], unparseable_count=counts['unparseable'],
                absent_count=counts['not_present'],
                false_positive_only_count=sum(v.get('false_positive_only',False) for v in values))


def timing_summary(rows):
    valid = lambda v: isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row['sample_id']].append(row)
    totals, incomplete = [], []
    for sid, group in by_sample.items():
        if all(valid(r['asr_latency_s']) for r in group):
            totals.append(dict(sample_id=sid, allocated_asr_latency_s=sum(r['asr_latency_s'] for r in group),
                               from_cache_count=sum(r['from_cache'] for r in group), audio_count=len(group)))
        else:
            incomplete.append(sid)
    modes = Counter('missing' if r['hypothesis_status']=='missing' else ('resumed' if r['resumed'] else ('cache' if r['from_cache'] else 'fresh')) for r in rows)
    fresh = [r['asr_latency_s'] for r in rows if not r['from_cache'] and not r['resumed'] and valid(r['asr_latency_s'])]
    return dict(mode_counts=dict(modes), per_file_s=percentiles([r['asr_latency_s'] for r in rows if valid(r['asr_latency_s'])]),
                fresh_per_file_s=percentiles(fresh), missing_or_invalid_latency_count=sum(not valid(r['asr_latency_s']) for r in rows),
                sample_total_allocated_asr_s=percentiles([r['allocated_asr_latency_s'] for r in totals]),
                samples_over_30s_allocated_asr=sum(r['allocated_asr_latency_s']>30 for r in totals),
                complete_sample_count=len(totals), incomplete_sample_ids=incomplete,
                sample_totals=totals,
                caveat='Equal batch-wall allocation, not sample wall time. Cache/resume is not live timing. CPU cannot predict competition latency; model load and pipeline overhead excluded.')


def evaluate(data_root, transcript_paths, out, scope='all_train', sample_ids=None, examples_per_type=10):
    out = ensure_work_path(out)
    data_root = data_root.resolve()
    if out.is_relative_to(data_root) or data_root.is_relative_to(out):
        raise ValueError('Output must not overlap DATA_ROOT')
    references = load_references(data_root)
    predictions, source_manifests, selected = {}, [], set(sample_ids or [])
    for path in transcript_paths:
        with path.open(encoding='utf-8-sig') as stream:
            for line_no,line in enumerate(stream,1):
                hyp = json.loads(line)
                if hyp.get('split')!='train':
                    raise ValueError(f'TRAIN only: {path}:{line_no}')
                key = ref_key(hyp)
                if key not in references:
                    raise ValueError(f'Unknown train reference identity: {key}')
                if key in predictions:
                    raise ValueError(f'Duplicate hypothesis identity: {key}')
                predictions[key] = hyp
                selected.add(key[0])
        manifest_path = path.parent/'run_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            source_manifests.append(dict(path=str(manifest_path), run_id=manifest.get('run_id'),
                                         device=manifest.get('device'), cache_mode=manifest.get('cache_mode'),
                                         total_wall_time_s=manifest.get('total_wall_time_s')))
            if scope=='transcript_samples':
                selected.update(manifest.get('selected_sample_ids',[]))
    all_ids = {k[0] for k in references}
    if selected-all_ids:
        raise ValueError(f'Unknown selected train IDs: {selected-all_ids}')
    if sample_ids:
        if set(k[0] for k in predictions)-set(sample_ids):
            raise ValueError('Predictions outside explicit --sample-ids')
        selected = set(sample_ids)
    if scope=='all_train' and not sample_ids:
        selected = all_ids
    if not selected:
        raise ValueError('No samples selected for evaluation')
    refs = {k:r for k,r in references.items() if k[0] in selected}
    rows, errors = [], []
    for key, reference in refs.items():
        hyp = predictions.get(key)
        status = 'missing' if hyp is None else ('error' if hyp.get('error') is not None or not isinstance(hyp.get('text_raw'),str) else 'ok')
        text = hyp['text_raw'] if status=='ok' else ''
        row = dict(reference, **score_utterance(reference['reference'],text))
        row.update(hypothesis_status=status, has_hypothesis=status=='ok',
                   error=None if status=='ok' else (hyp.get('error') or 'Invalid text_raw' if hyp else 'Missing hypothesis'),
                   from_cache=bool(hyp.get('from_cache',False)) if hyp else False,
                   resumed=bool(hyp.get('resumed',False)) if hyp else False,
                   asr_latency_s=hyp.get('asr_latency_s') if hyp else None,
                   batch_id=hyp.get('batch_id') if hyp else None,
                   batch_size=hyp.get('batch_size') if hyp else None,
                   batch_wall_time_s=hyp.get('batch_wall_time_s') if hyp else None,
                   normalization_version=NORMALIZATION_VERSION,
                   entity_errors=entity_comparison(reference['reference'],text),
                   money_relaxed=relaxed_money_comparison(reference['reference'],text),
                   time_relaxed=relaxed_time_comparison(reference['reference'],text),
                   parser_uncertainties=dict(reference=parser_uncertainties(reference['reference']),hypothesis=parser_uncertainties(text)),
                   relative_time_errors=relative_time_comparison(reference['reference'],text))
        rows.append(row)
        if status!='ok':
            errors.append({k:row[k] for k in ('sample_id','kind','turn_index','audio_path','hypothesis_status','error')})
    metrics = dict(scope=scope if not sample_ids else 'explicit_samples', selected_sample_ids=sorted(selected),
                   counts=dict(total_audio=len(rows), hypothesis_records=len(predictions),
                               with_hypothesis=sum(r['has_hypothesis'] for r in rows),
                               errors=len(errors), explicit_errors=sum(r['hypothesis_status']=='error' for r in rows),
                               missing=sum(r['hypothesis_status']=='missing' for r in rows)),
                   normalization_version=NORMALIZATION_VERSION, entity_rules_version=ENTITY_RULES_VERSION,
                   money_relaxed_version=MONEY_RELAXED_VERSION, relative_time_rules_version=RELATIVE_TIME_RULES_VERSION,
                   time_rules_version=TIME_RULES_VERSION,
                   wer_cer={}, entity_metrics={}, money_relaxed={}, time_relaxed={}, relative_time={}, latency=timing_summary(rows),
                   sources=source_manifests,
                   conventions='WER case/punctuation sensitive whitespace tokens; CER includes spaces. Missing/error hypothesis empty, included as deletions. Strict entity multiset spans; weak regex labels, not annotated truth.')
    for group in ('all','current','history'):
        subset = rows if group=='all' else [r for r in rows if r['kind']==group]
        metrics['wer_cer'][group] = aggregate(subset)
        metrics['entity_metrics'][group] = {k:status_summary([r['entity_errors'][k] for r in subset]) for k in CATEGORIES}
        metrics['money_relaxed'][group] = status_summary([r['money_relaxed'] for r in subset])
        metrics['money_relaxed'][group]['fraction_presence_mismatch_count']=sum(r['money_relaxed']['fraction_presence_mismatch'] for r in subset if r['money_relaxed']['status']!='not_present')
        metrics['time_relaxed'][group]=status_summary([r['time_relaxed'] for r in subset])
        metrics['relative_time'][group] = status_summary([r['relative_time_errors'] for r in subset])
    per_phrase = {}
    for phrase in RELATIVE_PHRASES:
        relevant = [r for r in rows if phrase in [s['canonical'] for s in r['relative_time_errors']['reference_spans']]]
        if relevant:
            per_phrase[phrase] = dict(sentences=len(relevant),
                sentences_missing_or_wrong_count=sum(
                    sum(s['canonical']==phrase for s in r['relative_time_errors']['reference_spans']) !=
                    sum(s['canonical']==phrase for s in r['relative_time_errors']['hypothesis_spans']) for r in relevant))
    metrics['relative_time']['per_reference_phrase'] = per_phrase
    # Rule review draws reference-only TRAIN examples, separately from smoke metrics.
    # Never uses hypotheses or public data to choose/modify patterns.
    examples = {k:[] for k in (*CATEGORIES,'relative_time')}
    for reference in references.values():
        spans = extract_entities(reference['reference'])
        spans['relative_time'] = relative_time_comparison(reference['reference'],'')['reference_spans']
        for kind in examples:
            if spans[kind] and len(examples[kind])<examples_per_type:
                examples[kind].append(dict(reference, spans=spans[kind]))
    rules = dict(weak_labels=True, entity_rules_version=ENTITY_RULES_VERSION,
                 money_relaxed_version=MONEY_RELAXED_VERSION, relative_time_rules_version=RELATIVE_TIME_RULES_VERSION,
                 time_rules_version=TIME_RULES_VERSION,
                 patterns={k:v.pattern for k,v in RULES.items()}, relative_phrases=list(RELATIVE_PHRASES),
                 examples_source='reference-only train, first matching utterances, not used in smoke score denominator',
                 limitations='Weak labels: multi-digit unsuffixed money tails/compound-scale rưỡi not guessed; fraction-presence guard can reject equivalent unscaled VND. No AM/PM/context inference, absent month/year remain unknown; bare Mai ambiguous. Unknown English/lowercase names can be missed and addresses overcaptured. No transcript edits.', examples=examples)
    worst = sorted(rows,key=lambda r:r['wer'],reverse=True)[:100]
    out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'per_utterance.jsonl',rows)
    write_jsonl(out/'errors.jsonl',errors)
    write_jsonl(out/'parser_uncertainties.jsonl',[{k:r[k] for k in ('sample_id','kind','turn_index','reference_norm','hypothesis_norm','parser_uncertainties')}
                for r in rows if r['parser_uncertainties']['reference'] or r['parser_uncertainties']['hypothesis']])
    write_jsonl(out/'worst_samples.jsonl',[dict(sample_id=r['sample_id'],kind=r['kind'],turn_index=r['turn_index'],
                audio_path=r['audio_path'],reference=r['reference_norm'],hypothesis=r['hypothesis_norm'],wer=r['wer'],
                incorrect_entity_types=[k for k,v in r['entity_errors'].items() if v['status']=='incorrect'],
                hypothesis_status=r['hypothesis_status']) for r in worst])
    atomic_json(out/'metrics.json',metrics)
    atomic_json(out/'entity_rules_and_examples.json',rules)
    atomic_json(out/'evaluation_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),argv=list(sys.argv),
                executable=sys.executable,transcripts=[dict(path=str(p.resolve()),sha256=audio_sha256(p)) for p in transcript_paths],
                reference_sha256=audio_sha256(data_root/'train'/'train.jsonl'),
                normalization_version=NORMALIZATION_VERSION,entity_rules_version=ENTITY_RULES_VERSION,
                scope=metrics['scope'],selected_sample_ids=sorted(selected)))
    report = ['# ASR evaluation — TRAIN smoke', '',
              f"Scope: {metrics['scope']}; {len(selected)} samples, {len(rows)} audio. Valid hypotheses {metrics['counts']['with_hypothesis']}, explicit errors {metrics['counts']['explicit_errors']}, missing {metrics['counts']['missing']}.",
              'Missing/error hypotheses are empty and counted as deletions. These are smoke results, not full-train estimates.', '',
              '| Group | N | WER micro | WER macro | CER micro | CER macro |','|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: f'{x:.4%}' if x is not None else 'N/A'
    for kind,s in metrics['wer_cer'].items():
        report.append(f"| {kind} | {s['n_utterances']} | {fmt(s['wer_micro'])} | {fmt(s['wer_macro'])} | {fmt(s['cer_micro'])} | {fmt(s['cer_macro'])} |")
    report += ['', 'WER retains case/punctuation; CER includes spaces. Whitespace tokens are not Vietnamese linguistic word segmentation.', '',
               f'## Weak entities ({ENTITY_RULES_VERSION})', '', '| Category | Sentences | Exact | Rate |','|---|---:|---:|---:|']
    for kind,s in metrics['entity_metrics']['all'].items():
        report.append(f"| {kind} | {s['sentences_with_entity']} | {s['exact_match_count']} | {fmt(s['exact_match_rate'])} |")
    report += ['', f'## Separate money relaxation ({MONEY_RELAXED_VERSION})', '',
               json.dumps(metrics['money_relaxed']['all'],ensure_ascii=False),
               'Conservative VND-value comparison only. Does not alter hypotheses, normalization or WER/CER.', '',
               'Fraction-presence mismatch is incorrect even if a truncated/base value would agree. Full fractional spans retained; unsupported forms unparseable, never silently clipped.', '',
               f'## Separate clock-value diagnostic ({TIME_RULES_VERSION})', '',
               json.dumps(metrics['time_relaxed']['all'],ensure_ascii=False),
               'kém subtracts minutes; rưỡi adds 30 minutes. Missing AM/PM is flagged; no context/calendar inference. Strict entity score unchanged.', '',
               f'## Relative-time phrases ({RELATIVE_TIME_RULES_VERSION})', '',
               json.dumps(metrics['relative_time']['all'],ensure_ascii=False),
               'Case-insensitive phrase multiplicity comparison, no calendar resolution. See metrics.json per_reference_phrase.', '',
               '## Timing', '',json.dumps({k:v for k,v in metrics['latency'].items() if k not in ('sample_totals','incomplete_sample_ids')},ensure_ascii=False), '',
               '## 10 error examples (or all available)', '']
    failures = [r for r in worst if r['wer']>0 or r['hypothesis_status']!='ok'][:10]
    for r in failures:
        report += [f"- {r['sample_id']} / {r['kind']} / {r['turn_index']}: WER={r['wer']:.4f}",
                   f"  Reference: {r['reference_norm']}",f"  Hypothesis: {r['hypothesis_norm']}"]
    report += ['', '## Risks / limitations', '',
               '- All entity labels are weak regex annotations. Name casing, overlapping categories and address span boundaries may inflate diagnostic errors.',
               '- Missing categories/rare slices have unknown accuracy, not 100%. Review up to 10 examples/type in entity_examples.md.',
               '- CPU timing and cached/resumed rows cannot certify real end-to-end latency; sample sums are batch allocations.',
               '- No public_test tuning, model changes, normalization changes or transcript repair performed.', '']
    (out/'asr_report.md').write_text('\n'.join(report),encoding='utf-8')
    review = ['# Weak-label rule review', '',f'ENTITY_RULES_VERSION={ENTITY_RULES_VERSION}. TRAIN reference-only; separate from smoke metrics.', '']
    for kind, items in examples.items():
        review += [f'## {kind}: {len(items)}/{examples_per_type} examples', '']
        for item in items:
            review += [f"- {item['sample_id']} / {item['kind']} / {item['turn_index']}",
                       f"  Reference: {item['reference']}",
                       '  Extracted: '+json.dumps(item['spans'],ensure_ascii=False)]
        review.append('')
    (out/'entity_examples.md').write_text('\n'.join(review),encoding='utf-8')
    review_uncertainties=[]
    seen=set()
    for items in examples.values():
        for item in items:
            key=ref_key(item)
            if key in seen:
                continue
            seen.add(key)
            issues=parser_uncertainties(item['reference'])
            if issues:
                review_uncertainties.append(dict(item,uncertainties=issues))
    write_jsonl(out/'rule_example_uncertainties.jsonl',review_uncertainties)
    uncertainty_report=['# Parser uncertainties — evaluation only','',
        'No new inference. Only existing smoke hypotheses and TRAIN reference rule-review examples.', '',
        '- Money: one trailing digit means tenths of immediately preceding scale; multi-digit tail without scale is ambiguous and rejected.',
        '- Compound-scale rưỡi is unparseable; no guess about which amount is halved.',
        '- Fraction-presence mismatch is incorrect by policy, including explicit zero tails or equivalent unscaled VND spellings; check whether this conservative rule is desired before full evaluation.',
        '- Time: missing AM/PM, unusual noon/night hours, or midnight kém crossing day boundary are flagged.',
        '- Date: missing month/year or ambiguous two-digit century stay unknown. A following lần is excluded; other non-calendar uses may remain false positives.',
        '- Mai after titles (including cô giáo/chị gái/anh trai/bác sĩ) is excluded as name; bare Mai remains uncertain, even sentence-initial. Compound names without titles may be false positives.',
        '- Name casing, phone word variants, unknown English lexicon and address boundaries retain earlier weak-label limits.', '',
        '## Smoke uncertainty examples','']
    for r in rows:
        for side,issues in r['parser_uncertainties'].items():
            if issues:
                uncertainty_report += [f"- {r['sample_id']} / {r['kind']} / {r['turn_index']} / {side}",
                    '  '+json.dumps(issues,ensure_ascii=False)]
    uncertainty_report += ['','## Uncertain TRAIN rule-review examples','']
    for item in review_uncertainties:
        uncertainty_report += [f"- {item['sample_id']} / {item['kind']} / {item['turn_index']}: {item['reference']}",
                                '  '+json.dumps(item['uncertainties'],ensure_ascii=False)]
    (out/'parser_uncertainties.md').write_text('\n'.join(uncertainty_report)+'\n',encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--transcripts',required=True,nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--scope',choices=('all_train','transcript_samples'),default='all_train')
    parser.add_argument('--sample-ids',nargs='+')
    args = parser.parse_args()
    result = evaluate(args.data_root,args.transcripts,args.out,args.scope,args.sample_ids)
    print(json.dumps(dict(counts=result['counts'],wer_cer=result['wer_cer']),indent=2))


if __name__=='__main__':
    main()
