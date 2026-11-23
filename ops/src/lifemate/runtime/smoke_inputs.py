"""Prepare bounded audio-only FIT inputs; official labels never enter outputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import wave

from .configuration import strict_json


def manifest():
    return strict_json((Path(__file__).resolve().parents[4] /
        'llm/baseline_50_2/assets.manifest.json').read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Invalid relative path')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Path escapes dataset')
    target = root.joinpath(*parts.parts).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError('Missing or escaping dataset file')
    return target


def clean_row(row):
    # Allowlist at every model-facing boundary. History tool results are prior
    # observations supplied by BTC, not current label.tool_results.
    context = row['context']
    clean_context = {k: deepcopy(context[k]) for k in (
        'current_time', 'timezone', 'location', 'user_profile') if k in context}
    clean_context['installed_apps'] = {
        app: {'tools': list(spec['tools'])} for app, spec in context['installed_apps'].items()}
    history = []
    for turn in row['history']:
        if turn['role'] == 'user':
            history.append({'role': 'user', 'audio': turn['audio']})
        elif turn['role'] == 'assistant':
            history.append({k: deepcopy(turn[k]) for k in ('role', 'text', 'tool_calls', 'tool_results') if k in turn})
        else:
            raise ValueError('Unexpected official history role')
    return {'id': row['id'], 'context': clean_context, 'history': history,
            'latest_audio': row['audio']}


def prepare_smoke(dataset_dir, private_data, selected_ids, output_dir, *, max_samples=12):
    selected_ids = list(selected_ids)
    if (type(max_samples) is not int or max_samples <= 0 or not selected_ids
            or len(selected_ids) > max_samples or len(set(selected_ids)) != len(selected_ids)
            or any(not isinstance(s, str) or not s for s in selected_ids)):
        raise ValueError('Expected bounded unique sample IDs')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Preserve existing output directory')
    root = Path(dataset_dir).resolve() / 'train'
    private = Path(private_data).resolve()
    pinned = manifest()
    spec = pinned['dataset']['train']
    if (digest(root/'train.jsonl') != spec['jsonl_sha256'] or
            digest(root/'train_tools_catalog.json') != spec['catalog_sha256']):
        raise ValueError('TRAIN JSONL/catalog differs from baseline50')
    runtime = {p.relative_to(root/'zaic_tools').as_posix(): digest(p)
               for p in sorted((root/'zaic_tools').rglob('*.py'))}
    if runtime != spec['runtime_py_sha256']:
        raise ValueError('TRAIN SDK differs from baseline50')
    for name in ('split.json', 'retrieval_index.json'):
        expected = pinned['files']['private_data/'+name]
        path = private/name
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Private split/index differs from baseline50')
    split = strict_json((private/'split.json').read_text(encoding='utf-8'))
    fit, val = set(split['train_fit']), set(split['val_main'])
    if fit & val or not set(selected_ids) <= fit:
        raise ValueError('Smoke IDs must belong to frozen baseline50 FIT')
    rows = [strict_json(line) for line in (root/'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate official IDs')
    clean, audio_checks = [], {}
    for sid in selected_ids:
        row = clean_row(by_id[sid])
        inside(root, 'train_environment/'+sid+'/state.json')  # existence only; never parse state
        for reference in [row['latest_audio']] + [t['audio'] for t in row['history'] if t['role']=='user']:
            path = inside(root, reference)
            if reference in audio_checks:
                continue
            with wave.open(str(path), 'rb') as wav:
                channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
                if (channels, width, rate) != (1, 2, 16000) or frames <= 0 or wav.getcomptype() != 'NONE':
                    raise ValueError('Expected nonempty mono PCM16 16kHz WAV')
                if len(wav.readframes(frames)) != frames*channels*width:
                    raise ValueError('Truncated WAV')
            audio_checks[reference] = {'sha256': digest(path), 'duration_seconds': frames/rate}
        clean.append(row)
    report = {'mode': 'baseline50_TRAIN_fit_audio_preparation', 'samples': len(clean),
              'sample_ids': selected_ids, 'audio': audio_checks,
              'train_jsonl_sha256': spec['jsonl_sha256'], 'catalog_sha256': spec['catalog_sha256'],
              'split_sha256': digest(private/'split.json'), 'runtime_py_sha256': runtime,
              'model_inference_run': False, 'model_assets_verified': False,
              'source': 'BTC_TRAIN; not team synthetic',
              'limitations': ['No model/GPU inference or quality certification.',
                              'Environment files checked for existence only; no state is exposed to model.']}
    output.mkdir(parents=True, exist_ok=False)
    for name, values in [('prepared_audio.jsonl', clean), ('expected_ids.jsonl', [{'id': s} for s in selected_ids])]:
        with (output/name).open('x', encoding='utf-8') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    report['prepared_input_sha256'] = digest(output/'prepared_audio.jsonl')
    report['expected_ids_sha256'] = digest(output/'expected_ids.jsonl')
    (output/'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report
