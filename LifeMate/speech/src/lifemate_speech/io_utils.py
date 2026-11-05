"""Audio-only dataset view. Gold transcripts, labels and context never escape loader."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import wave


def speech_work_dir() -> Path:
    """Private output root; override when code is checked out inside the team repo."""
    configured = os.environ.get('LIFEMATE_SPEECH_WORK_DIR')
    return (Path(configured).expanduser() if configured else Path(__file__).resolve().parents[3] / 'speech_work').resolve()


def ensure_work_path(path: Path) -> Path:
    work = speech_work_dir()
    path = path.resolve()
    if not path.is_relative_to(work.resolve()):
        raise ValueError(f'Output/cache must be inside {work}')
    return path


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def audio_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_audio(split_dir: Path, raw: str) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw:
        raise ValueError('Missing/invalid audio path')
    rel = PurePosixPath(raw.replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts or ':' in raw:
        raise ValueError(f'Unsafe audio path: {raw!r}')
    path = split_dir.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(split_dir.resolve()):
        raise ValueError('Audio path escapes split')
    return path, path.relative_to(split_dir.resolve()).as_posix()


def load_audio_refs(data_root: Path, split: str, limit: int | None = None, sample_ids=None) -> list[dict]:
    if split not in ('train', 'public_test') or (limit is not None and limit < 1):
        raise ValueError('Invalid split or limit')
    refs, ids = [], set()
    wanted = set(sample_ids) if sample_ids is not None else None
    with (data_root / split / f'{split}.jsonl').open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if limit is not None and len(ids) >= limit:
                break
            row = json.loads(line)
            sid = row['id']
            if wanted is not None and sid not in wanted:
                continue
            if not isinstance(sid, str) or sid in ids:
                raise ValueError(f'Duplicate/invalid sample id at line {line_no}')
            ids.add(sid)
            for i, turn in [(None, row)] + [(i, t) for i, t in enumerate(row.get('history', [])) if t.get('role') == 'user']:
                raw = turn.get('audio')
                error, path, relative = None, None, raw
                try:
                    path, relative = resolve_audio(data_root / split, raw)
                except (ValueError, TypeError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                refs.append(dict(sample_id=sid, split=split, kind='current' if i is None else 'history',
                                 turn_index=i, audio_path=relative, path=path, input_error=error))
    if wanted is not None and wanted != ids:
        raise ValueError(f'Selected sample IDs not found (or excluded by limit): {sorted(wanted - ids)}')
    return refs


def wav_info(path: Path) -> dict:
    with wave.open(str(path), 'rb') as wav:
        rate, frames = wav.getframerate(), wav.getnframes()
        if rate != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != 'NONE':
            raise ValueError('Expected 16 kHz mono 16-bit PCM; no implicit resampling')
        if not frames:
            raise ValueError('Empty audio')
        duration = frames / rate
        if duration > 30:
            raise ValueError(f'AudioTooLong: {duration:.6f}s >30s; chunking not enabled; refusing silent truncation')
        return dict(audio_duration_s=duration, n_frames=frames, sample_rate=rate)


def ref_key(row):
    return row['sample_id'], row['kind'], row['turn_index']
