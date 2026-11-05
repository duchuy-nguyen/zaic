"""Content-addressed cache; sample IDs are deliberately absent from keys."""
import hashlib
import json
from pathlib import Path
from .io_utils import atomic_json, ensure_work_path
from .normalize import NORMALIZATION_VERSION, normalize


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def cache_key(audio_sha256: str, asr_revision: str, decoding_config: dict,
              normalization_version: str = NORMALIZATION_VERSION) -> str:
    return config_hash(dict(audio_sha256=audio_sha256, asr_revision=asr_revision,
                            decoding_config_hash=config_hash(decoding_config), normalization_version=normalization_version))


class TranscriptCache:
    def __init__(self, directory: Path):
        self.directory = ensure_work_path(directory)

    def get(self, key):
        path = self.directory / key[:2] / f'{key}.json'
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('cache_key') != key or row.get('error') is not None or not isinstance(row.get('text_raw'), str):
            raise ValueError('Invalid cache payload')
        if normalize(row['text_raw']) != row.get('text_norm'):
            raise ValueError('Cache normalization mismatch')
        return row

    def put(self, key, row):
        if row.get('error') is not None:
            raise ValueError('Never cache failed transcriptions')
        atomic_json(self.directory / key[:2] / f'{key}.json', dict(row, cache_key=key))
