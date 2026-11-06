"""Content-preserving normalization only: Unicode NFC and whitespace."""
import unicodedata

NORMALIZATION_VERSION = 'nfc-whitespace-v1'


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError('normalize expects str')
    return ' '.join(unicodedata.normalize('NFC', text).split())
