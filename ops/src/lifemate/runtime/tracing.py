"""Private JSON artifacts with conservative secret-key redaction."""
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import re
from pathlib import Path
from typing import Any

_SECRET = re.compile(
    r"secret|password|passwd|authorization|api.?key|cookie|credential|(?:^|[_-])token$", re.I)
_BEARER = re.compile(r"(?i)\bBearer\s+\S+")


def safe_json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SECRET.search(str(k)) else safe_json(v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, str):
        return _BEARER.sub("Bearer [REDACTED]", value)
    if value is None or type(value) in (bool, int, float):
        return value
    return "[unsupported value]"


class JsonlWriter:
    def __init__(self, path: Path):
        self.file = path.open("x", encoding="utf-8")

    def __enter__(self):
        return self

    def __call__(self, event: Any) -> None:
        self.file.write(json.dumps(safe_json(event), ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()

    def __exit__(self, *args):
        self.file.close()
