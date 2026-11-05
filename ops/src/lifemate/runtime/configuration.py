"""Strict development configuration loading, without guessed model settings."""
import json
import math
from pathlib import Path

from .contracts import RuntimeConfig


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite_json_number")


def strict_json(text):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_json_number")
        return number
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant,
                      parse_float=finite_float)


def load_config(path: Path) -> RuntimeConfig:
    value = strict_json(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("config_must_be_object")
    from dataclasses import fields
    expected = {field.name for field in fields(RuntimeConfig)}
    if set(value) != expected:
        raise ValueError("config_requires_exact_runtime_fields")
    return RuntimeConfig(**value)
