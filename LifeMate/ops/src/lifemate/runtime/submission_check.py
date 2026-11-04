"""Structural preflight only; does not execute tools, score or authenticate hashes."""
from pathlib import Path
from .configuration import strict_json


class InputError(ValueError):
    pass


def expected_ids(path: Path):
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = strict_json(line)
            except ValueError:
                raise InputError(f"invalid_expected_json_line_{line_number}") from None
            if type(row) is not dict or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise InputError(f"invalid_expected_id_line_{line_number}")
            if row["id"] in ids:
                raise InputError(f"duplicate_expected_id_line_{line_number}")
            ids.add(row["id"])
    if not ids:
        raise InputError("empty_expected_ids")
    return ids


def check_submission(path: Path, expected: set[str], max_bytes: int = 5 * 1024 * 1024):
    if not expected or any(not isinstance(i, str) or not i.strip() for i in expected):
        raise InputError("invalid_expected_ids")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise InputError("invalid_size_limit")
    errors = []
    seen = set()
    rows = 0
    def error(code, line=None, count=None):
        item = {"code": code}
        if line is not None:
            item["line"] = line
        if count is not None:
            item["count"] = count
        errors.append(item)
    size = path.stat().st_size
    if size > max_bytes:
        error("file_size_exceeded")
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                rows += 1
                try:
                    row = strict_json(line)
                except ValueError:
                    error("invalid_json", number)
                    continue
                if type(row) is not dict:
                    error("row_must_be_object", number)
                    continue
                if set(row) != {"id", "tool_calls", "state_hash"}:
                    error("invalid_row_fields", number)
                identifier = row.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    error("invalid_id", number)
                else:
                    if identifier in seen:
                        error("duplicate_id", number)
                    seen.add(identifier)
                    if identifier not in expected:
                        error("unknown_id", number)
                if not isinstance(row.get("state_hash"), str) or not row["state_hash"].strip():
                    error("invalid_state_hash", number)
                calls = row.get("tool_calls")
                if type(calls) is not list:
                    error("tool_calls_must_be_list", number)
                    continue
                for call in calls:
                    if (type(call) is not dict or set(call) != {"name", "arguments"}
                            or not isinstance(call.get("name"), str) or not call["name"].strip()
                            or type(call.get("arguments")) is not dict):
                        error("invalid_tool_call", number)
    except UnicodeError:
        error("invalid_utf8")
    missing = expected - seen
    if missing:
        error("missing_ids", count=len(missing))
    return {"valid": not errors, "expected_count": len(expected), "row_count": rows,
            "unique_id_count": len(seen), "size_bytes": size, "max_bytes": max_bytes,
            "errors": errors, "checks": "structural_only",
            "not_checked": ["catalog_argument_schema", "tool_permissions", "call_provenance",
                            "state_hash_authenticity", "task_quality", "runtime_deadline"]}
