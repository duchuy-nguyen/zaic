"""Versioned provenance manifest. Unknown metadata stays null, never invented."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import platform
from pathlib import Path
import subprocess

from .contracts import RuntimeConfig


def provenance_template():
    return {
        "data": {"archive_sha256": None, "split_name": None, "split_manifest_sha256": None,
                 "catalog_sha256": None},
        "environment": {"runtime_revision": None, "runtime_sha256": None},
        "model": {"base_revision": None, "adapter_revision": None,
                  "logical_parameters": None, "tokenizer_revision": None,
                  "template_sha256": None, "prompt_sha256": None},
        "asr": {"revision": None, "config_sha256": None},
        "dependencies": {"lock_sha256": None, "container_digest": None},
        "hardware": {"gpu": None, "driver": None, "cuda": None},
        "inference": {"seed": None, "decoding": None, "precision": None,
                      "context_limit": None, "transcript_source": None},
        "evaluation": {"scorer_version": None, "timing_mode": None},
    }


def validate_provenance(metadata):
    template = provenance_template()
    if type(metadata) is not dict or set(metadata) != set(template):
        raise ValueError("invalid_provenance_sections")
    numeric = {"logical_parameters", "seed", "context_limit"}
    for section, fields in template.items():
        values = metadata[section]
        if type(values) is not dict or set(values) != set(fields):
            raise ValueError("invalid_provenance_fields")
        for key, value in values.items():
            if value is None:
                continue
            if key in numeric:
                if type(value) is not int or value < (0 if key == "seed" else 1):
                    raise ValueError("invalid_provenance_number")
                if key == "context_limit" and value > 16384:
                    raise ValueError("context_limit_exceeded")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError("invalid_provenance_text")
            if key.endswith("sha256"):
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("invalid_sha256")
            if isinstance(value, str) and ("Bearer " in value or "://" in value):
                raise ValueError("provenance_requires_identifiers_not_urls_or_credentials")
    return metadata


def code_identity(root: Path):
    digest = hashlib.sha256()
    paths = list((root / "src").rglob("*.py"))
    paths += list((root / "scripts").rglob("*.py"))
    paths += [root / "pyproject.toml"]
    for path in sorted(p for p in paths if p.is_file()):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(str(len(data)).encode() + b"\0" + data)
    revision = dirty = None
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, timeout=2, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                   capture_output=True, text=True, timeout=2, check=True).stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_revision": revision, "git_dirty": dirty, "source_sha256": digest.hexdigest()}


def build_manifest(root: Path, config: RuntimeConfig, *, mode: str, provenance=None):
    metadata = validate_provenance(provenance if provenance is not None else provenance_template())
    missing = [f"{section}.{key}" for section, fields in metadata.items()
               for key, value in fields.items() if value is None]
    return {
        "manifest_version": 1, "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "code": code_identity(root),
        "python": platform.python_version(), "platform": platform.platform(),
        "provenance": metadata, "missing_provenance": missing,
        "provenance_complete": not missing, "official_submission": False,
        "clock_policy": "cooperative monotonic; demo deadline uses fake clock",
        "deadline_scope": "runner only; excludes ASR/input and result serialization",
    }
