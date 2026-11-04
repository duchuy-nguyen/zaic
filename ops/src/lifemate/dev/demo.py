"""Offline CPU demo. Output is explicitly mock, never a submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from lifemate.runtime.contracts import Sample
from lifemate.runtime.configuration import load_config, strict_json
from lifemate.runtime.manifest import build_manifest
from lifemate.runtime.runner import Runner
from lifemate.runtime.tracing import JsonlWriter, safe_json
from .mocks import MockEnvironment, scenarios

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ops/runtime.example.json")
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        provenance = strict_json(args.provenance.read_text(encoding="utf-8")) if args.provenance else None
        manifest = build_manifest(ROOT, config, mode="development_mock_only", provenance=provenance)
    except (OSError, ValueError, TypeError, UnicodeError):
        parser.error("invalid config/provenance; use check_config.py")
    stamp = datetime.now(timezone.utc)
    directory = args.output_dir or ROOT / "runs" / stamp.strftime("mock_%Y%m%dT%H%M%S_%fZ")
    # Fail on an existing run instead of silently overwriting results.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "run_manifest.json").write_text(
        json.dumps(safe_json(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with JsonlWriter(directory / "tool_traces.jsonl") as trace, JsonlWriter(directory / "mock_results.jsonl") as results:
        for name, backend, clock in scenarios():
            result = Runner(lambda sample: MockEnvironment(), config,
                            clock=clock or time.monotonic, trace=trace).run(Sample(name), backend)
            results(result)
            print(f"{name}: {result.stop_reason.value}; executed={len(result.executed_calls)}")
    print(f"Mock artifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
