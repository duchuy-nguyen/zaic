"""Validate runtime config, optionally validate a provenance template."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lifemate.runtime.configuration import load_config, strict_json
from lifemate.runtime.manifest import validate_provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.provenance:
            validate_provenance(strict_json(args.provenance.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, UnicodeError):
        print(json.dumps({"valid": False, "error": "invalid_or_unreadable_config"}))
        return 2
    print(json.dumps({"valid": True, "config": asdict(config)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
