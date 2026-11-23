"""Audit pilot resume tar.gz without loading archived code or checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from lifemate.runtime.pilot_audit import audit_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_archive(args.archive)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, tarfile.TarError):
        print(json.dumps({'structurally_valid': False, 'error': 'invalid_or_incomplete_pilot_archive'}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['structurally_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
