"""Read-only structural submission validation against dataset/manifest ID JSONL."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lifemate.runtime.submission_check import check_submission, expected_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--expected-ids", type=Path, required=True,
                        help='UTF-8 JSONL with an "id" field per row (other fields ignored)')
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        result = check_submission(args.submission, expected_ids(args.expected_ids), args.max_bytes)
    except (OSError, ValueError, UnicodeError):
        print(json.dumps({"valid": False, "error": "invalid_or_unreadable_input"}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
