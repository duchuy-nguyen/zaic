"""Prepare pinned TRAIN-fit audio inputs, without loading model/runtime binaries."""
import argparse
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from lifemate.runtime.smoke_inputs import prepare_smoke
from lifemate.runtime.configuration import strict_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--private-data', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True, help='Handoff selection.json: samples list with id fields')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=12)
    args = parser.parse_args(argv)
    try:
        selection = strict_json(args.selection.read_text(encoding='utf-8'))
        report = prepare_smoke(args.dataset_dir, args.private_data,
            [row['id'] for row in selection['samples']], args.output_dir, max_samples=args.max_samples)
    except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error) as exc:
        print(json.dumps({'valid': False, 'error_type': type(exc).__name__}))
        return 2
    print(json.dumps({'valid': True, 'samples': report['samples'], 'audio_count': len(report['audio']),
                      'model_inference_run': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
