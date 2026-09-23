#!/usr/bin/env python3
"""Evaluate one externally supplied event-level NPZ with the paper's fixed protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from unified_metrics import evaluate


def load_events(path, *, energy_key='energy_keV'):
    with np.load(path, allow_pickle=False) as archive:
        required = ('label', 'score', energy_key, 'event_id')
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f'Missing event-level fields: {missing}')
        data = {key: archive[key] for key in archive.files}
    n = len(data['score'])
    for key in required:
        if data[key].ndim != 1 or len(data[key]) != n:
            raise ValueError(f'{key} must be an event-aligned one-dimensional array')
    ids = data['event_id'].astype(str)
    if len(np.unique(ids)) != n or np.any(ids == ''):
        raise ValueError('event_id must be nonempty and unique within this test population')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--energy-unit', choices=('keV', 'MeV'), default='keV')
    parser.add_argument('--energy-key', default=None, help='Defaults to energy_keV for keV or energy for MeV')
    parser.add_argument('--weights-output', type=Path, help='Optional separate event-aligned matching arrays')
    args = parser.parse_args()
    key = args.energy_key or ('energy_keV' if args.energy_unit == 'keV' else 'energy')
    data = load_events(args.predictions, energy_key=key)
    result, arrays = evaluate(data['label'], data['score'], data[key],
        group=data.get('group'), weight=data.get('weight'), energy_unit=args.energy_unit,
        return_arrays=True)
    result['input'] = {'path': str(args.predictions),
        'sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        'energy_key': key, 'energy_unit': args.energy_unit,
        'score_direction': 'higher supports label 1; supplied scores unchanged',
        'event_id_sha256': hashlib.sha256(data['event_id'].astype(str).tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.weights_output:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.weights_output, event_id=data['event_id'], **arrays)
    print(json.dumps({'output': str(args.output), 'n_input': result['n_input'],
        'I': result['independence']['I'], 'matched_auc': result['matching']['matched_auc'],
        'matching_status': result['matching']['status'], 'protocol_sha256': result['protocol_sha256']}))


if __name__ == '__main__':
    main()
