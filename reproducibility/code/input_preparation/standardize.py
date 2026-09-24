#!/usr/bin/env python3
"""Create an aligned EnergyBench NPZ from native scalar predictions and physical energy."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmark'))
from metrics import to_keV


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def event_ids(data, key):
    if key not in data:
        raise ValueError(f'Predictions must contain physical event IDs in {key}; array length alone is insufficient')
    ids = np.asarray(data[key]).astype(str)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(np.isin(ids, ['', 'None', 'nan'])):
        raise ValueError('Event IDs must be one-dimensional, nonempty and unique')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-key', default='score')
    parser.add_argument('--label-key', default='label')
    parser.add_argument('--event-id-key', default='event_id')
    parser.add_argument('--energy-key', default='energy_keV')
    parser.add_argument('--energy-unit', choices=('keV', 'MeV'), required=True)
    parser.add_argument('--energy-metadata', type=Path,
                        help='Optional physical-energy NPZ; rows are joined by event_id, never by position')
    parser.add_argument('--positive-label', type=int, choices=(0, 1), default=1)
    parser.add_argument('--score-label', type=int, choices=(0, 1), default=1,
                        help='Native label favored by a larger scalar score')
    parser.add_argument('--weight-key', help='Optional native base-weight array')
    parser.add_argument('--group-key', help='Optional native physical-category array')
    args = parser.parse_args()
    data = read(args.input)
    ids = event_ids(data, args.event_id_key)
    score = np.asarray(data[args.score_key], dtype=np.float64)
    label = np.asarray(data[args.label_key])
    if score.ndim != 1 or label.ndim != 1 or len(score) != len(ids) or len(label) != len(ids):
        raise ValueError('Supply one scalar score and binary label per event; use the model exporter for multi-output logits')
    if not np.all(np.isin(label, [0, 1])):
        raise ValueError('Native labels must be binary 0/1')
    if args.energy_metadata:
        meta = read(args.energy_metadata)
        meta_ids = event_ids(meta, args.event_id_key)
        order = np.argsort(meta_ids)
        positions = np.searchsorted(meta_ids[order], ids)
        if np.any(positions >= len(meta_ids)) or not np.array_equal(meta_ids[order][positions], ids):
            raise ValueError('Physical-energy metadata does not cover the exact prediction event IDs')
        rows = order[positions]
        energy = np.asarray(meta[args.energy_key], dtype=np.float64)[rows]
        if args.label_key in meta and not np.array_equal(meta[args.label_key][rows], label):
            raise ValueError('Native labels disagree after the physical-event ID join')
    else:
        energy = np.asarray(data[args.energy_key], dtype=np.float64)
    if energy.ndim != 1 or len(energy) != len(ids):
        raise ValueError('Energy must provide one original physical value per event')
    out = {'score': score if args.score_label == args.positive_label else -score,
           'label': (label == args.positive_label).astype(np.int8),
           'energy_keV': to_keV(energy, args.energy_unit), 'event_id': ids}
    for source, destination in [(args.weight_key, 'weight'), (args.group_key, 'group')]:
        if source:
            out[destination] = data[source]
            if out[destination].ndim != 1 or len(out[destination]) != len(ids):
                raise ValueError(f'{source} is not event-aligned')
    if args.output.resolve() == args.input.resolve() or (args.energy_metadata and args.output.resolve() == args.energy_metadata.resolve()):
        raise ValueError('Choose a new output file to preserve the input predictions and metadata')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **out)
    print(json.dumps({'events': len(ids), 'output': str(args.output),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
