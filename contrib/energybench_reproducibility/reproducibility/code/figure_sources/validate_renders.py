#!/usr/bin/env python3
"""Compare the six generated active figure PDFs with the supplied paper assets."""
from pathlib import Path
import argparse
import hashlib
import json
ROOT = Path(__file__).resolve().parents[2]
NAMES = ('mjd_motivation_main_v2_generated.pdf', 'mjd_low_high_waveforms.pdf',
         'next_capacity_scores.pdf', 'energy_bias_spectrum.pdf',
         'energy_threshold_tradeoff.pdf', 'supernemo_extent_energy_population.pdf')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/render/figures')
    args = parser.parse_args()
    records = []
    for name in NAMES:
        reference = ROOT / 'paper/wing_contribution/figures' / name
        generated = args.output_dir / name
        records.append({'figure': name, 'reference_sha256': digest(reference),
                        'generated_sha256': digest(generated),
                        'byte_identical': digest(reference) == digest(generated)})
    result = {'all_pass': all(row['byte_identical'] for row in records), 'records': records}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['all_pass'] else 1)
if __name__ == '__main__':
    main()
