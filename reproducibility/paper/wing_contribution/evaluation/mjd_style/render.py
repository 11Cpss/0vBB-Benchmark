#!/usr/bin/env python3
"""Render the active MJD low/high waveform figure from exact selected arrays."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/energybench-mjd-style-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from paper_style import apply_paper_style

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'data/mjd_waveform_population.csv').open() as handle:
        population = list(csv.DictReader(handle))
    summary = json.loads((ROOT / 'data/mjd_waveform_summary.json').read_text())
    assert len(population) == 410
    assert all(int(row['detector']) == 133 and int(row['run_number']) == 50928 for row in population)
    cohort = np.array([row['cohort'] for row in population])
    energy = np.array([float(row['energy_keV']) for row in population])
    rms = np.array([float(row['normalized_baseline_rms']) for row in population])
    rho = float(spearmanr(energy, rms).statistic)
    assert abs(rho - summary['spearman_energy_normalized_baseline_rms']) < 1e-12
    with np.load(ROOT / 'data/selected_waveforms_exact.npz', allow_pickle=False) as packed:
        chosen = packed['population_positions']
        centered = np.zeros((410, 3800), dtype=np.float32)
        normalized = np.zeros_like(centered)
        centered[chosen] = packed['centered']
        normalized[chosen] = packed['normalized']
    ids = np.array([int(row['event_id']) for row in population])
    assert ids[chosen].tolist() == [2407207, 2585508]
    q25, q75 = summary['energy_quartiles_keV']
    plt.rcdefaults()
    apply_paper_style()
    scope = dict(plt=plt, np=np, Line2D=Line2D, title_weight='normal',
                 cohort=cohort, energy=energy, rms=rms, rho=rho,
                 detector=133, run=50928, examples=chosen,
                 low=cohort == 'Low energy', high=cohort == 'High energy',
                 q25=q25, q75=q75, meta={'id': ids}, indices=np.arange(410),
                 waves=np.empty((0, 3800), dtype=np.float32),
                 centered=centered, normalized=normalized)
    snippet = ROOT / 'low_high_plot.py.inc'
    exec(compile(snippet.read_text(), str(snippet), 'exec'), scope)
    figure = scope['fig']
    output = args.output_dir / 'mjd_low_high_waveforms.pdf'
    figure.savefig(output, dpi=220, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(figure)
    print(json.dumps({'output': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__ == '__main__':
    main()
