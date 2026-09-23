#!/usr/bin/env python3
"""Raw SuperNEMO geometry versus measured energy; no learned model.

Predefined descriptors: radius of gyration, covariance anisotropy, hit count.
Fixed 250 keV energy bins, all test events in each class, median/IQR shown for
bins with at least 20 finite events. Sources are read-only; all artifacts and
runtime caches remain inside this paper revision. No fitted track is inferred.
"""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
assert ROOT.is_relative_to(Path('/home/wenyu/iclr final paper').resolve())
for name in ['.runtime/tmp', '.runtime/cache', '.runtime/mpl', 'figures', 'evidence']:
    path = ROOT/name
    assert path.resolve().is_relative_to(ROOT)
    path.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(ROOT/'.runtime/tmp'), TMP=str(ROOT/'.runtime/tmp'),
    TEMP=str(ROOT/'.runtime/tmp'), XDG_CACHE_HOME=str(ROOT/'.runtime/cache'),
    MPLCONFIGDIR=str(ROOT/'.runtime/mpl'), PYTHONDONTWRITEBYTECODE='1')
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

MANIFEST = Path('/home/wenyu/SuperNEMO/data/manifests/split_manifest.json')
CATEGORIES = ['Bi214', '0nubb']
BIN_WIDTH = 250.
MIN_BIN_COUNT = 20
FEATURES = ['radius_gyration_mm', 'anisotropy', 'n_tracker_hits']
WINDOWS = {'low': (500., 1000.), 'high': (2200., 2700.)}


def reduce_geometry(xyz, first, sizes):
    """Equal-hit-weight population covariance, using centered second moments."""
    means = np.add.reduceat(xyz, first, axis=0)/sizes[:, None]
    centered = xyz-np.repeat(means, sizes, axis=0)
    covariance = np.empty((len(first), 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(a, 3):
            values = np.add.reduceat(centered[:, a]*centered[:, b], first)/sizes
            covariance[:, a, b] = covariance[:, b, a] = values
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() >= -1.e-7
    eigenvalues = np.maximum(eigenvalues, 0.)
    trace = np.trace(covariance, axis1=1, axis2=2)
    assert np.all(trace >= 0.)
    rg = np.sqrt(trace)
    anisotropy = np.full(len(trace), np.nan)
    positive = trace > 0.
    anisotropy[positive] = (3.*eigenvalues[positive, -1]/trace[positive]-1.)/2.
    assert np.all((anisotropy[positive] >= -1.e-12)&(anisotropy[positive] <= 1.+1.e-12))
    anisotropy[positive] = np.clip(anisotropy[positive], 0., 1.)
    return rg, anisotropy, covariance


def extract(manifest, category):
    record = next(f for f in manifest['files'] if f['source_key'] == category)
    path = Path(manifest['data_root'])/record['file_name']
    assert path.stat().st_size == record['bytes']
    index_path = MANIFEST.parent/record['index_file']
    offsets = np.load(index_path, allow_pickle=False, mmap_mode='r')
    assert offsets.dtype == np.int64
    assert offsets[0] == 0 and offsets[-1] == record['profile']['row_count']
    values = {key: [] for key in ['event_ordinal', 'energy_keV', *FEATURES]}
    checks = {'individually_recomputed_covariances': 0, 'maximum_covariance_abs_error': 0.,
        'maximum_translation_covariance_abs_error': 0., 'n_raw_tracker_rows': 0}
    with h5py.File(path, 'r') as handle:
        for part in manifest['splits']['test']:
            if part['source_key'] != category:
                continue
            for start in range(part['event_start'], part['event_stop'], 4096):
                stop = min(start+4096, part['event_stop'])
                a, b = int(offsets[start]), int(offsets[stop])
                first = offsets[start:stop]-a
                sizes = np.diff(offsets[start:stop+1])
                assert np.all(sizes > 0)
                ids = np.arange(start, stop)
                assert np.array_equal(handle['ev_no'][a:b], np.repeat(ids, sizes))
                assert np.all(handle['label'][a:b] == category.encode())
                e1, e2 = handle['E1'][a:b], handle['E2'][a:b]
                assert np.array_equal(e1, np.repeat(e1[first], sizes))
                assert np.array_equal(e2, np.repeat(e2[first], sizes))
                energy = e1[first].astype(np.float64)+e2[first].astype(np.float64)
                xyz = np.column_stack([handle[key][a:b] for key in ['tX', 'tY', 'tZ']]).astype(np.float64)
                assert np.isfinite(xyz).all() and np.isfinite(energy).all()
                rg, shape, covariance = reduce_geometry(xyz, first, sizes)
                # Deterministic spot checks: first, middle, last event per chunk.
                for j in sorted(set([0, len(first)//2, len(first)-1])):
                    pts = xyz[first[j]:first[j]+sizes[j]]
                    centered = pts-pts.mean(axis=0)
                    independent = centered.T@centered/len(pts)
                    error = float(np.max(np.abs(independent-covariance[j])))
                    assert np.allclose(independent, covariance[j], rtol=1.e-12, atol=1.e-8)
                    translated = pts+np.array([31., -47., 101.])
                    translated -= translated.mean(axis=0)
                    translation_error = float(np.max(np.abs(translated.T@translated/len(pts)-independent)))
                    assert translation_error < 1.e-7
                    checks['maximum_covariance_abs_error'] = max(checks['maximum_covariance_abs_error'], error)
                    checks['maximum_translation_covariance_abs_error'] = max(checks['maximum_translation_covariance_abs_error'], translation_error)
                    checks['individually_recomputed_covariances'] += 1
                for key, arr in [('event_ordinal', ids), ('energy_keV', energy),
                        ('radius_gyration_mm', rg), ('anisotropy', shape), ('n_tracker_hits', sizes)]:
                    values[key].append(arr)
                checks['n_raw_tracker_rows'] += len(xyz)
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    n = len(values['energy_keV'])
    assert n == manifest['counts']['energy']['test']['by_category'][category]
    assert np.unique(values['event_ordinal']).size == n
    old = json.loads((ROOT/'evidence/raw_topology_diagnostic.json').read_text())['results'][category]
    energy_hash = hashlib.sha256(values['energy_keV'].astype('<f8').tobytes()).hexdigest()
    count_hash = hashlib.sha256(values['n_tracker_hits'].astype('<i8').tobytes()).hexdigest()
    assert energy_hash == old['energy_sha256']
    assert count_hash == old['n_tracker_hits_sha256']
    checks.update(source_file=str(path), index_file=str(index_path), n_events=n,
        energy_sha256=energy_hash, n_tracker_hits_sha256=count_hash,
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        zero_variance_events=int(np.isnan(values['anisotropy']).sum()))
    print(category, json.dumps(checks), flush=True)
    return values, checks


def describe(values):
    return {'n': len(values), 'minimum': float(np.min(values)), 'maximum': float(np.max(values)),
        'mean': float(np.mean(values)), 'p25': float(np.percentile(values, 25)),
        'median': float(np.median(values)), 'p75': float(np.percentile(values, 75))}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads(MANIFEST.read_text())
    values, checks = {}, {}
    for category in CATEGORIES:
        values[category], checks[category] = extract(manifest, category)
    # Zero-anchored fixed-width edges cover every event; the last bin includes
    # its right edge, matching np.histogram's final-edge convention.
    max_energy = max(v['energy_keV'].max() for v in values.values())
    assert min(v['energy_keV'].min() for v in values.values()) >= 0.
    edges = np.arange(0., BIN_WIDTH*np.ceil(max_energy/BIN_WIDTH)+BIN_WIDTH/2., BIN_WIDTH)
    rows, summary, tables = [], {}, {}
    for category, data in values.items():
        energy = data['energy_keV']
        ids = np.minimum(np.searchsorted(edges, energy, side='right')-1, len(edges)-2)
        counts = np.bincount(ids, minlength=len(edges)-1)
        assert counts.sum() == len(energy)
        summary[category] = {'n_events': len(energy), 'energy': describe(energy),
            'bin_counts': counts.tolist(), 'features': {}}
        tables[category] = {}
        for feature in FEATURES:
            finite = np.isfinite(data[feature])
            cohorts = {}
            for name, (lo, hi) in WINDOWS.items():
                mask = (energy >= lo)&(energy < hi)&finite
                cohorts[name] = describe(data[feature][mask])
            summary[category]['features'][feature] = {
                'all_finite': describe(data[feature][finite]), 'n_nonfinite': int((~finite).sum()),
                'spearman_with_energy': float(spearmanr(energy[finite], data[feature][finite]).statistic),
                'predefined_energy_cohorts': cohorts}
            feature_rows = []
            for k in range(len(edges)-1):
                sample = data[feature][(ids == k)&finite]
                q = np.percentile(sample, [25, 50, 75]) if len(sample) else [np.nan]*3
                record = {'category': category, 'feature': feature, 'energy_left_keV': float(edges[k]),
                    'energy_right_keV': float(edges[k+1]), 'bin_center_keV': float((edges[k]+edges[k+1])/2.),
                    'bin_events': int(counts[k]), 'finite_events': len(sample),
                    'q25': float(q[0]), 'median': float(q[1]), 'q75': float(q[2]),
                    'shown_as_trend': int(len(sample) >= MIN_BIN_COUNT)}
                rows.append(record)
                feature_rows.append(record)
            tables[category][feature] = feature_rows
            covered = sum(row['finite_events'] for row in feature_rows if row['shown_as_trend'])
            summary[category]['features'][feature]['trend_covered_events'] = covered
            summary[category]['features'][feature]['trend_coverage_fraction'] = covered/len(energy)
    # Event-level derived observables are local audit data and must not be uploaded.
    with gzip.open(ROOT/'evidence/supernemo_geometry_event_descriptors_LOCAL_ONLY.csv.gz', 'wt', newline='') as stream:
        fields = ['category', 'event_ordinal', 'energy_keV', *FEATURES]
        writer = csv.writer(stream)
        writer.writerow(fields)
        for category, data in values.items():
            for row in zip(*(data[key] for key in fields[1:])):
                writer.writerow([category, *row])
    write_csv(ROOT/'figures/supernemo_geometry_energy_binned.csv', rows)
    result = {'manifest': str(MANIFEST), 'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'split': 'test', 'energy_definition': 'E1 + E2; reconstructed calorimeter electron energies, keV',
        'coordinates': 'All stored tX,tY,tZ tracker hit centers in mm; equal hit weights; no trajectory fit',
        'descriptor_definitions': {'covariance': 'C = (1/n) sum_j (r_j - mean(r))(r_j - mean(r))^T',
            'radius_gyration_mm': 'sqrt(trace(C))',
            'anisotropy': '(3 lambda_max(C)/trace(C) - 1)/2; undefined for zero trace',
            'n_tracker_hits': 'number of raw tracker rows in the event'},
        'predefined_descriptors': FEATURES, 'energy_bin_width_keV': BIN_WIDTH,
        'energy_bin_edges_keV': edges.tolist(), 'bin_boundary': 'left closed/right open, except inclusive last right edge',
        'trend_min_finite_events': MIN_BIN_COUNT, 'cohort_windows_keV': WINDOWS,
        'normalization': 'None; physical coordinates retained and moments centered per event',
        'uncertainty_display': 'Middle 50 percent of events (IQR), not an uncertainty interval for the median',
        'checks': checks, 'summary': summary,
        'scope': 'Descriptive within-class topology-energy relationships; not independence score, model inference, fitted track length, or proof of topology energy separability.'}
    (ROOT/'evidence/geometry_energy.json').write_text(json.dumps(result, indent=2)+'\n')
    make_plot(tables, summary, edges)
    for category in CATEGORIES:
        print(category, 'correlations', {key: summary[category]['features'][key]['spearman_with_energy'] for key in FEATURES}, flush=True)


def make_plot(tables, summary, edges):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.3, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555555',
        'text.color': '#222222', 'axes.labelcolor': '#222222'})
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.55), sharex=True, sharey='col')
    colors = {'Bi214': '#286A9B', '0nubb': '#C47723'}
    names = {'Bi214': r'$^{214}$Bi', '0nubb': r'$0\nu\beta\beta$'}
    titles = ['Spatial extent', 'Shape anisotropy', 'Hit multiplicity']
    ylabels = [r'$R_g$ (mm)', r'$a$ (dimensionless)', 'Tracker hits']
    largest_hit_quartile = max(r['q75'] for c in CATEGORIES
        for r in tables[c]['n_tracker_hits'] if np.isfinite(r['q75']))
    hit_upper = 5.*np.ceil(1.08*largest_hit_quartile/5.)
    for ri, category in enumerate(CATEGORIES):
        for ci, feature in enumerate(FEATURES):
            ax = axes[ri, ci]
            records = tables[category][feature]
            x = np.array([r['bin_center_keV'] for r in records])
            shown = np.array([r['shown_as_trend'] for r in records], dtype=bool)
            med, q25, q75 = [np.array([r[key] if r['shown_as_trend'] else np.nan for r in records]) for key in ['median', 'q25', 'q75']]
            ax.fill_between(x, q25, q75, color=colors[category], alpha=.17, linewidth=0)
            ax.plot(x, med, color=colors[category], linewidth=1.5, marker='o', markersize=2.8)
            # Sparse bins remain visible as open points; no connecting trend/IQR.
            sparse = np.array([(0 < r['finite_events'] < MIN_BIN_COUNT) for r in records])
            if sparse.any():
                ax.plot(x[sparse], np.array([r['median'] for r in records])[sparse],
                    linestyle='none', marker='o', markersize=3, markerfacecolor='white',
                    markeredgecolor=colors[category], markeredgewidth=.8)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xticks(np.arange(0, edges[-1]+1, 1000))
            ax.set_ylabel(ylabels[ci])
            ax.grid(axis='y', color='#e6e6e6', linewidth=.5)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(titles[ci], loc='left', fontsize=9.2, pad=5)
            if ri == 1:
                ax.set_xlabel('Event energy (keV)')
            if ci == 0:
                ax.text(.04, .94, names[category]+f"; $n={summary[category]['n_events']:,}$",
                    transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 1.})
            if ci == 1:
                ax.set_ylim(0., 1.)
            if ci == 2:
                ax.set_ylim(0., hit_upper)
    fig.suptitle('Raw tracker topology versus reconstructed energy', x=.085, y=.995, ha='left', fontsize=11)
    fig.text(.085, .941, 'Median and middle 50% of events in 250 keV bins; no learned model', fontsize=8.3, color='#555555')
    fig.subplots_adjust(left=.085, right=.99, top=.83, bottom=.10, hspace=.24, wspace=.38)
    for suffix in ['pdf', 'png']:
        fig.savefig(ROOT/'figures'/f'supernemo_geometry_energy.{suffix}', dpi=240, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        audit = json.loads((ROOT/'evidence/geometry_energy.json').read_text())
        cached = {c: {f: [] for f in FEATURES} for c in CATEGORIES}
        with (ROOT/'figures/supernemo_geometry_energy_binned.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                for key in ['energy_left_keV', 'energy_right_keV', 'bin_center_keV', 'q25', 'median', 'q75']:
                    row[key] = float(row[key])
                for key in ['bin_events', 'finite_events', 'shown_as_trend']:
                    row[key] = int(row[key])
                cached[row['category']][row['feature']].append(row)
        make_plot(cached, audit['summary'], np.array(audit['energy_bin_edges_keV']))
    else:
        main()
