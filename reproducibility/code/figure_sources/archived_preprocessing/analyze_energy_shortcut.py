#!/usr/bin/env python3
"""Real 0nu/Bi214 energy-only diagnostic; no training or model inference.

The 2200-keV cut is the user's illustrative threshold, not a test-optimized cut.
Metric calls reuse the audited array-only evaluator; all writes stay in ROOT.
"""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
assert ROOT.is_relative_to(Path('/home/wenyu/iclr final paper').resolve())
for name in ['.runtime/tmp', '.runtime/cache', '.runtime/mpl', 'figures', 'evidence']:
    p = ROOT / name
    assert p.resolve().is_relative_to(ROOT)
    p.mkdir(exist_ok=True, parents=True)
os.environ.update(TMPDIR=str(ROOT/'.runtime/tmp'), TMP=str(ROOT/'.runtime/tmp'),
                  TEMP=str(ROOT/'.runtime/tmp'), XDG_CACHE_HOME=str(ROOT/'.runtime/cache'),
                  MPLCONFIGDIR=str(ROOT/'.runtime/mpl'), PYTHONDONTWRITEBYTECODE='1')
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MANIFEST = Path('/home/wenyu/SuperNEMO/data/manifests/split_manifest.json')
SOURCE = Path('/home/wenyu/summer/src/energybench')


def load_metric(name):
    package = 'shortcut_audited_metrics'
    if package not in sys.modules:
        p = types.ModuleType(package)
        p.__path__ = [str(SOURCE)]
        sys.modules[package] = p
    spec = importlib.util.spec_from_file_location(package+'.'+name, SOURCE/(name+'.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract(manifest, cat):
    record = next(f for f in manifest['files'] if f['source_key'] == cat)
    path = Path(manifest['data_root'])/record['file_name']
    assert path.stat().st_size == record['bytes']
    offsets = np.load(MANIFEST.parent/record['index_file'], allow_pickle=False, mmap_mode='r')
    energies, event_ids = [], []
    with h5py.File(path, 'r') as f:
        for part in manifest['splits']['test']:
            if part['source_key'] != cat:
                continue
            for start in range(part['event_start'], part['event_stop'], 4096):
                stop = min(start+4096, part['event_stop'])
                a, b = int(offsets[start]), int(offsets[stop])
                first = offsets[start:stop] - a
                nrows = np.diff(offsets[start:stop+1])
                e1, e2 = f['E1'][a:b], f['E2'][a:b]
                ids = np.arange(start, stop)
                assert np.array_equal(f['ev_no'][a:b], np.repeat(ids, nrows))
                assert np.array_equal(e1, np.repeat(e1[first], nrows))
                assert np.array_equal(e2, np.repeat(e2[first], nrows))
                assert f['label'][a].decode() == cat
                energies.append(e1[first].astype(float)+e2[first].astype(float))
                event_ids.append(ids)
    result, ids = np.concatenate(energies), np.concatenate(event_ids)
    assert result.size == manifest['counts']['energy']['test']['by_category'][cat]
    assert np.isfinite(result).all() and np.unique(ids).size == ids.size
    return result, ids


def write_csv(name, rows):
    with (ROOT/'figures'/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads(MANIFEST.read_text())
    values, ids = {}, {}
    for cat in ['0nubb', 'Bi214']:
        values[cat], ids[cat] = extract(manifest, cat)
        print(cat, len(values[cat]), flush=True)
    # Verify against the preceding independently extracted spectrum.
    previous = json.loads((ROOT.parent/'supernemo_0nu_comparison_20260904/supernemo_0nu_vs_bi214_summary.json').read_text())
    for cat in values:
        assert hashlib.sha256(values[cat].astype('<f8').tobytes()).hexdigest() == previous['classes'][cat]['event_energy_sha256']
    e = np.concatenate([values['0nubb'], values['Bi214']])
    y = np.concatenate([np.ones(len(values['0nubb']), dtype=int), np.zeros(len(values['Bi214']), dtype=int)])
    categories = np.where(y == 1, '0nubb', 'Bi214')
    a = np.ones(len(e))
    load_metric('utils')
    roc = load_metric('roc')
    dep = load_metric('dependence')
    dep.distance_correlation = lambda *args, **kwargs: float('nan')
    matched = roc.evaluate_energy_matched_roc(y, e, e, a, positive_label=1,
        n_bins=6, min_per_class=20, target='overlap', target_tpr=.9,
        support_trim_quantile=.005, n_bootstrap=0, random_state=42)
    coverage = min(matched.coverage.signal_matched_weight_fraction, matched.coverage.background_matched_weight_fraction)
    valid_bins = sum(b.valid for b in matched.bins)
    formal_valid = matched.status == 'ok' and coverage >= .5 and valid_bins >= 2
    common_mask = (e >= matched.common_support[0]) & (e <= matched.common_support[1])
    common_roc = roc.weighted_roc_curve(y[common_mask], e[common_mask], a[common_mask])
    # Check the exact pooled-AUC decomposition for the monotone energy score.
    # Matching uses left-closed bins; the final right edge is included.
    decomposition = .5
    for k, item in enumerate(matched.bins):
        if not item.valid:
            continue
        mask = common_mask & (e >= item.low) & ((e < item.high) if k < len(matched.bins)-1 else (e <= item.high))
        local_auc = roc.weighted_roc_curve(y[mask], e[mask], a[mask]).auc
        decomposition += item.target_mass**2 * (local_auc - .5)
        np.testing.assert_allclose(item.signal_matched_mass, item.background_matched_mass, rtol=0, atol=2e-12)
    np.testing.assert_allclose(decomposition, matched.matched_auc, rtol=0, atol=2e-12)
    independence = dep.evaluate_dependence(e, e, y, categories, a,
        n_energy_bins=8, n_score_bins=20, min_per_bin=20, threshold=None, seed=42)
    threshold = 2200.
    tpr, fpr = float(np.mean(values['0nubb'] >= threshold)), float(np.mean(values['Bi214'] >= threshold))
    threshold_result = dict(threshold_keV=threshold, rule='predict 0nubb if E >= 2200 keV',
        tpr=tpr, fpr=fpr, background_rejection=1-fpr, balanced_accuracy=(tpr+1-fpr)/2,
        true_positive=int(np.sum(values['0nubb'] >= threshold)), false_positive=int(np.sum(values['Bi214'] >= threshold)),
        n_signal=len(values['0nubb']), n_background=len(values['Bi214']), selected_by='user-specified illustration, not optimized on test')
    # Inclusive AUC is independently verified as a tie-aware pair probability.
    bg = np.sort(values['Bi214'])
    wins = np.searchsorted(bg, values['0nubb'], side='left')
    ties = np.searchsorted(bg, values['0nubb'], side='right') - wins
    direct_auc = float(np.mean((wins + .5*ties)/len(bg)))
    np.testing.assert_allclose(direct_auc, matched.inclusive_auc, rtol=0, atol=2e-14)
    # Store scalar diagnostics and complete bin metadata without huge ROC arrays.
    summary = matched.to_dict()
    for key in ['inclusive', 'matched']:
        if summary[key] is not None:
            summary[key] = {k:v for k,v in summary[key].items() if k not in ['fpr','tpr','thresholds']}
    summary.update(threshold=threshold_result, formal_matched_metric_valid=formal_valid,
        energy_independence_score=independence['overall_energy_independence_score'],
        independence_protocol={'n_energy_bins':8,'n_score_bins':20,'min_per_bin':20,
                               'threshold':None,'seed':42,'weights':'original unit event weights'},
        common_support_unweighted_auc=common_roc.auc,
        monotone_score_auc_decomposition=decomposition,
        diagnostic_score='s = E1 + E2, reconstructed keV; no trained model',
        purpose='0nubb/Bi214 motivation only; formal model benchmark remains 2nu/Bi214',
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        metric_source_sha256={name:hashlib.sha256((SOURCE/(name+'.py')).read_bytes()).hexdigest() for name in ['utils','roc','dependence']},
        dcor='skipped; not a headline metric',
        display_bins_keV=50, additional_energy_selection=None)
    (ROOT/'evidence/energy_shortcut_analysis.json').write_text(json.dumps(summary, indent=2)+'\n')

    edges = np.arange(0., 3300.+50, 50.)
    histogram_rows = []
    density = {}
    for cat, energy in values.items():
        counts = np.histogram(energy, bins=edges)[0]
        assert counts.sum() == energy.size
        density[cat] = counts/(len(energy)*50.)*1000.
        for k, n in enumerate(counts):
            histogram_rows.append(dict(category=cat, left_keV=edges[k], right_keV=edges[k+1],
                count=int(n), n_events=len(energy), density_per_keV=float(density[cat][k]/1000.)))
    write_csv('energy_shortcut_histograms.csv', histogram_rows)
    # Downsampling is for ROC rendering only; AUC uses every unique score.
    roc_rows = []
    for name, curve in [('inclusive', matched.inclusive), ('support_only', common_roc), ('matched', matched.matched)]:
        if curve is not None:
            for k in np.unique(np.linspace(0, len(curve.fpr)-1, min(2500, len(curve.fpr)), dtype=int)):
                roc_rows.append(dict(distribution=name, fpr=float(curve.fpr[k]), tpr=float(curve.tpr[k])))
    write_csv('energy_shortcut_roc.csv', roc_rows)
    cuts = np.arange(0., 3300.+10., 10.)
    efficiencies = {}
    for cat, energy in values.items():
        efficiencies[cat] = (len(energy)-np.searchsorted(np.sort(energy), cuts, side='left'))/len(energy)
    scan = [dict(threshold_keV=float(c), signal_efficiency=float(t), background_rejection=float(1-f),
                 balanced_accuracy=float((t+1-f)/2)) for c,t,f in zip(cuts,efficiencies['0nubb'],efficiencies['Bi214'])]
    write_csv('energy_threshold_scan.csv', scan)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#444444',
        'text.color':'#222222','axes.labelcolor':'#222222'})
    blue, orange, gray = '#286A9B', '#C47723', '#555555'
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), gridspec_kw={'width_ratios':[1.15,1]})
    ax = axes[0]
    for cat,label,color,style in [('0nubb',r'$0\nu\beta\beta$',blue,'-'),('Bi214',r'$^{214}$Bi',orange,'--')]:
        ax.stairs(density[cat], edges, label=label, color=color, linestyle=style, linewidth=1.6)
    ax.axvline(threshold, color=gray, linestyle=':', linewidth=1.2)
    ax.text(2150,2.85,'2200 keV',ha='right',va='top',fontsize=8,color=gray)
    ax.set(xlim=(0,3300),ylim=(0,3.1), xlabel=r'$E_1+E_2$ (keV)', ylabel=r'Density ($10^{-3}$ keV$^{-1}$)')
    ax.set_xticks([0,1000,2000,3000])
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    ax.set_title('(a) Reconstructed energy',loc='left',fontsize=10)
    ax = axes[1]
    ax.plot(matched.inclusive.fpr, matched.inclusive.tpr, color=blue, linewidth=1.7,
            label=f'Inclusive AUC = {matched.inclusive_auc:.3f}')
    ax.plot(common_roc.fpr,common_roc.tpr,color=gray,linestyle='-.',linewidth=1.1,
            label=f'Support only = {common_roc.auc:.3f}')
    if formal_valid:
        ax.plot(matched.matched.fpr,matched.matched.tpr,color=orange,linestyle='--',linewidth=1.6,
                label=f'Matched AUC = {matched.matched_auc:.3f}')
    ax.plot([0,1],[0,1],color='#999999',linestyle=':',linewidth=1)
    ax.scatter([fpr],[tpr],color=blue,edgecolors='white',s=30,zorder=5)
    ax.annotate('2200 keV cut',xy=(fpr,tpr),xytext=(.37,.74),fontsize=8,
                arrowprops={'arrowstyle':'-','color':gray,'linewidth':.8})
    ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Background acceptance (FPR)',ylabel='Signal efficiency (TPR)')
    ax.set_xticks([0,.5,1])
    ax.set_yticks([0,.5,1])
    ax.legend(frameon=False,loc='lower right',fontsize=7.7)
    ax.set_title(r'(b) Energy-only score $s=E_1+E_2$',loc='left',fontsize=10)
    for ax in axes:
        ax.grid(axis='y',color='#e6e6e6',linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle(r'SuperNEMO $0\nu\beta\beta$ vs $^{214}$Bi: energy-only diagnostic', x=.085,y=.985,ha='left',fontsize=11)
    fig.text(.085,.883,'196,608 signal / 262,144 background test events; 50 keV display bins; unit area per class',fontsize=7.7,color=gray)
    fig.subplots_adjust(left=.085,right=.99,bottom=.2,top=.80,wspace=.37)
    for suffix in ['pdf','png']:
        fig.savefig(ROOT/'figures'/('energy_shortcut_motivation.'+suffix),dpi=240,bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9,3.9))
    ax.plot(cuts,efficiencies['0nubb'],color=blue,label=r'Signal efficiency ($0\nu\beta\beta$)',linewidth=1.6)
    ax.plot(cuts,1-efficiencies['Bi214'],color=orange,linestyle='--',label=r'Background rejection ($^{214}$Bi)',linewidth=1.6)
    ax.axvline(threshold,color=gray,linestyle=':',linewidth=1.2)
    ax.scatter([threshold,threshold],[tpr,1-fpr],s=25,color=[blue,orange],edgecolors='white',zorder=5)
    ax.text(100,.82,f'At 2200 keV:\nSignal efficiency = {100*tpr:.1f}%\nBackground rejection = {100*(1-fpr):.1f}%',fontsize=9,va='top')
    ax.set(xlim=(0,3300),ylim=(0,1.03),xlabel=r'Energy threshold $t$ (keV), predict signal if $E\geq t$',ylabel='Event fraction')
    fig.legend(*ax.get_legend_handles_labels(),frameon=False,fontsize=8.1,
               loc='lower left',bbox_to_anchor=(.105,.795),ncol=2,columnspacing=1.)
    ax.grid(axis='y',color='#e6e6e6',linewidth=.5)
    ax.set_axisbelow(True)
    fig.suptitle(r'SuperNEMO $0\nu\beta\beta$ vs $^{214}$Bi: threshold trade-off',x=.12,y=.99,ha='left',fontsize=11)
    fig.text(.12,.91,'Same test events; 2200 keV is illustrative, not optimized on the test set',fontsize=8,color=gray)
    fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.77)
    for suffix in ['pdf','png']:
        fig.savefig(ROOT/'figures'/('energy_threshold_tradeoff.'+suffix),dpi=240,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'threshold':threshold_result,'auc':direct_auc,'matched_auc':matched.matched_auc,
        'coverage':matched.coverage.to_dict(),'valid_bins':valid_bins,'formal_valid':formal_valid,
        'common_support_auc':common_roc.auc,
        'independence':summary['energy_independence_score']},indent=2))


if __name__ == '__main__':
    main()
