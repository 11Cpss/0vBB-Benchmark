#!/usr/bin/env python3
"""Plot audited raw-tracker extent versus energy; no model execution."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = Path('/home/wenyu/iclr final paper').resolve()
assert ROOT.is_relative_to(ALLOWED)
for sub in ['tmp', 'cache', 'mpl']:
    (ROOT / '.runtime' / sub).mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(ROOT / '.runtime/tmp'),
                  XDG_CACHE_HOME=str(ROOT / '.runtime/cache'),
                  MPLCONFIGDIR=str(ROOT / '.runtime/mpl'),
                  PYTHONDONTWRITEBYTECODE='1')
import hashlib
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import ScalarFormatter

SOURCE = ALLOWED / 'wing_huang_revision4_20260904/evidence/supernemo_geometry_event_descriptors_LOCAL_ONLY.csv.gz'
AUDIT = SOURCE.parent / 'geometry_energy.json'
CATS = [('Bi214', r'$^{214}$Bi'), ('0nubb', r'$0\nu\beta\beta$')]
BLUE, ORANGE = '#2474A5', '#D5772B'
E_EDGES = np.arange(0., 3300., 100.)
R_EDGES = np.arange(0., 2240., 20.)
MIN_COUNT = 20

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(x):
    return dict(n=len(x), minimum=float(x.min()), maximum=float(x.max()),
                median=float(np.median(x)), p25=float(np.quantile(x, .25)),
                p75=float(np.quantile(x, .75)))

def main():
    audit = json.loads(AUDIT.read_text())
    df = pd.read_csv(SOURCE)
    required = ['energy_keV', 'radius_gyration_mm', 'anisotropy', 'n_tracker_hits']
    assert np.isfinite(df[required].to_numpy()).all()
    assert not df[['category', 'event_ordinal']].duplicated().any()
    out = {'source': str(SOURCE), 'source_sha256': sha(SOURCE),
           'audit_source': str(AUDIT), 'audit_source_sha256': sha(AUDIT),
           'manifest': audit['manifest'], 'manifest_sha256': audit['manifest_sha256'],
           'energy_definition': audit['energy_definition'],
           'radius_gyration_definition': audit['descriptor_definitions']['radius_gyration_mm'],
           'covariance_definition': audit['descriptor_definitions']['covariance'],
           'split': 'test', 'energy_bin_width_keV': 100,
           'radius_bin_width_mm': 20,
           'energy_edges_keV': E_EDGES.tolist(), 'radius_edges_mm': R_EDGES.tolist(),
           'minimum_events_per_energy_column': MIN_COUNT,
           'heatmap_normalization': 'count / all finite events in each energy bin; displayed Rg range 0 to 800 mm',
           'heatmap_color_scale': 'shared linear probability per 20 mm bin',
           'quartile_definition': 'Sort each class by energy then event ordinal; take first N//4 and last N//4 events. Middle half excluded only from ECDF comparison.',
           'cdf': 'Exact empirical CDF, every event retained; log extent axis includes full support.',
           'classes': {}}
    hist_rows, cdf_rows = [], []
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.labelsize':10.5, 'axes.titlesize':11,
                         'pdf.fonttype':42, 'ps.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.edgecolor':'#555555', 'text.color':'#242424'})
    fig = plt.figure(figsize=(10.6, 8.2))
    gs = fig.add_gridspec(2, 2, left=.095, right=.88, top=.85, bottom=.20,
                          hspace=.50, wspace=.26, height_ratios=[1., .9])
    cmap = LinearSegmentedColormap.from_list('tracker_blue', ['#FFFFFF','#D4E5EF','#7AB1CF','#2474A5','#14415F'])
    cmap.set_bad('#EDEDED')
    norm = Normalize(0, .13)
    for col, (cat, title) in enumerate(CATS):
        sub = df[df.category.eq(cat)].sort_values('event_ordinal')
        e = sub.energy_keV.to_numpy(dtype=np.float64)
        r = sub.radius_gyration_mm.to_numpy(dtype=np.float64)
        count_hash = hashlib.sha256(sub.n_tracker_hits.to_numpy(dtype=np.int64).tobytes()).hexdigest()
        energy_hash = hashlib.sha256(e.tobytes()).hexdigest()
        assert len(sub) == audit['checks'][cat]['n_events']
        assert energy_hash == audit['checks'][cat]['energy_sha256']
        assert count_hash == audit['checks'][cat]['n_tracker_hits_sha256']
        h, _, _ = np.histogram2d(e, r, bins=[E_EDGES, R_EDGES])
        n = np.histogram(e, E_EDGES)[0]
        assert h.sum() == len(sub)
        assert np.array_equal(h.sum(axis=1).astype(int), n)
        p = np.divide(h, n[:, None], out=np.zeros_like(h), where=n[:, None]>0)
        assert np.allclose(p[n>0].sum(axis=1), 1)
        supported = n >= MIN_COUNT
        masked = np.ma.masked_array(p.T, mask=np.broadcast_to(~supported, p.T.shape))
        med = np.array([np.median(r[(e>=lo) & (e<hi)]) if nn>=MIN_COUNT else np.nan
                        for lo,hi,nn in zip(E_EDGES[:-1],E_EDGES[1:],n)])
        ranks = sub.sort_values(['energy_keV', 'event_ordinal'])
        size = len(ranks)//4
        low, high = ranks.iloc[:size], ranks.iloc[-size:]
        cs = {'n_events':len(sub), 'energy_sha256':energy_hash, 'n_tracker_hits_sha256':count_hash,
              'energy': stats(e), 'radius':stats(r),
              'n_missing':0, 'n_heatmap_display_tail_above_800mm':int((r>800).sum()),
              'heatmap_display_tail_fraction':float((r>800).mean()),
              'n_energy_columns_below_minimum':int((~supported).sum()),
              'n_events_in_unsupported_energy_columns':int(n[~supported].sum()),
              'energy_bin_counts':n.tolist(), 'cohorts':{}}
        ax = fig.add_subplot(gs[0,col])
        mesh=ax.pcolormesh(E_EDGES,R_EDGES,masked,cmap=cmap,norm=norm,rasterized=True,shading='flat')
        ax.plot((E_EDGES[:-1]+E_EDGES[1:])/2,med,color='#242424',lw=1.6,label='Bin median')
        ax.set(xlim=(300,3200),ylim=(0,800),xlabel='Reconstructed event energy (keV)',
               ylabel=r'Tracker spatial extent $R_g$ (mm)' if col==0 else '',
               title=rf'{title}: all {len(sub):,} test events')
        ax.set_xticks([500,1000,1500,2000,2500,3000])
        ax.set_yticks([0,200,400,600,800])
        ax.text(.035,.95,'Conditional extent distribution',transform=ax.transAxes,
                va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
        ax.legend(loc='upper right',fontsize=8,frameon=False)
        ec = fig.add_subplot(gs[1,col])
        for name, cohort, color, ls in [('low',low,BLUE,'-'),('high',high,ORANGE,'--')]:
            rr = np.sort(cohort.radius_gyration_mm.to_numpy())
            ee = cohort.energy_keV.to_numpy()
            cs['cohorts'][name] = {'energy':stats(ee), 'radius':stats(rr),
                'boundary_energy_ties_in_full_class':int((e==(ee.max() if name=='low' else ee.min())).sum()),
                'n_above_800mm':int((rr>800).sum())}
            xs, multiplicity = np.unique(rr, return_counts=True)
            ys = np.cumsum(multiplicity) / len(rr)
            # Every distinct extent is retained: the plotted ECDF is not sampled.
            ec.step(xs,ys,where='post',color=color,ls=ls,lw=1.9,
                    label=f'{"Lowest" if name=="low" else "Highest"} energy quartile: {ee.min():.0f}–{ee.max():.0f} keV')
            for xx,yy in zip(xs,ys):
                cdf_rows.append([cat,name,float(xx),float(yy),len(rr)])
        median_low = cs['cohorts']['low']['radius']['median']
        median_high = cs['cohorts']['high']['radius']['median']
        cs['high_minus_low_median_mm'] = median_high-median_low
        cs['high_vs_low_median_percent_change'] = (median_high/median_low-1)*100
        ec.axhline(.5,color='#BBBBBB',lw=.7,zorder=0)
        ec.set(xlim=(60,2250),ylim=(0,1.015),xlabel=r'Tracker spatial extent $R_g$ (mm; log scale)',
               ylabel='Fraction of events at or below extent' if col==0 else '',
               title=f'Equal-sized energy cohorts: {size:,} events each',xscale='log')
        ec.set_xticks([100,200,300,500,1000,2000])
        ec.xaxis.set_major_formatter(ScalarFormatter())
        ec.minorticks_off()
        ec.legend(loc='upper left',bbox_to_anchor=(-.015,-.205),fontsize=8.2,frameon=False,
                  handlelength=2.4, borderaxespad=0)
        ec.text(.96,.08,f'Median extent: {median_low:.0f} → {median_high:.0f} mm',
                transform=ec.transAxes,va='bottom',ha='right',fontsize=9.5)
        for k in range(len(n)):
            for j in range(len(R_EDGES)-1):
                hist_rows.append([cat,float(E_EDGES[k]),float(E_EDGES[k+1]),float(R_EDGES[j]),
                                  float(R_EDGES[j+1]),int(h[k,j]),int(n[k]),float(p[k,j]),bool(supported[k])])
        out['classes'][cat] = cs
    heatmap_pos = ax.get_position()
    cax = fig.add_axes([.903,heatmap_pos.y0,.016,heatmap_pos.height])
    cb = fig.colorbar(mesh,cax=cax)
    cb.set_label('Conditional probability per 20 mm bin',fontsize=9)
    cb.set_ticks([0,.04,.08,.12])
    fig.suptitle('SuperNEMO tracker extent versus reconstructed energy',x=.095,y=.978,ha='left',fontsize=15)
    fig.text(.095,.935,r"Raw hit coordinates only: $R_g$ is the RMS distance from an event's hit-cloud centroid.",fontsize=10)
    fig.text(.095,.902,'Top: within-energy-bin probabilities. Bottom: full extent distributions, grouped by energy alone.',fontsize=10)
    t1=out['classes']['Bi214']['heatmap_display_tail_fraction']*100
    t2=out['classes']['0nubb']['heatmap_display_tail_fraction']*100
    fig.text(.095,.017,f'Heatmaps: 100 keV × 20 mm bins; gray columns have fewer than 20 events. Extents above 800 mm\n'
             f'({t1:.2f}% of Bi-214; {t2:.2f}% of 0νββ) are outside the heatmap view, but included in normalization and both CDFs.',
             fontsize=8.5,linespacing=1.4)
    for ext in ['pdf','png']:
        fig.savefig(ROOT / 'figures' / f'supernemo_extent_energy_population.{ext}',dpi=210)
    plt.close(fig)
    with (ROOT/'figures/supernemo_conditional_extent_histogram.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['category','energy_left_keV','energy_right_keV','radius_left_mm','radius_right_mm',
                                   'count','n_in_energy_bin','probability_given_energy_bin','display_supported']);w.writerows(hist_rows)
    # ECDF values are event-derived detail, therefore remain local only in evidence/.
    import gzip
    with gzip.open(ROOT/'evidence/supernemo_cohort_ecdf_LOCAL_ONLY.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(['category','energy_cohort','radius_gyration_mm','ecdf','cohort_n']);w.writerows(cdf_rows)
    (ROOT/'evidence/supernemo_population_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__ == '__main__':
    main()
