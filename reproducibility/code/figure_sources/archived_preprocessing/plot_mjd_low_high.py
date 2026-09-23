"""Controlled MJD waveform and residual-energy diagnostic; no training/inference.

Chart contract: show what energy changes in actual same-class/same-detector/
same-run waveforms, before and after the current classification preprocessing.
Use static 2x2 waveform panels and two population scatter plots. Blue/orange
encode predeclared bottom/top energy quartiles; middle energies are neutral.
All 410 waveforms in the most populous clean detector/run are used; examples
are nearest cohort energy medians, chosen without waveform inspection.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
ALLOWED = Path('/home/wenyu/iclr final paper').resolve()
if not ROOT.is_relative_to(ALLOWED):
    raise RuntimeError('All generated files must remain inside the authorized paper directory.')
for key, sub in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'mpl'}.items():
    path = ROOT / '.runtime' / sub
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['TMP'] = os.environ['TMPDIR']
os.environ['TEMP'] = os.environ['TMPDIR']
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import csv
import json
import hashlib
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA = Path('/home/klz/Data/zeronu_benchmark/MJD')
with np.load(ROOT / 'evidence/test_scalar_metadata.npz') as packed:
    meta = {key: packed[key] for key in packed.files}
clean = meta['clean'] & np.isfinite(meta['energy_label'])
pairs, counts = np.unique(np.c_[meta['detector'][clean], meta['run_number'][clean]], axis=0, return_counts=True)
order = np.lexsort((pairs[:,1], pairs[:,0], -counts))
detector, run = map(int, pairs[order[0]])
indices = np.flatnonzero(clean & (meta['detector'] == detector) & (meta['run_number'] == run))
energy = meta['energy_label'][indices]
q25, q75 = np.quantile(energy, [.25,.75])
low, high = energy <= q25, energy >= q75
cohort = np.where(low, 'Low energy', np.where(high, 'High energy', 'Middle energy'))
waves = np.empty((len(indices), 3800),dtype=np.float32)
for shard in np.unique(meta['shard'][indices]):
    local = np.flatnonzero(meta['shard'][indices] == shard)
    rows = meta['row'][indices[local]]
    assert np.all(np.diff(rows) > 0)
    with h5py.File(DATA / f'MJD_Test_{shard}.hdf5','r') as f:
        waves[local] = np.asarray(f['raw_waveform'][rows], dtype=np.float32)
baseline = waves[:,:200].mean(axis=1,dtype=np.float64).astype(np.float32)
centered = np.asarray(waves-baseline[:,None],dtype=np.float32)
amplitude = np.max(np.abs(centered),axis=1)
normalized = centered.copy()
np.divide(normalized, amplitude[:,None], out=normalized, where=amplitude[:,None] > 0)
rms = np.sqrt(np.mean(centered[:,:200].astype(np.float64)**2,axis=1))
relative_rms = np.sqrt(np.mean(normalized[:,:200].astype(np.float64)**2,axis=1))
assert np.all(np.isfinite(waves)) and np.all(amplitude > 0)
assert np.allclose(relative_rms, rms/amplitude, rtol=1e-6)

# Deterministic examples, nearest each cohort median energy, tie by ID then row.
examples = []
for mask in (low, high):
    members = np.flatnonzero(mask)
    order = np.lexsort((meta['row'][indices[members]], meta['id'][indices[members]],
                        np.abs(energy[members] - np.median(energy[members]))))
    examples.append(int(members[order[0]]))
example_records = []
for j in examples:
    k = indices[j]
    example_records.append({
        'cohort':str(cohort[j]),'shard':int(meta['shard'][k]),'row':int(meta['row'][k]),
        'id':int(meta['id'][k]), 'detector':detector, 'run_number':run,
        'tp0':int(meta['tp0'][k]),'energy_keV':float(energy[j]),
        'baseline_adc':float(baseline[j]),'max_abs_amplitude_adc':float(amplitude[j]),
        'baseline_rms_adc':float(rms[j]),'normalized_baseline_rms':float(relative_rms[j])})
np.savez_compressed(ROOT/'evidence/controlled_waveforms.npz', global_test_index=indices,
                   energy_keV=energy, waveform_float32=waves, centered=centered, normalized=normalized)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.facecolor':'white'})
BLUE, ORANGE, MID = '#256C9C', '#D57627', '#92989F'
colors = [BLUE, ORANGE]
sample = np.arange(waves.shape[1])
fig, axs = plt.subplots(2,2,figsize=(10.3,7.2),sharex=True,sharey='row')
fig.subplots_adjust(left=.10,right=.985,bottom=.125,top=.82,hspace=.31,wspace=.14)
fig.suptitle('MJD low- and high-energy waveforms',x=.10,y=.975,ha='left',fontsize=16,weight='bold')
fig.text(.10,.928,f'Clean test events only | Detector {detector}, run {run} | One event near each cohort median energy',fontsize=10)
fig.text(.10,.892,f'Low cohort: {energy[low].min():.1f}–{q25:.1f} keV (n={low.sum()}); high cohort: {q75:.1f}–{energy[high].max():.1f} keV (n={high.sum()})',fontsize=10)
for col,j in enumerate(examples):
    title=f'{cohort[j]}: {energy[j]:.1f} keV'
    for row, values in enumerate([centered[j], normalized[j]]):
        ax=axs[row,col]
        ax.plot(sample,values,color=colors[col],lw=1)
        ax.axhline(0,color='#6E747B',lw=.6,zorder=0)
        ax.set_xlim(0,3799)
        ax.grid(axis='y',color='#E5E8EB',lw=.6)
        ax.set_title(title if row==0 else 'Actual amplitude-normalized input',loc='left')
    axs[0,col].text(.03,.85,f'Event ID {meta["id"][indices[j]]}',transform=axs[0,col].transAxes,fontsize=9)
    # The same original samples, expanded vertically and horizontally.
    inset=axs[1,col].inset_axes([.53,.16,.43,.30])
    inset.plot(sample[:200],normalized[j,:200],color=colors[col],lw=.75)
    inset.axhline(0,color='#9B9B9B',lw=.5)
    inset.set_xlim(0,199)
    limit=max(np.abs(normalized[examples,:200]).max()*1.15,.002)
    inset.set_ylim(-limit,limit)
    inset.set_xticks([0,100,199]); inset.tick_params(labelsize=7,pad=1)
    inset.set_title('Baseline zoom: first 200 samples',fontsize=8,pad=3)
axs[0,0].set_ylabel('Baseline-subtracted amplitude\n(ADC counts)')
axs[1,0].set_ylabel('Normalized amplitude\n(dimensionless)')
for ax in axs[-1]: ax.set_xlabel('Sample index')
axs[1,0].set_ylim(-.065,1.08)
fig.text(.10,.045,'Current preprocessing: subtract the mean of samples 0–199, then divide each event by its maximum absolute amplitude.',fontsize=9)
fig.text(.10,.023,'Axes share limits within each row; baseline insets also share limits. Time sampling interval is not assumed.',fontsize=9,color='#50555A')
for suffix in ('pdf','png'): fig.savefig(ROOT/f'figures/mjd_low_high_waveforms.{suffix}',dpi=220)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10.3,4.85))
fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.77,wspace=.30)
fig.suptitle('MJD waveform amplitude and residual noise versus energy',x=.09,y=.975,ha='left',fontsize=15,weight='bold')
fig.text(.09,.914,f'All {len(indices)} clean test events in detector {detector}, run {run}; no waveform-shape selection',fontsize=10)
fig.text(.09,.872,'Blue: lowest energy quartile | Gray: middle half | Orange: highest energy quartile',fontsize=9.5)
for ax,values,title,ylabel in zip(axs,[amplitude,relative_rms],
 ['(a) Before amplitude normalization','(b) After amplitude normalization'],
 ['Maximum absolute amplitude (ADC counts)','Baseline RMS / maximum amplitude']):
    for mask,color,marker in [(~(low|high),MID,'o'),(low,BLUE,'o'),(high,ORANGE,'^')]:
        ax.scatter(energy[mask],values[mask],c=color,s=16,marker=marker,alpha=.65,linewidths=0,rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Calibrated energy (keV)'); ax.set_ylabel(ylabel)
    ax.set_title(title,loc='left')
    ax.grid(which='major',color='#E5E8EB',lw=.6)
    corr=spearmanr(energy,values).statistic
    xpos, align = (.04, 'left') if ax is axs[0] else (.97, 'right')
    ax.text(xpos,.95,rf'Spearman $\rho$ = {corr:.3f}',transform=ax.transAxes,ha=align,va='top',fontsize=10)
fig.text(.09,.085,'The leading pulse amplitude is removed by preprocessing, but relative baseline noise can still encode energy.',fontsize=9.5)
fig.text(.09,.045,'RMS uses samples 0–199. This descriptive proxy is not a classifier score or proof that a trained model uses energy.',fontsize=9,color='#50555A')
for suffix in ('pdf','png'): fig.savefig(ROOT/f'figures/mjd_energy_noise_relationship.{suffix}',dpi=220)
plt.close(fig)

with (ROOT/'figures/mjd_waveform_population.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['global_test_index','shard','row','event_id','detector','run_number','cohort','energy_keV','baseline_mean_adc','max_abs_amplitude_adc','baseline_rms_adc','normalized_baseline_rms'])
    for j,k in enumerate(indices):
        w.writerow([int(k),int(meta['shard'][k]),int(meta['row'][k]),int(meta['id'][k]),detector,run,cohort[j],energy[j],baseline[j],amplitude[j],rms[j],relative_rms[j]])
with (ROOT/'figures/mjd_example_waveforms.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['cohort','event_id','energy_keV','sample_index','baseline_subtracted_adc','normalized_amplitude'])
    for j in examples:
        for t in sample:w.writerow([cohort[j],int(meta['id'][indices[j]]),energy[j],int(t),centered[j,t],normalized[j,t]])
summary={
 'selection': 'Most populous (detector,run) among clean test events. Ties resolve by smaller detector, then smaller run. Bottom and top energy quartiles selected without waveform inspection. Example nearest each cohort median energy; ties by event ID then row.',
 'detector':detector,'run_number':run,'events':len(indices),'energy_quartiles_keV':[float(q25),float(q75)],
 'low_events':int(low.sum()),'high_events':int(high.sum()),'example_events':example_records,
 'spearman_energy_max_amplitude':float(spearmanr(energy,amplitude).statistic),
 'spearman_energy_absolute_baseline_rms':float(spearmanr(energy,rms).statistic),
 'spearman_energy_normalized_baseline_rms':float(spearmanr(energy,relative_rms).statistic),
 'low_median_relative_baseline_rms':float(np.median(relative_rms[low])),
 'high_median_relative_baseline_rms':float(np.median(relative_rms[high])),
 'low_median_absolute_baseline_rms_adc':float(np.median(rms[low])),
 'high_median_absolute_baseline_rms_adc':float(np.median(rms[high])),
 'low_high_relative_noise_ratio':float(np.median(relative_rms[low])/np.median(relative_rms[high])),
 'zero_amplitude_events':int(np.sum(amplitude==0)), 'finite_waveforms':bool(np.isfinite(waves).all()),
 'global_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
 'classification_preprocessing':{'source':'/home/wenyu/MJD/mjdbench/data.py:314-326','baseline_samples':200,'arithmetic':'Read float32; first200 mean in float64 then castfloat32; subtract float32; divide maxabsolute per waveform where scale>0.'},
 'units':{'energy':'keV; current README and official release','waveform':'ADC counts; official release','time':'sample index; no sampling interval assumed'},
 'interpretation':'Illustrates energy information surviving per-event amplitude normalization as relative baseline noise in a controlled cohort. Does not establish trained model dependence, causality, a universal detector effect, or expected AUC improvement. Formal classification is clean versus non-clean, not low versus high energy.'
}
(ROOT/'evidence/mjd_waveform_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
