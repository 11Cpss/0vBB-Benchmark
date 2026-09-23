"""Read scalar test metadata only; all generated files remain in this branch."""
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
import json
import numpy as np
import h5py
DATA = Path('/home/klz/Data/zeronu_benchmark/MJD')
fields = ['energy_label','id','detector','run_number','tp0',
          'psd_label_low_avse','psd_label_high_avse','psd_label_dcr','psd_label_lq']
parts = {key: [] for key in fields + ['shard','row']}
files = sorted(DATA.glob('MJD_Test_*.hdf5'))
for shard, path in enumerate(files):
    with h5py.File(path, 'r') as f:
        for key in fields:
            parts[key].append(f[key][:])
        parts['shard'].append(np.full(len(f['id']), shard, dtype=np.int16))
        parts['row'].append(np.arange(len(f['id']), dtype=np.int32))
arrays = {key: np.concatenate(value) for key, value in parts.items()}
arrays['clean'] = np.all(np.column_stack([arrays[key] == 1 for key in fields[5:]]), axis=1)
np.savez_compressed(ROOT / 'evidence/test_scalar_metadata.npz', **arrays)
detectors, counts = np.unique(arrays['detector'][arrays['clean']], return_counts=True)
order = np.lexsort((detectors,-counts))
most_populous_detector = int(detectors[order[0]])
pairs, pair_counts = np.unique(np.c_[arrays['detector'][arrays['clean']],arrays['run_number'][arrays['clean']]],axis=0,return_counts=True)
pair_order = np.lexsort((pairs[:,1],pairs[:,0],-pair_counts))
detector, run = map(int,pairs[pair_order[0]])
mask = arrays['clean'] & (arrays['detector'] == detector) & (arrays['run_number'] == run) & np.isfinite(arrays['energy_label'])
q = np.quantile(arrays['energy_label'][mask],[0,.25,.5,.75,1])
summary = {
 'source_files': [str(p) for p in files], 'events':len(arrays['id']),
 'clean_events':int(arrays['clean'].sum()), 'selected_detector':detector, 'selected_run':run,
 'most_populous_detector_if_run_ignored':most_populous_detector,
 'detector_run_selection':'Most clean test events per (detector,run); smaller detector and run IDs break ties, determined without waveform inspection.',
 'selected_detector_run_clean_events':int(mask.sum()), 'energy_quartiles_keV':q.tolist(),
 'detectors_clean_counts':dict(zip(map(str,detectors.tolist()),counts.tolist())),
 'waveform_cohort_rule':'Bottom and top energy quartiles within clean events of the selected detector and run; no shape-based selection.',
 'run_count':int(len(np.unique(arrays['run_number'][mask])))
}
(ROOT / 'evidence/test_metadata_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary,indent=2))
