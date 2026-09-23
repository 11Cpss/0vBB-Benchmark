#!/usr/bin/env python3
"""Direct, independent raw scalar/event and original prediction checks."""
from pathlib import Path
import hashlib,json
import h5py,numpy as np
A=Path(__file__).resolve().parent;C=A.parent;R=C.parent

def read(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(x,y):return bool(np.array_equal(x,y))
exo=json.loads((C/'exo/manifest.json').read_text())['records']
mjd=json.loads((C/'mjd/manifest.json').read_text())['ready']
checks=[];raw=[]
# Recover raw EXO physics by independent file/event-number join, using no loader.
a=read(exo[0]['standardized_file']);split=[x.split('::') for x in a['event_id']];files=np.array([x[1] for x in split]);ids=np.array([int(x[2]) for x in split]);root=Path('/home/klz/Data/zeronu_benchmark/EXO-200')
for file in np.unique(files):
 ix=np.flatnonzero(files==file)
 with h5py.File(root/file,'r') as h:
  ev=np.asarray(h['event_number']).ravel();en=np.asarray(h['Rotated_energy']).ravel();charge=np.asarray(h['Charge_cluster_number']).ravel()
  assert len(np.unique(ev))==len(ev)
  lookup={int(v):j for j,v in enumerate(ev)};pos=np.array([lookup[int(v)] for v in ids[ix]])
  assert same(en[pos],a['energy_keV'][ix]);assert same((charge[pos]==1).astype('int8'),a['label'][ix])
  raw.append({'dataset':'EXO-200','path':str(root/file),'n_saved_events_checked':len(ix),'event_number_sha256':ash(ev),'Rotated_energy_sha256':ash(en),'charge_cluster_number_sha256':ash(charge),'all_pass':True})
# Recover all six MJD raw shard scalar arrays in explicit Test0..5 order.
a=read(mjd[0]['standardized_predictions']);root=Path('/home/klz/Data/zeronu_benchmark/MJD');start=0
for i in range(6):
 path=root/('MJD_Test_'+str(i)+'.hdf5')
 with h5py.File(path,'r') as h:
  en=np.asarray(h['energy_label']);rid=np.asarray(h['id']);flags=[np.asarray(h[k]) for k in ['psd_label_low_avse','psd_label_high_avse','psd_label_dcr','psd_label_lq']];label=np.logical_and.reduce(flags).astype('int8')
  n=len(en);end=start+n;event=np.array([path.name+'::row='+str(j)+'::id='+str(int(rid[j])) for j in range(n)])
  assert same(a['energy_keV'][start:end],en);assert same(a['label'][start:end],label);assert same(a['event_id'][start:end],event)
  raw.append({'dataset':'MJD','path':str(path),'n_saved_events_checked':n,'raw_energy_sha256':ash(en),'id_sha256':ash(rid),'four_PSD_flags_sha256':[ash(x) for x in flags],'all_pass':True});start=end
assert start==390000==len(a['label'])
classic=read(R/'sources/next_mjd/MJD_official_test_metadata.npz')
assert same(a['energy_keV'],classic['energy_label']) and same(a['label'],classic['label']) and same(a['event_id'],classic['physical_event_id'])
for r in exo:
 a=read(r['standardized_file']);o=read(r['prediction_file']);native=read(r['native_prediction_file']);cl=read(r['test_split']['classic_prediction'])
 c={'source_hash':sha(r['prediction_file'])==r['prediction_sha256'],'native_source_hash':sha(r['native_prediction_file'])==r['native_prediction_sha256'],'event_identity':same(a['event_id'],o['event_id']),'label_flip':same(a['label'],1-o['label']),'score_negation_no_sigmoid':same(a['score'],-o['score']),'raw_keV_unchanged':same(a['energy_keV'],o['energy_condition']),'native_score_identical':same(o['score'],native['prediction']),'native_label_identical':same(o['label'],native['target']),'unit_weights':same(a['weight'],o['sample_weight']) and bool(np.all(a['weight']==1)),'true_physical_groups':same(a['group'],o['category']),'split_is_test':set(o['split'])=={'test'},'classic_complete_metadata_match':all(same(o[k],cl[k]) for k in ['event_id','label','category','energy_condition','sample_weight','group_id','split'])}
 checks.append({'dataset':'EXO-200','model_key':r['model_key'],'all_pass':all(c.values()),'checks':c})
for r in mjd:
 a=read(r['standardized_predictions']);o=read(r['source_predictions']);native_path=Path(r['source_predictions']).parents[1]/'predictions.npz';native=read(native_path)
 c={'all_declared_source_hashes':all(sha(p)==h for p,h in r['source_files'].items()),'score_unchanged_no_sigmoid':same(a['score'],o['score']),'labels_unchanged':same(a['label'],o['label']),'native_score_identical':same(o['score'],native['prediction']),'native_label_identical':same(o['label'],native['target']),'cached_float32_energy_equals_cast_original':same(o['energy_kev'],a['energy_keV'].astype('float32').astype('float64')),'cached_MeV_matches_source_float32':same(o['energy_mev'],a['energy_keV'].astype('float32').astype('float64')/1000),'classic_original_energy_equal':same(a['energy_keV'],classic['energy_label']),'classic_labels_equal':same(a['label'],classic['label']),'classic_event_ids_equal':same(a['event_id'],classic['physical_event_id']),'unit_weights':bool(np.all(a['weight']==1)),'true_groups':same(a['group'],np.where(a['label']==1,'clean','nonclean')),'overflow_not_clipped':int(np.sum(a['energy_keV']>3000))==67}
 checks.append({'dataset':'MJD','model_key':r['model_key'],'all_pass':all(c.values()),'checks':c})
result={'all_pass':len(checks)==15 and all(x['all_pass'] for x in checks),'n_models':len(checks),'method':'Independent reads of all raw EXO test file/event IDs and MJD official shard/row/ID scalar fields; exact comparisons to original exported and native prediction arrays, original labels/scores/energy, classic event metadata, and source hashes. No dataset loader or metric function imported.','raw_checks':raw,'model_checks':checks}
(A/'independent_sources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'all_pass':result['all_pass'],'n_models':len(checks),'n_raw_events':sum(x['n_saved_events_checked'] for x in raw),'failures':[x for x in checks if not x['all_pass']]},indent=2));raise SystemExit(0 if result['all_pass'] else 1)
