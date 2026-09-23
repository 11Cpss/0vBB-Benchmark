#!/usr/bin/env python3
"""Verify every canonical NEXT test event against original HDF5 and ordinal split slices."""
from pathlib import Path
import json,hashlib,h5py,numpy as np
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/wenyu/summer/03_training_runs/energybench_campaigns/20260808_energybench_rewrite_v2')
DATA=Path('/home/klz/Data/zeronu_benchmark/NEXT')
def main():
 manifest=json.loads((ROOT/'event_split.json').read_text())
 p=ROOT/'reevaluation_20260818/cnn_004_multiview_late_fusion/classification/predictions.npz'
 with np.load(p) as z:pred={k:z[k] for k in ['event_id','energy','label','category']}
 lookup={v:i for i,v in enumerate(pred['event_id'])};seen=set();records=[];worst=0.
 for j,spec in enumerate(manifest['splits']['test']):
  source=DATA/spec['relative_path']
  with h5py.File(source,'r') as f:rows=f['MC/hits/table'][:]
  ids=rows['values_block_0'][:,0];energy=rows['values_block_1'][:,3]
  starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1];ends=np.r_[starts[1:],len(ids)]
  selection=range(spec['event_start'],spec['event_stop']);n=0;metadata_hash=hashlib.sha256()
  for k in selection:
   a,b=starts[k],ends[k];key=f"NEXT::{spec['relative_path']}::{ids[a]}";assert key not in seen;seen.add(key);i=lookup[key]
   e=float(np.sum(energy[a:b],dtype=np.float64));error=abs(e-pred['energy'][i]);worst=max(worst,error)
   assert error==0.,(key,e,pred['energy'][i]);assert pred['label'][i]==spec['label'];assert pred['category'][i]==spec['category']
   expected=b'Signal' if spec['label']==1 else b'Bkg';assert np.all(rows['values_block_2'][a:b,0]==expected)
   metadata_hash.update(key.encode());metadata_hash.update(np.float64(e).tobytes());n+=1
  assert n==spec['event_count']
  records.append({'path':str(source),'size':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns,'event_start':spec['event_start'],'event_stop':spec['event_stop'],'n_test':n,'test_id_energy_sha256':metadata_hash.hexdigest()})
  if (j+1)%100==0:print('verified raw source files',j+1,'events',len(seen),flush=True)
 assert seen==set(lookup)
 result={'n_test_events':len(seen),'n_source_files':len(records),'all_event_ids_match_official_split':True,'all_physical_energy_sums_exactly_match_saved_predictions':True,'all_raw_labels_match':True,'max_energy_difference_MeV':worst,'source_files':records}
 (OUT/'next_raw_event_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(seen),len(records),flush=True)
if __name__=='__main__':main()
