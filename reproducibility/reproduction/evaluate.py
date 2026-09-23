#!/usr/bin/env python3
"""Evaluate standardized event archives with explicit frozen paper profiles."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROFILES={'strict600':ROOT/'paper/wing_contribution/evaluation/core/unified_metrics.py',
          'overflow601':ROOT/'paper/wing_contribution/evaluation/legacy_v2/core/unified_metrics.py'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def evaluator(profile):
    spec=importlib.util.spec_from_file_location('energybench_'+profile,PROFILES[profile]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def run(path,profile,energy_key='energy_keV',energy_unit='keV',expected=None):
    m=evaluator(profile)
    with np.load(path,allow_pickle=False) as f:
        required={'score','label','event_id',energy_key}
        if not required.issubset(f.files):raise ValueError(f'Missing fields: {sorted(required-set(f.files))}')
        arrays={k:np.array(f[k]) for k in f.files}
    n=len(arrays['score']);ids=arrays['event_id']
    if ids.ndim!=1 or len(ids)!=n or len(np.unique(ids))!=n or np.any(ids.astype(str)==''):
        raise ValueError('event_id must be nonempty, unique, and aligned with scores')
    result=m.evaluate(arrays['label'],arrays['score'],arrays[energy_key],group=arrays.get('group'),weight=arrays.get('weight'),energy_unit=energy_unit)
    output={'input_sha256':sha(path),'profile':profile,'evaluator_sha256':sha(PROFILES[profile]),'metrics':result}
    if expected:
        observed={'I':result['independence']['I'],'matched_auc':result['matching']['matched_auc'],'inclusive_auc':result['inclusive_auc'],'n_events':result['n_input']}
        for key,value in expected.items():
            if value is None:
                if observed[key] is not None:raise AssertionError(f'{key}: expected missing')
            elif observed[key] is None or abs(observed[key]-value)>2e-12:raise AssertionError(f'{key}: {observed[key]} != {value}')
        output['matches_frozen_paper_result']=True
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input',type=Path);g.add_argument('--manifest',type=Path)
    p.add_argument('--profile',choices=PROFILES);p.add_argument('--data-root',type=Path)
    p.add_argument('--use-original-inputs',action='store_true',help='Read original paths from the local provenance manifest; no source files are written')
    p.add_argument('--energy-key',default='energy_keV');p.add_argument('--energy-unit',choices=('keV','MeV'),default='keV')
    p.add_argument('--dataset',action='append',help='Filter manifest by exact dataset name; may be repeated')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.input:
        if not a.profile:p.error('--input requires explicit --profile')
        out=run(a.input,a.profile,a.energy_key,a.energy_unit);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'I':out['metrics']['independence']['I'],'matched_auc':out['metrics']['matching']['matched_auc'],'output':str(a.output)}));return
    if a.profile:p.error('Manifest mode uses each record\'s profile; do not override it')
    if a.use_original_inputs==bool(a.data_root):p.error('Choose exactly one of --data-root or --use-original-inputs')
    records=json.loads(a.manifest.read_text())['records']
    if a.dataset:records=[r for r in records if r['dataset'] in a.dataset]
    if not records:p.error('No selected manifest records')
    resolved=[(r,Path(r['original_input']) if a.use_original_inputs else a.data_root/r['relative_input']) for r in records]
    missing=[str(path) for _,path in resolved if not path.is_file()]
    if missing:raise FileNotFoundError('Required archives unavailable:\n'+'\n'.join(missing))
    a.output.mkdir(parents=True,exist_ok=True);summaries=[]
    for r,path in resolved:
        if sha(path)!=r['input_sha256']:raise ValueError(f'Input hash mismatch: {r["id"]}')
        out=run(path,r['profile'],expected=r['expected'])
        if out['metrics']['protocol_sha256']!=r['protocol_sha256']:raise ValueError('Protocol fingerprint mismatch')
        target=a.output/(r['id']+'.json');target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        summaries.append({'id':r['id'],'profile':r['profile'],'matches_frozen_paper_result':True})
        print(r['id'],'PASS',flush=True)
    (a.output/'receipt.json').write_text(json.dumps({'records':summaries,'all_pass':True,'external_event_inputs_required':True},indent=2)+'\n')
if __name__=='__main__':main()
