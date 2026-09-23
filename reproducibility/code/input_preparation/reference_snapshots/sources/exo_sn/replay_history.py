#!/usr/bin/env python3
"""Verify archived EXO/SN metrics from events; isolate physical-range changes."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/wenyu/summer/src')
from energybench.dependence import evaluate_dependence
from energybench.roc import evaluate_energy_matched_roc

def run(a,mask):
    y,s,e,w,g=(a[k][mask] for k in ['label','score','energy_keV','weight','group'])
    # Only I is compared; a cap of four avoids computing unrelated O(n^2)
    # distance-correlation diagnostics. It cannot affect histogram-based I.
    dep=evaluate_dependence(s,e,y,category=g,sample_weight=w,n_energy_bins=8,n_score_bins=20,min_per_bin=20,distance_correlation_max_samples=4)
    roc=evaluate_energy_matched_roc(y,s,e,w,positive_label=1,n_bins=6,min_per_class=20,target='overlap',n_bootstrap=0,random_state=42,support_trim_quantile=.005)
    return dict(I=dep['overall_energy_independence_score'],group_I={k:v['energy_independence_score'] for k,v in dep['groups'].items()},matched_auc=roc.matched_auc,inclusive_auc=roc.inclusive.auc,matching=roc.to_dict())

def main():
    rows=[]
    for rec in json.loads((ROOT/'manifest.json').read_text())['records']:
        if rec['status']!='ready': continue
        with np.load(rec['standardized_file']) as z:a={k:z[k] for k in z.files}
        old=run(a,np.ones(len(a['score']),dtype=bool)); physical=run(a,(a['energy_keV']>=0)&(a['energy_keV']<=3000))
        errors={'I':abs(old['I']-rec['old_I']),'matched_auc':abs(old['matched_auc']-rec['old_matched_auc']),'inclusive_auc':abs(old['inclusive_auc']-rec['old_inclusive_auc'])}
        assert max(errors.values())<1e-11,(rec['dataset'],rec['model_key'],errors)
        rows.append(dict(dataset=rec['dataset'],model_key=rec['model_key'],old_replay=old,physical_range_only_historical_Q8_Q6=physical,range_only_delta_I=physical['I']-old['I'],range_only_delta_matched_auc=physical['matched_auc']-old['matched_auc'],absolute_replay_error=errors))
        print(rec['dataset'],rec['model_key'],errors,'range delta I',physical['I']-old['I'],flush=True)
        (ROOT/'historical_replay_and_range_attribution.json').write_text(json.dumps({'records':rows,'note':'Historical protocol reproduced from events (Q8 per-category independence, Q6 pooled-class-balanced matching). Range-only counterfactual retains these old histogram rules; it is not a reportable unified result. I ignores the unrelated distance-correlation cap=4. The unified-minus-range-only difference includes the global 5-keV energy grid and harmonized retained-population histograms/20-event qualification.'},indent=2)+'\n')

if __name__=='__main__':main()
