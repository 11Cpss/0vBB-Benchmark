#!/usr/bin/env python3
"""Replay original NEXT/SuperNEMO Transformer metrics from native event archives.

Uses the two copied historical packages in separate processes. Does not train,
infer, resample, convert score direction, or substitute a newer grid/profile.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(dataset,manifest,output):
    path=HERE/'original'/('evalutaions_workflow' if dataset=='NEXT' else 'exo200_detector/frozen_energybench')
    sys.path.insert(0,str(path))
    import numpy as np
    if dataset=='NEXT':
        from energybench.metrics import evaluate_energy_matched_classification,evaluate_energy_dependence
    else:
        from energybench.roc import evaluate_energy_matched_roc
        from energybench.dependence import evaluate_dependence
    results=[]
    for r in json.loads(manifest.read_text())['records']:
        if r['dataset']!=dataset:continue
        p=ROOT/r['bundle_file'];assert sha(p)==r['sha256']
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if dataset=='NEXT':
            cfg=dict(r['historical_config'],distance_correlation_max_samples=4)
            auc=evaluate_energy_matched_classification(a['label'],a['score'],a['energy'],a['sample_weight'],cfg)
            dep=evaluate_energy_dependence(a['label'],a['score'],a['energy'],a['sample_weight'],a['category'],None,cfg)
            values=dict(I=dep['overall_energy_independence_score'],matched_auc=auc['matched_auc'],inclusive_auc=auc['inclusive']['auc'])
            diagnostics=dict(status=auc.get('matched_auc_status',auc.get('status')),coverage=auc.get('coverage'),protocol=dep['protocol'])
        else:
            cfg=r['historical_config'];c=cfg['classification'];d=cfg['dependence']
            auc=evaluate_energy_matched_roc(a['label'],a['score'],a['energy_condition'],a['sample_weight'],
                positive_label=int(c['positive_label']),n_bins=c['energy_bins'],min_per_class=c['min_per_class'],target=c['matching_target'],
                target_tpr=c['target_tpr'],n_bootstrap=0,random_state=cfg['runtime']['seed'],support_trim_quantile=c['support_trim_quantile'],energy_roi=c['energy_roi'])
            dep=evaluate_dependence(a['score'],a['energy_condition'],a['label'],category=a['category'],sample_weight=a['sample_weight'],
                n_energy_bins=d['energy_bins'],n_score_bins=d['score_bins'],min_per_bin=d['min_per_bin'],distance_correlation_max_samples=4,seed=cfg['runtime']['seed'])
            values=dict(I=dep['overall_energy_independence_score'],matched_auc=auc.matched_auc,inclusive_auc=auc.inclusive.auc)
            diagnostics=dict(status=auc.status,coverage=dataclasses.asdict(auc.coverage),effective_sample_sizes=dataclasses.asdict(auc.effective_sample_sizes),
                             common_support=auc.common_support,bins_requested=auc.n_bins_requested,bins_actual=auc.n_bins_actual)
        errors={k:abs(values[k]-r['expected_source_metrics'][k]) for k in values}
        assert max(errors.values())<1e-12,(dataset,r['model_key'],errors)
        result=dict(dataset=dataset,model_key=r['model_key'],n_events=len(a['label']),metrics=values,expected_source_metrics=r['expected_source_metrics'],absolute_error=errors,
                    native_input_sha256=r['sha256'],diagnostics=diagnostics,source_package=str(path.relative_to(ROOT)),
                    unrelated_diagnostic_optimization='Only distance-correlation max_samples is reduced from 1200 to 4; this has no effect on histogram I or either AUC. No distance-correlation result is reported.')
        results.append(result)
        print(dataset,r['model_key'],errors,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(records=results),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'provenance/transformer_historical_inputs.json');p.add_argument('--output-dir',type=Path,default=ROOT/'validation/transformer_historical_replay');p.add_argument('--worker',choices=['NEXT','SuperNEMO']);a=p.parse_args()
    if a.worker:return worker(a.worker,a.manifest,a.output_dir/(a.worker+'.json'))
    for ds in ['NEXT','SuperNEMO']:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'validation/transformer_mpl'))
        subprocess.run([sys.executable,'-B',__file__,'--manifest',str(a.manifest),'--output-dir',str(a.output_dir),'--worker',ds],env=env,check=True)
    print('All 12 historical Transformer metric replays passed.')
if __name__=='__main__':main()
