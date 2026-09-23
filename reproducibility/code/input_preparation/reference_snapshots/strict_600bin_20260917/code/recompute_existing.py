"""Actually reevaluate the 8 selected MJD/EXO-200 classic event sets under strict v3."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
from unified_metrics import evaluate,load_config,fingerprint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'transformer_exo_mjd_20260917/baseline/wing_contribution/figures/data/unified_results.json'
CONFIG=load_config()
EVALUATOR=Path(__file__).with_name('unified_metrics.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    source=json.loads(BASE.read_text())
    manifest=[]
    for record in source['records']:
        if record['dataset'] not in ('MJD','EXO-200') or record['model_key'] not in ('mvcnn','gine','bigru','mamba'):continue
        assert record['status']=='recomputed'
        input_path=Path(record['standardized_file'])
        assert sha(input_path)==record['input_sha256'],f'Input changed: {input_path}'
        with np.load(input_path,allow_pickle=False) as f:
            data={k:f[k] for k in ('label','score','energy_keV','group','weight','event_id')}
        assert len(np.unique(data['event_id']))==len(data['label'])
        result=evaluate(data['label'],data['score'],data['energy_keV'],data['group'],data['weight'])
        np.testing.assert_allclose(result['inclusive_auc'],record['inclusive_auc'],rtol=0,atol=2e-15)
        assert all(len(g['bin_counts'])==600 for g in result['independence']['groups'].values())
        assert all(len(a)==600 for a in result['matching']['bin_counts'])
        detail=ROOT/'results/detail'/f"{record['dataset']}__{record['model_key']}.json"
        detail.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        old={k:record.get(k) for k in ('I','matched_auc','inclusive_auc','protocol_version','protocol_sha256','detail_file')}
        d=result['independence'];m=result['matching']
        record.update(previous_protocol_metrics=old,protocol_version=result['protocol_version'],protocol_sha256=result['protocol_sha256'],evaluator_sha256=sha(EVALUATOR),
            I=d['I'],min_group_I=d['min_group_I'],matched_auc=m['matched_auc'],matching_status=m['status'],inclusive_auc=result['inclusive_auc'],
            n_events=result['n_input'],detail_file=str(detail),valid_matching_bins=m['valid_bin_count'],
            valid_I_bins={k:v['valid_bin_count'] for k,v in d['groups'].items()},coverage={k:v['matched_fraction_original_finite'] for k,v in m['classes'].items()},
            matched_ess={k:v['matched_ess'] for k,v in m['classes'].items()},common_support_keV=m['common_support_keV'],
            delta_I=None if d['I'] is None else d['I']-float(record['old_I']),
            delta_matched_auc=None if m['matched_auc'] is None else m['matched_auc']-float(record['old_matched_auc']),
            delta_from_v2_I=None if d['I'] is None else d['I']-old['I'],delta_from_v2_matched_auc=None if m['matched_auc'] is None or old['matched_auc'] is None else m['matched_auc']-old['matched_auc'])
        gm={str(c):str(np.unique(data['group'][data['label']==c])[0]) for c in (0,1)}
        assert all(len(np.unique(data['group'][data['label']==c]))==1 for c in (0,1))
        manifest.append({'dataset':record['dataset'],'model_key':record['model_key'],'standardized_file':str(input_path),'input_sha256':sha(input_path),'detail_file':str(detail),'detail_sha256':sha(detail),'group_for_label':gm})
        print(record['dataset'],record['model_key'],'I=',d['I'],'AUC=',m['matched_auc'],m['status'],flush=True)
        (ROOT/'results/existing_results.json').write_text(json.dumps(source,indent=2,allow_nan=False)+'\n')
        (ROOT/'results/existing_manifest.json').write_text(json.dumps({'records':manifest},indent=2)+'\n')
    assert len(manifest)==8
    print('Recomputed the 8 selected classic event sets; inclusive AUC preserved.',flush=True)

if __name__=='__main__':main()
