"""Integrate reviewed event evaluations and prepare source-backed paper data."""
from pathlib import Path
import copy,csv,hashlib,json,subprocess
import numpy as np
from unified_metrics import load_config,fingerprint

ROOT=Path(__file__).resolve().parents[1]
TRANS=ROOT.parent/'transformer_exo_mjd_20260917'
WING=ROOT.parent.parent/'overleaf/wing_contribution'
CONFIG=load_config()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    baseline=json.loads((TRANS/'baseline/wing_contribution/figures/data/unified_results.json').read_text())
    source=copy.deepcopy(baseline)
    all_recomputed=json.loads((ROOT/'results/existing_results.json').read_text())
    classic_keys={(ds,k) for ds in ('MJD','EXO-200') for k in ('mvcnn','gine','bigru','mamba')}
    updates={(r['dataset'],r['model_key']):r for r in all_recomputed['records'] if (r['dataset'],r['model_key']) in classic_keys}
    source['records']=[copy.deepcopy(updates.get((r['dataset'],r['model_key']),r)) for r in source['records']]
    assert len(json.loads((ROOT/'results/existing_manifest.json').read_text())['records'])==8
    lookup={(r['dataset'],r['model_key']):r for r in source['records']}
    for folder,ds in [('exo','EXO-200'),('mjd','MJD')]:
        manifest_path=TRANS/folder/'strict600/manifest.json'
        manifest=json.loads(manifest_path.read_text())
        summaries=json.loads((manifest_path.parent/'summary.json').read_text())
        rows=manifest.get('records',manifest.get('ready'))
        summaries=summaries['records'] if isinstance(summaries,dict) else summaries
        summaries={r.get('model_key',r.get('model')):r for r in summaries}
        assert len(rows)==(9 if folder=='exo' else 6)
        for entry in rows:
            key=entry['model_key'];previous=lookup[ds,key];s=summaries[key]
            path=entry.get('standardized_file',entry.get('standardized_predictions'))
            assert sha(path)==entry['standardized_sha256']
            native_detail=entry.get('detail_file',entry.get('detail_json'))
            packed=json.loads(Path(native_detail).read_text());detail=packed.get('metrics',packed)
            assert detail['protocol_sha256']==fingerprint(CONFIG)
            assert detail['independence']['I']==s['new_I']
            assert detail['matching']['matched_auc']==s['new_matched_auc']
            target=ROOT/'results/detail'/f'{ds}__{key}.json';write(target,detail)
            d,m=detail['independence'],detail['matching']
            previous.update(status='recomputed',comparison_eligible=True,
                architecture_id=entry.get('model_id',entry.get('run_id')),params=str(entry.get('parameter_count',entry.get('exact_parameter_count'))),
                source_manifest=str(manifest_path),input_source=entry.get('original_prediction_file',entry.get('source_predictions')),
                standardized_file=path,input_sha256=sha(path),detail_file=str(target),source_detail_file=native_detail,
                protocol_version=detail['protocol_version'],protocol_sha256=detail['protocol_sha256'],evaluator_sha256=sha(Path(__file__).with_name('unified_metrics.py')),
                I=d['I'],min_group_I=d['min_group_I'],matched_auc=m['matched_auc'],matching_status=m['status'],inclusive_auc=detail['inclusive_auc'],n_events=detail['n_input'],
                valid_matching_bins=m['valid_bin_count'],valid_I_bins={k:g['valid_bin_count'] for k,g in d['groups'].items()},
                coverage={k:c['matched_fraction_original_finite'] for k,c in m['classes'].items()},matched_ess={k:c['matched_ess'] for k,c in m['classes'].items()},common_support_keV=m['common_support_keV'],
                old_I=s['old_I'],old_matched_auc=s['old_matched_auc'],old_inclusive_auc=s['old_inclusive_auc'],delta_I=s['new_I']-s['old_I'],delta_matched_auc=s['new_matched_auc']-s['old_matched_auc'],
                notes='Recomputed from original event predictions under strict 600-bin v3; saved checkpoint, test population and score representation retained.')
    campaign=json.loads((WING/'evaluation/campaigns/transformer_exo_mjd_20260917.json').read_text())
    source['transformer_reevaluation']=campaign
    selected=classic_keys|{(r['dataset'],r['model_key']) for r in campaign['selected_models']}
    assert len(selected)==23
    source['protocol_registry']={fingerprint(baseline['protocol']):baseline['protocol'],fingerprint(CONFIG):CONFIG}
    source['active_reevaluation']={'scope':'MJD and EXO-200 only; MJD RoPE excluded','protocol_sha256':fingerprint(CONFIG),'selected_models':[{'dataset':d,'model_key':k} for d,k in sorted(selected)]}
    for old,new in zip(baseline['records'],source['records']):
        if (old['dataset'],old['model_key']) not in selected:assert old==new
    write(ROOT/'results/results_full_precision.json',source)
    write(ROOT/'results/selected_results_full_precision.json',{'protocol':CONFIG,'scope':source['active_reevaluation'],'records':[r for r in source['records'] if (r['dataset'],r['model_key']) in selected]})
    write(WING/'figures/data/unified_results.json',source)
    # Every diagnostic table is derived from the same evaluated records, including
    # the explicitly referenced earlier NEXT PointMamba set.
    old_diagnostics=json.loads(subprocess.check_output(['git','show','4851898:wing_contribution/evaluation/data/diagnostics.json'],cwd=WING.parent))
    diagnostics_by_key={(r['dataset'],r['model_key']):r for r in old_diagnostics['records']}
    comparison=[];groups=[];losses=[];event_checks={};manifest=[]
    for r in source['records']:
        if (r['dataset'],r['model_key']) not in selected:continue
        assert r['status']=='recomputed'
        d=json.loads(Path(r['detail_file']).read_text());m=d['matching'];ind=d['independence']
        assert d['protocol_sha256']==fingerprint(CONFIG)
        with np.load(r['standardized_file'],allow_pickle=False) as f:
            gm={str(c):str(np.unique(f['group'][f['label']==c])[0]) for c in (0,1)}
            metadata={k:hashlib.sha256(np.ascontiguousarray(f[k]).tobytes()).hexdigest() for k in ('event_id','label','energy_keV','group','weight')}
        manifest.append({'dataset':r['dataset'],'model_key':r['model_key'],'standardized_file':r['standardized_file'],'input_sha256':r['input_sha256'],'detail_file':r['detail_file'],'detail_sha256':sha(r['detail_file']),'group_for_label':gm,'event_metadata_hashes':metadata})
        pop_group=r['dataset']
        if (r['dataset'],r['model_key']) in {(a['dataset'],a['model_key']) for a in campaign['selected_models']}:
            pop_group+=' Transformers'
        if pop_group not in event_checks:event_checks[pop_group]=metadata
        else:assert metadata==event_checks[pop_group],f'Unmatched event metadata: {pop_group} {r["model_key"]}'
        c={k:{'n':v['n'],'base_mass':v['base_mass']} for k,v in m['classes']['0'].items() if isinstance(v,dict)}
        diagnostics_by_key[r['dataset'],r['model_key']]={'dataset':r['dataset'],'model_key':r['model_key'],'protocol_sha256':r['protocol_sha256'],'source_input_sha256':r['input_sha256'],'source_detail_sha256':sha(r['detail_file']),'group_for_label':gm,
            'n_events':r['n_events'],'inclusive_auc':r['inclusive_auc'],'matched_auc':r['matched_auc'],'I':r['I'],
            'independence':{'min_group_I':ind['min_group_I'],'groups':{k:{field:g[field] for field in ('I_g','valid_bin_count','retained_fraction_original_finite','retained','sparse_excluded','range_excluded','above_range','below_range')} for k,g in ind['groups'].items()}},
            'matching':{k:m[k] for k in ('status','common_support_keV','valid_bin_count','classes','diagnostic_auc_not_for_reporting')}}
        previous=r.get('previous_protocol_metrics',{})
        row={k:r.get(k) for k in ('dataset','model_key','n_events','old_I','I','old_matched_auc','matched_auc','inclusive_auc','delta_I','delta_matched_auc','valid_matching_bins','input_source','standardized_file','input_sha256','protocol_version','protocol_sha256')}
        row.update(previous_paper_I=previous.get('I',r['old_paper_I']),previous_paper_matched_auc=previous.get('matched_auc',r['old_paper_matched_auc']),matching_status=m['status'],minimum_group_I=ind['min_group_I'])
        row['delta_from_previous_paper_I']=r['I']-float(row['previous_paper_I'])
        row['delta_from_previous_paper_matched_auc']=r['matched_auc']-float(row['previous_paper_matched_auc'])
        row['notes']=r.get('notes','Reused original classic predictions; changed v2 range handling only.')
        for k in ('0','1'):
            cl=m['classes'][k];row.update({f'coverage_{k}':cl['matched_fraction_original_finite'],f'ESS_{k}':cl['matched_ess'],f'above_range_{k}':cl['above_range']['n'],f'below_range_{k}':cl['below_range']['n']})
            losses.append({'dataset':r['dataset'],'model_key':r['model_key'],'label':k,**{f'{stage}_n':cl[stage]['n'] for stage in ('input','original_finite','energy_population','range_excluded','below_range','above_range','support_excluded_after_range','sparse_excluded_after_support','matched')},'coverage_original_finite':cl['matched_fraction_original_finite'],'matched_ESS':cl['matched_ess']})
        comparison.append(row)
        for name,g in ind['groups'].items():groups.append({'dataset':r['dataset'],'model_key':r['model_key'],'group':name,'I_g':g['I_g'],'n_original_finite':g['original_finite']['n'],'n_in_range':g['energy_population']['n'],'n_retained':g['retained']['n'],'valid_bins':g['valid_bin_count'],'retained_fraction_original_finite':g['retained_fraction_original_finite'],'range_excluded':g['range_excluded']['n'],'sparse_excluded':g['sparse_excluded']['n']})
    assert len(manifest)==23
    write(ROOT/'results/input_manifest.json',{'records':manifest})
    write(ROOT/'audit/event_population_checks.json',event_checks)
    write(WING/'evaluation/data/diagnostics.json',{'description':'Reviewed aggregate display data; per-record frozen protocol fingerprints distinguish the MJD/EXO-200 v3 reevaluation from unchanged v2 records. Never used as event input.','records':list(diagnostics_by_key.values())})
    write_csv(ROOT/'results/new_old_comparison.csv',comparison);write_csv(ROOT/'results/group_independence.csv',groups);write_csv(ROOT/'results/class_coverage_losses.csv',losses)
    print('Integrated 23 full-precision MJD/EXO-200 event evaluations; all unselected result records unchanged.')

if __name__=='__main__':main()
