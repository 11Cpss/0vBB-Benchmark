#!/usr/bin/env python3
"""Evaluate the audited saved MJD Transformer events under final strict600 protocol.

Only energy metrics exclude physical energies outside0--3000keV. Inclusive
scores/labels and all source identities are unchanged. No inference or training.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
sys.dont_write_bytecode = True
from recompute_mjd import HERE, DEFAULT_PAPER, save, sha, small


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=DEFAULT_PAPER/'wing_contribution/evaluation')
    parser.add_argument('--output',type=Path,default=HERE/'strict600')
    args=parser.parse_args()
    sys.path.insert(0,str(args.evaluation))
    from unified_metrics import evaluate,load_config,fingerprint
    config=load_config()
    assert config['energy']['n_bins']==600, 'Final evaluator must have exactly600 bins'
    assert 'overflow' not in config['protocol_version'].lower()
    assert fingerprint(config)=='6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953'
    assert sha(args.evaluation/'core/unified_metrics.py')=='a243160b0a529a26a38dc6be14baa6f17a09618fea57c5a80de21822369e06b0'
    out=args.output
    (out/'details').mkdir(parents=True,exist_ok=True)
    save(out/'protocol.json',config)
    sources=json.loads((HERE/'manifest.json').read_text())
    manifest={k:v for k,v in sources.items() if k not in ['ready','protocol_version','protocol_sha256',
        'protocol_file_sha256','evaluator_file_sha256','evaluator_path','recompute_script_sha256']}
    manifest.update(protocol_version=config['protocol_version'],protocol_sha256=fingerprint(config),
        protocol_file_sha256=sha(args.evaluation/'protocol.json'),
        evaluator_file_sha256=sha(args.evaluation/'core/unified_metrics.py'),
        evaluator_path=str(args.evaluation/'core/unified_metrics.py'),
        recompute_script_sha256=sha(Path(__file__)),v2_intermediate_manifest=str(HERE/'manifest.json'),ready=[])
    rows=[]
    for src in sources['ready']:
        key=src['model_key']
        assert sha(src['standardized_predictions'])==src['standardized_sha256']
        with np.load(src['standardized_predictions']) as z:
            a={k:z[k] for k in z.files}
        result=evaluate(a['label'],a['score'],a['energy_keV'],a['group'],a['weight'],config=config)
        f32=evaluate(a['label'],a['score'],a['legacy_energy_float32_keV'],a['group'],a['weight'],config=config)
        old=src['old_metrics']
        v2=json.loads(Path(src['detail_json']).read_text())
        stages={
            'old_saved_836_bins':old,
            'v2_float64_with_overflow_intermediate':small(v2['metrics']),
            'v3_float32_strict600_diagnostic':small(f32),
            'v3_float64_strict600_final':small(result)}
        assert abs(result['inclusive_auc']-old['inclusive_auc'])<1e-12
        assert max(max(g['valid_bin_indices']) for g in result['independence']['groups'].values())<600
        outside=(~np.isfinite(a['energy_keV']))|(a['energy_keV']<0)|(a['energy_keV']>3000)
        report=dict(dataset='MJD',model_key=key,run_id=src['run_id'],metrics=result,
            change_attribution=dict(stages=stages,
                legacy_replay=str(HERE/'legacy_replay'/('MJD_'+key+'.json')),
                prior_v2_details=src['detail_json'],
                range_excluded_by_class={str(k):int((outside&(a['label']==k)).sum()) for k in (0,1)},
                final_vs_original_delta={k:stages['v3_float64_strict600_final'][k]-old[k] for k in old},
                final_vs_v2_overflow_delta={k:stages['v3_float64_strict600_final'][k]-stages['v2_float64_with_overflow_intermediate'][k] for k in old},
                float64_restoration_delta={k:stages['v3_float64_strict600_final'][k]-stages['v3_float32_strict600_diagnostic'][k] for k in old}))
        detail=out/'details'/('MJD_'+key+'.json')
        save(detail,report)
        record=dict(src,detail_json=str(detail),protocol_version=config['protocol_version'],
                    protocol_sha256=fingerprint(config))
        manifest['ready'].append(record)
        row=dict(dataset='MJD',model=key,run_id=src['run_id'],params=src['exact_parameter_count'],n=len(a['label']),
            old_I=old['I'],new_I=result['independence']['I'],delta_I=result['independence']['I']-old['I'],
            old_matched_auc=old['matched_auc'],new_matched_auc=result['matching']['matched_auc'],
            delta_matched_auc=result['matching']['matched_auc']-old['matched_auc'],
            old_inclusive_auc=old['inclusive_auc'],new_inclusive_auc=result['inclusive_auc'],
            min_group_I=result['independence']['min_group_I'],matching_status=result['matching']['status'],
            matching_valid_bins=result['matching']['valid_bin_count'],
            **{f'I_{g}':r['I_g'] for g,r in result['independence']['groups'].items()},
            **{f'I_bins_{g}':r['valid_bin_count'] for g,r in result['independence']['groups'].items()},
            **{f'coverage_class{k}':r['matched_fraction_original_finite'] for k,r in result['matching']['classes'].items()},
            **{f'ESS_class{k}':r['matched_ess'] for k,r in result['matching']['classes'].items()},
            source=src['source_predictions'],protocol_version=config['protocol_version'],protocol_sha256=fingerprint(config))
        rows.append(row)
        save(out/'summary.json',rows)
        save(out/'manifest.json',manifest)
        with (out/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
        print(key,'I',row['new_I'],'matched',row['new_matched_auc'],flush=True)
    assert len(rows)==6


if __name__=='__main__':
    main()
