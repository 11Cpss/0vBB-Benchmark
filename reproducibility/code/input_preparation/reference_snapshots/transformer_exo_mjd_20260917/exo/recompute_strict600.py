#!/usr/bin/env python3
"""Reevaluate the nine audited, unchanged EXO Transformer event bundles under v3."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
OUT = BASE / 'strict600'
EVAL = Path('/home/wenyu/iclr final paper/overleaf/wing_contribution/evaluation')
VERSION = 'EnergyBench-unified-5keV-range-v3.0.0'
FINGERPRINT = '6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953'
IMPLEMENTATION_SHA = 'a243160b0a529a26a38dc6be14baa6f17a09618fea57c5a80de21822369e06b0'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'detail').mkdir(exist_ok=True)
    source = EVAL / 'core/unified_metrics.py'
    assert sha(source) == IMPLEMENTATION_SHA
    spec = importlib.util.spec_from_file_location('exo_strict600', source)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    config = json.loads((EVAL / 'protocol.json').read_text())
    assert config['protocol_version'] == VERSION and metrics.fingerprint(config) == FINGERPRINT
    save(OUT / 'protocol_used.json', config)
    (OUT / 'unified_metrics.py').write_bytes(source.read_bytes())
    origin = json.loads((BASE / 'manifest.json').read_text())
    previous = {r['model_key']:r for r in json.loads((BASE / 'summary.json').read_text())['records']}
    rows, group_rows, provenance = [], [], []
    for r in origin['records']:
        p = Path(r['standardized_file'])
        assert sha(p) == r['standardized_sha256']
        with np.load(p, allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        result = metrics.evaluate(a['label'], a['score'], a['energy_keV'],
                                  group=a['group'], weight=a['weight'],
                                  energy_unit='keV', config=config)
        assert result['energy_bin_count'] == 600
        assert abs(result['inclusive_auc'] - r['historical_replay']['inclusive_auc']) < 1e-12
        result.update(dataset='EXO-200', model_key=r['model_key'], model_id=r['model_id'],
                      input_sha256=sha(p), evaluator_sha256=sha(source))
        target = OUT / 'detail' / f"EXO_200__{r['model_key']}.json"
        save(target, result)
        old, intermediate = r['historical_replay'], previous[r['model_key']]
        row = dict(dataset='EXO-200', model_key=r['model_key'], model_id=r['model_id'],
                   parameter_count=r['checkpoint']['parameter_count'],
                   old_I=old['I'], new_I=result['independence']['I'],
                   delta_I=result['independence']['I']-old['I'],
                   v2_intermediate_I=intermediate['new_I'],
                   strict_minus_v2_I=result['independence']['I']-intermediate['new_I'],
                   old_matched_auc=old['matched_auc'], new_matched_auc=result['matching']['matched_auc'],
                   delta_matched_auc=result['matching']['matched_auc']-old['matched_auc'] if result['matching']['matched_auc'] is not None else None,
                   v2_intermediate_matched_auc=intermediate['new_matched_auc'],
                   strict_minus_v2_matched_auc=result['matching']['matched_auc']-intermediate['new_matched_auc'] if result['matching']['matched_auc'] is not None else None,
                   old_inclusive_auc=old['inclusive_auc'], new_inclusive_auc=result['inclusive_auc'],
                   n_events=result['n_input'], common_support_keV=result['matching']['common_support_keV'],
                   valid_matching_bins=result['matching']['valid_bin_count'], matched_status=result['matching']['status'],
                   min_group_I=result['independence']['min_group_I'], above_range_n=result['n_above_range'],
                   below_range_n=result['n_below_range'], protocol_version=VERSION, protocol_sha256=FINGERPRINT,
                   standardized_file=str(p), detail_file=str(target))
        for c, role in [('1', 'signal'), ('0', 'background')]:
            item = result['matching']['classes'][c]
            for dest, field in [('coverage', 'matched_fraction_original_finite'), ('ESS', 'matched_ess')]:
                row[role+'_'+dest] = item[field]
            for field in ['input', 'original_finite', 'energy_population', 'range_excluded', 'above_range',
                          'below_range', 'common_support', 'matched', 'support_excluded_after_range',
                          'sparse_excluded_after_support']:
                row[role+'_'+field+'_n'] = item[field]['n']
        for name, g in result['independence']['groups'].items():
            assert len(g['bin_counts']) == 600
            group_rows.append(dict(dataset='EXO-200', model_key=r['model_key'], group=name,
                         old_I_g=old['group_I'][name], new_I_g=g['I_g'], delta_I_g=g['I_g']-old['group_I'][name],
                         valid_bin_count=g['valid_bin_count'], input_n=g['original_finite']['n'],
                         energy_population_n=g['energy_population']['n'], retained_n=g['retained']['n'],
                         sparse_excluded_n=g['sparse_excluded']['n'], range_excluded_n=g['range_excluded']['n'],
                         above_range_n=g['above_range']['n'], below_range_n=g['below_range']['n'],
                         retained_fraction_original_finite=g['retained_fraction_original_finite'],
                         retained_fraction_energy_population=g['retained_fraction_energy_population'],
                         group_aggregation_mass=g['group_aggregation_mass']))
        rows.append(row)
        provenance.append(dict(dataset='EXO-200', model_key=r['model_key'], model_id=r['model_id'],
                     standardized_file=str(p), standardized_sha256=sha(p), detail_file=str(target), detail_sha256=sha(target),
                     source_manifest=str(BASE/'manifest.json'), source_manifest_sha256=sha(BASE/'manifest.json'),
                     original_prediction_file=r['prediction_file'], original_prediction_sha256=r['prediction_sha256'],
                     checkpoint=r['checkpoint']['checkpoint'], checkpoint_sha256=r['checkpoint']['checkpoint_sha256'],
                     parameter_count=r['checkpoint']['parameter_count'], protocol_version=VERSION, protocol_sha256=FINGERPRINT))
        print(r['model_key'], 'I', repr(row['new_I']), 'matched', repr(row['new_matched_auc']), flush=True)
    save(OUT/'summary.json', dict(records=rows, group_records=group_rows))
    save(OUT/'manifest.json', dict(records=provenance, protocol_version=VERSION, protocol_sha256=FINGERPRINT,
                     implementation_file=str(source), implementation_sha256=sha(source),
                     source_audit=str(BASE/'source_audit.md'), operation='Read-only reuse of nine original full-test standardized bundles; strict 600-bin reevaluation; no inference or training.'))
    for filename, records in [('summary.csv', rows), ('group_summary.csv', group_rows)]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save(OUT/'validation.json', dict(model_count=len(rows), all_600_bins=True,
                  all_matching_reportable=all(r['matched_status']=='ok' for r in rows),
                  unchanged_input_hashes=True, inclusive_max_absolute_change=max(abs(r['new_inclusive_auc']-r['old_inclusive_auc']) for r in rows),
                  above_range_n=rows[0]['above_range_n'], below_range_n=rows[0]['below_range_n'],
                  all_group_I_decrease=all(r['delta_I_g']<0 for r in group_rows),
                  protocol_version=VERSION, protocol_sha256=FINGERPRINT,
                  implementation_sha256=sha(source), script_sha256=sha(__file__)))

if __name__ == '__main__':
    main()
