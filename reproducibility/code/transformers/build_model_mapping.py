#!/usr/bin/env python3
"""Link each of the 36 current-paper Transformer entries to source artifacts."""
from pathlib import Path
import csv,hashlib,json

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SRC=Path('/home/klz/Data/zeronu_benchmark/Transformer_Approach')
SN=Path('/home/wenyu/SuperNEMO')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def artifact(p):
    p=Path(p)
    return dict(path=str(p),exists=p.is_file(),sha256=sha(p) if p.is_file() else None,packaged=False)
def main():
    evidence=json.loads((HERE/'configs/paper_reference/classification_table_evidence.json').read_text())
    paper={(r['dataset'],r['model_key']):r for r in evidence['records'] if r['model_key'].startswith(('entity_','region_','summary_'))}
    assert len(paper)==36
    ids={'entity_mlp':'transformer_001_sampled_hits_coordinate_mlp','region_mlp':'transformer_002_voxel_coordinate_mlp',
         'region_fourier':'transformer_003_voxel_fourier_xyz','entity_fourier':'transformer_004_sampled_hits_fourier_xyz',
         'summary_mlp':'transformer_005_summary_features_coordinate_mlp','summary_fourier':'transformer_006_summary_features_fourier_xyz'}
    rows=[]
    for (dataset,key),p in paper.items():
        family,pe=key.split('_');r=dict(dataset=dataset,model_key=key,paper_matched_auc=p['matched_auc'],paper_I=p['independence'],paper_params=p.get('params'),
            paper_status=p['status'],paper_worksheet=p.get('worksheet'),paper_row=p.get('row'),paper_source_cells=p.get('source_cells'),source_training_config=None,
            final_evaluation='strict600' if dataset=='EXO-200' or (dataset=='MJD' and pe!='rope') else 'historical; do not silently substitute strict600 or overflow601')
        if pe=='rope' and dataset!='EXO-200':
            r.update(status='missing_exact_training_source_and_checkpoint',source_code=None,checkpoint=None,predictions=None,
                     limitation='Current paper retains recorded values, but no corresponding RoPE implementation/config/checkpoint/prediction was located. The recovered MLP/Fourier package does not implement RoPE. No EXO implementation or unrelated Transformer was substituted.')
            rows.append(r);continue
        if dataset in ('EXO-200','MJD'):
            ds='exo200_detector' if dataset=='EXO-200' else 'mjd_detector'
            token={'entity':'pulse_entities','region':'raw_patches','summary':'segment_summary'}[family]
            enc={'mlp':'coordinate_mlp','fourier':'fourier_coordinates','rope':'rope'}[pe]
            runid=f'classification__{token}__{enc}';run=SRC/ds/'results/transformer_official_v1'/runid
            cfg=json.loads((run/'run_config.json').read_text())
            pred=run/'energybench_predictions.npz' if dataset=='EXO-200' else run/'energybench_classification/predictions.npz'
            r.update(run_id=runid,source_code=f'code/transformers/original/{ds}',source_training_config=str((HERE/'configs'/ds/runid/'run_config.json').relative_to(ROOT)),
               representation=cfg['representation'],training=cfg['training'],data=cfg['data'],exact_parameter_count=cfg['parameter_count'],
               checkpoint=artifact(run/'best.pt'),predictions=artifact(pred),status='exact_saved_run_audited',
               limitation='Weights and event predictions are referenced and hashed but intentionally not packaged. Strict600 uses restored original physical energy, not the historical evaluator output.')
        elif dataset=='NEXT':
            runid=ids[key];campaign='final' if family=='summary' else 'final_cached_v1';run=SRC/'next_detector/results'/campaign/runid
            cfg=json.loads((run/'representation_config.json').read_text());met=json.loads((run/'evaluation/metrics.json').read_text())
            checks={}
            for field,actual in [('matched_auc',met['matched_auc']),('independence',met['energy_independence_score'])]:
                recorded=str(p[field]);dp=len(recorded.split('.')[1]) if '.' in recorded else 0
                checks[field]=dict(paper_recorded=recorded,source_full_precision=actual,decimal_places=dp,
                    equal_at_recorded_precision=(f'{actual:.{dp}f}'==recorded),absolute_difference=abs(float(recorded)-actual))
                assert checks[field]['equal_at_recorded_precision']
            assert int(float(p['params']))==cfg['parameter_count']
            r.update(run_id=runid,source_code='code/transformers/original/next_detector',source_training_config=str((HERE/'configs/next'/campaign/runid/'representation_config.json').relative_to(ROOT)),
              representation=cfg,exact_parameter_count=cfg['parameter_count'],paper_source_comparison=checks,
              checkpoint=artifact(run/'training/best_model.pt'),predictions=artifact(run/'evaluation/predictions.npz'),source_metrics=artifact(run/'evaluation/metrics.json'),
              status='saved_run_matches_all_recorded_metric_digits_and_exact_parameter_count',
              limitation='Paper workbook stores 5–6 decimal metric precision rather than the source full-precision binary floats. All stored digits and exact parameter counts match this uniquely named completed run; this is not proof of an unrecorded workbook event fingerprint. Preserve the original NEXT evaluation profile.')
        else:
            runid=ids[key];run=SN/'outputs/classification'/runid;cfg=json.loads((run/'run_config.json').read_text())
            mp=run/'test_evaluation/energybench/.energybench/metrics.json';met=json.loads(mp.read_text())
            actual_auc=met['classification']['aggregates']['matched_auc_macro'];actual_I=met['energy_dependence']['overall_energy_independence_score']
            assert float(p['matched_auc'])==actual_auc and float(p['independence'])==actual_I
            r.update(run_id=runid,source_code='code/transformers/published_supernemo',source_training_config=str((HERE/'configs/published_supernemo'/runid/'run_config.json').relative_to(ROOT)),
              representation=dict(model=cfg['model'],tokenization=cfg['tokenization']),training=cfg['training'],data=cfg['data'],
              checkpoint=artifact(run/'best.pt'),predictions=artifact(run/'test_evaluation/test_predictions.npz'),source_metrics=artifact(mp),
              status='exact_saved_run_linked_metrics_match_paper',exact_parameter_count=int(float(p['params'])),
              historical_matching_bins=6,historical_I_energy_bins_per_category=8,
              limitation='The canonical Transformer_Approach SuperNEMO package uses different IDs and has only one completed local run; it is retained as a related source snapshot and must not replace this published implementation.')
        rows.append(r)
    payload=dict(paper_snapshot='code/transformers/configs/paper_reference/classification_table_evidence.json',rows=rows,
        missing_model_count=sum(r['status'].startswith('missing') for r in rows),
        search=dict(roots=['/home/klz/Data/zeronu_benchmark/Transformer_Approach','/home/wenyu','/home filename search excluding environment caches'],
          excluded_candidate='/home/klz/Data/pelon/MJD/transformer(1).ipynb is unrelated Optuna energy regression, contains no RoPE, and is not a source for the classification matrix.'),
        no_training_or_inference_performed=True)
    (ROOT/'provenance/transformer_model_mapping.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (ROOT/'provenance/transformer_model_mapping.csv').open('w',newline='') as f:
        fields=['dataset','model_key','run_id','status','source_code','source_training_config','exact_parameter_count','paper_matched_auc','paper_I','final_evaluation','limitation']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(models=len(rows),missing=payload['missing_model_count'])))
if __name__=='__main__':main()
