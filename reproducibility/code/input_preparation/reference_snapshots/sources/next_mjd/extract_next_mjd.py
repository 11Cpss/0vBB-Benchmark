#!/usr/bin/env python3
"""Extract immutable saved scores plus physical energies; no training or summary-to-score reconstruction."""
from pathlib import Path
import hashlib, json, csv, sys
import numpy as np
import h5py
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1]/'code'))
from unified_metrics import to_keV
PAPER=Path('/home/wenyu/iclr final paper/overleaf')
NEXT=Path('/home/wenyu/summer/03_training_runs/energybench_campaigns/20260808_energybench_rewrite_v2')
MJD=Path('/home/wenyu/MJD/outputs/classification')
MAIN={'cnn_004_multiview_late_fusion':'mvcnn','gnn_001_static_gine':'gine','seq_001_bigru':'bigru','ssm_001_pointmamba':'mamba'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def oldmetric(m):return {'I':m['energy_independence_score'],'matched_auc':m['matched_auc'],'inclusive_auc':m['auc'],'groups':m['energy_dependence']['groups'],'matching':m['classification']}
def basic(dataset,arch,p,m,z,energy,group,extra=None):
 assert set(np.unique(z['label']))=={0,1}
 assert np.all(z['split']=='test')
 assert len(np.unique(z['event_id']))==len(z['score'])
 assert np.all(np.isfinite(z['score'])) and np.all(z['sample_weight']==1)
 d={'score':z['score'].astype(np.float64),'label':z['label'].astype(np.int8),'energy_keV':energy.astype(np.float64),'group':group.astype(str),'event_id':z['event_id'].astype(str),'weight':z['sample_weight'].astype(np.float64),'split':z['split'].astype(str),'legacy_energy_keV':z['energy']*1000.}
 if extra:d.update(extra)
 target=OUT/f'{dataset}__{arch}.npz';np.savez_compressed(target,**d)
 r={'dataset':dataset,'architecture_id':arch,'model_key':MAIN.get(arch,arch),'block':'classic','main_table':arch in MAIN,'status':'ready','standardized_predictions':str(target),'standardized_sha256':sha(target),'source_predictions':str(p),'source_predictions_sha256':sha(p),'source_metrics_json':str(p.with_name('metrics.json')),'source_metrics':str(p.with_name('metrics.json')),'source_metrics_sha256':sha(p.with_name('metrics.json')),'n_events':len(d['score']),'event_id_sha256':arrsha(d['event_id']),'label_sha256':arrsha(d['label']),'energy_keV_sha256':arrsha(d['energy_keV']),'source_score_dtype':str(z['score'].dtype),'score_transform':'none; lossless float32 to float64 promotion','old_metrics':oldmetric(m),'old':oldmetric(m),'legacy_protocol':m['protocol'],'class_counts':{str(c):int(np.sum(d['label']==c)) for c in (0,1)},'energy_location_counts':{str(c):{'below_0':int(np.sum((d['label']==c)&(energy<0))),'above_3000':int(np.sum((d['label']==c)&(energy>3000))),'nonfinite':int(np.sum((d['label']==c)&~np.isfinite(energy)))} for c in (0,1)},'paper_locations':['wing_contribution/tables/benchmark_main.tex'] if arch in MAIN else []}
 return r,d

def main():
 rows=[];alignment=[]
 split_path=NEXT/'event_split.json';manifest=json.loads(split_path.read_text());expected={x['relative_path']:x for x in manifest['splits']['test']}
 ref=None
 for mp in sorted((NEXT/'reevaluation_20260818').glob('*/classification/metrics.json')):
  arch=mp.parent.parent.name;p=mp.with_name('predictions.npz');m=json.loads(mp.read_text());z=load(p)
  assert len(z['score'])==manifest['counts']['test']==116549
  group=z['category'];assert np.all((group=='0nubb')==(z['label']==1));assert set(group)=={'0nubb','Bi214'}
  source_paths,counts=np.unique(z['group_id'],return_counts=True)
  assert set(source_paths)==set(expected)
  assert all(int(n)==expected[str(s)]['event_count'] for s,n in zip(source_paths,counts))
  if ref is None:ref=z
  eq={k:bool(np.array_equal(z[k],ref[k])) for k in ('event_id','label','category','energy','sample_weight','group_id','split')};assert all(eq.values())
  record=json.loads(mp.with_name('reevaluation_record.json').read_text());original=load(record['source_predictions'])
  assert all(np.array_equal(z[k],original[k]) for k in z)
  r,d=basic('NEXT',arch,p,m,z,to_keV(z['energy'],'MeV'),group,{'source_file':z['group_id']})
  r.update(positive_class='0nubb (raw Signal); 0=Bi214 (raw Bkg)',score_representation='raw classification logit; larger favors 0nubb',energy_source='float64 sum of original float32 per-hit physical deposited energies in MC/hits/table values_block_1[:,3]; original unit MeV',energy_conversion='multiply original saved MeV by 1000; no clipping',test_split='canonical event-count 80/10/10, seed 42; class stratified file order and ordinal slices',test_split_manifest=str(split_path),test_split_manifest_sha256=sha(split_path),test_file_count=len(source_paths),original_predictions=record['source_predictions'],original_predictions_sha256=sha(record['source_predictions']))
  summary_path=NEXT/'runs'/arch/'classification/run_summary.json';summary=json.loads(summary_path.read_text());r['run_summary']=str(summary_path);r['run_summary_sha256']=sha(summary_path);r['best_epoch']=summary['best_epoch'];r['params']=summary.get('parameter_count');r['checkpoint_candidates']=[str(q) for q in (summary_path.parent/'training').glob('*best*')]
  r['paper_locations']+=['wing_contribution/tables/next_classification.tex']
  if arch not in {'cnn_001_two_conv_baseline','cnn_002_global_energy_skip','classic_001_topology_xgboost'}:r['paper_locations']+=['wing_contribution/figures/data/next_capacity_plot.csv','wing_contribution/sections/next_capacity_figure.tex']
  rows.append(r);alignment.append({'dataset':'NEXT','architecture_id':arch,'metadata_equal_to_first_model':eq,'reevaluation_equal_to_original_prediction':True,'test_manifest_file_counts_match':True})
  print('extracted NEXT',arch,flush=True)
 # Read actual calibrated scalar metadata from official shards, never cached summaries.
 config=json.loads((MJD/'cnn_004_multiview_late_fusion/run_config.json').read_text())
 paths=sorted(Path(config['data']['data_root']).glob('MJD_Test_*.hdf5'));assert len(paths)==6
 chunks=[];raw_manifest=[]
 for p in paths:
  with h5py.File(p,'r') as f:
   fields={k:f[k][:] for k in ['energy_label','id','run_number','detector','tp0','psd_label_low_avse','psd_label_high_avse','psd_label_dcr','psd_label_lq']}
  n=len(fields['id']);fields['source_shard']=np.full(n,p.name);fields['source_row']=np.arange(n);chunks.append(fields)
  raw_manifest.append({'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'n_events':n,'scalar_dataset_sha256':{k:arrsha(v) for k,v in fields.items() if k not in ('source_shard','source_row')}})
 raw={k:np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
 labels=np.all(np.stack([raw[k] for k in ['psd_label_low_avse','psd_label_high_avse','psd_label_dcr','psd_label_lq']]),axis=0).astype(np.int8)
 e=raw['energy_label'];physical_ids=np.array([f"{shard}::row={i}::id={rawid}" for shard,i,rawid in zip(raw['source_shard'],raw['source_row'],raw['id'])])
 np.savez_compressed(OUT/'MJD_official_test_metadata.npz',**raw,label=labels,physical_event_id=physical_ids)
 dump('MJD_official_test_shards.json',raw_manifest)
 ref=None
 for arch in MAIN:
  p=MJD/arch/'energybench_classification/clean/predictions.npz';z=load(p);m=json.loads(p.with_name('metrics.json').read_text());original=load(MJD/arch/'predictions.npz')
  assert np.array_equal(labels,z['label']) and np.array_equal(labels,original['target'])
  assert np.array_equal(z['score'],original['prediction'].astype(np.float32))
  assert np.array_equal(z['energy'],np.clip(e.astype(np.float32).astype(np.float64)/1000,0,3))
  assert np.array_equal(z['event_id'],np.array([f'mjd-test-{i}' for i in range(len(e))]))
  if ref is None:ref=z
  eq={k:bool(np.array_equal(z[k],ref[k])) for k in ('event_id','label','energy','sample_weight','group_id','split')};assert all(eq.values())
  r,d=basic('MJD',arch,p,m,z,e,np.where(labels==1,'signal','background'),{'physical_event_id':physical_ids,'source_shard':raw['source_shard'],'source_row':raw['source_row'],'raw_id':raw['id'],'run_number':raw['run_number'],'detector':raw['detector'],'energy_float32_keV':e.astype(np.float32).astype(np.float64)})
  r.update(positive_class='clean: all four reference PSD flags equal 1; 0=nonclean',score_representation=('four auxiliary PSD pass logits combined to logit(product of sigmoid probabilities), larger favors clean; additional four-label supervision' if arch=='gnn_001_static_gine' else 'raw binary clean-vs-nonclean classification logit, larger favors clean'),energy_source='official MJD_Test_0..5.hdf5 energy_label float64, calibrated keV',energy_conversion='none; recover original float64 physical energy before historical float32 metadata cast and [0,3000] clipping',test_split='all six official Test shards concatenated lexicographically, row order; unchanged full 390000-event inclusive population',test_raw_metadata=str(OUT/'MJD_official_test_metadata.npz'),test_raw_shard_manifest=str(OUT/'MJD_official_test_shards.json'),run_config=str(MJD/arch/'run_config.json'),run_config_sha256=sha(MJD/arch/'run_config.json'),checkpoint_candidates=[str(MJD/arch/'best.pt')],original_predictions=str(MJD/arch/'predictions.npz'),original_predictions_sha256=sha(MJD/arch/'predictions.npz'),float32_then_clip_exactly_reproduces_saved_energy=True,source_labels_exactly_match=True,source_scores_exactly_match=True,physical_energy_restored=True)
  if arch=='ssm_001_pointmamba':r['historical_note']='Workbook AP5 incorrectly linked a regression result and truncated values; old raw metrics read from independently located classification metrics.json.'
  rows.append(r);alignment.append({'dataset':'MJD','architecture_id':arch,'metadata_equal_to_first_model':eq,'raw_labels_exact_match':True,'raw_scores_exact_match':True,'float32_then_clip_exact_match':True})
  print('extracted MJD',arch,flush=True)
 # Source inventory provides only paper identifiers and historical reported numbers, never metric inputs.
 evidence=json.loads((PAPER/'wing_contribution/figures/data/classification_table_evidence.json').read_text())
 blocked=[]
 for r in evidence['records']:
  if r['dataset'] in ('NEXT','MJD') and (r['model_key'] not in MAIN.values() or (r['dataset']=='NEXT' and r['model_key']=='mamba')):
   blocked.append({'dataset':r['dataset'],'model_key':r['model_key'],'architecture_id':None,'block':'classic' if r['model_key']=='mamba' else 'transformer','main_table':True,'status':'blocked_missing_event_inputs','standardized_predictions':None,'source_predictions':None,'old_I_reported':r.get('independence'),'old_matched_auc_reported':r.get('matched_auc'),'old_precision':'only source worksheet decimal string; original metric output unavailable','reported_source':str(PAPER/'wing_contribution/figures/data/classification_table_evidence.json'),'worksheet':r['worksheet'],'worksheet_row':r['row'],'paper_locations':['wing_contribution/tables/benchmark_main.tex'],'reason':('Canonical rewrite campaign has no completed PointMamba classification run. Older NEXTALT PointMamba result uses a different retained run/test population and is not substituted.' if r['model_key']=='mamba' else 'No attributable saved event predictions, training configuration or matching checkpoint found after broad account search; cannot verify units, label/score direction, test IDs or either historical protocol. Workbook values are not recomputation inputs.')})
 dump('manifest.json',{'schema':'energybench-source-manifest-v1','ready':rows,'blocked':blocked,'note':'All standardized NPZ preserve full inclusive test population; energy filtering and the separate >3000 keV overflow category are deferred to the shared evaluator.'})
 dump('alignment_checks.json',alignment)
 dump('mjd_energy_recovery.json',{'n_events':len(e),'raw_dtype':str(e.dtype),'finite':int(np.isfinite(e).sum()),'below_zero':int(np.sum(e<0)),'above_3000':int(np.sum(e>3000)),'above_3000_by_class':{str(c):int(np.sum((labels==c)&(e>3000))) for c in (0,1)},'raw_min':float(e.min()),'raw_max':float(e.max()),'float32_metadata_max_absolute_difference_keV':float(np.max(abs(e-e.astype(np.float32)))),'fixed_bin_membership_float32_vs_float64_disagreements':int(np.sum(np.searchsorted(np.arange(0,3005,5),e,side='right')!=np.searchsorted(np.arange(0,3005,5),e.astype(np.float32),side='right'))),'restored_physical_id_unique':len(np.unique(physical_ids))})
 print('READY',len(rows),'BLOCKED',len(blocked),flush=True)
if __name__=='__main__':main()
