#!/usr/bin/env python3
from pathlib import Path
import json,sys,hashlib,numpy as np
OUT=Path(__file__).resolve().parent/'exploratory'
sys.path.insert(0,str(OUT.parents[2]/'code'))
from unified_metrics import to_keV,evaluate
CITED={'cnn_004_multiview_late_fusion','cnn_005_multiscale_projection','cnn_006_dense_3d_resnet','gnn_001_static_gine','seq_001_bigru','seq_002_dilated_tcn','ssm_001_pointmamba'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 rows=[];align=[];ref=None;computed=[]
 for p in sorted(Path('/home/wenyu/summer/04_evaluations').glob('NEXTALT_*/predictions_test.npz')):
  with np.load(p) as z:a={k:z[k] for k in z}
  arch=p.parent.name.removeprefix('NEXTALT_').removesuffix('_test')
  # Seven numerical appendix references plus graph candidates needed to verify GINE's lead.
  if arch not in CITED and not arch.startswith('gnn_'):continue
  mp=p.parent/'evaluation_test/.energybench/metrics.json';rp=mp.with_name('resolved_manifest.json');m=json.loads(mp.read_text());resolved=json.loads(rp.read_text());meta=json.loads(a['__metadata__'].item());pair=m['classification']['pairs'][0]
  assert len(a['score'])==115499 and meta['energy_unit']=='MeV' and meta['positive_class']=='0nubb' and meta['score_space']=='logit'
  assert np.all(a['split']=='test') and np.all((a['category']=='0nubb')==(a['label']==1))
  if ref is None:ref=a
  eq={k:bool(np.array_equal(a[k],ref[k])) for k in ('event_id','energy_condition','label','category','sample_weight','split','group_id')};assert all(eq.values()),(arch,eq)
  d={'score':a['score'].astype(float),'label':a['label'],'energy_keV':to_keV(a['energy_condition'],'MeV'),'group':a['category'],'event_id':a['event_id'],'weight':a['sample_weight'].astype(float),'source_file':a['group_id'],'legacy_energy_keV':a['energy_condition']*1000.,'split':a['split']}
  dest=OUT/f'NEXT-exploratory__{arch}.npz';np.savez_compressed(dest,**d)
  r={'dataset':'NEXT-exploratory','architecture_id':arch,'model_key':'exploratory_'+arch,'block':'classic_exploratory','main_table':False,'status':'ready','standardized_predictions':str(dest),'standardized_sha256':sha(dest),'source_predictions':str(p),'source_predictions_sha256':sha(p),'source_metrics':str(mp),'source_metrics_json':str(mp),'source_metrics_sha256':sha(mp),'source_resolved_manifest':str(rp),'source_resolved_manifest_sha256':sha(rp),'old_metrics':{'I':m['energy_dependence']['overall_energy_independence_score'],'matched_auc':pair['matched_auc'],'inclusive_auc':pair['inclusive_auc']},'legacy_protocol':{'classification':resolved['classification'],'dependence':resolved['dependence']},'old_protocol_fingerprint':m['protocol_fingerprint'],'old_evaluator_code_fingerprint':m['evaluator']['code_fingerprint'],'n_events':len(a['score']),'event_id_sha256':ah(a['event_id']),'energy_keV_sha256':ah(d['energy_keV']),'label_sha256':ah(d['label']),'positive_class':'0nubb=1, Bi214=0','score_representation':'raw classifier logit; larger favors 0nubb','score_transform':'none','energy_source':meta['energy_condition_derivation'],'energy_conversion':'MeV to keV using shared exact canonical-edge handling','test_split':'earlier file-split 80/10/10 seed 42; distinct from canonical event-count split','prediction_metadata':meta,'paper_locations':['wing_contribution/sections/appendix_models.tex:8'] if arch in CITED else ['context: graph candidates for appendix_models.tex GINE-lead claim'],'class_counts':{str(c):int(np.sum(a['label']==c)) for c in (0,1)},'energy_location_counts':{str(c):{'below_0':int(np.sum((a['label']==c)&(d['energy_keV']<0))),'above_3000':int(np.sum((a['label']==c)&(d['energy_keV']>3000))),'nonfinite':int(np.sum((a['label']==c)&~np.isfinite(d['energy_keV'])))} for c in (0,1)}}
  r['old']=r['old_metrics'];rows.append(r);align.append({'architecture_id':arch,'metadata_equal_to_first':eq})
  result=evaluate(d['label'],d['score'],d['energy_keV'],d['group'],d['weight']);computed.append({'source':r,'result':result})
  print(arch,'old',pair['matched_auc'],'new',result['matching']['matched_auc'],'I',result['independence']['I'],flush=True)
 (OUT/'manifest.json').write_text(json.dumps({'schema':'energybench-source-manifest-v1','ready':rows,'blocked':[]},indent=2)+'\n')
 (OUT/'results.json').write_text(json.dumps(computed,indent=2)+'\n')
 (OUT/'alignment.json').write_text(json.dumps(align,indent=2)+'\n')
if __name__=='__main__':main()
