#!/usr/bin/env python3
"""Independent full-event metric audit: entropy-form JS and ROC integration.
Does not import shared evaluation code. Writes only to this audit directory.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
A=Path(__file__).resolve().parent
C=A.parent
P=C.parents[1]/'overleaf'
BASELINE={(r['dataset'],r['model_key']):r for r in json.loads((A/'expected_v3_models.json').read_text())['records']}
EXPECTED=set(BASELINE)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantile(x,w,probs):
 ix=np.argsort(x,kind='mergesort');x=x[ix];w=w[ix]
 return np.interp(probs,(np.cumsum(w)-w/2)/w.sum(),x)
def roc(y,s,w):
 m=np.isfinite(s)&(w>0);y=y[m];s=s[m];w=w[m]
 if not np.any(y==0) or not np.any(y==1):return None
 ix=np.argsort(-s,kind='mergesort');s=s[ix];y=y[ix];w=w[ix]
 ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
 t=np.r_[0,np.cumsum(w*(y==1))[ends]];f=np.r_[0,np.cumsum(w*(y==0))[ends]]
 integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
 return float(integrate(t/t[-1],f/f[-1]))
def ent(p):
 x=np.asarray(p);x=x[x>0];return float(-np.sum(x*np.log(x)))
def equal(x,y,tol=5e-12):
 if x is None or y is None:return x is None and y is None
 return bool(abs(x-y)<=tol)
def arr_eq(x,y,tol=5e-12):return bool(np.allclose(x,y,atol=tol,rtol=0))
def bins(e):
 out=np.full(len(e),-1,dtype=int);m=np.isfinite(e)&(e>=0)&(e<=3000)
 out[m]=np.minimum(np.floor(e[m]/5).astype(int),599)
 return out

def population_meta(a):
 g=np.asarray(a.get('group',a['label'])).astype(str)
 _,first,inv=np.unique(g,return_index=True,return_inverse=True)
 ranks=np.empty(len(first),dtype=int);ranks[np.argsort(first)]=np.arange(len(first))
 # MJD old standardized inputs carry convenience IDs plus physical IDs;
 # newer source audit carries the physical ID directly. Group names also
 # differ (signal/background versus clean/nonclean) but partitions coincide.
 return {'label':a['label'],'energy':a['energy_keV'],'weight':a.get('weight',np.ones(len(g))),
         'group_partition':ranks[inv], 'event_id':a.get('physical_event_id',a['event_id']).astype(str)}

def check_row(row,reference):
 path=Path(row['standardized_file']);detail=Path(row['detail_file'])
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 y=a['label'];s=np.asarray(a['score'],dtype=float);e=np.asarray(a['energy_keV'],dtype=float)
 w=np.asarray(a.get('weight',np.ones(len(s))),dtype=float);g=np.asarray(a.get('group',y)).astype(str);event=a['event_id'].astype(str)
 d=json.loads(detail.read_text());d=d.get('metrics',d);D=d['independence'];M=d['matching'];b=bins(e)
 finite=np.isfinite(s)&np.isfinite(e)&(w>0);energy=finite&(e>=0)&(e<=3000)
 checks={'v3_protocol_fingerprint':d['protocol_sha256']=='6d696d87f1307b3f6d27bb9ef7ec443e1d8d98b9d95e90713f553c70edb42953','binary_labels':bool(np.all(np.isin(y,[0,1]))),'finite_nonnegative_base_weights':bool(np.all(np.isfinite(w)&(w>=0))),'aligned_arrays':all(x.ndim==1 and len(x)==len(s) for x in [y,e,w,g,event]),'unique_event_ids':len(np.unique(event))==len(event),'n_input':len(s)==d['n_input'],'inclusive_auc':equal(roc(y,s,w),d['inclusive_auc']),'strict600_matching_shape':np.asarray(M['bin_counts']).shape==(2,600),'baseline_inclusive_population_preserved':equal(roc(y,s,w),BASELINE[row['dataset'],row['model_key']]['baseline_inclusive_auc']),'original_standardized_input_hash_unchanged':sha(path)==BASELINE[row['dataset'],row['model_key']]['input_sha256']}
 declared=row.get('input_sha256',row.get('standardized_sha256'))
 if declared is not None:checks['standardized_hash_matches_declared']=sha(path)==declared
 ds=row['dataset'];meta=population_meta(a)
 if ds in reference:checks['same_test_events_and_physical_metadata']=all(np.array_equal(meta[k],reference[ds][k]) for k in meta)
 else:reference[ds]={k:v.copy() for k,v in meta.items()};checks['same_test_events_and_physical_metadata']=True
 groups={};group_mass={};group_checks={}
 for name in sorted(set(g)):
  sel=energy&(g==name);raw=finite&(g==name);counts=np.bincount(b[sel],minlength=600);eligible=counts>=20
  keep=sel&np.isin(b,np.flatnonzero(eligible));out=D['groups'][name];population_key='energy_population' if 'energy_population' in out else 'in_range'
  chk={'count_array':np.array_equal(counts,out['bin_counts']),'valid_bin_indices':np.array_equal(np.flatnonzero(eligible),out['valid_bin_indices']),'n_original_finite':int(raw.sum())==out['original_finite']['n'],'n_energy_population':int(sel.sum())==out[population_key]['n'],'n_retained':int(keep.sum())==out['retained']['n'],'pre_sparse_group_mass':equal(w[sel].sum(),out['group_aggregation_mass'],1e-7),'sparse_exclusion':int(np.sum(sel&~keep))==out['sparse_excluded']['n'],'group_lower_exclusion':int(np.sum(raw&(e<0)))==out['below_range']['n'],'group_upper_exclusion':int(np.sum(raw&(e>3000)))==out['above_range']['n'],'group_range_exclusion':int(np.sum(raw&~energy))==out['range_excluded']['n'],'group_original_denominator':equal(float(w[keep].sum()/w[raw].sum()) if w[raw].sum() else None,out['retained_fraction_original_finite'])}
  if not np.any(keep):chk['I_g_missing']=out['I_g'] is None;group_checks[name]=chk;continue
  se=np.unique(quantile(s[keep],w[keep],np.linspace(0,1,21)))
  if len(se)==1:
   span=max(abs(se[0])*1e-9,1e-12);se=np.array([se[0]-span,se[0]+span])
  else:se[0]=np.nextafter(se[0],-np.inf);se[-1]=np.nextafter(se[-1],np.inf)
  scorebin=np.minimum(np.searchsorted(se,s[keep],side='right')-1,len(se)-2)
  H=np.zeros((600,len(se)-1));np.add.at(H,(b[keep],scorebin),w[keep]);H=H[eligible]
  mass=H.sum(axis=1);hist=H/mass[:,None];pool=H.sum(axis=0)/H.sum()
  js=np.array([ent((h+pool)/2)-.5*(ent(h)+ent(pool)) for h in hist]);div=float(np.dot(mass/mass.sum(),js));val=float(np.clip(1-np.sqrt(max(div,0)/np.log(2)),0,1))
  groups[name]=val;group_mass[name]=float(w[sel].sum())
  chk.update(I_g=equal(val,out['I_g']),score_edges=arr_eq(se,out['score_edges']),retained_pooled_histogram=arr_eq(pool,out['pooled_histogram']),bin_histograms=arr_eq(hist,out['bin_histograms']),bin_weights=arr_eq(mass/mass.sum(),out['bin_probability']),bin_JS=arr_eq(js,out['bin_JS_nats']))
  group_checks[name]={k:bool(v) for k,v in chk.items()}
 I=sum(groups[k]*group_mass[k] for k in groups)/sum(group_mass.values()) if groups else None
 checks['I_overall']=equal(I,D['I']);checks['I_min']=equal(min(groups.values()) if groups else None,D['min_group_I']);checks['I_all_group_details']=all(all(v.values()) for v in group_checks.values())
 common=np.zeros(len(s),dtype=bool);mw=np.zeros(len(s));counts=np.zeros((2,600),dtype=int);mass=np.zeros((2,600));target=np.zeros(600)
 lo=hi=None;status='not_estimable_missing_energy_population_class';matched=None;diag=None
 if all(np.any(energy&(y==c)) for c in [0,1]):
  limits=[quantile(e[energy&(y==c)],w[energy&(y==c)],[.005,.995]) for c in [0,1]];lo=max(t[0] for t in limits);hi=min(t[1] for t in limits)
  if hi<lo:status='not_estimable_disjoint_support'
  else:
   common=energy&(e>=lo)&(e<=hi)
   for c in [0,1]:
    m=common&(y==c);counts[c]=np.bincount(b[m],minlength=600);mass[c]=np.bincount(b[m],weights=w[m],minlength=600)
   valid=np.all(counts>=20,axis=0)
   if not np.any(valid):status='not_estimable_no_eligible_bins'
   else:
    fractions=mass/mass.sum(axis=1)[:,None];target=np.minimum(fractions[0],fractions[1]);target[~valid]=0;target/=target.sum()
    for c in [0,1]:
     ix=np.flatnonzero(common&(y==c));mw[ix]=w[ix]*target[b[ix]]/mass[c,b[ix]]
    diag=roc(y,s,mw);coverage=[float(w[(mw>0)&(y==c)].sum()/w[finite&(y==c)].sum()) for c in [0,1]]
    if min(coverage)<.5:status='not_estimable_low_coverage'
    elif valid.sum()<2 and lo!=hi:status='not_estimable_too_few_bins'
    else:status='ok';matched=diag
 valid=np.all(counts>=20,axis=0);retained=mw>0
 checks.update(matching_status=status==M['status'],matched_auc=equal(matched,M['matched_auc']),diagnostic_auc=equal(diag,M['diagnostic_auc_not_for_reporting']),common_support=(M['common_support_keV'] is None if lo is None else arr_eq([lo,hi],M['common_support_keV'])),matching_count_array=np.array_equal(counts,M['bin_counts']),matching_base_mass=arr_eq(mass,M['bin_base_mass'],1e-7),overlap_target=arr_eq(target,M['target_mass']),valid_bins=np.array_equal(np.flatnonzero(valid),M['valid_bin_indices']),common_support_auc=equal(roc(y[common],s[common],w[common]),M['common_support_auc']),retained_auc=equal(roc(y[retained],s[retained],w[retained]),M['retained_unweighted_auc']))
 class_checks={};coverage=[];ess=[]
 for c in [0,1]:
  sel=y==c;out=M['classes'][str(c)];population_key='energy_population' if 'energy_population' in out else 'in_range';negative_key='negative_energy' if 'negative_energy' in out else 'below_range';den=w[finite&sel].sum();cov=float(w[retained&sel].sum()/den) if den else None;ww=mw[sel];eff=float(ww.sum()**2/np.dot(ww,ww)) if np.any(ww>0) else 0.;coverage.append(cov);ess.append(eff)
  stages={'original_finite':finite&sel,population_key:energy&sel,negative_key:finite&sel&(e<0),'above_range':finite&sel&(e>3000),'range_excluded':finite&sel&~energy,'common_support':common&sel,'matched':retained&sel,'support_excluded_after_range':energy&sel&~common,'sparse_excluded_after_support':common&sel&~retained}
  ch={k:int(mask.sum())==out[k]['n'] and equal(w[mask].sum(),out[k]['base_mass'],1e-7) for k,mask in stages.items()}
  ch.update(coverage=equal(cov,out['matched_fraction_original_finite']),ESS=equal(eff,out['matched_ess'],1e-6),class_weight=equal(ww.sum(),out['matched_weight_sum']),unit_mass_or_unmatched=equal(ww.sum(),1 if np.any(retained) else 0),loss_decomposition=out['original_finite']['n']==sum(out[k]['n'] for k in ['range_excluded','support_excluded_after_range','sparse_excluded_after_support','matched']))
  class_checks[str(c)]={k:bool(v) for k,v in ch.items()}
 checks['all_class_diagnostics']=all(all(v.values()) for v in class_checks.values())
 checks={k:bool(v) for k,v in checks.items()}
 return {'dataset':ds,'model_key':row['model_key'],'all_pass':all(checks.values()),'checks':checks,'group_checks':group_checks,'class_checks':class_checks,'independent_metrics':{'inclusive_auc':roc(y,s,w),'I':I,'I_groups':groups,'matched_auc':matched,'matching_status':status,'valid_matching_bins':int(valid.sum()),'coverage_negative_positive':coverage,'ESS_negative_positive':ess},'n_events':len(s),'standardized_file':str(path),'standardized_sha256':sha(path),'detail_file':str(detail),'detail_sha256':sha(detail),'protocol_sha256':d['protocol_sha256']}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('manifests',nargs='+',type=Path);ap.add_argument('--output',type=Path,default=A/'independent_numeric_results_v3.json');ap.add_argument('--cache',type=Path);args=ap.parse_args()
 rows=[]
 for p in args.manifests:
  x=json.loads(p.read_text());rs=x if isinstance(x,list) else x.get('records',x.get('ready',[]))
  for rr in rs:
   r=dict(rr);r['standardized_file']=r.get('standardized_file',r.get('standardized_predictions'));r['detail_file']=r.get('detail_file',r.get('detail_json'))
   if (r['dataset'],r['model_key']) in EXPECTED and r.get('standardized_file') and r.get('detail_file'):rows.append(r)
 keys=[(r['dataset'],r['model_key']) for r in rows];assert len(keys)==len(set(keys)),'duplicate model identity'
 reference={};reports=[]
 cache={(r['dataset'],r['model_key']):r for r in json.loads(args.cache.read_text())['records']} if args.cache else {}
 for row in rows:
  saved=cache.get((row['dataset'],row['model_key']))
  if saved and saved['all_pass'] and sha(row['standardized_file'])==saved['standardized_sha256'] and sha(row['detail_file'])==saved['detail_sha256']:
   out=saved
   if row['dataset'] not in reference:
    with np.load(row['standardized_file'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
    reference[row['dataset']]=population_meta(a)
  else:out=check_row(row,reference)
  reports.append(out);print(out['dataset'],out['model_key'],out['all_pass'],out['independent_metrics'],flush=True)
 got=set(keys);result={'all_pass':all(r['all_pass'] for r in reports) and got==EXPECTED,'requested_model_count':61,'models_reviewed':len(reports),'missing_model_keys':sorted(EXPECTED-got),'excluded_by_user':['MJD/'+f+'_rope' for f in ['entity','region','summary']],'method':'Strict 0–3000keV/600-bin full-event independent histogram counts and entropy-form JS, descending ROC trapezoidal integration, independently reconstructed support/overlap/coverage/ESS. No shared metric helper imported.','source_manifests':[str(p) for p in args.manifests],'records':reports}
 args.output.write_text(json.dumps(result,indent=2)+'\n');print('all_pass',result['all_pass']);raise SystemExit(0 if result['all_pass'] else 1)
if __name__=='__main__':main()
