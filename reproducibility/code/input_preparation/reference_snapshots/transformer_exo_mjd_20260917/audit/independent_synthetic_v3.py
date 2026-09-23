#!/usr/bin/env python3
"""Independent equations and pairwise enumeration; does not reuse evaluator helpers."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;ROOT=A.parent.parent;P=ROOT.parent/'overleaf';sys.path.insert(0,str(P/'wing_contribution/evaluation'));import unified_metrics as u

def quantile(x,w,p):
 ix=sorted(range(len(x)),key=lambda j:x[j]);xs=np.asarray([x[j] for j in ix]);ws=np.asarray([w[j] for j in ix]);centers=(np.cumsum(ws)-ws/2)/sum(ws)
 return np.interp(p,centers,xs)
def auc(y,s,w):
 p=np.flatnonzero((y==1)&np.isfinite(s)&(w>0));n=np.flatnonzero((y==0)&np.isfinite(s)&(w>0))
 if not len(p) or not len(n):return None
 d=s[p,None]-s[n][None,:];return float(np.sum(w[p,None]*w[n][None,:]*((d>0)+.5*(d==0)))/(sum(w[p])*sum(w[n])))
def ref(y,s,e,w,g):
 finite=np.isfinite(s)&np.isfinite(e)&(w>0);inside=finite&(e>=0)&(e<=3000);b=np.full(len(e),-1)
 for k in range(600):b[(e>=5*k)&((e<5*(k+1))|((k==599)&(e==3000)))]=k
 groups={}
 for name in sorted(set(g)):
  ids=np.flatnonzero(inside&(g==name));kept=[j for j in ids if sum(b[ids]==b[j])>=20]
  if not kept:groups[name]=None;continue
  p=np.linspace(0,1,21);edges=np.unique(quantile(s[kept],w[kept],p))
  if len(edges)==1:
   span=max(abs(s[kept[0]])*1e-9,1e-12);edges=np.array([s[kept[0]]-span,s[kept[0]]+span])
  else:edges[0]=np.nextafter(edges[0],-np.inf);edges[-1]=np.nextafter(edges[-1],np.inf)
  pooled=np.histogram(s[kept],edges,weights=w[kept])[0];pooled=pooled/sum(pooled);d=0
  for k in sorted(set(b[kept])):
   loc=[j for j in kept if b[j]==k];h=np.histogram(s[loc],edges,weights=w[loc])[0];h=h/sum(h);m=(h+pooled)/2
   term=sum(h[z]*np.log(h[z]/m[z]) if h[z]>0 else 0 for z in range(len(h)))+sum(pooled[z]*np.log(pooled[z]/m[z]) if pooled[z]>0 else 0 for z in range(len(h)))
   d+=sum(w[loc])/sum(w[kept])*term/2
  groups[name]=float(1-np.sqrt(max(d,0)/np.log(2)))
 validgroups=[g0 for g0 in groups if groups[g0] is not None];mass={g0:sum(w[inside&(g==g0)]) for g0 in validgroups};I=sum(mass[g0]*groups[g0] for g0 in validgroups)/sum(mass.values()) if validgroups else None
 mw=np.zeros(len(e));vb=[];support=None;status='not_estimable_missing_energy_population_class';formal=None
 if all(any(inside&(y==c)) for c in [0,1]):
  qs=[quantile(e[inside&(y==c)],w[inside&(y==c)],[.005,.995]) for c in [0,1]];lo=max(qs[0][0],qs[1][0]);hi=min(qs[0][1],qs[1][1]);support=[lo,hi]
  if lo>hi:status='not_estimable_disjoint_support'
  else:
   common=inside&(e>=lo)&(e<=hi);tot=[sum(w[common&(y==c)]) for c in [0,1]];target={}
   for k in sorted(set(b[common])):
    ids=[np.flatnonzero(common&(y==c)&(b==k)) for c in [0,1]]
    if min(map(len,ids))>=20:vb.append(k);target[k]=min(sum(w[ids[0]])/tot[0],sum(w[ids[1]])/tot[1])
   if not vb:status='not_estimable_no_eligible_bins'
   else:
    den=sum(target.values())
    for k in vb:
     q=target[k]/den
     for c in [0,1]:
      ids=np.flatnonzero(common&(y==c)&(b==k));mw[ids]=q*w[ids]/sum(w[ids])
    cov=[sum(w[(mw>0)&(y==c)])/sum(w[finite&(y==c)]) for c in [0,1]]
    if min(cov)<.5:status='not_estimable_low_coverage'
    elif len(vb)<2 and lo!=hi:status='not_estimable_too_few_bins'
    else:status='ok';formal=auc(y,s,mw)
 return dict(I=I,groups=groups,weights=mw,formal=formal,diagnostic=auc(y,s,mw),status=status,validbins=vb,support=support,inclusive=auc(y,s,w))

def same(a,b,atol=4e-13):
 if a is None or b is None:return a is None and b is None
 return bool(abs(a-b)<=atol)
checks=[];rng=np.random.default_rng(9202611)
for i in range(48):
 n=240 if i<24 else 360;y=rng.integers(0,2,n);s=rng.integers(-3,5,n).astype(float) if i%3==0 else rng.normal(size=n)
 e=rng.choice([0.,5.,1005.,1015.,2995.,3000.,-1.,3001.],n,p=[.23,.2,.14,.1,.05,.03,.05,.2]) if i%2==0 else rng.choice([1.,6.,11.],n)+rng.uniform(0,2,n)
 w=np.ones(n) if i%3==0 else rng.uniform(.1,4,n);g=np.where(y==1,'s','b')
 if i%4==0:s[:4]=np.nan;e[4:7]=np.nan;w[7:10]=0
 if i%7==0:s[:]=2
 r,a=u.evaluate(y,s,e,group=g,weight=w,return_arrays=True);z=ref(y,s,e,w,g)
 c={'case':i,'inclusive':same(r['inclusive_auc'],z['inclusive']),'I':same(r['independence']['I'],z['I']),'group_I':all(same(r['independence']['groups'][k]['I_g'],v) for k,v in z['groups'].items()),'matched_formal':same(r['matching']['matched_auc'],z['formal']),'diagnostic':same(r['matching']['diagnostic_auc_not_for_reporting'],z['diagnostic']),'weights':bool(np.allclose(a['matched_weight'],z['weights'],atol=1e-15,rtol=1e-13)),'validbins':r['matching']['valid_bin_indices']==z['validbins'],'status':r['matching']['status']==z['status'],'actual_status':r['matching']['status']};checks.append(c)
# Dedicated degeneracies: one constant energy permitted; nonconstant one bin disallowed; zero groups, disjoint, unit equivalence at every edge.
for name,e in [('constant',np.full(80,5.)),('single_nonconstant_bin',np.tile(np.linspace(5.1,8.9,40),2)),('all_out',np.full(80,3001.)),('disjoint',np.repeat([5.,50.],40))]:
 y=np.repeat([0,1],40);r=u.evaluate(y,np.ones(80),e);checks.append(dict(case=name,inclusive_half=r['inclusive_auc']==.5,I_constant=(r['independence']['I']==1 if name in ['constant','single_nonconstant_bin','disjoint'] else r['independence']['I'] is None),status=r['matching']['status'],formal=r['matching']['matched_auc']))
E=np.arange(601)*5.;bins=u.energy_bin_indices(E);converted=u.to_keV(E/1000,'MeV');checks.append(dict(case='all_601_boundaries_unit_equivalence',passed=bool(np.array_equal(bins,u.energy_bin_indices(converted))),failed_energy_keV=E[bins!=u.energy_bin_indices(converted)].tolist()))
# Score=f(bin) with unequal class spectra, exact 5-keV bin masses matching.
e=np.concatenate([np.repeat([1.,6.],[40,100]),np.repeat([1.,6.],[120,30])]);y=np.r_[np.zeros(140,dtype=int),np.ones(150,dtype=int)];r=u.evaluate(y,np.floor(e/5),e);checks.append(dict(case='bin_only_score',passed=same(r['matching']['matched_auc'],.5),matched_auc=r['matching']['matched_auc']))
# Strict range plus unchanged reporting denominator at exactly constant common energy.
y=np.repeat([0,1],100);e=np.tile(np.r_[np.full(40,5.),np.full(60,3001.)],2);r=u.evaluate(y,np.ones(200),e)
checks.append(dict(case='constant_support_does_not_waive_original_coverage',passed=(r['matching']['status']=='not_estimable_low_coverage' and r['matching']['matched_auc'] is None and r['matching']['diagnostic_auc_not_for_reporting']==.5 and r['inclusive_auc']==.5 and all(c['above_range']['n']==60 and c['matched_fraction_original_finite']==.4 for c in r['matching']['classes'].values()))))
x=np.array([-1.,0.,np.nextafter(5.,-np.inf),5.,2995.,3000.,np.nextafter(3000.,np.inf),4000.,np.inf,np.nan]);expected=np.array([-1,0,0,1,599,599,-1,-1,-1,-1]);checks.append(dict(case='strict_endpoints_no_overflow',passed=bool(np.array_equal(u.energy_bin_indices(x),expected))))
expected_degenerate={'constant':('ok',.5),'single_nonconstant_bin':('not_estimable_too_few_bins',None),'all_out':('not_estimable_missing_energy_population_class',None),'disjoint':('not_estimable_disjoint_support',None)}
for c in checks:
 if c['case'] in expected_degenerate:
  status,value=expected_degenerate[c['case']];c['expected_status_and_formal']=c['status']==status and same(c['formal'],value)
all_pass=all(all(v for v in c.values() if isinstance(v,bool)) for c in checks)
summary=dict(all_pass=all_pass,evaluator_sha256=hashlib.sha256((P/'wing_contribution/evaluation/core/unified_metrics.py').read_bytes()).hexdigest(),config_sha256=u.fingerprint(u.load_config()),method='48 strict-range independent full-equation synthetic evaluations, brute-force cross-class AUC pairs; dedicated gates, missingness, weights, constants, endpoints and all601 unit boundaries',checks=checks)
(A/'independent_synthetic_v3.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2));raise SystemExit(0 if all_pass else 1)
