#!/usr/bin/env python3
"""Reconstruct active extent plotting arrays from bundled event descriptors.
The exact selected descriptors are supplied alongside this script.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,io,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/extent_inputs');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 source=Path(__file__).resolve().parent/'data/supernemo_geometry_event_descriptors.csv.gz';expected=ROOT/'paper/wing_contribution/evaluation/appendix_figures/data';refcdf=Path(__file__).resolve().parent/'data/supernemo_cohort_ecdf.csv.gz'
 frame=pd.read_csv(source);assert len(frame)==458752;assert not frame[['category','event_ordinal']].duplicated().any()
 ee=np.arange(0.,3300.,100.);rr=np.arange(0.,2240.,20.);hrows=[];mrows=[];crows=[]
 for cat in ['Bi214','0nubb']:
  d=frame[frame.category.eq(cat)].sort_values('event_ordinal');e=d.energy_keV.to_numpy();r=d.radius_gyration_mm.to_numpy();assert np.isfinite(e).all() and np.isfinite(r).all()
  h,_,_=np.histogram2d(e,r,bins=[ee,rr]);n=np.histogram(e,ee)[0];assert h.sum()==len(d);assert np.array_equal(h.sum(1),n)
  probability=np.divide(h,n[:,None],out=np.zeros_like(h),where=n[:,None]>0)
  for k in range(len(n)):
   median=float(np.median(r[(e>=ee[k])&(e<ee[k+1])])) if n[k]>=20 else ''
   mrows.append([cat,float((ee[k]+ee[k+1])/2),median,int(n[k])])
   for j in range(len(rr)-1):hrows.append([cat,float(ee[k]),float(ee[k+1]),float(rr[j]),float(rr[j+1]),int(h[k,j]),int(n[k]),float(probability[k,j]),bool(n[k]>=20)])
  ranked=d.sort_values(['energy_keV','event_ordinal']);size=len(ranked)//4
  for cohort,sub in [('low',ranked.iloc[:size]),('high',ranked.iloc[-size:])]:
   values,count=np.unique(sub.radius_gyration_mm.to_numpy(),return_counts=True);cdf=np.cumsum(count)/size
   crows.extend([cat,cohort,float(x),float(y),size] for x,y in zip(values,cdf))
 def write(name,header,rows):
  out=a.output_dir/name
  with out.open('w',newline='') as f:w=csv.writer(f,lineterminator='\n');w.writerow(header);w.writerows(rows)
  return out
 hist=write('supernemo_conditional_extent_histogram.csv',['category','energy_left_keV','energy_right_keV','radius_left_mm','radius_right_mm','count','n_in_energy_bin','probability_given_energy_bin','display_supported'],hrows)
 median=write('extent_energy_medians.csv',['category','energy_center_keV','median_radius_mm','n_events'],mrows)
 cdf=write('supernemo_cohort_ecdf.csv',['category','energy_cohort','radius_gyration_mm','ecdf','cohort_n'],crows)
 # Text-normalized equality also verifies complete ECDF rows, not a sampled curve.
 checks={'histogram_exact_normalized_csv':hist.read_text()==(expected/hist.name).read_text(),'medians_exact_normalized_csv':median.read_text()==(expected/median.name).read_text(),'complete_ecdf_exact_normalized_csv':cdf.read_text()==gzip.open(refcdf,'rt').read()}
 result={'all_pass':all(checks.values()),'checks':checks,'n_source_events':len(frame),'n_ecdf_rows':len(crows),'input_sha256':digest(source),'output_sha256':{x.name:digest(x) for x in [hist,median,cdf]},'method':'Exact plotted bins and stable energy/event-ordinal quartiles; no event sampling, fitting, metric change or source-file mutation.'}
 (a.output_dir/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['all_pass'] else 1)
if __name__=='__main__':main()
