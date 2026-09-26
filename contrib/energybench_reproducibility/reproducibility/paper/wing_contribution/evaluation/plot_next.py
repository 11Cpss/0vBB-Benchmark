#!/usr/bin/env python3
"""Render publication NEXT capacity figures from regenerated, source-backed CSV."""
from pathlib import Path
import argparse,csv,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(Path(__import__('tempfile').gettempdir())/'energybench-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,NullFormatter
from scipy.stats import spearmanr,rankdata
from result_profiles import profile,fingerprint
from paper_style import apply_paper_style, PAPER_WIDTH_IN
GROUPS={
 'Projection and voxel models':('#276A98','o'),
 'Point-set models':('#D17B28','^'),
 'Graph models':('#798D35','s'),
 'Sequence, state-space, topology and hybrid models':('#B75E91','D')}
RETAINED={'cnn_004_multiview_late_fusion','gnn_001_static_gine','seq_001_bigru'}
EXCLUDED={'cnn_001_two_conv_baseline','cnn_002_global_energy_skip','classic_001_topology_xgboost','ssm_001_pointmamba'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=ROOT/'figures/data');parser.add_argument('--output-dir',type=Path,default=ROOT/'figures');args=parser.parse_args();args.source=args.data_dir/'next_capacity_plot.csv'
 with args.source.open(newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==len({r['architecture_id'] for r in rows})==19
 assert not ({r['architecture_id'] for r in rows}&EXCLUDED)
 assert {r['protocol'] for r in rows}=={profile("overflow601")['protocol_version']}
 assert {int(r['n_events']) for r in rows}=={116549}
 results=json.loads((args.data_dir/'unified_results.json').read_text())['records']
 original={r['architecture_id']:r for r in results if r['dataset']=='NEXT'}
 for row in rows:
  source=original[row['architecture_id']]
  assert source['protocol_sha256']==fingerprint(profile("overflow601"))
  assert float(row['matched_auc'])==source['matched_auc']
  assert float(row['energy_independence_score'])==source['I']
  assert float(row['auc'])==source['inclusive_auc']
  assert float(row['pairwise_ranking_error'])==1.-source['matched_auc']
  assert 50000<int(row['trainable_parameters'])<1600000
  row['retained_architecture']=row['architecture_id'] in RETAINED
 lookup={r['architecture_id']:r for r in rows}
 apply_paper_style()
 fig,axes=plt.subplots(1,2,figsize=(PAPER_WIDTH_IN,3.5));fig.subplots_adjust(left=.12,right=.975,bottom=.19,top=.735,wspace=.46)
 keys=('pairwise_ranking_error','energy_independence_score')
 for ax,key in zip(axes,keys):
  for group,(color,marker) in GROUPS.items():
   subset=[r for r in rows if r['model_group']==group]
   ax.scatter([int(r['trainable_parameters']) for r in subset],[float(r[key]) for r in subset],c=color,marker=marker,s=31,linewidths=.45,edgecolors='white',zorder=4)
  selected=[r for r in rows if r['retained_architecture']]
  ax.scatter([int(r['trainable_parameters']) for r in selected],[float(r[key]) for r in selected],s=83,marker='o',facecolors='none',edgecolors='#111111',linewidths=.9,zorder=5)
  ax.set_xscale('log');ax.set_xlim(50000,1600000);ax.set_xticks([100000,300000,1000000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:'1M' if value==1000000 else f'{value/1000:.0f}k'));ax.xaxis.set_minor_formatter(NullFormatter())
  ax.set_xlabel('Trainable parameters (log scale)',labelpad=5);ax.set_box_aspect(1);ax.grid(axis='y',color='#E3E6E9',linewidth=.55,zorder=0);ax.set_axisbelow(True)
 axes[0].set_yscale('log');axes[0].set_ylim(.0035,.17);axes[0].set_yticks([.005,.01,.02,.05,.1]);axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'));axes[0].yaxis.set_minor_formatter(NullFormatter());axes[0].set_ylabel(r'Ranking error $1-\mathrm{AUC}_{\mathrm{match}}$',labelpad=5);axes[0].set_title('(a) Ranking error ↓',loc='left',pad=8)
 axes[1].set_ylim(.9670,.9754);axes[1].set_yticks([.968,.970,.972,.974]);axes[1].set_ylabel(r'Energy independence $I$',labelpad=5);axes[1].set_title('(b) Energy independence ↑',loc='left',pad=8)
 annotations=[(0,'cnn_004_multiview_late_fusion','Multi-view\nCNN',(.28,.64)),(0,'seq_001_bigru','BiGRU',(.77,.32)),(0,'gnn_001_static_gine','GINE',(.45,.045)),(0,'gnn_003_egnn','EGNN',(.46,.94)),(1,'seq_002_dilated_tcn','TCN',(.49,.94))]
 for panel,model,label,position in annotations:
  row=lookup[model];axes[panel].annotate(label,xy=(int(row['trainable_parameters']),float(row[keys[panel]])),xytext=position,textcoords='axes fraction',fontsize=8,arrowprops={'arrowstyle':'-','color':'#6B7177','linewidth':.55},annotation_clip=False,zorder=6)
 handles=[]
 for group,(color,marker) in GROUPS.items():
  label=group if not group.startswith('Sequence,') else 'Sequence, state-space, topology\nand hybrid models'
  handles.append(Line2D([0],[0],ls='none',marker=marker,color=color,markeredgecolor='white',markeredgewidth=.45,markersize=6.,label=label))
 # The caption supplies the population and figure title; use the top band for keys.
 ring=Line2D([0],[0],ls='none',marker='o',markersize=7.5,markerfacecolor='none',markeredgecolor='#111111',markeredgewidth=.9)
 ring.set_label('Retained architectures')
 fig.legend(handles=handles+[ring],loc='upper left',bbox_to_anchor=(.104,1.0),ncol=2,frameon=False,columnspacing=1.5,handletextpad=.5,handlelength=1.2,labelspacing=.45)
 fig.text(.12,.04,'Two smallest CNNs omitted; no scaling curve is fitted.',fontsize=8,color='#555555')
 fig.canvas.draw();panel_inches=[[float(ax.get_window_extent().width/fig.dpi),float(ax.get_window_extent().height/fig.dpi)] for ax in axes]
 assert all(abs(w-h)<1e-9 for w,h in panel_inches)
 # Numeric scatter clipping check: every retained data mark is within its explicit axes.
 for ax,key in zip(axes,keys):
  assert all(ax.get_ylim()[0]<float(r[key])<ax.get_ylim()[1] for r in rows)
 out=args.output_dir;out.mkdir(parents=True,exist_ok=True);hashes={}
 for suffix in ('pdf','png'):
  target=out/f'next_capacity_scores.{suffix}';kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
  fig.savefig(target,dpi=300,**kwargs);hashes[target.name]=sha(target)
 plt.close(fig)
 params=np.array([int(r['trainable_parameters']) for r in rows]);correlations={}
 for key in ('auc','matched_auc','energy_independence_score'):
  values=np.array([float(r[key]) for r in rows]);rho=float(spearmanr(params,values).statistic);np.testing.assert_allclose(rho,np.corrcoef(rankdata(params),rankdata(values))[0,1],atol=1e-14);correlations[key]=rho
 evidence={'source_csv':str(args.source),'source_csv_sha256':sha(args.source),'full_precision_results':str(args.data_dir/'unified_results.json'),'protocol_version':profile("overflow601")['protocol_version'],'protocol_sha256':fingerprint(profile("overflow601")),'plotted_ids':[r['architecture_id'] for r in rows],'points_per_panel':19,'same_test_events':116549,'each_value_matches_recomputed_results':True,'no_point_jitter_or_score_adjustment':True,'square_panel_inches':panel_inches,'all_points_within_limits':True,'spearman_parameters':correlations,'palette':GROUPS,'output_sha256':hashes,'interpretation':'Single-run point estimates across heterogeneous models; no significance or scaling-law claim.'}
 (out/'next_capacity_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
