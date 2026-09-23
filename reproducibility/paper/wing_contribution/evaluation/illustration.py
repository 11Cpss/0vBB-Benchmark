"""Portable preparation and rendering of the separate SuperNEMO 0nu energy illustration."""
from pathlib import Path
import argparse,csv,hashlib,json,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'energybench-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_npz import load_events
from legacy_v2.core.unified_metrics import evaluate,load_config,fingerprint
from paper_style import apply_paper_style, PAPER_WIDTH_IN
ROOT=Path(__file__).resolve().parent.parent
BLUE,ORANGE,GRAY='#286A9B','#C47723','#555555'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def roc_curve(y,s,w):
    use=np.isfinite(s)&(w>0);y,s,w=y[use],s[use],w[use]
    order=np.argsort(-s,kind='stable');y,s,w=y[order],s[order],w[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(w*(y==1))[ends];fp=np.cumsum(w*(y==0))[ends]
    return np.r_[0.,fp/fp[-1]],np.r_[0.,tp/tp[-1]],np.r_[np.inf,s[ends]]

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def read_prepared(data_dir):
    evidence=json.loads((data_dir/'supernemo_matching_evidence.json').read_text())
    assert evidence['protocol_sha256']==fingerprint(load_config()),'Prepared data use a different protocol'
    with (data_dir/'supernemo_matching_50keV.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    hist_arrays={};display_edges=None
    for kind in ('original','retained','matched'):
        for category in ('0nubb','Bi214'):
            selected=[r for r in rows if r['distribution']==kind and r['category']==category]
            edges=np.array([float(r['left_keV']) for r in selected]+[float(selected[-1]['right_keV'])])
            if display_edges is None:display_edges=edges
            np.testing.assert_array_equal(edges,display_edges)
            np.testing.assert_array_equal([float(r['right_keV']) for r in selected],edges[1:])
            np.testing.assert_allclose(np.diff(edges),50,rtol=0,atol=0)
            density=np.array([float(r['density_per_keV']) for r in selected])
            assert np.all(density>=0)
            np.testing.assert_allclose(np.dot(density,np.diff(edges)),1,rtol=0,atol=1e-12)
            for r,d,width in zip(selected,density,np.diff(edges)):
                np.testing.assert_allclose(d,float(r['base_or_matched_mass'])/float(r['normalizing_class_mass'])/width,rtol=0,atol=1e-14)
            hist_arrays[kind,category]=density
    with (data_dir/'supernemo_matching_roc.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    curves={}
    for kind in ('original','common_support','retained','matched'):
        selected=[r for r in rows if r['distribution']==kind]
        x=np.array([float(r['fpr']) for r in selected]);y=np.array([float(r['tpr']) for r in selected]);auc=float(np.trapezoid(y,x))
        assert np.all(np.diff(x)>=0) and np.all(np.diff(y)>=0)
        np.testing.assert_allclose([x[0],y[0],x[-1],y[-1]],[0,0,1,1],rtol=0,atol=1e-12)
        np.testing.assert_allclose(auc,evidence['stages'][kind]['auc'],rtol=0,atol=1e-10)
        curves[kind]=(x,y,auc)
    np.testing.assert_allclose(curves['matched'][2],evidence['diagnostic_matched_auc'],rtol=0,atol=1e-10)
    return evidence,display_edges,hist_arrays,curves

def prepare(predictions,output_dir):
    data=load_events(predictions);data.setdefault('weight',np.ones(len(data['score'])))
    assert np.array_equal(data['score'],data['energy_keV']),'This separate illustration requires the fixed score s=E in keV'
    assert np.all(np.isfinite(data['energy_keV'])) and np.all(data['energy_keV']>=0)
    if 'group' in data:np.testing.assert_array_equal(data['group'],np.where(data['label']==1,'0nubb','Bi214'))
    metrics,mask=evaluate(data['label'],data['score'],data['energy_keV'],data.get('group'),data['weight'],return_arrays=True)
    y,s,e=data['label'],data['score'],data['energy_keV'];w=data['weight'];mw=mask['matched_weight']
    assert metrics['matching']['diagnostic_auc_not_for_reporting'] is not None,'Matching is not estimable for this illustration'
    display_edges=np.arange(0.,max(3300.,np.ceil(e.max()/50)*50)+1,50.)
    histograms=[]
    for kind,weights in [('original',w),('retained',w*mask['retained']),('matched',mw)]:
        for c,category in [(1,'0nubb'),(0,'Bi214')]:
            take=y==c;counts=np.histogram(e[take],bins=display_edges,weights=weights[take])[0];density=counts/weights[take].sum()/np.diff(display_edges)
            for i,d in enumerate(density):histograms.append(dict(distribution=kind,category=category,left_keV=display_edges[i],right_keV=display_edges[i+1],base_or_matched_mass=counts[i],normalizing_class_mass=weights[take].sum(),density_per_keV=d))
    matching=metrics['matching'];curve_rows=[];stages={}
    for kind,weights,reference in [('original',w,metrics['inclusive_auc']),('common_support',w*mask['common_support'],matching['common_support_auc']),('retained',w*mask['retained'],matching['retained_unweighted_auc']),('matched',mw,matching['diagnostic_auc_not_for_reporting'])]:
        fpr,tpr,thresholds=roc_curve(y,s,weights);auc=float(np.trapezoid(tpr,fpr));np.testing.assert_allclose(auc,reference,rtol=0,atol=1e-10)
        stages[kind]=dict(auc=auc,signal_events=int(((y==1)&(weights>0)).sum()),background_events=int(((y==0)&(weights>0)).sum()))
        for a,b,t in zip(fpr,tpr,thresholds):curve_rows.append(dict(distribution=kind,fpr=a,tpr=b,threshold_keV=float(t) if np.isfinite(t) else 'inf'))
    threshold=2200.;tpr=float(np.average(e[y==1]>=threshold,weights=w[y==1]));fpr=float(np.average(e[y==0]>=threshold,weights=w[y==0]))
    evidence=dict(protocol_version=metrics['protocol_version'],protocol_sha256=metrics['protocol_sha256'],energy_protocol=load_config()['energy'],source=str(predictions),source_sha256=sha(predictions),display_bin_width_keV=50,display_range_keV=[0,float(display_edges[-1])],stages=stages,I=metrics['independence']['I'],matching_status=matching['status'],formal_matched_auc=matching['matched_auc'],diagnostic_matched_auc=matching['diagnostic_auc_not_for_reporting'],common_support_keV=matching['common_support_keV'],valid_matching_bins=matching['valid_bin_count'],coverage_original_finite={str(c):matching['classes'][str(c)]['matched_fraction_original_finite'] for c in (0,1)},threshold=dict(keV=threshold,tpr=tpr,fpr=fpr,balanced_accuracy=.5*(tpr+1-fpr)))
    output_dir.mkdir(parents=True,exist_ok=True);write_csv(output_dir/'supernemo_matching_50keV.csv',histograms);write_csv(output_dir/'supernemo_matching_roc.csv',curve_rows)
    (output_dir/'supernemo_matching_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    read_prepared(output_dir)
    return evidence

def prepare_main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--predictions',type=Path,help='Optional external event NPZ for a fresh metric calculation');parser.add_argument('--data-dir',type=Path,default=ROOT/'figures/data');parser.add_argument('--output-dir',type=Path,help='Required with --predictions; never overwrite reviewed data implicitly');args=parser.parse_args()
    if args.predictions:
        if args.output_dir is None:parser.error('--predictions requires --output-dir')
        evidence=prepare(args.predictions,args.output_dir)
    else:
        if args.output_dir is not None:parser.error('--output-dir requires --predictions')
        evidence,*_=read_prepared(args.data_dir)
    print(json.dumps({'prepared_data_valid':True,'protocol':evidence['protocol_version'],'I':evidence['I'],'formal_matched_auc':evidence['formal_matched_auc']}))

def plot_main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=ROOT/'figures/data');parser.add_argument('--output-dir',type=Path,default=ROOT/'figures');args=parser.parse_args()
    evidence,display_edges,hist_arrays,curves=read_prepared(args.data_dir)
    protocol=load_config();matching={'matched_auc':evidence['formal_matched_auc'],'common_support_keV':evidence['common_support_keV'],'valid_bin_count':evidence['valid_matching_bins']};metrics={'independence':{'I':evidence['I']}}
    valid=matching['matched_auc'] is not None;covs={int(c):v for c,v in evidence['coverage_original_finite'].items()};fpr,tpr=(evidence['threshold'][k] for k in ('fpr','tpr'))
    OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    apply_paper_style()
    fig=plt.figure(figsize=(PAPER_WIDTH_IN,2.65))
    grid=fig.add_gridspec(1,3,left=.095,right=.97,top=.86,bottom=.34,wspace=.42)
    before=fig.add_subplot(grid[0]);after=fig.add_subplot(grid[1],sharex=before,sharey=before);roc=fig.add_subplot(grid[2])
    for ax,kind in [(before,'original'),(after,'matched')]:
        for category,label,color,style in [('0nubb',r'$0\nu\beta\beta$',BLUE,'-'),('Bi214',r'$^{214}$Bi',ORANGE,'--')]:
            ax.stairs(hist_arrays[kind,category]*1000,display_edges,color=color,linestyle=style,linewidth=1.1,label=label,zorder=3)
        ax.set(xlim=(0,display_edges[-1]),ylim=(0,3.1),xlabel='Energy (keV)');ax.set_xticks([0,1000,2000,3000]);ax.set_yticks([0,1,2,3])
    before.set_ylabel(r'Density ($10^{-3}$ keV$^{-1}$)',labelpad=3);before.set_title('(a) Original spectra',loc='left',pad=6)
    after.set_title('(b) Matched diagnostic' if not valid else '(b) After matching',loc='left',pad=6);after.tick_params(labelleft=False)
    before.legend(frameon=False,loc='upper left',handlelength=1.8,borderaxespad=.1,labelspacing=.35)
    for edge in matching['common_support_keV']:after.axvline(edge,color='#b4b4b4',linewidth=.6,linestyle=':',zorder=1)
    after.text(.045,.965,'5 keV matching bins',transform=after.transAxes,va='top',fontsize=8,color=GRAY)
    for kind,color,style,label in [('original',BLUE,'-','Original'),('retained',GRAY,'-.','Retained'),('matched',ORANGE,'--','Matched' if valid else 'Matched*')]:
        x,z,a=curves[kind];roc.plot(x,z,color=color,linestyle=style,linewidth=1.1,label=f'{label}  {a:.4f}')
    roc.plot([0,1],[0,1],color='#999999',linestyle=':',linewidth=.8)
    roc.scatter([fpr],[tpr],color=BLUE,edgecolors='white',s=21,linewidths=.7,zorder=5)
    roc.annotate('2200 keV',xy=(fpr,tpr),xytext=(.30,.70),fontsize=8,arrowprops={'arrowstyle':'-','color':GRAY,'linewidth':.7})
    roc.set(xlim=(0,1),ylim=(0,1),xlabel='Background acceptance',ylabel='Signal efficiency');roc.set_xticks([0,.5,1]);roc.set_yticks([0,.5,1]);roc.set_title('(c) Energy-only ROC',loc='left',pad=6)
    roc.legend(frameon=False,loc='lower right',title='AUC',fontsize=7,title_fontsize=7.5,handlelength=.9,handletextpad=.35,borderaxespad=.1,labelspacing=.18)
    for ax in [before,after,roc]:ax.set_box_aspect(1);ax.grid(axis='y',color='#e6e6e6',linewidth=.5);ax.set_axisbelow(True);ax.tick_params(width=.7,length=3,pad=2)
    overflow='overflow' in json.dumps(protocol.get('energy',{})).lower()
    scope='5 keV + overflow' if overflow else '0–3000 keV'
    fig.text(.095,.135,rf"$I = {metrics['independence']['I']:.3f}$; retained: {100*covs[1]:.1f}% signal, {100*covs[0]:.1f}% background (original counts).",fontsize=8,color=GRAY)
    message=f"{matching['valid_bin_count']} matching bins ({scope}); spectra displayed in 50 keV bins."
    if not valid:message='* Diagnostic only: signal coverage < 50%; formal matched AUC unavailable.'
    fig.text(.095,.06,message,fontsize=8,color=GRAY)
    for suffix in ['pdf','png']:
        kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
        fig.savefig(OUT/f'supernemo_energy_matching_auc.{suffix}',dpi=300,facecolor='white',**kwargs)
    plt.close(fig)
    print(json.dumps({'output_dir':str(OUT),'protocol':evidence['protocol_version'],'source_data_unchanged':True}))
