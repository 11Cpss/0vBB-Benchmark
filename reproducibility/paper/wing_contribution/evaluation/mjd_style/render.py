#!/usr/bin/env python3
"""Portable replay of two existing MJD plots, using frozen plotted data only.

python render.py --output-dir /path/to/figures --title-weight normal
The bold option is only a style-control replay for verification.
"""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/energybench-mjd-style-mpl')
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from paper_style import apply_paper_style
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_signature(a):
    a=np.ascontiguousarray(a)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def geometry(fig):
    axes=[]
    for ax in fig.findobj(Axes):
        axes.append({'position':list(ax.get_position().bounds),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'lines':[{'x':array_signature(line.get_xdata()),'y':array_signature(line.get_ydata()),'color':line.get_color(),'style':line.get_linestyle(),'width':line.get_linewidth()} for line in ax.lines],'collections':[{'offsets':array_signature(c.get_offsets()),'sizes':array_signature(c.get_sizes())} for c in ax.collections]})
    return axes
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--title-weight',choices=['normal','bold'],default='normal');parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    examples=list(csv.DictReader((ROOT/'data/mjd_example_waveforms.csv').open()))
    population=list(csv.DictReader((ROOT/'data/mjd_waveform_population.csv').open()))
    summary=json.loads((ROOT/'data/mjd_waveform_summary.json').read_text())
    assert len(examples)==7600 and len(population)==410
    assert all(int(row['detector'])==133 and int(row['run_number'])==50928 for row in population)
    cohort=np.array([r['cohort'] for r in population]);energy=np.array([float(r['energy_keV']) for r in population]);rms=np.array([float(r['normalized_baseline_rms']) for r in population])
    rho=float(spearmanr(energy,rms).statistic)
    assert abs(rho-summary['spearman_energy_normalized_baseline_rms'])<1e-12
    common=dict(plt=plt,np=np,Line2D=Line2D,title_weight=args.title_weight,cohort=cohort,energy=energy,rms=rms,rho=rho)
    receipt={'title_weight':args.title_weight,'inputs':{p.name:sha(p) for p in sorted((ROOT/'data').iterdir())},'population_events':410,'example_event_ids':[2407207,2585508],'outputs':{}}
    for name,snippet in [('mjd_motivation_main','main_plot.py.inc'),('mjd_low_high_waveforms','low_high_plot.py.inc')]:
        plt.rcdefaults();apply_paper_style();scope=dict(common)
        if name=='mjd_motivation_main':scope['examples']=examples
        else:
            packed=np.load(ROOT/'data/selected_waveforms_exact.npz',allow_pickle=False)
            chosen=packed['population_positions'];centered=np.zeros((410,3800),dtype=np.float32);normalized=np.zeros_like(centered)
            centered[chosen]=packed['centered'];normalized[chosen]=packed['normalized']
            ids=np.array([int(r['event_id']) for r in population]);assert ids[chosen].tolist()==receipt['example_event_ids']
            q25,q75=summary['energy_quartiles_keV'];low=cohort=='Low energy';high=cohort=='High energy'
            scope.update(detector=133,run=50928,examples=chosen,low=low,high=high,q25=q25,q75=q75,meta={'id':ids},indices=np.arange(410),waves=np.empty((0,3800),dtype=np.float32),centered=centered,normalized=normalized)
        exec(compile((ROOT/snippet).read_text(),str(ROOT/snippet),'exec'),scope)
        fig=scope['fig'];plot_geometry=geometry(fig)
        output=args.output_dir/f'{name}.pdf';fig.savefig(output,dpi=300 if name=='mjd_motivation_main' else 220,metadata={'CreationDate':None,'ModDate':None})
        receipt['outputs'][name]={'path':str(output),'sha256':sha(output),'figure_inches':fig.get_size_inches().tolist(),'geometry':plot_geometry,'geometry_sha256':hashlib.sha256(json.dumps(plot_geometry,sort_keys=True).encode()).hexdigest()}
        plt.close(fig)
    if args.receipt:args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='geometry'} for k,v in receipt['outputs'].items()},indent=2))
if __name__=='__main__':main()
