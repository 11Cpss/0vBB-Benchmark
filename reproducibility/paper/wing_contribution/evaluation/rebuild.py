#!/usr/bin/env python3
"""Re-render reviewed paper tables and figures without external prediction files."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from tables import render_all
from illustration import read_prepared
from unified_metrics import load_config,fingerprint
ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'figures/data')
    parser.add_argument('--output-dir',type=Path,default=ROOT,help='Output root containing tables/ and figures/; defaults to wing_contribution')
    parser.add_argument('--compile',action='store_true',help='Compile the paper in place with latexmk after re-rendering')
    args=parser.parse_args()
    if args.compile and args.output_dir.resolve()!=ROOT.resolve():parser.error('--compile requires the default in-place output directory')
    data_files=list(args.data_dir.glob('*'));before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files if p.is_file()}
    read_prepared(args.data_dir)
    names=render_all(args.data_dir,args.output_dir/'tables')
    for script in ('plot_next_capacity.py','plot_supernemo_matching.py'):
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/script),'--data-dir',str(args.data_dir),'--output-dir',str(args.output_dir/'figures')],check=True)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files if p.is_file()}
    assert before==after,'Rendering must not modify reviewed source data'
    registry=json.loads((args.data_dir/'unified_results.json').read_text())['protocol_registry']
    report={'default_evaluator_protocol':load_config()['protocol_version'],'default_evaluator_sha256':fingerprint(load_config()),'rendered_protocol_registry':registry,'output_dir':str(args.output_dir),'tables':names,'figures':['next_capacity_scores.pdf','next_capacity_scores.png','supernemo_energy_matching_auc.pdf','supernemo_energy_matching_auc.png'],'reviewed_data_unchanged':True,'event_data_required':False,'operation':'render reviewed aggregate data; no metric recomputation'}
    (args.output_dir/'rebuild_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.compile:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','iclr2027_conference.tex'],cwd=ROOT.parent,check=True)
    print(json.dumps({k:v for k,v in report.items() if k!='rendered_protocol_registry'},indent=2))

if __name__=='__main__':main()
