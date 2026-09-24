#!/usr/bin/env python3
"""Portable entry points. Writes only below the selected output directory."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper';WING=PAPER/'wing_contribution'

def call(args,log,cwd=ROOT):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(log.parent/'mpl_cache')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as out:
        subprocess.run([str(x) for x in args],cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('PASS',log.name,flush=True)

def render(output):
    output.mkdir(parents=True,exist_ok=True)
    py=[sys.executable,'-B']
    call(py+[WING/'scripts/rebuild_unified_evaluation.py','--output-dir',output],output/'logs/tables_and_matching.log')
    tables={}
    for path in (output/'tables').glob('*.tex'):
        reference=WING/'tables'/path.name
        tables[path.name]=path.read_bytes()==reference.read_bytes()
    if not all(tables.values()):raise AssertionError(tables)
    call(py+[WING/'scripts/plot_mjd_motivation_v2.py','--output-stem',output/'figures/mjd_motivation_main_v2_generated'],output/'logs/mjd_motivation.log')
    call(py+[WING/'evaluation/mjd_style/render.py','--output-dir',output/'figures'],output/'logs/mjd_waveforms.log')
    call(py+[WING/'scripts/plot_appendix_figures.py','--extent-cdf',ROOT/'code/figure_sources/data/supernemo_cohort_ecdf.csv.gz','--output-dir',output/'figures'],output/'logs/appendix_figures.log')
    record={'all_generated_tables_byte_match_final_paper':tables,'event_evaluation_performed':False,'figures_generated_from_final_display_inputs':True,'source_paper_modified':False,'dataset_example_pdf_generation':'Original generating code unavailable; eight supplied PDF assets remain in paper/dataset_description/Images.'}
    (output/'render_validation.json').write_text(json.dumps(record,indent=2)+'\n')

def compile_paper(output,engine,only_cached):
    work=output/'manuscript'
    if work.exists():raise FileExistsError(f'Choose a new output directory; existing build preserved: {work}')
    shutil.copytree(PAPER,work,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Always compile the supplied source in a separate build directory.
    (work/'iclr2027_conference.pdf').unlink(missing_ok=True)
    binary=shutil.which(engine) or (str(Path(engine).resolve()) if Path(engine).is_file() else None)
    if not binary:raise FileNotFoundError(f'Install {engine}, or pass --engine /path/to/tectonic')
    if Path(binary).name.startswith('tectonic'):
        args=[binary,'--keep-logs','--keep-intermediates']
        if only_cached:args.append('--only-cached')
        args+=['-Z',f'search-path={ROOT / "environments/texfonts"}','iclr2027_conference.tex']
    else:args=[binary,'-pdf','-interaction=nonstopmode','-halt-on-error','iclr2027_conference.tex']
    call(args,output/'logs/compile.log',cwd=work)
    pdf=work/'iclr2027_conference.pdf';shutil.copy2(pdf,output/'paper.pdf')
    receipt={'engine':binary,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'built_from_current_working_tree_snapshot':True,'original_files_modified':False}
    (output/'compile_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','render','compile'))
    p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--engine',default='latexmk');p.add_argument('--only-cached',action='store_true');a=p.parse_args();output=a.output.resolve()
    if output==PAPER or PAPER in output.parents:raise ValueError('Write builds outside the frozen paper snapshot')
    if a.action=='render':render(output)
    elif a.action=='compile':compile_paper(output,a.engine,a.only_cached)
    else:
        call([sys.executable,'-B',ROOT/'benchmark/tests/test_metrics.py'],output/'logs/numerical_tests.log')
        call([sys.executable,'-B',ROOT/'code/input_preparation/tests/test_standardize.py'],output/'logs/event_alignment_tests.log')
if __name__=='__main__':main()
