#!/usr/bin/env python3
"""Inventory literal active LaTeX inputs, figure assets, and table environments."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(s):return '\n'.join(re.split(r'(?<!\\)%',line,maxsplit=1)[0] for line in s.splitlines())
def main():
 p=argparse.ArgumentParser();p.add_argument('--paper-dir',type=Path,default=Path(__file__).resolve().parents[2]/'paper');p.add_argument('--output',type=Path);a=p.parse_args();root=a.paper_dir.resolve();seen=set();files=[];includes=[];figures=[];tables=[]
 def visit(rel):
  f=root/rel
  if not f.suffix:f=f.with_suffix('.tex')
  if not f.exists():raise FileNotFoundError(f)
  rel=f.relative_to(root).as_posix()
  if rel in seen:return
  seen.add(rel);src=clean(f.read_text());files.append({'path':rel,'sha256':sha(f)})
  for match in re.finditer(r'\\(?:input|include)\s*\{([^}]+)\}',src):
   child=match.group(1);includes.append({'from':rel,'line':src[:match.start()].count('\n')+1,'to':child});visit(child)
  for block in re.finditer(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}',src,re.S):
   kind=block.group(1);body=block.group(0);labels=re.findall(r'\\label\{([^}]+)\}',body)
   if kind.startswith('table'):tables.append({'source_tex':rel,'line':src[:block.start()].count('\n')+1,'labels':labels,'rendering':'native LaTeX table; see mapping for generated versus manually maintained source'})
  for match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}',src):
   target=match.group(1);asset=root/target
   if not asset.suffix:
    asset=next((root/(target+ext) for ext in ['.pdf','.png','.jpg','.eps'] if (root/(target+ext)).exists()),asset)
   figs={'asset':asset.relative_to(root).as_posix(),'source_tex':rel,'line':src[:match.start()].count('\n')+1,'exists':asset.exists(),'sha256':sha(asset) if asset.exists() else None,'bytes':asset.stat().st_size if asset.exists() else None}
   if target.startswith('dataset_description/Images/'):
    figs.update(status='archived_asset_only_generator_not_located',generator=None,source_commit='9360b04fa52568562adcf1702231527d06a1146a',limitation='Eight PDFs were imported through Overleaf on 2026-09-21. No matching generator, source event selection manifest, or exact figure input was found locally. Existing PDF bytes reproduce the manuscript panel, but regeneration from raw events is not established.')
   elif 'mjd_motivation_main_v2_generated' in target:
    figs.update(status='script_and_frozen_inputs_available',generator='wing_contribution/scripts/plot_mjd_motivation_v2.py',inputs=['wing_contribution/evaluation/mjd_style/data/mjd_example_waveforms.csv','wing_contribution/figures/data/supernemo_matching_50keV.csv','wing_contribution/figures/data/supernemo_matching_roc.csv','wing_contribution/figures/data/supernemo_matching_evidence.json'],note='Current uncommitted figure selected by active manuscript. Panel (a) MJD; panels (b,c) separate SuperNEMO 0nu/Bi214 energy-only illustration, not trained MJD output.')
   elif 'mjd_low_high_waveforms' in target:
    figs.update(status='script_and_frozen_inputs_available',generator='wing_contribution/evaluation/mjd_style/render.py',inputs=['wing_contribution/evaluation/mjd_style/data/selected_waveforms_exact.npz','wing_contribution/evaluation/mjd_style/data/mjd_waveform_summary.json'])
   elif 'next_capacity_scores' in target:
    figs.update(status='script_and_frozen_inputs_available',generator='wing_contribution/scripts/plot_next_capacity.py',inputs=['wing_contribution/figures/data/unified_results.json','wing_contribution/figures/data/next_capacity_plot.csv','wing_contribution/figures/data/next_unified_inventory.csv'],note='Retained NEXT v2 profile; inspect per-record protocol tags rather than relabeling as v3.')
   elif any(x in target for x in ['energy_bias_spectrum','energy_threshold_tradeoff','supernemo_extent_energy_population']):
    figs.update(status='script_and_frozen_inputs_available',generator='wing_contribution/scripts/plot_appendix_figures.py',inputs=['wing_contribution/evaluation/appendix_figures/data/'])
    if 'supernemo_extent' in target:figs['external_input_bundled']='code/figure_sources/data/supernemo_cohort_ecdf.csv.gz'
   else:figs.update(status='needs_review',generator=None)
   figures.append(figs)
 visit('iclr2027_conference.tex')
 for row in tables:
  name=Path(row['source_tex']).name
  if name=='benchmark_main.tex':row.update(generator='wing_contribution/scripts/build_classification_table.py',inputs=['wing_contribution/figures/data/unified_results.json','wing_contribution/figures/data/classification_transformer_workbook.json','wing_contribution/figures/data/classification_table_additions.json'])
  elif name in ['matching_diagnostics.tex','evaluation_losses.tex','independence_diagnostics.tex','transformer_diagnostics.tex','next_classification.tex']:row.update(generator='wing_contribution/evaluation/tables.py',inputs=['wing_contribution/evaluation/data/diagnostics.json','wing_contribution/figures/data/unified_results.json'])
  else:row.update(generator=None,status='manually_maintained_LaTeX',note='The active LaTeX source is the reproducible table artifact; no numeric table-generation script is asserted. Transformer tokenization details describe recorded implementations, not a plot.')
 data={'schema_version':1,'entry_point':'iclr2027_conference.tex','paper_root_at_inventory':str(root),'method':'Recursively follow literal input/include commands after stripping unescaped percent comments; enumerate only active figure/table environments and includegraphics. No conditional content branches affect these manuscript includes.','active_tex_files':files,'include_edges':includes,'figures':figures,'tables':tables,'summary':{'active_tex_files':len(files),'figure_assets':len(figures),'table_environments':len(tables),'asset_only_figures':sum(x['status']=='archived_asset_only_generator_not_located' for x in figures),'missing_assets':[x['asset'] for x in figures if not x['exists']]},'excluded_inactive_examples':['dataset_description/Images/*.png','wing_contribution/figures/mjd_motivation_main.pdf','wing_contribution/figures/supernemo_energy_matching_auc.pdf','transformer_contribution/tables/tokenization_summary.tex']}
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps(data['summary'],indent=2))
if __name__=='__main__':main()
