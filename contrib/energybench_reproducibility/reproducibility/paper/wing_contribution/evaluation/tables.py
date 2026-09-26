"""Render published tables from final repository data without reading event files."""
from pathlib import Path
import csv,importlib.util,json
from result_profiles import profile,fingerprint
ROOT=Path(__file__).resolve().parent.parent
NAMES={'mvcnn':'Multi-view CNN','gine':'Static GINE','bigru':'BiGRU','mamba':'PointMamba-lite'}
DS=('NEXT','MJD','EXO-200','SuperNEMO')

def table(caption,label,columns,lines):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering\footnotesize',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\setlength{\tabcolsep}{4pt}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{'+columns+'}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}}',r'\end{table}',''])

def render_all(data_dir,output_dir):
    source=json.loads((data_dir/'unified_results.json').read_text());rr=source['records'];lookup={(r['dataset'],r['model_key']):r for r in rr}
    registry=source.get('protocol_registry',{fingerprint(source['protocol']):source['protocol']})
    assert all(key==fingerprint(config) for key,config in registry.items())
    assert fingerprint(profile("strict600")) in registry
    details=json.loads((ROOT/'evaluation/data/diagnostics.json').read_text())['records'];details={(r['dataset'],r['model_key']):r for r in details}
    for key,d in details.items():
        r=lookup[key];assert d['protocol_sha256']==r['protocol_sha256'] and r['protocol_sha256'] in registry;assert d['source_input_sha256']==r['input_sha256']
    spec=importlib.util.spec_from_file_location('paper_table_builder',ROOT/'scripts/build_classification_table.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    main=json.loads((data_dir/'main_table_results.json').read_text())['records']
    rendered={'benchmark_main.tex':builder.render(main)}
    provenance=json.loads((data_dir/'next_exploratory_provenance.json').read_text())
    earlier=lookup[provenance['dataset'],provenance['model_key']]
    assert earlier['status']=='recomputed'
    assert earlier['architecture_id']==provenance['architecture_id']=='ssm_001_pointmamba'
    assert earlier['input_sha256']==provenance['prediction_input_sha256']
    assert earlier['n_events']==provenance['n_events']==115499
    assert earlier['protocol_sha256']==fingerprint(source['protocol'])
    earlier_params=provenance['parameter_count'];assert isinstance(earlier_params,int) and earlier_params>0
    with (data_dir/'next_unified_inventory.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    ready={r['architecture_id']:r for r in rr if r['dataset']=='NEXT' and r['status']=='recomputed'}
    lines=[r'Model & Params (M) & Inclusive AUC $\uparrow$ & Matched AUC $\uparrow$ & $I$ $\uparrow$ \\',r'\midrule'];last=None
    for row in rows:
        arch=row['architecture_id'];r=ready.get(arch);group=row['model_group']
        params=row['trainable_parameters']
        if arch=='ssm_001_pointmamba':r=earlier;params=earlier_params
        if group!=last:
            if last is not None:lines.append(r'\midrule')
            lines.append(r'\multicolumn{5}{l}{\textit{'+group+r'}} \\');last=group
        retained=arch in ('cnn_004_multiview_late_fusion','gnn_001_static_gine','seq_001_bigru','ssm_001_pointmamba');cells=[row['display_name']+(r'$^{\dagger}$' if retained else ''),f'{int(params)/1e6:.4f}' if params else r'\textemdash']
        for field in ('inclusive_auc','matched_auc','I'):
            value=r[field] if r else None;text=r'\textemdash' if value is None else f'{value:.6f}'
            cells.append(text)
        lines.append(' & '.join(cells)+r' \\')
    assert len(rows)==23
    rendered['next_classification.tex']=table('NEXT classification candidates reevaluated with 600 regular 5 keV bins on 0--3000 keV plus one overflow bin above 3000 keV. Params (M) counts trainable neural parameters in millions. A dash denotes an inapplicable parameter count. $^{\\dagger}$Architecture instantiated across datasets; no new model is fitted.','tab:next-classification','lrrrr',lines)
    lines=[r'Dataset & Common support (keV) & Bins & $\gamma_1$ & $\gamma_0$ & ESS$_1$ & ESS$_0$ \\',r'\midrule'];losses=[r'Dataset & $N_1/N_0$ & Range loss & $E>3000$ & $I$ sparse loss & Support loss & Matching sparse loss \\',r'\midrule']
    for ds in DS:
        d=details[ds,'mvcnn'];m=d['matching'];c=m['classes'];lo,hi=m['common_support_keV']
        lines.append(f'{ds} & [{lo:.2f}, {hi:.2f}] & {m["valid_bin_count"]} & {c["1"]["matched_fraction_original_finite"]:.4f} & {c["0"]["matched_fraction_original_finite"]:.4f} & {c["1"]["matched_ess"]:.0f} & {c["0"]["matched_ess"]:.0f}'+r' \\')
        gm=d['group_for_label'];ig=d['independence']['groups'];pair=lambda field:'/'.join(str(c[k][field]['n']) for k in ('1','0'));isp='/'.join(str(ig[gm[k]]['sparse_excluded']['n']) for k in ('1','0'))
        above='/'.join(str(c[k].get('above_range',c[k].get('overflow'))['n']) for k in ('1','0'))
        losses.append(f'{ds} & {pair("input")} & {pair("range_excluded")} & {above} & {isp} & {pair("support_excluded_after_range")} & {pair("sparse_excluded_after_support")}'+r' \\')
    rendered['matching_diagnostics.tex']=table('Matching diagnostics for the multi-view CNN prediction sets under their dataset-specific profiles (Table~\\ref{tab:metric-protocols}). All reevaluated classic models within each displayed test population use identical test event IDs, labels, physical energies, and base weights. Coverage $\\gamma_c$ uses the original finite score/energy class base mass before energy-range exclusion; ESS uses matched weights.','tab:matching-diagnostics','lrrrrrr',lines)
    rendered['evaluation_losses.tex']=table('Disjoint matching losses and the separate independence sparse-bin loss, reported as positive/negative event counts. $N_1/N_0$ is the original test population. Range loss precedes support estimation, then matching sparse-bin exclusion. For MJD and EXO-200, range loss includes energies below 0 or above 3000 keV; the $E>3000$ column is a subset of that loss. For retained NEXT and SuperNEMO results, only negative energies are excluded and $E>3000$ counts are retained overflow diagnostics. The separate $I$ sparse loss is measured after the applicable range filtering. Counts are shared by reevaluated classic models within each displayed test population.','tab:evaluation-losses','lrrrrrr',losses)
    lines=[r'Dataset / model & $I_1$ & $I_0$ & $\min_g I_g$ & Bins$_{1/0}$ & Retained$_{1/0}$ \\',r'\midrule']
    for ds in DS:
        for key,name in NAMES.items():
            r=earlier if (ds,key)==('NEXT','mamba') else lookup[ds,key]
            if r['status']!='recomputed':continue
            full=details[r['dataset'],r['model_key']];d=full['independence'];gm=full['group_for_label'];a,b=(d['groups'][gm[k]] for k in ('1','0'))
            lines.append(f'{ds} / {name} & {a["I_g"]:.4f} & {b["I_g"]:.4f} & {d["min_group_I"]:.4f} & {a["valid_bin_count"]}/{b["valid_bin_count"]} & {a["retained_fraction_original_finite"]:.4f}/{b["retained_fraction_original_finite"]:.4f}'+r' \\')
    rendered['independence_diagnostics.tex']=table('Class-level independence diagnostics under the profiles in Table~\\ref{tab:metric-protocols}. Retained fractions divide eligible-bin base mass by original finite score/energy class base mass, before energy-range exclusion. Score histograms retain up to 20 weighted score-quantile bins. The audit files include all extended NEXT configurations and full precision.','tab:independence-diagnostics','lrrrrr',lines)
    lines=[r'Dataset / Transformer & $I_1/I_0$ & Bins$_{I,1/0}$ & Retained$_{I,1/0}$ & $K_{\rm match}$ & $\gamma_1/\gamma_0$ & ESS$_{1/0}$ \\',r'\midrule']
    for selected in source.get('transformer_reevaluation',{}).get('selected_models',[]):
        ds,key=selected['dataset'],selected['model_key'];d=details[ds,key];m=d['matching'];c=m['classes'];gm=d['group_for_label'];ig=d['independence']['groups'];a,b=(ig[gm[k]] for k in ('1','0'))
        name=key.replace('_','--')
        lines.append(f'{ds} / {name} & {a["I_g"]:.4f}/{b["I_g"]:.4f} & {a["valid_bin_count"]}/{b["valid_bin_count"]} & {a["retained_fraction_original_finite"]:.4f}/{b["retained_fraction_original_finite"]:.4f} & {m["valid_bin_count"]} & {c["1"]["matched_fraction_original_finite"]:.4f}/{c["0"]["matched_fraction_original_finite"]:.4f} & {c["1"]["matched_ess"]:.0f}/{c["0"]["matched_ess"]:.0f}'+r' \\')
    rendered['transformer_diagnostics.tex']=table('Diagnostics for the 15 reevaluated MJD and EXO-200 Transformers using 600 fixed 5 keV bins on 0--3000 keV. All match the CNN test identities, class partitions, physical energies and unit weights within their dataset, so common support and matching losses equal Tables~\\ref{tab:matching-diagnostics}--\\ref{tab:evaluation-losses}. Retained fractions and coverage use original finite-pair class mass before range exclusion. The minimum group score is the smaller of the two displayed $I_g$ values. All formal matched AUCs pass the reporting gates.','tab:transformer-diagnostics','lrrrrrr',lines)
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,text in rendered.items():(output_dir/name).write_text(text)
    return list(rendered)
