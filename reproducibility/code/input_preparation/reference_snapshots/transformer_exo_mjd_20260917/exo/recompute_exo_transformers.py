#!/usr/bin/env python3
"""Audit/replay EXO Transformer event predictions with the frozen unified v2.

Read-only source campaign. No inference, optimization, training, model change,
test-set change or writes to /home/klz. All outputs are adjacent to this script.
"""
from __future__ import annotations
import csv,dataclasses,hashlib,importlib.util,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
SOURCE=Path('/home/klz/Data/zeronu_benchmark/Transformer_Approach/exo200_detector')
PAPER=Path('/home/wenyu/iclr final paper/overleaf')
EVALUATION=PAPER/'wing_contribution/evaluation'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'runtime/mpl'))
sys.path[:0]=[str(SOURCE/'frozen_energybench'),str(SOURCE)]
import numpy as np
import torch
from exobench.config import DataConfig
from exo_transformer.energy_aware import build_test_metadata
from exo_transformer.model import EXOTransformerClassifier
from exo_transformer.tokenization import TokenizationConfig
from energybench.dependence import evaluate_dependence
from energybench.roc import evaluate_energy_matched_roc

MODEL_MAP={f'{family}_{encoding}':f'classification__{token}__{pe}' for family,token in [('entity','pulse_entities'),('region','raw_patches'),('summary','segment_summary')] for encoding,pe in [('mlp','coordinate_mlp'),('fourier','fourier_coordinates'),('rope','rope')]}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def model_check(run,config,provenance):
    checkpoint=run/'best.pt';assert sha(checkpoint)==provenance['checkpoint_sha256']
    representation=config['representation'];tokens=representation['tokenization']
    selected={k:tokens[k] for k in TokenizationConfig.__dataclass_fields__ if k in tokens}
    model_args={k:representation[k] for k in ['position_encoding','d_model','nhead','num_layers','dim_feedforward','dropout','num_frequencies']}
    if representation.get('rope_base') is not None:model_args['rope_base']=representation['rope_base']
    model=EXOTransformerClassifier(tokenization_config=TokenizationConfig(**selected),**model_args)
    packed=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(packed['model_state_dict'],strict=True)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count==config['parameter_count']
    summary=json.loads((run/'run_summary.json').read_text())
    assert summary['parameter_count']==count and summary['best_epoch']==packed['epoch']
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'best_epoch':int(packed['epoch']),'validation_selection_auc':float(packed['score']),'parameter_count':int(count),'parameter_count_validation':'Instantiated exact source architecture/configuration; strict-loaded original best.pt with weights_only=True; counted requires_grad parameters; equals run_config and run_summary. No forward pass.', 'representation':representation,'run_summary':summary}

def replay(a):
    i=evaluate_dependence(a['score'],a['energy_keV'],a['label'],category=a['group'],sample_weight=a['weight'],n_energy_bins=8,n_score_bins=20,min_per_bin=20,distance_correlation_max_samples=4,seed=42)
    auc=evaluate_energy_matched_roc(a['label'],a['score'],a['energy_keV'],a['weight'],positive_label=1,n_bins=6,min_per_class=20,target='overlap',target_tpr=.9,n_bootstrap=0,random_state=42,support_trim_quantile=.005)
    diagnostics=auc.to_dict()
    for curve_name in ['inclusive','matched']:
        if diagnostics.get(curve_name):
            diagnostics[curve_name]={k:v for k,v in diagnostics[curve_name].items() if k not in ['fpr','tpr','thresholds']}
    return {'I':i['overall_energy_independence_score'],'group_I':{k:g['energy_independence_score'] for k,g in i['groups'].items()},'matched_auc':auc.matched_auc,'inclusive_auc':auc.inclusive.auc,'independence_energy_bin_count':{k:len(g['events_by_energy_bin']) for k,g in i['groups'].items()},'matching':diagnostics,'note':'Histogram I replay exact; unrelated distance-correlation diagnostic capped at4 to avoid unnecessary O(n^2) calculations; this cannot affect I. Full historical ROC arrays are available from the hashed original metrics file and are not duplicated in this manifest.'}

def main():
    (OUT/'standardized').mkdir(exist_ok=True);(OUT/'detail').mkdir(exist_ok=True)
    implementation=EVALUATION/'core/unified_metrics.py';spec=importlib.util.spec_from_file_location('exo_unified_v2',implementation);unified=importlib.util.module_from_spec(spec);spec.loader.exec_module(unified)
    config=json.loads((EVALUATION/'protocol.json').read_text());assert config['protocol_version']=='EnergyBench-unified-5keV-overflow-v2.0.0';assert unified.fingerprint(config)=='f862fc8eb3b03cb02640347098f9b19a71e3ea7b8ad1298b868b96de4154aa88';save(OUT/'protocol_used.json',config)
    table=PAPER/'wing_contribution/figures/data/classification_table_evidence.json';paper_records={(r['dataset'],r['model_key']):r for r in json.loads(table.read_text())['records']}
    campaign=SOURCE/'results/transformer_official_v1';first=campaign/next(iter(MODEL_MAP.values()));first_config=json.loads((first/'run_config.json').read_text())
    canonical=build_test_metadata(DataConfig(**first_config['data']))
    split_path=SOURCE/'manifests/exo200_v1_split.json';split_manifest=json.loads(split_path.read_text())
    assert first_config['counts']==split_manifest['counts'];assert first_config['split_runs']==split_manifest['runs']
    canonical_keys=['event_id','label','category','energy_condition','sample_weight','group_id','split']
    classic_path=Path('/home/wenyu/EXO200/outputs/classification/cnn_004_multiview_late_fusion/energybench_predictions.npz');classic=load_npz(classic_path)
    for key in canonical_keys:np.testing.assert_array_equal(canonical[key],classic[key],err_msg='Transformer canonical vs classic '+key)
    source_code={str(p):sha(p) for p in [SOURCE/'exo_transformer/energy_aware.py',SOURCE/'exo_transformer/model.py',SOURCE/'exo_transformer/tokenization.py',SOURCE/'exo_transformer/positional_encoding.py',SOURCE/'exo_transformer/rotary_attention.py',SOURCE/'exobench/data.py',SOURCE/'exobench/config.py',SOURCE/'exobench/training.py',split_path,implementation,EVALUATION/'protocol.json']}
    source_code.update({str(p):sha(p) for p in sorted((SOURCE/'notebooks').glob('*.ipynb'))})
    records=[];rows=[];groups=[]
    for key,model_id in MODEL_MAP.items():
        run=campaign/model_id;prediction=run/'energybench_predictions.npz';raw=run/'predictions.npz';metrics=run/'energybench_classification/.energybench/metrics.json';resolved=metrics.parent/'resolved_manifest.json'
        a=load_npz(prediction);native=load_npz(raw);provenance=json.loads((run/'energybench_input_provenance.json').read_text());metadata=json.loads(a['__metadata__'].item());run_config=json.loads((run/'run_config.json').read_text());report=json.loads(metrics.read_text());oldconfig=json.loads(resolved.read_text())
        assert provenance==metadata
        assert sha(raw)==provenance['prediction_source_sha256'];assert sha(run/'run_config.json')==provenance['run_config_sha256'];assert sha(prediction)==report['input']['sha256']
        assert sha(SOURCE/'exobench/data.py')==provenance['exobench_split_source_sha256']
        assert run_config['counts']==split_manifest['counts'] and run_config['split_runs']==split_manifest['runs']
        assert oldconfig['classification']['positive_label']=='0' and oldconfig['classification']['score_direction']=='lower'
        assert oldconfig['classification']['energy_bins']==6 and oldconfig['dependence']['energy_bins']==8
        assert oldconfig['dataset']['energy_unit']=='keV' and provenance['energy_field']=='Rotated_energy'
        for col in canonical_keys:np.testing.assert_array_equal(a[col],canonical[col],err_msg=key+' '+col)
        np.testing.assert_array_equal(a['score'],native['prediction']);np.testing.assert_array_equal(a['label'],native['target'])
        assert np.all(a['sample_weight']==1) and np.all(np.isfinite(a['score']))
        frozen={}
        for file in ['roc.py','dependence.py','utils.py']:
            source=SOURCE/'frozen_energybench/energybench'/file;frozen[file]=sha(source);assert frozen[file]==provenance['energybench_source_files_sha256'][file]
        checkpoint=model_check(run,run_config,provenance)
        standard=dict(event_id=a['event_id'].astype(str),label=(1-a['label']).astype('int8'),score=-a['score'].astype('float64'),energy_keV=a['energy_condition'].astype('float64'),group=a['category'].astype(str),weight=a['sample_weight'].astype('float64'))
        standard_path=OUT/'standardized'/f'EXO_200__{key}.npz';np.savez_compressed(standard_path,**standard)
        old=replay(standard);old_I=report['energy_dependence']['overall_energy_independence_score'];old_auc=report['classification']['aggregates']['matched_auc_macro'];old_inc=report['classification']['aggregates']['inclusive_auc_macro']
        errors={'I':abs(old['I']-old_I),'matched_auc':abs(old['matched_auc']-old_auc),'inclusive_auc':abs(old['inclusive_auc']-old_inc)};assert max(errors.values())<1e-12
        result=unified.evaluate(standard['label'],standard['score'],standard['energy_keV'],group=standard['group'],weight=standard['weight'],energy_unit='keV',config=config)
        result.update(dataset='EXO-200',model_key=key,model_id=model_id,input_sha256=sha(standard_path),evaluator_sha256=sha(implementation))
        assert abs(result['inclusive_auc']-old_inc)<1e-12
        detail_path=OUT/'detail'/f'EXO_200__{key}.json';save(detail_path,result)
        paper=paper_records['EXO-200',key];paper_compare={'I':float(paper['independence']),'matched_auc':float(paper['matched_auc']),'params':int(float(paper['params'])),'source':str(table),'source_sha256':sha(table),'note':'Current repository table evidence is used only to identify/compare published old values, never as recomputation input.'}
        assert abs(paper_compare['I']-old_I)<5.1e-11 and abs(paper_compare['matched_auc']-old_auc)<5.1e-11 and paper_compare['params']==checkpoint['parameter_count']
        record={'dataset':'EXO-200','model_key':key,'model_id':model_id,'status':'recomputed','standardized_file':str(standard_path),'standardized_sha256':sha(standard_path),'detail_file':str(detail_path),'detail_sha256':sha(detail_path),'prediction_file':str(prediction),'prediction_sha256':sha(prediction),'native_prediction_file':str(raw),'native_prediction_sha256':sha(raw),'metrics_file':str(metrics),'metrics_sha256':sha(metrics),'resolved_manifest_file':str(resolved),'resolved_manifest_sha256':sha(resolved),'run_config_file':str(run/'run_config.json'),'run_config_sha256':sha(run/'run_config.json'),'provenance':provenance,'checkpoint':checkpoint,'old_protocol':oldconfig,'historical_replay':old,'historical_replay_absolute_error':errors,'published_old':paper_compare,'n_events':len(standard['label']),'label_definition':{'original':'0=signal (Charge_cluster_number==1); 1=background (Charge_cluster_number>1)','standard':'1=signal, 0=background','conversion':'1 - stored_label'},'score_definition':{'original_space':'one background logit, saved float32 then losslessly exported as float64','standard_space':'signal-oriented scalar logit','conversion':'-stored_logit, no sigmoid'},'energy_definition':{'physical_field':'Rotated_energy','original_unit':'keV','conversion':'none','clipped':False,'provenance':'Raw HDF5 recomputed by canonical source build_test_metadata and exactly matched against every prediction bundle'},'test_split':{'split_manifest':str(split_path),'split_manifest_sha256':sha(split_path),'test_runs':split_manifest['runs']['test'],'event_id_format':'EXO200::<source HDF5 filename>::<event_number>','n_unique_ids':int(np.unique(standard['event_id']).size),'same_order_and_metadata_as_classic':True,'classic_prediction':str(classic_path),'classic_prediction_sha256':sha(classic_path),'identical_columns':canonical_keys},'array_sha256':{k:array_sha(v) for k,v in standard.items()},'frozen_historical_source_sha256':frozen,'paper_locations':['wing_contribution/tables/benchmark_main.tex','wing_contribution/figures/data/classification_table_evidence.json'],'protocol_version':result['protocol_version'],'protocol_sha256':result['protocol_sha256']}
        records.append(record)
        row={'dataset':'EXO-200','model_key':key,'model_id':model_id,'parameter_count':checkpoint['parameter_count'],'old_I':old_I,'new_I':result['independence']['I'],'delta_I':result['independence']['I']-old_I,'old_matched_auc':old_auc,'new_matched_auc':result['matching']['matched_auc'],'delta_matched_auc':result['matching']['matched_auc']-old_auc if result['matching']['matched_auc'] is not None else None,'old_inclusive_auc':old_inc,'new_inclusive_auc':result['inclusive_auc'],'n_events':result['n_input'],'common_support_keV':result['matching']['common_support_keV'],'valid_matching_bins':result['matching']['valid_bin_count'],'matched_status':result['matching']['status'],'min_group_I':result['independence']['min_group_I'],'overflow_n':result['n_overflow'],'protocol_version':result['protocol_version'],'protocol_sha256':result['protocol_sha256'],'standardized_file':str(standard_path),'detail_file':str(detail_path)}
        for c,role in [('1','signal'),('0','background')]:
            v=result['matching']['classes'][c]
            for dest,field in [('coverage','matched_fraction_original_finite'),('ESS','matched_ess')]:row[role+'_'+dest]=v[field]
            for field in ['input','negative_energy','overflow','common_support','matched','support_excluded_after_range','sparse_excluded_after_support']:
                if field in v:row[role+'_'+field+'_n']=v[field]['n']
        for name,g in result['independence']['groups'].items():groups.append({'dataset':'EXO-200','model_key':key,'group':name,'old_I_g':old['group_I'][name],'new_I_g':g['I_g'],'delta_I_g':g['I_g']-old['group_I'][name],'valid_bin_count':g['valid_bin_count'],'input_n':g['original_finite']['n'],'retained_n':g['retained']['n'],'sparse_excluded_n':g['sparse_excluded']['n'],'retained_fraction_original_finite':g['retained_fraction_original_finite'],'overflow_n':g['bin_counts'][600]})
        rows.append(row)
        save(OUT/'manifest.json',{'records':records,'source_code_sha256':source_code,'raw_test_metadata_array_sha256':{k:array_sha(canonical[k]) for k in canonical_keys},'raw_hdf5_inventory':[{'file':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(Path(first_config['data']['data_root']).glob('*.h5'))],'excluded_run_directories':[str(p) for p in campaign.glob('*partial*')],'protocol_scope_note':'Frozen protocol user_scope is historical release metadata; 2026-09-17 user authorization extends this unchanged statistical protocol to EXO/MJD Transformers.','operation':'Read-only reuse of saved prediction/checkpoint and original test metadata. No inference or training.'})
        save(OUT/'summary.json',{'records':rows,'group_records':groups})
        print(key,f'I {old_I:.12f} -> {result["independence"]["I"]:.12f}',f'AUC {old_auc:.12f} -> {result["matching"]["matched_auc"]:.12f}',f'params {checkpoint["parameter_count"]}',flush=True)
    for file,data in [('summary.csv',rows),('group_summary.csv',groups)]:
        with (OUT/file).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print('All nine EXO Transformer variants audited and recomputed.',flush=True)

if __name__=='__main__':main()
