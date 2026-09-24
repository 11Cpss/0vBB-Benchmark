#!/usr/bin/env python3
"""Train or export EXO/MJD predictions with aligned physical energy and event IDs.

The publication adapters preserve model inputs and native label/score direction.
Final metric computation uses the public EnergyBench entry points.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,sys
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['EXO-200','MJD'],required=True);p.add_argument('--model-key',required=True)
    p.add_argument('--mode',choices=['train','test'],default='test');p.add_argument('--data-root',type=Path)
    p.add_argument('--output-dir',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--device')
    p.add_argument('--describe',action='store_true');a=p.parse_args()
    records=json.loads((HERE.parents[1]/'results/transformer_models.json').read_text())['rows']
    found=[r for r in records if r['dataset']==a.dataset and r['model_key']==a.model_key]
    if not found or not found[0].get('source_training_config'):p.error('No source-backed configuration for this entry.')
    r=found[0];cfg=json.loads((HERE.parents[1]/r['source_training_config']).read_text())
    if a.describe:
        print(json.dumps(cfg,indent=2));return
    if a.data_root is None or a.output_dir is None:p.error('--data-root and --output-dir are required for execution')
    if a.mode=='test' and a.checkpoint is None:p.error('--mode test requires an explicit original --checkpoint')
    out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('--output-dir must be new or empty; existing results are never overwritten')
    source=HERE/'detectors'/('exo200' if a.dataset=='EXO-200' else 'mjd')
    sys.path.insert(0,str(source));import torch
    if a.dataset=='EXO-200':
        from exobench import DataConfig,TrainingConfig,prepare_dataset,train_model,evaluate_model,set_seed
        from exo_transformer import TokenizationConfig,EXOTransformerClassifier,validate_split_manifest
        cls=EXOTransformerClassifier
    else:
        from mjdbench import DataConfig,TrainingConfig,prepare_dataset,train_model,evaluate_model,set_seed
        from mjd_transformer import TokenizationConfig,MJDTransformer
        cls=MJDTransformer
    dc=DataConfig(**{**cfg['data'],'data_root':str(a.data_root.resolve())})
    tc=TrainingConfig(**{**cfg['training'],**({'device':a.device} if a.device else {})})
    extra={} if a.dataset=='EXO-200' else {'task':'classification'}
    loaders=prepare_dataset(data_config=dc,batch_size=tc.batch_size,num_workers=tc.num_workers,**extra)
    assert loaders.counts==cfg['counts'],'Dataset/split counts differ from the source run'
    if a.dataset=='EXO-200':validate_split_manifest(loaders,data_root=a.data_root)
    rep=cfg['representation'];token=TokenizationConfig(**{k:v for k,v in rep['tokenization'].items() if k in TokenizationConfig.__dataclass_fields__})
    kw={k:rep[k] for k in ['position_encoding','d_model','nhead','num_layers','dim_feedforward','dropout','num_frequencies']}
    if a.dataset=='EXO-200' and rep.get('rope_base') is not None:kw['rope_base']=rep['rope_base']
    set_seed(tc.seed,tc.deterministic);model=cls(tokenization_config=token,**extra,**kw)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==cfg['parameter_count']
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps({**cfg,'data':dc.to_dict(),'training':tc.to_dict()},indent=2)+'\n')
    if a.mode=='train':
        train_model(model,loaders.train_loader,loaders.validation_loader,config=tc,output_dir=out,**extra)
    else:
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==r['checkpoint']['sha256'],'Checkpoint is not the archived original for this row'
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    evaluate_model(model,loaders.test_loader,device=tc.device,output_dir=out,use_amp=tc.use_amp,amp_precision=tc.amp_precision,**extra)
    print('predictions.npz contains native score/label, physical energy_keV and event_id. Standardize the EXO positive-class direction before EnergyBench evaluation; MJD already uses positive label 1.')
if __name__=='__main__':main()
