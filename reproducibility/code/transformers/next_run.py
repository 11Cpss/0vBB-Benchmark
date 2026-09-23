#!/usr/bin/env python3
"""Use original NEXT APIs with an explicit archived configuration and output path.

This new convenience wrapper has only been syntax/constructor checked, not trained.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-key',required=True);p.add_argument('--mode',choices=['train','test'],default='test')
    p.add_argument('--data-root',type=Path);p.add_argument('--split-manifest',type=Path);p.add_argument('--token-cache',type=Path)
    p.add_argument('--output-dir',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--describe',action='store_true');a=p.parse_args()
    records=json.loads((ROOT/'provenance/transformer_model_mapping.json').read_text())['rows']
    found=[r for r in records if r['dataset']=='NEXT' and r['model_key']==a.model_key]
    if not found or not found[0]['source_training_config']:p.error('No source-backed configuration for this entry')
    r=found[0];cfg=json.loads((ROOT/r['source_training_config']).read_text())
    if a.describe:print(json.dumps(cfg,indent=2));return
    if any(x is None for x in [a.data_root,a.split_manifest,a.output_dir]):p.error('Require --data-root, --split-manifest and --output-dir')
    if a.mode=='test' and a.checkpoint is None:p.error('--mode test requires the explicit original --checkpoint')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):p.error('Output must be new or empty')
    sys.path[:0]=[str(HERE/'original/evalutaions_workflow'),str(HERE/'original/next_detector')]
    import torch
    from energybench import TrainingConfig,EvaluationConfig,prepare_dataset,set_seed,train_model,evaluate_classification
    from next_transformer import TokenizationConfig,NEXTTokenBuilder,NEXTTransformerClassifier,prepare_cached_dataset,validate_token_cache
    tok=TokenizationConfig(**{k:v for k,v in cfg.items() if k in TokenizationConfig.__dataclass_fields__})
    tc=TrainingConfig(**cfg.get('training_config',{'num_workers':8}))
    ec=EvaluationConfig(**cfg.get('evaluation_config',{}))
    if 'cache_manifest_path' in cfg:
        if a.token_cache is None:p.error('This recorded run used a token cache; supply its validated --token-cache directory')
        validate_token_cache(a.token_cache,a.data_root,a.split_manifest,tok)
        data=prepare_cached_dataset(a.token_cache,batch_size=tc.batch_size,num_workers=tc.num_workers,seed=tc.seed,
             trim_padding=cfg.get('trim_padding',False),compact_training_batches=cfg.get('compact_training_batches',False))
    else:
        data=prepare_dataset(a.data_root,batch_size=tc.batch_size,mode='classification',split_fractions=(.8,.1,.1),seed=tc.seed,
             num_workers=tc.num_workers,manifest_path=a.split_manifest,max_files_per_class=None,input_builder=NEXTTokenBuilder(tok))
    reference=json.loads((HERE/'configs/next/event_split.json').read_text())
    runtime=json.loads(Path(data.manifest_path).read_text())
    for key in ['settings','counts','splits']:assert runtime[key]==reference[key],f'Original NEXT split differs: {key}'
    set_seed(tc.seed,tc.deterministic)
    model=NEXTTransformerClassifier(**{k:cfg[k] for k in ['position_encoding','feature_dim','d_model','nhead','num_layers','dim_feedforward','dropout','num_frequencies']})
    assert sum(q.numel() for q in model.parameters() if q.requires_grad)==cfg['parameter_count']
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'representation_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if a.mode=='train':train_model(model,data.train_loader,data.validation_loader,config=tc,task='classification',output_dir=a.output_dir/'training',overwrite=False)
    else:
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==r['checkpoint']['sha256'],'Checkpoint is not the archived original for this row'
        model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    evaluate_classification(model,data.test_loader,device=tc.device,output_dir=a.output_dir/'evaluation',config=ec,overwrite=False)
if __name__=='__main__':main()
