#!/usr/bin/env python3
"""Read trusted local paper checkpoints and export configuration only, never weights."""
from pathlib import Path
import hashlib
import json
import math
import torch

BUNDLE=Path(__file__).resolve().parents[2]
KEEP={'format_version','schema_version','task','architecture_id','architecture','model_name','input_kind',
      'epoch','best_epoch','kind','config','model_config','training_config','data_config','representation',
      'projection','representation_config','split_config','data_selection','class_map','positive_class','split_seed','split_fractions','seed','model',
      'feature_extractor','backend','selection_split','selection_metric','model_source','training_source'}
DROP={'model_state_dict','state_dict','optimizer_state_dict','optimizer','scheduler','scaler',
      'booster_raw_base64','booster_raw','history','weights'}

def serial(value):
    if isinstance(value,torch.Tensor):return {'omitted_tensor':True,'shape':list(value.shape)}
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items() if k not in DROP}
    if isinstance(value,(list,tuple)):return [serial(x) for x in value]
    if isinstance(value,Path):return str(value)
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):return value if math.isfinite(value) else str(value)
    return str(value)

def main():
    records=[]
    for r in json.loads((BUNDLE/'provenance/classic_run_index.json').read_text())['records']:
        p=Path(r['checkpoint'])
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual==r['checkpoint_sha256'],p
        ckpt=json.loads(p.read_text()) if p.suffix=='.json' else torch.load(p,map_location='cpu',weights_only=False)
        assert isinstance(ckpt,dict),p
        records.append(dict(dataset=r['dataset'],model_key=r['model_key'],population=r['population'],
            checkpoint=str(p),checkpoint_sha256=actual,original_keys=list(ckpt),
            configuration=serial({k:v for k,v in ckpt.items() if k in KEEP})))
    (BUNDLE/'provenance/classic_checkpoint_metadata.json').write_text(json.dumps(dict(
        records=records,checkpoints_included=False,note='Derived configuration view of trusted original checkpoints. Tensor weights, optimizer states and tree payloads are excluded. Original checkpoint hashes bind these metadata to the exact selected artifact.'),indent=2)+'\n')
    print('Exported configuration-only metadata for',len(records),'checkpoints')

if __name__=='__main__':main()
