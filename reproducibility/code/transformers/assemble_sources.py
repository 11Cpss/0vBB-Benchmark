#!/usr/bin/env python3
"""Copy source snapshots without modifying originals or distributing event/model arrays."""
from pathlib import Path
import hashlib, json, shutil

BUNDLE = Path(__file__).resolve().parents[2]
DEST = BUNDLE/'code/transformers'
SOURCE = Path('/home/klz/Data/zeronu_benchmark/Transformer_Approach')
PAPER = Path('/home/wenyu/iclr final paper/overleaf')
SN = Path('/home/wenyu/SuperNEMO')
records=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def copy(src, dst):
    src,dst=Path(src),Path(dst)
    if not src.is_file():return
    digest=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    assert sha(dst)==digest
    records.append(dict(source=str(src),bundle=str(dst.relative_to(BUNDLE)),bytes=src.stat().st_size,
                        source_sha256=digest,bundle_sha256=digest,operation='byte-identical copy'))

def package(rel):
    for src in sorted((SOURCE/rel).rglob('*')):
        if src.is_file() and '__pycache__' not in src.parts and src.suffix in ('.py','.md','.toml','.txt'):
            copy(src,DEST/'original'/src.relative_to(SOURCE))

def main():
    for name in ['README.md','pyproject.toml','LICENSE','requirements.txt']:
        copy(SOURCE/name,DEST/'original'/name)
    for rel in ['evalutaions_workflow/energybench','evalutaions_workflow/simple_energybench','evalutaions_workflow/tests',
                'next_detector/next_transformer','next_detector/tests','mjd_detector/mjd_transformer','mjd_detector/mjdbench',
                'exo200_detector/exo_transformer','exo200_detector/exobench','exo200_detector/frozen_energybench',
                'supernemo_detector/supernemo_transformer','supernemo_detector/supernemobench']:
        package(rel)
    for rel in ['evalutaions_workflow/README.md','evalutaions_workflow/requirements.txt','next_detector/CACHE_GUIDE.md',
                'next_detector/scripts/build_token_cache.py','next_detector/scripts/benchmark_token_cache.py',
                'next_detector/scripts/measure_voxel_padding.py','exo200_detector/README.md','supernemo_detector/README.md']:
        copy(SOURCE/rel,DEST/'original'/rel)
    notebooks={
      'next_detector':['next_energybench_train.ipynb','next_full_Model_run_with_cache.ipynb','next_energybench_final_results.ipynb',
                      'Tests/Exploration/next_energybench_cached_optimized.ipynb'],
      'mjd_detector':['mjd_transformer_train.ipynb','mjd_transformer_results.ipynb','mjd_transformer_results_energy_aware.ipynb'],
      'exo200_detector':['exo_transformer_train.ipynb','exo_transformer_results.ipynb','exo_transformer_energybench_results_executed.ipynb'],
      'supernemo_detector':['supernemo_transformer_train.ipynb','supernemo_transformer_results.ipynb','supernemo_token_count_audit.ipynb']}
    for ds,names in notebooks.items():
        for n in names:
            rel=Path(ds)/'notebooks'/n;copy(SOURCE/rel,DEST/'original'/rel)
    for src in sorted((SOURCE/'exo200_detector/manifests').glob('*.json')):copy(src,DEST/'original'/src.relative_to(SOURCE))
    copy(SOURCE/'next_detector/results/event_split.json',DEST/'configs/next/event_split.json')
    for ds in ['exo200_detector','mjd_detector']:
        for run in sorted((SOURCE/ds/'results/transformer_official_v1').glob('classification__*')):
            if 'partial' in run.name:continue
            for name in ['run_config.json','run_summary.json','energybench_input_provenance.json']:
                copy(run/name,DEST/'configs'/ds/run.name/name)
            copy(run/'energybench_classification/.energybench/resolved_manifest.json',DEST/'configs'/ds/run.name/'historical_evaluation_manifest.json')
    for campaign in ['final_cached_v1','final']:
        for run in sorted((SOURCE/'next_detector/results'/campaign).glob('transformer_*')):
            if 'partial' in run.name or not (run/'evaluation/metrics.json').is_file():continue
            for name in ['representation_config.json','evaluation/metrics.json']:
                copy(run/name,DEST/'configs/next'/campaign/run.name/name)
    # The published SuperNEMO six are from the wenyu implementation, not the
    # differently named single canonical completed run in Transformer_Approach.
    for name in ['supernemobench','architectures']:
        for src in sorted((SN/name).rglob('*.py')):
            if '__pycache__' not in src.parts:copy(src,DEST/'published_supernemo'/src.relative_to(SN))
    for name in ['README.md','requirements.txt','evaluation/supernemo_2nu_vs_bi214.json','data/manifests/split_manifest.json']:
        copy(SN/name,DEST/'published_supernemo'/name)
    for run in sorted((SN/'outputs/classification').glob('transformer_*')):
        copy(run/'run_config.json',DEST/'configs/published_supernemo'/run.name/'run_config.json')
        copy(run/'test_evaluation/energybench/.energybench/resolved_manifest.json',DEST/'configs/published_supernemo'/run.name/'historical_evaluation_manifest.json')
    copy(SOURCE/'pyproject.toml',BUNDLE/'environments/transformers/source_pyproject.toml')
    copy(SOURCE/'evalutaions_workflow/requirements.txt',BUNDLE/'environments/transformers/source_next_requirements.txt')
    copy(SN/'requirements.txt',BUNDLE/'environments/transformers/source_published_supernemo_requirements.txt')
    for name in ['classification_table_evidence.json','classification_transformer_workbook.json']:
        copy(PAPER/'wing_contribution/figures/data'/name,DEST/'configs/paper_reference'/name)
    payload=dict(source_root=str(SOURCE),files=records,excluded=['raw datasets','weights/checkpoints','event predictions','token caches','logs/history','partial/discarded runs','regression-only notebooks/results','kernel-interrupt/archive orchestration shell script'],
      notebooks='Original selected notebooks are byte-identical, including their existing cell outputs; these are provenance, not fresh validation.',
      source_mutation='No original file was changed; all writes are inside this new bundle.')
    p=BUNDLE/'provenance/transformer_source_manifest.json';p.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),manifest=str(p))))

if __name__=='__main__':main()
