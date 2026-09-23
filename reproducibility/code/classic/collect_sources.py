#!/usr/bin/env python3
"""Collect immutable classic-model source snapshots and small run metadata.

Reads original workspaces; writes only this new reproducibility bundle. Does not
copy datasets, trained weights, predictions, binary caches, figures or logs.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

BUNDLE = Path(__file__).resolve().parents[2]
CLASSIC = BUNDLE/'code/classic'
AUDIT = Path('/home/wenyu/iclr final paper/unified_5kev_evaluation')
ARCHS = ['cnn_004_multiview_late_fusion','gnn_001_static_gine','seq_001_bigru','ssm_001_pointmamba']
TEXT_EXTENSIONS = {'.py','.json','.yaml','.yml','.toml','.md','.txt','.sh','.lock','.sha256'}
rows=[]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def copy(source,dest,purpose):
    source=Path(source);dest=BUNDLE/dest
    assert source.is_file(),source
    if dest.exists():
        assert digest(source)==digest(dest),f'Existing bundle copy differs: {dest}'
    else:
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    sh=digest(source);assert sh==digest(dest)
    if not any(r['bundle_path']==str(dest.relative_to(BUNDLE)) for r in rows):
        rows.append(dict(source=str(source),bundle_path=str(dest.relative_to(BUNDLE)),
            bytes=source.stat().st_size,sha256=sh,purpose=purpose,byte_identical=True))
    return str(dest.relative_to(BUNDLE))


def tree(source,dest,purpose):
    source=Path(source)
    for f in sorted(source.rglob('*')):
        if not f.is_file() or any(x in f.parts for x in ['__pycache__','.pytest_cache','.git']):continue
        if f.suffix in TEXT_EXTENSIONS or f.name in ['LICENSE','.python-version']:
            copy(f,Path(dest)/f.relative_to(source),purpose)


def main():
    for ds,package in [('MJD','mjdbench'),('EXO200','exobench'),('SuperNEMO','supernemobench')]:
        source=Path('/home/wenyu')/ds;dest=Path('code/classic')/ds
        for folder in [package,'architectures','scripts','tests']:
            if (source/folder).exists():tree(source/folder,dest/folder,'Original dataset training/inference/preprocessing source; unmodified')
        for name in ['README.md','requirements.txt','pyproject.toml']:
            copy(source/name,dest/name,'Original documentation/dependency declaration; unmodified')
        copy(source/'requirements.txt',Path('environments/classic')/ds/'requirements.txt','Original dependency declaration')
        for arch in ARCHS:
            run=source/'outputs/classification'/arch
            for name in ['run_config.json','history.json','run_summary.json']:
                if (run/name).is_file():copy(run/name,dest/'run_records/classification'/arch/name,'Selected paper run metadata; historical metrics are not final evaluator inputs')
        if ds=='SuperNEMO':
            tree(source/'evaluation',dest/'evaluation','Historical native evaluation dependency; not the final paper metric authority')
            copy(source/'data/manifests/split_manifest.json',dest/'run_records/split_manifest.json','Saved official source/locality-block split metadata; event-offset caches omitted')
    source=Path('/home/wenyu/summer');dest=Path('code/classic/NEXT')
    for folder in ['01_code/architectures','src/next_alt','src/next_cnn','src/energybench','evalutaions_workflow','requirements','manifests','tests']:
        tree(source/folder,dest/folder,'Original NEXT architecture/training/inference dependency; historical evaluators retained without fixes')
    for name in ['README.md','LICENSE','pyproject.toml','uv.lock','.python-version']:
        if (source/name).exists():copy(source/name,dest/name,'Original project metadata/dependencies; unmodified')
    for name in ['ALTERNATIVE_ARCHITECTURES.md','NONTRANSFORMER_V2_TRAINING.md','USAGE_GUIDE_EN.md']:
        copy(source/'docs'/name,dest/'docs'/name,'Original source documentation; historical protocol caveats apply')
    tree(source/'requirements','environments/classic/NEXT','Original dependency declarations')
    copy(source/'uv.lock','environments/classic/NEXT/uv.lock','Original environment lock; not a new resolution')
    campaign=source/'03_training_runs/energybench_campaigns/20260808_energybench_rewrite_v2'
    for name in ['manifest.json','event_split.json']:
        copy(campaign/name,dest/'run_records/canonical_campaign'/name,'Original retained-campaign configuration and exact event split')
    canonical=json.loads((AUDIT/'sources/next_mjd/manifest.json').read_text())['ready']
    exploratory=json.loads((AUDIT/'sources/next_mjd/exploratory/manifest.json').read_text())['ready']
    exosn=json.loads((AUDIT/'sources/exo_sn/manifest.json').read_text())['records']
    selected=[]
    for r in canonical:
        if r['dataset']=='NEXT':
            run=Path(r['run_summary']).parent
            summary=json.loads(Path(r['run_summary']).read_text())
            arch=r['architecture_id'];meta=[]
            for name in ['run_summary.json','training/history.json']:
                if (run/name).exists():meta.append(copy(run/name,dest/'run_records/canonical_campaign/runs'/arch/'classification'/name,'Original retained-campaign model configuration/history'))
            ck=run/'training/best_model.pt'
            selected.append(dict(dataset='NEXT',population='canonical_116549',architecture_id=arch,
                model_key=r['model_key'],checkpoint=str(ck),checkpoint_sha256=digest(ck),
                checkpoint_included=False,prediction_source=r['source_predictions'],
                prediction_sha256=r['source_predictions_sha256'],run_records=meta,
                training_entry=str(dest/'01_code/architectures'/arch/'train_classification.py'),
                source_summary=r['run_summary'],best_epoch=summary['best_epoch']))
        elif r['dataset']=='MJD':
            arch=r['architecture_id'];run=Path('/home/wenyu/MJD/outputs/classification')/arch;ck=run/'best.pt'
            selected.append(dict(dataset='MJD',population='official_test_390000',architecture_id=arch,
                model_key=r['model_key'],checkpoint=str(ck),checkpoint_sha256=digest(ck),checkpoint_included=False,
                prediction_source=r['source_predictions'],prediction_sha256=r['source_predictions_sha256'],
                run_records=[f'code/classic/MJD/run_records/classification/{arch}/run_config.json'],
                training_entry=f'code/classic/MJD/architectures/{arch}/train_classification.py'))
    for r in exploratory:
        arch=r['architecture_id'];p=r['prediction_metadata'];ck=Path(p['checkpoint'])
        selected.append(dict(dataset='NEXT',population='exploratory_115499',architecture_id=arch,
            model_key=r['model_key'],checkpoint=str(ck),checkpoint_sha256=digest(ck),checkpoint_included=False,
            prediction_source=r['source_predictions'],prediction_sha256=r['source_predictions_sha256'],
            checkpoint_epoch=p.get('checkpoint_epoch'),
            inference_adapter=p.get('adapter'),
            training_entry='code/classic/NEXT/src/next_alt/training.py:main_for_architecture (earlier file-split API; current architecture CLI wrappers use canonical workflow_runner)',
            source_config_note='Original config/representation recorded in copied exploratory audit manifest'))
    for r in exosn:
        if r.get('status')!='ready' or r.get('model') not in ARCHS:continue
        ds={'EXO-200':'EXO200','SuperNEMO':'SuperNEMO'}.get(r['dataset'])
        if ds is None:continue
        arch=r['model'];ck=Path(r['checkpoint'])
        selected.append(dict(dataset=r['dataset'],population='heldout_140383' if ds=='EXO200' else 'classification_2nu_vs_Bi214',
            architecture_id=arch,model_key=r['model_key'],checkpoint=str(ck),checkpoint_sha256=digest(ck),checkpoint_included=False,
            prediction_source=r['prediction_file'],prediction_sha256=r['prediction_sha256'],
            run_records=[f'code/classic/{ds}/run_records/classification/{arch}/run_config.json'],
            training_entry=f'code/classic/{ds}/architectures/{arch}/train_classification.py' if ds=='EXO200' else 'code/classic/SuperNEMO/supernemobench/workflow.py'))
    for f in ['next_mjd/manifest.json','next_mjd/exploratory/manifest.json','exo_sn/manifest.json']:
        copy(AUDIT/'sources'/f,Path('provenance/classic_source_audits')/f,'Prior event-input audit; external predictions are referenced but not copied')
    for f in (source/'03_training_runs/campaigns/20260803_200356/ssm_001_pointmamba/attempt_001').glob('*'):
        if f.name in ['run_summary.json','config.snapshot.yaml','history.json']:
            copy(f,dest/'run_records/exploratory_pointmamba'/f.name,'Published earlier PointMamba-lite run configuration/history')
    assert len(selected)==45,len(selected)
    manifest=dict(collection_date='2026-09-23',original_files_modified=False,
        copied_files=len(rows),copied_bytes=sum(r['bytes'] for r in rows),
        policy='Byte-identical snapshots. Historical metric implementations are preserved, not repaired; use the bundle shared final profile for paper metrics.',files=rows)
    (BUNDLE/'provenance/classic_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (BUNDLE/'provenance/classic_run_index.json').write_text(json.dumps(dict(records=selected,
        note='22 canonical NEXT runs,11 earlier NEXT runs,4 MJD,4 EXO-200,4 SuperNEMO; 45 prediction/checkpoint provenances. NEXT has23 distinct classification architecture IDs across populations.'),indent=2)+'\n')
    with (BUNDLE/'provenance/classic_source_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
