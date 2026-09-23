#!/usr/bin/env python3
"""Validate bundled classic source snapshots without loading data or training."""
from pathlib import Path
import ast
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parents[2]
CLASSIC = BUNDLE / 'code/classic'
DEST = BUNDLE / 'validation/classic_source_checks'
ARCHS = ['cnn_004_multiview_late_fusion', 'gnn_001_static_gine', 'seq_001_bigru', 'ssm_001_pointmamba']

def run(job):
    name, cwd, args, paths = job
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(DEST/'matplotlib'),
               PYTHONPATH=os.pathsep.join(map(str, paths)))
    result = subprocess.run([sys.executable, '-B', *args], cwd=cwd, env=env,
                            text=True, capture_output=True, timeout=120)
    log = DEST / (name + '.txt')
    log.write_text(result.stdout + result.stderr)
    return dict(name=name, cwd=str(cwd.relative_to(BUNDLE)), command=[sys.executable, '-B', *args],
                returncode=result.returncode, log=str(log.relative_to(BUNDLE)))

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source = json.loads((BUNDLE/'provenance/classic_source_manifest.json').read_text())
    failures=[]
    for row in source['files']:
        f=BUNDLE/row['bundle_path']
        if hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']: failures.append(str(f))
    syntax=[]
    for f in sorted(CLASSIC.rglob('*.py')):
        try: ast.parse(f.read_text(), filename=str(f))
        except Exception as exc: syntax.append(dict(path=str(f.relative_to(BUNDLE)),error=str(exc)))
    jobs=[]
    for ds in ['MJD','EXO200']:
        root=CLASSIC/ds
        for arch in ARCHS:
            jobs.append((ds+'_'+arch,root,[f'architectures/{arch}/train_classification.py','--help'],[root]))
        package='mjdbench' if ds=='MJD' else 'exobench'
        code=f"import {package}.data as data, {package}.training as training; import pathlib; root=pathlib.Path.cwd(); assert pathlib.Path(data.__file__).is_relative_to(root); assert pathlib.Path(training.__file__).is_relative_to(root); print(data.__file__); print(training.__file__)"
        jobs.append((ds+'_package_import',root,['-c',code],[root]))
    root=CLASSIC/'MJD'
    jobs.append(('MJD_checkpoint_cli',root,['scripts/run_checkpoint_inference.py','--help'],[root]))
    root=CLASSIC/'NEXT'
    for f in sorted((root/'01_code/architectures').glob('*/train_classification.py')):
        jobs.append(('NEXT_'+f.parent.name,root,[str(f.relative_to(root)),'--help'],[root/'src']))
    for f in ['run_energybench_campaign.py','nontransformer_campaign.py']:
        jobs.append(('NEXT_'+f[:-3],root,['01_code/architectures/'+f,'--help'],[root/'src']))
    jobs.append(('NEXT_inference_cli',root,['-m','energybench','predict','--help'],[root/'src']))
    jobs.append(('NEXT_old_adapters',root,['-c',"import next_cnn.adapter, next_alt.adapter; print(next_cnn.adapter.__file__); print(next_alt.adapter.__file__)"],[root/'src']))
    root=CLASSIC/'SuperNEMO'
    jobs.append(('SuperNEMO_workflow',root,['-m','supernemobench.workflow','--help'],[root,CLASSIC/'NEXT/src']))
    jobs.append(('SuperNEMO_package_import',root,['-c',"import supernemobench.data, supernemobench.models, supernemobench.training; print(supernemobench.data.__file__); print(supernemobench.models.__file__); print(supernemobench.training.__file__)"],[root,CLASSIC/'NEXT/src']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: checks=list(pool.map(run,jobs))
    versions={}
    for package in ['numpy','torch','h5py','matplotlib','PyYAML','pandas','pyarrow','tqdm','xgboost','pytest','scipy','scikit-learn']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,
        note='Existing environment used for source-only smoke checks; not asserted to be the original training environment. No dependencies installed.')
    (BUNDLE/'validation/classic_validation_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    result=dict(source_hash_checks=len(source['files']),hash_failures=failures,
        parsed_python_files=len(list(CLASSIC.rglob('*.py'))),syntax_failures=syntax,
        process_checks=checks,training_executed=False,data_loaded=False,
        all_pass=not failures and not syntax and all(j['returncode']==0 for j in checks))
    (BUNDLE/'validation/classic_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='process_checks'},indent=2))
    print('Process results:', len(checks),'passed:',sum(r['returncode']==0 for r in checks))
    for r in checks:
        if r['returncode']: print('FAILED',r['name'],r['log'])
    return int(not result['all_pass'])

if __name__=='__main__': raise SystemExit(main())
