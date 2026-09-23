#!/usr/bin/env python3
"""Read-only syntax/import/constructor checks, never training or model inference."""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-originals',action='store_true',help='Also read the original local source paths; omit on another machine')
    args=parser.parse_args()
    syntax=[]
    for p in sorted(HERE.rglob('*.py')):
        try:compile(p.read_text(),str(p),'exec');status='pass'
        except Exception as e:status=f'{type(e).__name__}: {e}'
        syntax.append(dict(path=str(p.relative_to(ROOT)),status=status))
    # Compile notebook cells after IPython's own transformation, without executing them.
    from IPython.core.inputtransformer2 import TransformerManager
    transform=TransformerManager();notebooks=[]
    for p in sorted((HERE/'original').rglob('*.ipynb')):
        for ix,c in enumerate(json.loads(p.read_text())['cells']):
            if c['cell_type']!='code':continue
            try:compile(transform.transform_cell(''.join(c['source'])),f'{p}:cell{ix}','exec');status='pass'
            except Exception as e:status=f'{type(e).__name__}: {e}'
            notebooks.append(dict(path=str(p.relative_to(ROOT)),cell=ix,status=status))
    checks={
      'NEXT':(['original/evalutaions_workflow','original/next_detector'],"import energybench, next_transformer; from next_transformer import NEXTTransformerClassifier; m=NEXTTransformerClassifier(position_encoding='coordinate_mlp'); print(energybench.__file__,next_transformer.__file__,sum(p.numel() for p in m.parameters()))"),
      'MJD':(['original/mjd_detector'],"import mjdbench,mjd_transformer; from mjd_transformer import MJDTransformer,TokenizationConfig; m=MJDTransformer(task='classification',tokenization_config=TokenizationConfig(tokenization='pulse_entities'),position_encoding='coordinate_mlp'); print(mjdbench.__file__,mjd_transformer.__file__,sum(p.numel() for p in m.parameters()))"),
      'EXO':(['original/exo200_detector','original/exo200_detector/frozen_energybench'],"import exobench,exo_transformer; from exo_transformer import EXOTransformerClassifier,TokenizationConfig; m=EXOTransformerClassifier(tokenization_config=TokenizationConfig(tokenization='pulse_entities'),position_encoding='rope'); print(exobench.__file__,exo_transformer.__file__,sum(p.numel() for p in m.parameters()))"),
      'SuperNEMO_canonical':(['original/supernemo_detector','original/exo200_detector/frozen_energybench'],"import supernemobench,supernemo_transformer; from supernemobench.models import build_model; m=build_model('transformer_001_entity_coordinate_mlp','classification'); print(supernemobench.__file__,supernemo_transformer.__file__,sum(p.numel() for p in m.parameters()))"),
      'SuperNEMO_published':(['published_supernemo','original/exo200_detector/frozen_energybench'],"import supernemobench,architectures; from supernemobench.models import build_model; m=build_model('transformer_001_sampled_hits_coordinate_mlp','classification'); print(supernemobench.__file__,architectures.__file__,sum(p.numel() for p in m.parameters()))")}
    imports=[]
    for name,(paths,code) in checks.items():
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(str(HERE/p) for p in paths),MPLCONFIGDIR=str(ROOT/'validation/transformer_mpl'))
        p=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd=HERE,capture_output=True,text=True,timeout=90)
        imports.append(dict(name=name,status='pass' if p.returncode==0 else 'fail',returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,paths=paths))
    manifest=json.loads((ROOT/'provenance/transformer_source_manifest.json').read_text());hashes=[]
    for r in manifest['files']:
        dest=ROOT/r['bundle'];d=hashlib.sha256(dest.read_bytes()).hexdigest()
        source_unchanged=None
        if args.check_originals:
            src=Path(r['source']);s=hashlib.sha256(src.read_bytes()).hexdigest();source_unchanged=s==r['source_sha256']
        hashes.append(dict(path=r['bundle'],source_unchanged=source_unchanged,bundle_identical=d==r['bundle_sha256']))
    versions={}
    for name in ['numpy','torch','pandas','h5py','matplotlib','scipy','scikit-learn','nbconvert','ipykernel','ipython','PyYAML','torch-geometric']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    env=dict(python=sys.version,executable=sys.executable,packages=versions,scope='Observed validation environment, not a claim about every original training environment.')
    (ROOT/'validation/transformer_validation_environment.json').write_text(json.dumps(env,indent=2)+'\n')
    report=dict(syntax=syntax,notebook_cell_syntax=notebooks,imports=imports,source_hash_recheck=hashes,
                no_training=True,no_model_forward=True,no_original_source_writes=True,original_source_paths_checked=args.check_originals,
                limitations='No data-dependent training/inference was executed. Import and constructor checks do not establish numerical checkpoint equivalence or hardware reproducibility.')
    (ROOT/'validation/transformer_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(py_files=len(syntax),py_failures=sum(r['status']!='pass' for r in syntax),notebook_cells=len(notebooks),cell_failures=sum(r['status']!='pass' for r in notebooks),imports=[(r['name'],r['status']) for r in imports],source_files=len(hashes),all_hashes_identical=all(r['source_unchanged'] is not False and r['bundle_identical'] for r in hashes))))

if __name__=='__main__':main()
