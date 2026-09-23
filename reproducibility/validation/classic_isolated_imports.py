#!/usr/bin/env python3
"""Four source-only constructor checks with a /home read firewall.

The -S worker installs its audit hook before processing this environment's .pth
files, then allows original .pth path additions to remain active. Project code
must still load exclusively from the bundle. No model forward/data access runs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

BUNDLE=Path(__file__).resolve().parents[1]
CLASSIC=BUNDLE/'code/classic'
VENV=Path('/home/wenyu/summer/.venv')
PYTHON_BUILDS=Path('/home/wenyu/summer/envs/python-builds')
DATASETS=['NEXT','MJD','EXO200','SuperNEMO']


def worker(dataset):
    allowed=[os.path.realpath(p) for p in [BUNDLE,VENV,PYTHON_BUILDS]]
    blocked=[]
    home_opens=0
    probe_active=False
    def audit(event,args):
        nonlocal home_opens
        if event!='open' or not args or isinstance(args[0],int):return
        raw=os.fsdecode(args[0])
        candidate=os.path.realpath(os.path.abspath(raw))
        if candidate=='/home' or candidate.startswith('/home/'):
            home_opens+=1
            if not any(candidate==base or candidate.startswith(base+os.sep) for base in allowed):
                blocked.append(dict(path=candidate,probe=probe_active))
                raise PermissionError('Isolation check blocked open outside bundle/interpreter: '+candidate)
    sys.addaudithook(audit)
    sys.dont_write_bytecode=True
    probe_active=True
    try:
        with open('/home/wenyu/MJD/README.md','rb') as f:
            raise AssertionError('Firewall failed: forbidden original source was opened')
    except PermissionError:
        pass
    probe_active=False
    # -S disables startup .pth processing. Process it here, after the audit hook.
    import site
    site.addsitedir(str(VENV/'lib/python3.11/site-packages'))
    roots=[CLASSIC/dataset]
    if dataset=='NEXT':roots=[CLASSIC/'NEXT/01_code/architectures',CLASSIC/'NEXT/src',CLASSIC/'NEXT/evalutaions_workflow']
    elif dataset=='SuperNEMO':roots.append(CLASSIC/'NEXT/src')
    sys.path[:0]=list(map(str,roots))
    import importlib
    if dataset=='NEXT':
        import workflow_runner
        import simple_energybench
        import next_alt.adapter
        from next_alt.registry import build_model
        model=build_model('cnn_004_multiview_late_fusion')
    elif dataset=='MJD':
        import mjdbench.data, mjdbench.training, mjdbench.workflow
        module=importlib.import_module('architectures.cnn_004_multiview_late_fusion.model')
        model=module.build_model('classification')
    elif dataset=='EXO200':
        import exobench.data, exobench.training, exobench.workflow
        module=importlib.import_module('architectures.cnn_004_multiview_late_fusion.model')
        model=module.build_model()
    else:
        import supernemobench.data, supernemobench.training, supernemobench.workflow
        from supernemobench.models import build_model
        model=build_model('cnn_004_multiview_late_fusion','classification')
    prefixes=('architectures','mjdbench','exobench','supernemobench','next_alt','next_cnn','simple_energybench','energybench','workflow_runner','workflow_models','workflow_data','classic_topology')
    modules={}
    for name,module in sorted(sys.modules.items()):
        if not any(name==p or name.startswith(p+'.') for p in prefixes):continue
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            assert path.is_relative_to(BUNDLE),(name,str(path))
            modules[name]=str(path.relative_to(BUNDLE))
        for namespace in getattr(module,'__path__',[]):
            assert Path(namespace).resolve().is_relative_to(BUNDLE),(name,namespace)
    violations=[x for x in blocked if not x['probe']]
    result=dict(dataset=dataset,passed=not violations,audit_hook_installed_before_pth=True,
        pth_processing_enabled=True,allowlisted_home_roots=allowed,forbidden_open_probe_blocked=True,
        blocked_nonprobe_opens=violations,home_open_events=home_opens,
        project_modules=modules,constructor_class=type(model).__module__+'.'+type(model).__qualname__,
        parameter_count=sum(x.numel() for x in model.parameters()),forward_executed=False,data_loaded=False)
    destination=BUNDLE/'validation'/('classic_isolated_'+dataset+'.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['project_modules','allowlisted_home_roots']},indent=2))
    return int(not result['passed'])


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':return worker(sys.argv[2])
    checks=[]
    for ds in DATASETS:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',MPLCONFIGDIR=str(BUNDLE/'validation/classic_isolated_mpl'))
        command=[str(VENV/'bin/python'),'-B','-S',str(Path(__file__).resolve()),'--worker',ds]
        p=subprocess.run(command,cwd=BUNDLE,env=env,capture_output=True,text=True,timeout=120)
        logfile=BUNDLE/'validation'/('classic_isolated_'+ds+'.log')
        logfile.write_text(p.stdout+p.stderr)
        checks.append(dict(dataset=ds,command=command,returncode=p.returncode,log=str(logfile.relative_to(BUNDLE)),
            detail='validation/classic_isolated_'+ds+'.json'))
        print(ds,p.returncode,p.stdout,p.stderr,flush=True)
    payload=dict(all_pass=all(x['returncode']==0 for x in checks),checks=checks,
        scope='Representative package imports and MV-CNN constructors for four datasets, under filesystem audit hook; no forward or data loads.',
        frozen_sources_modified=False,environment_pth_processed_after_firewall=True)
    (BUNDLE/'validation/classic_isolation_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    return int(not payload['all_pass'])

if __name__=='__main__':raise SystemExit(main())
