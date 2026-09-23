#!/usr/bin/env python3
"""Read archived event predictions and verify their physical source identities.

Run with /home/wenyu/summer/.venv/bin/python -B extract_sources.py.
Only this directory receives writes. No training/inference or source mutation.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[2] / 'overleaf'
BASELINE = ROOT.parents[1] / 'baseline/overleaf'
if BASELINE.exists():
    PAPER_BASELINE = BASELINE
else:
    PAPER_BASELINE = PAPER
CLASSICS = {'mvcnn':'cnn_004_multiview_late_fusion','gine':'gnn_001_static_gine','bigru':'seq_001_bigru','mamba':'ssm_001_pointmamba'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()

def ash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()

def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def verify_exo(a):
    """Join every saved event ID back to original HDF5 energy and charge label."""
    data_root=Path('/home/klz/Data/zeronu_benchmark/EXO-200')
    triples=[x.split('::') for x in a['event_id']]
    files=np.array([x[1] for x in triples]); numbers=np.array([int(x[2]) for x in triples])
    checks=[]
    for name in np.unique(files):
        p=data_root/name; ix=np.flatnonzero(files==name)
        with h5py.File(p,'r') as h:
            original=np.asarray(h['event_number']).reshape(-1)
            assert np.unique(original).size==len(original)
            order=np.argsort(original); rows=order[np.searchsorted(original[order],numbers[ix])]
            assert np.array_equal(original[rows],numbers[ix])
            e=np.asarray(h['Rotated_energy'],dtype=np.float64).reshape(-1)[rows]
            charge=np.asarray(h['Charge_cluster_number']).reshape(-1)[rows]
            np.testing.assert_array_equal(e,a['energy_condition'][ix])
            np.testing.assert_array_equal((charge>1).astype(int),a['label'][ix])
            assert np.all(charge>=1)
            checks.append(dict(path=str(p),events=int(len(ix)),raw_size=p.stat().st_size,event_energy_sha256=ash(e),run_number=str(h.attrs.get('run_number'))))
    return dict(method='Every event joined by source file and event_number; raw Rotated_energy and Charge_cluster_number checked exactly',files=checks)

def verify_sn(a):
    """Verify all event IDs, manifest memberships and float64 E1+E2 values."""
    p=Path('/home/wenyu/SuperNEMO/data/manifests/split_manifest.json'); manifest=json.loads(p.read_text())
    parts=manifest['splits']['test']; checks=[]
    for source_key,category in [('2nubb','2nu'),('Bi214','Bi214')]:
        record=next(x for x in manifest['files'] if x['source_key']==source_key)
        raw=Path(manifest['data_root'])/record['file_name']; off=p.parent/record['index_file']
        offsets=np.load(off,mmap_mode='r',allow_pickle=False)
        expected_ids=[]; energies=[]; tracker_count=0
        with h5py.File(raw,'r') as h:
            for part in parts:
                if part['source_key']!=source_key: continue
                for start in range(part['event_start'],part['event_stop'],4096):
                    stop=min(start+4096,part['event_stop']); first_row,last_row=int(offsets[start]),int(offsets[stop])
                    first=offsets[start:stop]-first_row; counts=np.diff(offsets[start:stop+1]); ids=np.arange(start,stop)
                    ev=np.asarray(h['ev_no'][first_row:last_row]); np.testing.assert_array_equal(ev,np.repeat(ids,counts))
                    e1=np.asarray(h['E1'][first_row:last_row]); e2=np.asarray(h['E2'][first_row:last_row])
                    np.testing.assert_array_equal(e1,np.repeat(e1[first],counts)); np.testing.assert_array_equal(e2,np.repeat(e2[first],counts))
                    assert np.all(h['label'][first_row:last_row]==source_key.encode())
                    energies.append(e1[first].astype('float64')+e2[first].astype('float64'))
                    expected_ids.extend(f'SuperNEMO::{source_key}::{i}' for i in ids)
                    tracker_count+=last_row-first_row
        ix=np.flatnonzero(a['category']==category); energies=np.concatenate(energies)
        np.testing.assert_array_equal(a['event_id'][ix],expected_ids)
        np.testing.assert_array_equal(a['energy_condition'][ix],energies)
        checks.append(dict(source_key=source_key,events=len(ix),raw_file=str(raw),raw_size=raw.stat().st_size,offset_file=str(off),offset_sha256=sha(off),tracker_rows_checked=tracker_count,event_energy_sha256=ash(energies)))
    return dict(method='All test events reconstructed in manifest order; repeated tracker rows and class labels checked; E1/E2 individually cast to float64 before summing',manifest=str(p),manifest_sha256=sha(p),files=checks)

def main():
    records=json.loads((PAPER_BASELINE/'wing_contribution/figures/data/classification_table_evidence.json').read_text())['records']
    selected=[r for r in records if r['dataset'] in ('EXO-200','SuperNEMO')]
    out=[]; references={}; raw_checks={}
    for rec in selected:
        ds,key=rec['dataset'],rec['model_key']; model=CLASSICS.get(key,rec.get('source_architecture_id'))
        run=Path('/home/wenyu')/('EXO200' if ds=='EXO-200' else 'SuperNEMO')/'outputs/classification'/model
        pred=run/('energybench_predictions.npz' if ds=='EXO-200' else 'test_evaluation/test_predictions.npz')
        metrics=run/('energybench_classification/.energybench/metrics.json' if ds=='EXO-200' else 'test_evaluation/energybench/.energybench/metrics.json')
        entry=dict(dataset=ds,model_key=key,model=model,paper_location='wing_contribution/tables/benchmark_main.tex',paper_record=rec)
        if not pred.exists():
            entry.update(status='missing_event_inputs',prediction_file=None,old_I=float(rec['independence']) if rec['independence'] else None,old_matched_auc=float(rec['matched_auc']) if rec['matched_auc'] else None,old_precision_note='Only workbook precision available; no full-precision saved evaluation',blocker='No attributable EXO Transformer predictions, checkpoint, or implementation found in current account; workbook values cannot be recomputed or assigned classic protocol')
            out.append(entry); continue
        a=read(pred); report=json.loads(metrics.read_text()); resolved=metrics.parent/'resolved_manifest.json'; cfg=json.loads(resolved.read_text()); metadata=json.loads(a['__metadata__'].item())
        run_config=run/'run_config.json'; training_config=json.loads(run_config.read_text())
        assert len(a['event_id'])==len(np.unique(a['event_id']))
        assert np.all(a['split']=='test')
        assert np.all(np.isfinite(a['score'])) and np.all(np.isfinite(a['energy_condition']))
        assert np.all(a['sample_weight']==1)
        if ds in references:
            for col in ['event_id','label','category','energy_condition','sample_weight','group_id','split']:
                np.testing.assert_array_equal(a[col],references[ds][col],err_msg=f'{ds}/{key} {col}')
        else:
            references[ds]=a
            raw_checks[ds]=verify_exo(a) if ds=='EXO-200' else verify_sn(a)
        if ds=='EXO-200':
            assert set(np.unique(a['group_id']).astype(int))==set(training_config['split_runs']['test'])
            assert training_config['counts']['test']==len(a['score'])
            for split in ['train','validation']:
                assert set(training_config['split_runs'][split]).isdisjoint(training_config['split_runs']['test'])
            np.testing.assert_array_equal(a['category'],np.where(a['label']==0,'signal','background'))
            native=read(run/'predictions.npz')
            np.testing.assert_array_equal(a['label'],native['target'])
            np.testing.assert_array_equal(a['score'],native['prediction'])
            checkpoint=Path(metadata['checkpoint']); assert sha(checkpoint)==metadata['checkpoint_sha256']
            assert sha(run/'predictions.npz')==metadata['prediction_source_sha256']
            label=(1-a['label']).astype('int8'); score=-a['score'].astype('float64')
            orientation='Stored label 0 = signal (n_CCL=1), stored logit increases toward background label 1. Standardized label=1-stored_label, score=-stored_logit.'
        else:
            recorded_provenance=metadata['dataset_provenance']
            assert sha(recorded_provenance['manifest_path'])==recorded_provenance['manifest_file_sha256']
            np.testing.assert_array_equal(a['label'],np.where(a['category']=='2nu',1,0))
            checkpoint=Path(metadata['checkpoint']['path']); assert sha(checkpoint)==metadata['checkpoint']['sha256']
            label=a['label'].astype('int8'); score=a['score'].astype('float64')
            orientation='2nu=1 signal, Bi214=0 background. Stored scalar logit increases toward 2nu; no sigmoid or sign conversion.'
        standard=ROOT/f"{ds.replace('-','').lower()}__{key}.npz"
        np.savez_compressed(standard,score=score,label=label,energy_keV=a['energy_condition'].astype('float64'),group=a['category'].astype(str),event_id=a['event_id'].astype(str),weight=a['sample_weight'].astype('float64'))
        old_I=report['energy_dependence']['overall_energy_independence_score']; pair=report['classification']['pairs'][0]
        assert abs(old_I-float(rec['independence']))<1e-14
        assert abs(pair['matched_auc']-float(rec['matched_auc']))<1e-14
        entry.update(status='ready',prediction_file=str(pred),prediction_sha256=sha(pred),standardized_file=str(standard),standardized_sha256=sha(standard),metrics_file=str(metrics),metrics_sha256=sha(metrics),resolved_manifest_file=str(resolved),resolved_manifest_sha256=sha(resolved),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),old_I=old_I,old_matched_auc=pair['matched_auc'],old_inclusive_auc=pair['inclusive_auc'],old_group_I={k:v['energy_independence_score'] for k,v in report['energy_dependence']['groups'].items()},old_protocol_fingerprint=report['protocol_fingerprint'],old_protocol={'classification':cfg['classification'],'dependence':cfg['dependence']},old_input_metadata=metadata,orientation=orientation,energy_source='Rotated_energy' if ds=='EXO-200' else 'float64(E1)+float64(E2)',energy_unit='keV',energy_clipped=False,n_events=len(score),class_counts={str(k):int((label==k).sum()) for k in [0,1]},out_of_range_counts={str(k):int(((label==k)&((a['energy_condition']<0)|(a['energy_condition']>3000))).sum()) for k in [0,1]},array_hashes={k:ash(a[k]) for k in ['event_id','label','energy_condition','category']},cross_model_event_match=True)
        entry['run_config_file']=str(run_config)
        entry['run_config_sha256']=sha(run_config)
        entry['split_details']=training_config.get('split_runs',metadata.get('dataset_provenance',{}).get('counts'))
        out.append(entry)
        print(ds,key,len(score),'verified',flush=True)
    (ROOT/'manifest.json').write_text(json.dumps({'records':out,'raw_source_checks':raw_checks,'notes':['All archived model inputs retained, including physical energies outside 0..3000 keV.','Workbook-derived values used only for inventory and old-value comparison, never as reevaluation inputs.','EXO missing Transformer entries require source artifacts; no inference performed because attributable checkpoints were not found.']},indent=2)+'\n')
    extract_illustration()

def extract_illustration():
    script=PAPER/'wing_contribution/scripts/prepare_supernemo_matching.py'
    spec=importlib.util.spec_from_file_location('original_illustration_extractor',script); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    manifest=json.loads(mod.MANIFEST.read_text()); energy_audit=json.loads(mod.ENERGY_AUDIT.read_text())
    assert sha(mod.MANIFEST)==energy_audit['manifest_sha256']
    energies=[]; labels=[]; ids=[]; groups=[]; source={}
    for category,label,_ in mod.CLASS_SPECS:
        energy,prov=mod.extract(manifest,category,energy_audit['classes'][category]); source[category]=prov
        category_ids=np.concatenate([np.arange(p['event_start'],p['event_stop']) for p in manifest['splits']['test'] if p['source_key']==category])
        energies.append(energy); labels.append(np.full(len(energy),label,dtype='int8')); groups.append(np.full(len(energy),category)); ids.extend(f'SuperNEMO::{category}::{i}' for i in category_ids)
    energy=np.concatenate(energies); label=np.concatenate(labels)
    target=ROOT/'supernemo_0nu_illustration__energy_only.npz'
    np.savez_compressed(target,score=energy,label=label,energy_keV=energy,group=np.concatenate(groups),event_id=np.array(ids),weight=np.ones(len(energy)))
    evidence=json.loads((PAPER_BASELINE/'wing_contribution/figures/data/supernemo_matching_evidence.json').read_text())
    out=dict(dataset='SuperNEMO-0nu-illustration',model_key='energy_only',standardized_file=str(target),standardized_sha256=sha(target),n_events=len(energy),source=source,energy_definition='float64(E1)+float64(E2), keV',label_definition='0nubb=1, Bi214=0',score_definition='E1+E2 keV',old_evidence_file=str(PAPER_BASELINE/'wing_contribution/figures/data/supernemo_matching_evidence.json'),old_evidence=evidence,paper_locations=['wing_contribution/sections/appendix_energy_illustration.tex','wing_contribution/sections/energybench_auc_figure.tex'],task_note='Separate 0nu/Bi214 energy-only illustration, never trained 2nu/Bi214 classification task')
    (ROOT/'illustration_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Illustration',len(energy),'verified',flush=True)

if __name__=='__main__': main()
