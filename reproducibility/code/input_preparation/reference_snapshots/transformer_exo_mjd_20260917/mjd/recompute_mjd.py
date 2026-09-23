#!/usr/bin/env python3
"""Audit saved MJD binary Transformer predictions and recompute frozen v2 metrics.

No training or inference; source tree and checkpoints are read-only. Full raw
HDF5 files are not hashed (waveforms are not evaluation inputs); every scalar
source field used for event identity, labels, and energy is hashed instead.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import h5py
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_PAPER = Path('/home/wenyu/iclr final paper/overleaf')
DEFAULT_SOURCE = Path('/home/klz/Data/zeronu_benchmark/Transformer_Approach')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def ready(x):
    if isinstance(x, dict):
        return {str(k): ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ready(v) for v in x]
    if isinstance(x, np.ndarray):
        return ready(x.tolist())
    if isinstance(x, np.generic):
        return ready(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save(path, data):
    Path(path).write_text(json.dumps(ready(data), indent=2, allow_nan=False) + '\n')


def omit_roc_arrays(value):
    """Retain metrics and support diagnostics without million-entry ROC arrays."""
    if isinstance(value, dict):
        return {k: omit_roc_arrays(v) for k, v in value.items()
                if k not in ('fpr', 'tpr', 'thresholds')}
    if isinstance(value, (list, tuple)):
        return [omit_roc_arrays(v) for v in value]
    return value


def small(r):
    return dict(inclusive_auc=r['inclusive_auc'], I=r['independence']['I'],
                min_group_I=r['independence']['min_group_I'],
                matched_auc=r['matching']['matched_auc'],
                I_groups={k: v['I_g'] for k, v in r['independence']['groups'].items()},
                I_valid_bins={k: v['valid_bin_count'] for k, v in r['independence']['groups'].items()},
                matching_valid_bins=r['matching']['valid_bin_count'],
                matching_status=r['matching']['status'],
                common_support_keV=r['matching']['common_support_keV'],
                matching_classes=r['matching']['classes'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation', type=Path, default=HERE/'reference_v2',
                    help='Frozen v2 evaluator for this historical intermediate audit')
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--data-root', type=Path, default=Path('/home/klz/Data/zeronu_benchmark/MJD'))
    ap.add_argument('--output', type=Path, default=HERE)
    args = ap.parse_args()
    out = args.output.resolve()
    for d in ['standardized', 'details', 'legacy_replay']:
        (out / d).mkdir(parents=True, exist_ok=True)
    evaluation = args.evaluation
    sys.path.insert(0, str(evaluation))
    from unified_metrics import evaluate, energy_bin_indices, load_config, fingerprint
    src = args.source / 'mjd_detector'
    sys.path.insert(0, str(src))
    from mjdbench.data import discover_files, MJDWaveformDataset
    from mjd_transformer.energy_aware import _build_model
    import torch
    sys.path.insert(0, str(args.source / 'evalutaions_workflow'))
    import simple_energybench.metrics as legacy
    cfg = load_config()
    assert fingerprint(cfg) == 'f862fc8eb3b03cb02640347098f9b19a71e3ea7b8ad1298b868b96de4154aa88'
    shards = discover_files(args.data_root, 'test')
    assert len(shards) == 6
    dataset = MJDWaveformDataset(shards, task='classification', baseline_samples=200,
                                 classification_amplitude_normalization=True)
    raw_fields = ['energy_label', 'id', 'run_number', 'detector', 'tp0',
                  'psd_label_low_avse', 'psd_label_high_avse', 'psd_label_dcr', 'psd_label_lq']
    arrays, shard_audit, identities = {k: [] for k in raw_fields}, [], []
    for shard, info in zip(shards, dataset._files):
        assert info.path == shard.resolve() and info.selected_rows is None
        with h5py.File(shard, 'r') as h:
            fields = {k: np.asarray(h[k][:]) for k in raw_fields}
        n = len(fields['energy_label'])
        for k, v in fields.items():
            assert len(v) == n
            arrays[k].append(v)
        clean = np.all(np.column_stack([fields[k] for k in raw_fields[5:]]), axis=1)
        assert np.array_equal(info.clean, clean)
        assert np.array_equal(info.energies, fields['energy_label'].astype('float32'))
        assert all(np.array_equal(info.metadata[k], fields[k]) for k in raw_fields[1:5])
        identities.extend(f'{shard.name}::row={i}::id={int(raw_id)}' for i, raw_id in enumerate(fields['id']))
        shard_audit.append(dict(path=str(shard), bytes=shard.stat().st_size,
            mtime_ns=shard.stat().st_mtime_ns, n=n,
            fields={k: dict(dtype=str(v.dtype), shape=list(v.shape), sha256=array_sha(v))
                    for k, v in fields.items()}))
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    energy = arrays['energy_label']
    energy32 = energy.astype('float32').astype('float64')
    label = np.all(np.column_stack([arrays[k] for k in raw_fields[5:]]), axis=1).astype('int8')
    event_id = np.asarray(identities)
    assert len(event_id) == len(np.unique(event_id)) == len(dataset) == 390000
    assert len(dataset._offsets) == 7
    group = np.where(label == 1, 'clean', 'nonclean')
    # This verifies Transformer identity against the earlier classic audit only
    # after independently rebuilding the event map from its own actual loader.
    classic_path = HERE.parents[1] / 'sources/next_mjd/MJD_official_test_metadata.npz'
    classic_comparison = {'path': str(classic_path), 'exists': classic_path.exists()}
    if classic_path.exists():
        with np.load(classic_path) as z:
            classic_comparison['keys'] = z.files
            for key in ['energy_label', 'label', 'physical_event_id', 'id', 'run_number', 'detector', 'tp0']:
                if key in z:
                    expected = dict(**arrays, label=label, physical_event_id=event_id)[key]
                    classic_comparison[key + '_exact_equal'] = bool(np.array_equal(expected, z[key]))
                    assert classic_comparison[key + '_exact_equal']
    boundary_changed = np.flatnonzero(energy_bin_indices(energy) != energy_bin_indices(energy32))
    identity_audit = dict(n=len(label), source_shards=shard_audit,
        source_loader=str(src/'mjdbench/data.py'), source_loader_sha256=sha(src/'mjdbench/data.py'),
        event_id_definition='shard basename::row=<zero-based within-shard row>::id=<stored id>',
        event_id_order_sha256=array_sha(event_id), ids_unique=True,
        mapping='Exact actual loader file order and no row filtering for classification; official test indices arange and shuffle=False; per-row saved labels and float32 energies asserted for every model.',
        id_array_sha256=array_sha(arrays['id']), raw_energy_sha256=array_sha(energy),
        raw_energy_unit='keV', original_dtype=str(energy.dtype),
        finite=int(np.isfinite(energy).sum()), negative=int((energy < 0).sum()),
        overflow_by_class={str(k): int(((label == k)&(energy > 3000)).sum()) for k in (0,1)},
        counts_by_class={str(k): int((label == k).sum()) for k in (0,1)},
        classic_comparison=classic_comparison,
        float32_to_float64_bin_changes=[dict(index=int(i), event_id=event_id[i], label=int(label[i]),
            float32_keV=float(energy32[i]), raw_float64_keV=float(energy[i])) for i in boundary_changed])
    save(out/'event_identity_audit.json', identity_audit)
    np.savez_compressed(out/'test_metadata.npz', event_id=event_id, energy_keV=energy, label=label,
                        group=group, **{k:v for k,v in arrays.items() if k != 'energy_label'})
    manifest = dict(dataset='MJD', protocol_version=cfg['protocol_version'],
        protocol_sha256=fingerprint(cfg), evaluator_path=str(evaluation/'core/unified_metrics.py'),
        evaluator_file_sha256=sha(evaluation/'core/unified_metrics.py'),
        protocol_file_sha256=sha(evaluation/'protocol.json'),
        source_root=str(src), event_audit=str(out/'event_identity_audit.json'),
        legacy_evaluator_path=str(Path(legacy.__file__)), legacy_evaluator_sha256=sha(legacy.__file__),
        ready=[], out_of_scope_user=[dict(model_key=t+'_rope', status='out_of_scope_user',
            reason='User explicitly said RoPE need not be handled; existing historical paper values remain outside this campaign.')
            for t in ['entity','region','summary']])
    rows = []
    for run in sorted((src/'results/transformer_official_v1').glob('classification__*')):
        config = json.loads((run/'run_config.json').read_text())
        representation = config['representation']
        token = {'pulse_entities':'entity','raw_patches':'region','segment_summary':'summary'}[representation['tokenization']['tokenization']]
        pos = {'coordinate_mlp':'mlp','fourier_coordinates':'fourier'}[representation['position_encoding']]
        key = token+'_'+pos
        orig_path = run/'predictions.npz'
        pred_path = run/'energybench_classification/predictions.npz'
        old_path = run/'energybench_classification/metrics.json'
        saved = json.loads(old_path.read_text())
        with np.load(pred_path) as z:
            prediction = {k: z[k] for k in z.files}
        with np.load(orig_path) as z:
            assert np.array_equal(prediction['score'], z['prediction'])
            assert np.array_equal(prediction['label'], z['target'])
        assert np.array_equal(prediction['label'], label)
        assert np.array_equal(prediction['energy_kev'], energy32)
        assert np.array_equal(prediction['energy_mev'], energy32 / 1000)
        assert config['counts']['test'] == len(label)
        score = prediction['score'].astype('float64')
        assert np.isfinite(score).all()
        # Verify saved model architecture and exact parameter count. No inference.
        checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        run_summary = json.loads((run/'run_summary.json').read_text())
        checkpoint_metadata = dict(epoch=int(checkpoint['epoch']),
                                   validation_score=float(checkpoint['score']))
        assert checkpoint_metadata['epoch'] == run_summary['best_epoch']
        assert checkpoint_metadata['validation_score'] == run_summary['best_validation_score']
        model = _build_model(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params == config['parameter_count']
        del model, checkpoint
        standard = out/'standardized'/f'MJD_{key}.npz'
        np.savez_compressed(standard, score=score, label=label, energy_keV=energy,
            group=group, event_id=event_id, weight=np.ones(len(label)),
            legacy_energy_float32_keV=energy32, legacy_energy_MeV=prediction['energy_mev'])
        # Replay the saved 836-bin observed-range protocol explicitly. Current
        # module defaults alone would incorrectly reject energies above 3000.
        settings = dict(saved['protocol'])
        settings['distance_correlation_max_samples'] = 4
        legacy.CANONICAL_ENERGY_MIN_KEV = saved['protocol']['energy_grid_min_kev']
        legacy.CANONICAL_ENERGY_MAX_KEV = saved['protocol']['energy_grid_max_kev']
        legacy.CANONICAL_ENERGY_BIN_COUNT = saved['protocol']['energy_grid_bin_count']
        assert legacy.CANONICAL_ENERGY_MAX_KEV == 4180.0 and legacy.CANONICAL_ENERGY_BIN_COUNT == 836
        w = np.ones(len(label))
        old_d = legacy.evaluate_energy_dependence(label, score, prediction['energy_mev'], w, None, None, settings)
        old_m = legacy.evaluate_energy_matched_classification(label, score, prediction['energy_mev'], w, settings)
        old = dict(I=saved['energy_independence_score'], matched_auc=saved['matched_auc'], inclusive_auc=saved['auc'])
        replay = dict(I=old_d['overall_energy_independence_score'], matched_auc=old_m['matched_auc'], inclusive_auc=old_m['inclusive_auc'])
        errors = {k: replay[k]-old[k] for k in old}
        assert all(abs(v)<1e-10 for v in errors.values()), (key, errors)
        save(out/'legacy_replay'/f'MJD_{key}.json', dict(saved=old, recomputed=replay,
            difference=errors, saved_protocol=saved['protocol'],
            diagnostic_only_change='dCor subsample reduced to4; headline I/AUC definitions/settings unchanged',
            dependence=old_d, matching=omit_roc_arrays(old_m)))
        stages = {'old_saved_836_bins': old, 'old_replayed_836_bins': replay}
        float32 = evaluate(label, score, energy32, group, w, config=cfg)
        result = evaluate(label, score, energy, group, w, config=cfg)
        assert abs(result['inclusive_auc']-old['inclusive_auc']) < 1e-12
        stages['unified_float32_with_overflow'] = small(float32)
        stages['unified_float64_with_overflow_final'] = small(result)
        # Distinguish MeV floating-edge assignment from changed overflow policy.
        legacy_edges = legacy.make_fixed_energy_bins(prediction['energy_mev'], 5.0, 'MeV')
        old_idx = legacy._bin_indices(prediction['energy_mev'], legacy_edges)
        new32_idx = energy_bin_indices(energy32)
        regular_changes = np.flatnonzero((old_idx != new32_idx)&(energy32 <= 3000))
        attribution = dict(stages=stages, old_replay_minus_saved=errors,
            overflow_policy='Historical 836 regular 5keV bins to4180; v2 uses600 regular bins through3000 plus one overflow bin.',
            old_to_v2_regular_bin_changes=[dict(index=int(i),event_id=event_id[i],
                energy_keV=float(energy32[i]), old_bin=int(old_idx[i]),new_bin=int(new32_idx[i])) for i in regular_changes],
            float32_to_float64_bin_changes=identity_audit['float32_to_float64_bin_changes'],
            deltas=[dict(before=a,after=b,**{k:stages[b][k]-stages[a][k] for k in old})
                for a,b in [('old_replayed_836_bins','unified_float32_with_overflow'),
                            ('unified_float32_with_overflow','unified_float64_with_overflow_final')]])
        detail = out/'details'/f'MJD_{key}.json'
        save(detail, dict(dataset='MJD', model_key=key, run_id=run.name,
            metrics=result, change_attribution=attribution))
        record = dict(dataset='MJD', model_key=key, run_id=run.name, status='ready',
            family='Transformer', architecture_id=key, exact_parameter_count=params,
            checkpoint_metadata=checkpoint_metadata,
            standardized_predictions=str(standard), standardized_sha256=sha(standard),
            detail_json=str(detail), source_predictions=str(pred_path),
            old_metrics=old, source_metrics_json=str(old_path), legacy_protocol=saved['protocol'],
            source_files={str(run/r):sha(run/r) for r in ['run_config.json','run_summary.json','best.pt','predictions.npz',
                'energybench_classification/predictions.npz','energybench_classification/metrics.json']},
            source_code={str(src/r):sha(src/r) for r in ['mjdbench/data.py','mjdbench/training.py',
                'mjd_transformer/model.py','mjd_transformer/energy_aware.py']},
            score_definition='Raw single binary logit from trained clean-vs-nonclean classifier; higher meansclean. No sigmoid/product/sign flip.',
            label_definition='1 iff low_avse,high_avse,dcr,lq PSD flags all true;0 otherwise.',
            energy_definition='Original per-row float64 HDF5 energy_label in keV, recovered only after exactfull-vector float32energy and label comparison with Transformer saved prediction cache; no clipping.',
            event_identity='Actual Transformer sorted six officialtestshards, allrows; exactsaved label andfloat32energy match; unique shard/row/rawid identifier.',
            test_event_id_sha256=array_sha(event_id), n_events=len(label),
            inclusive_population_unchanged=True, prediction_scores_equal_training_saved=True,
            inference_run=False, training_run=False)
        manifest['ready'].append(record)
        save(out/'manifest.json',manifest)
        row = dict(dataset='MJD',model=key,run_id=run.name,params=params,n=len(label),
            old_I=old['I'],new_I=result['independence']['I'],delta_I=result['independence']['I']-old['I'],
            old_matched_auc=old['matched_auc'],new_matched_auc=result['matching']['matched_auc'],
            delta_matched_auc=result['matching']['matched_auc']-old['matched_auc'],
            old_inclusive_auc=old['inclusive_auc'],new_inclusive_auc=result['inclusive_auc'],
            min_group_I=result['independence']['min_group_I'], matching_status=result['matching']['status'],
            matching_valid_bins=result['matching']['valid_bin_count'],
            **{f'I_{g}':r['I_g'] for g,r in result['independence']['groups'].items()},
            **{f'I_bins_{g}':r['valid_bin_count'] for g,r in result['independence']['groups'].items()},
            **{f'coverage_class{k}':r['matched_fraction_original_finite'] for k,r in result['matching']['classes'].items()},
            **{f'ESS_class{k}':r['matched_ess'] for k,r in result['matching']['classes'].items()},
            source=str(pred_path),protocol_version=cfg['protocol_version'],protocol_sha256=fingerprint(cfg))
        rows.append(row)
        save(out/'summary.json', rows)
        with (out/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
        print(key, 'I', row['new_I'], 'matched', row['new_matched_auc'], 'params',params,flush=True)
    assert len(rows)==6
    save(out/'manifest.json',manifest)


if __name__ == '__main__':
    main()
