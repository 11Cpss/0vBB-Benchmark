#!/usr/bin/env python3
"""Train or export native predictions for the six published SuperNEMO Transformers."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--manifest-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    rows = json.loads((ROOT/'results/transformer_models.json').read_text())['rows']
    found = [r for r in rows if r['dataset'] == 'SuperNEMO' and r['model_key'] == args.model_key]
    if not found or not found[0].get('source_training_config'):
        parser.error('No source-backed configuration for this model.')
    row = found[0]
    config = json.loads((ROOT/row['source_training_config']).read_text())
    if args.describe:
        print(json.dumps(config, indent=2))
        return
    if any(value is None for value in (args.data_root, args.manifest_path, args.output_dir)):
        parser.error('Require --data-root, --manifest-path and --output-dir.')
    if args.mode == 'test' and args.checkpoint is None:
        parser.error('Test mode requires the explicit original --checkpoint.')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be new or empty.')
    reference = json.loads((HERE/'detectors/supernemo/data/manifests/split_manifest.json').read_text())
    runtime_manifest = json.loads(args.manifest_path.read_text())
    for key in ('splits', 'counts', 'settings', 'grouping', 'input_policy'):
        if runtime_manifest[key] != reference[key]:
            raise ValueError(f'Original SuperNEMO split differs: {key}')
    sys.path.insert(0, str(HERE/'detectors/supernemo'))
    import torch
    from architectures import TrackerHitTransformer
    from supernemobench.config import DataConfig, TrainingConfig
    from supernemobench.data import prepare_dataset
    from supernemobench.tokenization import SuperNEMOTrackerTokenizationConfig
    from supernemobench.training import set_seed, train_model, _epoch, _device, _amp
    from supernemobench.evaluation import dataset_provenance, classification_bundle, _save_bundle_atomic

    data_config = DataConfig(**{**config['data'], 'data_root': str(args.data_root.resolve()),
                                'manifest_path': str(args.manifest_path.resolve())})
    training = TrainingConfig(**{**config['training'], **({'device': args.device} if args.device else {})})
    tokenization = SuperNEMOTrackerTokenizationConfig(**config['tokenization'])
    data = prepare_dataset(task='classification', input_kind='sequence', data_config=data_config,
                           tokenization_config=tokenization, batch_size=training.batch_size,
                           num_workers=training.num_workers)
    if data.counts != config['counts']:
        raise ValueError('Test/train/validation counts differ from the paper run.')
    set_seed(training.seed, training.deterministic)
    model = TrackerHitTransformer(**config['model'])
    model.architecture_id = config['architecture_id']
    model.model_name = config['model_name']
    model.input_kind = 'sequence'
    model.task = 'classification'
    model.tokenization_config = tokenization
    provenance = dataset_provenance(manifest_path=args.manifest_path, data_root=args.data_root,
                    task='classification', counts=data.counts, data_config=data_config.to_dict(),
                    input_kind='sequence', representation_config=tokenization.to_dict())
    output.mkdir(parents=True, exist_ok=True)
    (output/'run_config.json').write_text(json.dumps({**config, 'data': data_config.to_dict(),
                         'training': training.to_dict(), 'dataset_provenance': provenance}, indent=2)+'\n')
    if args.mode == 'train':
        train_model(model, data.train_loader, data.validation_loader, 'classification', training,
                    output, model_config=config['model'], provenance=provenance)
        checkpoint_path = output/'best.pt'
    else:
        checkpoint_path = args.checkpoint
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != row['checkpoint']['sha256']:
            raise ValueError('Checkpoint is not the recorded original for this paper row.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['model_config'] != config['model'] or checkpoint['architecture_id'] != config['architecture_id']:
            raise ValueError('Checkpoint architecture/config does not match the selected row.')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    device = _device(training.device)
    model.to(device)
    amp_enabled, amp_dtype = _amp(device, training.use_amp, training.amp_precision)
    loss, metrics, target, score, event_id, category, energy, group_id, split = _epoch(
        model, data.test_loader, task='classification', device=device, optimizer=None,
        gradient_clip_norm=training.gradient_clip_norm, amp_enabled=amp_enabled,
        amp_dtype=amp_dtype, collect_outputs=True)
    bundle = classification_bundle(event_id=event_id, label=target, category=category, score=score,
             energy_condition=energy, group_id=group_id, split=split,
             metadata={'dataset_provenance': provenance, 'native_metrics': {**metrics, 'loss': loss},
                       'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()})
    _save_bundle_atomic(bundle, output/'test_predictions.npz')
    print('Native keV predictions written. Use the SuperNEMO paper profile in benchmark/ for final metrics.')


if __name__ == '__main__':
    main()
