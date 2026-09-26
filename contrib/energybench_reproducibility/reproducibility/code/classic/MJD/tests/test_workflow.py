"""Synthetic checks for the MJD data and paired-model contracts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from mjdbench import (
    DataConfig,
    MJDWaveformDataset,
    TrainingConfig,
    WaveformCNNClassifier,
    WaveformCNNRegressor,
    evaluate_model,
    inspect_data_root,
    prepare_dataset,
    train_model,
)
from mjdbench.training import _clean_logit_from_auxiliary


def _write_shard(path: Path, events: int, *, seed: int) -> None:
    generator = np.random.default_rng(seed)
    waveform = generator.normal(size=(events, 128)).astype(np.float32)
    waveform += np.linspace(0.0, 2.0, events, dtype=np.float32)[:, None]
    labels = generator.integers(0, 2, size=(events, 4), dtype=np.int8)
    labels[:2] = 1
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw_waveform", data=waveform)
        handle.create_dataset("energy_label", data=np.linspace(100, 2500, events))
        handle.create_dataset("psd_label_low_avse", data=labels[:, 0])
        handle.create_dataset("psd_label_high_avse", data=labels[:, 1])
        handle.create_dataset("psd_label_dcr", data=labels[:, 2])
        handle.create_dataset("psd_label_lq", data=labels[:, 3])
        handle.create_dataset("tp0", data=np.arange(events))
        handle.create_dataset("detector", data=np.full(events, 111))
        handle.create_dataset("run_number", data=np.full(events, seed))
        handle.create_dataset("id", data=np.arange(seed * 1000, seed * 1000 + events))


def test_schema_loading_clean_filter_and_preprocessing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "MJD_Train_0.hdf5"
        _write_shard(path, 20, seed=1)
        report = inspect_data_root(temporary)
        assert report[0]["status"] == "ok"
        assert report[0]["waveform_shape"] == [128]

        classification = MJDWaveformDataset(
            [path], task="classification", baseline_samples=16
        )
        example = classification[0]
        assert example["inputs"].shape == (1, 128)
        assert example["labels"].shape == (4,)
        assert torch.max(torch.abs(example["inputs"])) <= 1.00001

        regression = MJDWaveformDataset(
            [path], task="regression", baseline_samples=16
        )
        with h5py.File(path, "r") as handle:
            expected_clean = int(
                np.sum(
                    np.column_stack(
                        [
                            handle["psd_label_low_avse"][:],
                            handle["psd_label_high_avse"][:],
                            handle["psd_label_dcr"][:],
                            handle["psd_label_lq"][:],
                        ]
                    ).all(axis=1)
                )
            )
        assert len(regression) == expected_clean
        assert bool(regression[0]["clean"])


def test_official_split_roles_and_paired_backbone() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_shard(root / "MJD_Train_0.hdf5", 30, seed=2)
        _write_shard(root / "MJD_Test_0.hdf5", 12, seed=3)
        data = prepare_dataset(
            task="classification",
            data_config=DataConfig(data_root=root, baseline_samples=16),
            batch_size=4,
            num_workers=0,
        )
        assert data.counts == {"train": 27, "validation": 3, "test": 12}
        batch = next(iter(data.train_loader))
        assert batch["inputs"].shape == (4, 1, 128)
        assert batch["labels"].shape == (4, 4)

        classifier = WaveformCNNClassifier(base_channels=4)
        regressor = WaveformCNNRegressor(base_channels=4)
        assert classifier(batch["inputs"]).shape == (4,)
        assert regressor(batch["inputs"]).shape == (4,)
        classifier_backbone = classifier.backbone.state_dict()
        regressor_backbone = regressor.backbone.state_dict()
        assert classifier_backbone.keys() == regressor_backbone.keys()
        for key in classifier_backbone:
            assert classifier_backbone[key].shape == regressor_backbone[key].shape


def test_training_checkpoint_and_evaluation_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _write_shard(root / "MJD_Train_0.hdf5", 30, seed=4)
        _write_shard(root / "MJD_Test_0.hdf5", 12, seed=5)
        data = prepare_dataset(
            task="classification",
            data_config=DataConfig(data_root=root, baseline_samples=16),
            batch_size=8,
            num_workers=0,
        )
        model = WaveformCNNClassifier(base_channels=2)
        output = root / "output"
        history = train_model(
            model,
            data.train_loader,
            data.validation_loader,
            task="classification",
            config=TrainingConfig(
                batch_size=8,
                epochs=1,
                early_stopping_patience=1,
                device="cpu",
            ),
            output_dir=output,
        )
        metrics = evaluate_model(
            model,
            data.test_loader,
            task="classification",
            device="cpu",
            output_dir=output,
        )
        assert len(history) == 1
        assert metrics["events"] == 12
        assert (output / "best.pt").is_file()
        assert (output / "history.json").is_file()
        assert (output / "metrics.json").is_file()
        assert (output / "predictions.npz").is_file()


def test_four_label_logits_produce_ordered_clean_logits() -> None:
    auxiliary_logits = torch.tensor(
        [
            [4.0, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0, -4.0],
            [-4.0, -4.0, -4.0, -4.0],
        ]
    )
    clean_logits = _clean_logit_from_auxiliary(auxiliary_logits)
    assert clean_logits.shape == (3,)
    assert torch.isfinite(clean_logits).all()
    assert clean_logits[0] > clean_logits[1] > clean_logits[2]
