"""Focused regression tests for the SuperNEMO Transformer workflow."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import supernemobench.training as training_module
from supernemobench.config import (
    ARCHITECTURES,
    TRANSFORMER_ARCHITECTURE_IDS,
    TrainingConfig,
)
from supernemobench.evaluation import (
    EVALUATION_MANIFEST,
    classification_bundle,
    dataset_provenance,
    energybench_code_sha256,
    evaluate_classification_bundle,
    file_sha256,
)
from supernemobench.models import build_model, registered_models
from supernemobench.training import train_model
from supernemobench.workflow import _load_checkpoint


TRANSFORMER_CASES = (
    (
        "transformer_001_sampled_hits_coordinate_mlp",
        "sampled_hits",
        "coordinate_mlp",
        4,
        111_233,
    ),
    (
        "transformer_002_voxel_coordinate_mlp",
        "voxel",
        "coordinate_mlp",
        4,
        111_233,
    ),
    (
        "transformer_003_voxel_fourier_xyz",
        "voxel",
        "fourier_xyz",
        4,
        113_537,
    ),
    (
        "transformer_004_sampled_hits_fourier_xyz",
        "sampled_hits",
        "fourier_xyz",
        4,
        113_537,
    ),
    (
        "transformer_005_summary_features_coordinate_mlp",
        "summary_features",
        "coordinate_mlp",
        6,
        111_361,
    ),
    (
        "transformer_006_summary_features_fourier_xyz",
        "summary_features",
        "fourier_xyz",
        6,
        113_665,
    ),
)
CANONICAL_CLASSIFICATION_COLUMNS = {
    "event_id",
    "label",
    "category",
    "score",
    "energy_condition",
    "sample_weight",
    "group_id",
    "split",
}


def _point_batch(feature_dim: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260828)
    coords = torch.randn(3, 7, 3, generator=generator)
    features = torch.randn(3, 7, feature_dim, generator=generator)
    mask = torch.tensor(
        (
            (True, True, True, True, True, True, True),
            (True, True, True, True, False, False, False),
            (True, True, False, False, False, False, False),
        ),
        dtype=torch.bool,
    )
    return {"coords": coords, "features": features, "mask": mask}


@pytest.mark.parametrize(
    (
        "architecture_id",
        "tokenization",
        "position_encoding",
        "feature_dim",
        "parameter_count",
    ),
    TRANSFORMER_CASES,
)
def test_transformer_registry_and_parameter_count(
    architecture_id: str,
    tokenization: str,
    position_encoding: str,
    feature_dim: int,
    parameter_count: int,
) -> None:
    assert architecture_id in registered_models()
    model = build_model(architecture_id, "classification")
    assert model.architecture_id == architecture_id
    assert model.position_encoding == position_encoding
    assert model.input_kind == "sequence"
    assert model.task == "classification"
    assert model.feature_dim == feature_dim
    assert model.tokenization_config.tokenization == tokenization
    assert ARCHITECTURES[architecture_id].tokenization == model.tokenization_config
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_active_registry_contains_exactly_four_baselines_and_six_transformers() -> None:
    registered = registered_models()
    assert len(registered) == 10
    assert tuple(identifier for identifier in registered if identifier.startswith("transformer_")) == (
        TRANSFORMER_ARCHITECTURE_IDS
    )
    assert not any(identifier.startswith("trf_") for identifier in registered)


@pytest.mark.parametrize(
    ("architecture_id", "feature_dim"),
    [(case[0], case[3]) for case in TRANSFORMER_CASES],
)
def test_transformer_gradients_and_set_invariances(
    architecture_id: str,
    feature_dim: int,
) -> None:
    model = build_model(architecture_id, "classification")
    batch = _point_batch(feature_dim)

    model.train()
    output = model(batch)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    model.eval()
    with torch.inference_mode():
        reference = model(batch)

        changed_padding = {name: value.clone() for name, value in batch.items()}
        invalid_slots = ~changed_padding["mask"]
        changed_padding["coords"][invalid_slots] = 1_000.0
        changed_padding["features"][invalid_slots] = -1_000.0
        changed_padding_output = model(changed_padding)

        padded = {
            "coords": torch.cat((batch["coords"], torch.full((3, 2, 3), 500.0)), dim=1),
            "features": torch.cat(
                (
                    batch["features"],
                    torch.full((3, 2, feature_dim), -500.0),
                ),
                dim=1,
            ),
            "mask": torch.cat(
                (batch["mask"], torch.zeros((3, 2), dtype=torch.bool)), dim=1
            ),
        }
        padded_output = model(padded)

        permutation = torch.tensor((5, 1, 6, 0, 3, 2, 4), dtype=torch.int64)
        permuted_output = model(
            {
                "coords": batch["coords"][:, permutation],
                "features": batch["features"][:, permutation],
                "mask": batch["mask"][:, permutation],
            }
        )

    torch.testing.assert_close(changed_padding_output, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(padded_output, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(permuted_output, reference, rtol=2e-5, atol=2e-6)

    empty_event = {name: value.clone() for name, value in batch.items()}
    empty_event["mask"].fill_(False)
    with pytest.raises(ValueError, match="at least one valid token"):
        model(empty_event)


def _small_bundle_inputs() -> dict[str, Any]:
    return {
        "event_id": np.asarray(["Bi214:1", "2nu:1", "Bi214:2", "2nu:2"]),
        "label": np.asarray([0, 1, 0, 1]),
        "category": np.asarray(["Bi214", "2nu", "Bi214", "2nu"]),
        "score": np.asarray([-1.2, 0.8, -0.5, 1.4], dtype=np.float64),
        "energy_condition": np.asarray([450.0, 1_500.0, 3_050.0, 3_500.0]),
        "group_id": np.asarray(["g0", "g1", "g2", "g3"]),
        "split": np.asarray(["test"] * 4),
        "metadata": {"purpose": "unit-test"},
    }


def test_classification_bundle_contract_and_label_semantics() -> None:
    protocol = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    assert protocol["classification"]["positive_label"] == "1"
    assert protocol["classification"]["signal_categories"] == ["2nu"]
    assert protocol["classification"]["background_categories"] == ["Bi214"]
    assert protocol["classification"]["score_space"] == "logit"
    assert protocol["columns"]["energy_condition"] == "energy_condition"

    inputs = _small_bundle_inputs()
    bundle = classification_bundle(**inputs)

    assert set(bundle.columns) == CANONICAL_CLASSIFICATION_COLUMNS
    assert bundle.require("event_id").dtype.kind == "U"
    assert bundle.require("category").dtype.kind == "U"
    assert bundle.require("group_id").dtype.kind == "U"
    assert bundle.require("split").dtype.kind == "U"
    assert bundle.require("label").dtype == np.dtype(np.int8)
    assert bundle.require("score").dtype == np.dtype(np.float32)
    assert bundle.require("energy_condition").dtype == np.dtype(np.float64)
    assert bundle.require("sample_weight").dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        bundle.require("energy_condition"), inputs["energy_condition"]
    )
    assert np.any(bundle.require("energy_condition") > 3_000.0)

    conflicting = dict(inputs)
    conflicting["label"] = np.asarray([1, 1, 0, 1])
    with pytest.raises(ValueError, match="category and label"):
        classification_bundle(**conflicting)


def _synthetic_evaluation_columns() -> dict[str, np.ndarray]:
    events_per_class = 600
    energy_grid = np.linspace(500.0, 3_500.0, events_per_class, dtype=np.float64)
    labels = np.repeat(np.asarray([0, 1], dtype=np.int8), events_per_class)
    categories = np.where(labels == 1, "2nu", "Bi214")
    generator = np.random.default_rng(20260828)
    scores = np.concatenate(
        (
            generator.normal(-0.8, 0.7, events_per_class),
            generator.normal(0.8, 0.7, events_per_class),
        )
    )
    return {
        "event_id": np.asarray(
            [f"{category}:{index}" for index, category in enumerate(categories)]
        ),
        "label": labels,
        "category": categories,
        "score": scores,
        "energy_condition": np.concatenate((energy_grid, energy_grid)),
        "group_id": np.asarray([f"synthetic-group-{index // 20}" for index in range(1_200)]),
        "split": np.asarray(["test"] * 1_200),
    }


def _temporary_dataset_provenance(
    tmp_path: Path,
    *,
    representation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frozen = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    counts = {
        "train": 9_600,
        "validation": 1_200,
        "test": 1_200,
        "total": 12_000,
    }
    data_root = (tmp_path / "synthetic-data").resolve()
    data_root.mkdir()
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "content_sha256": frozen["dataset"]["dataset_version"],
                "data_root": str(data_root),
                "counts": {"classification": counts},
                "inventory": {"kind": "synthetic-test-release"},
                "grouping": {"event_identity": ["category", "event_id"]},
                "input_policy": {
                    "kind": "synthetic-canonical-predictions",
                    "excluded_from_model": [
                        "E1",
                        "E2",
                        "tR",
                        "dY",
                        "dZ",
                        "theta",
                        "phiS",
                        "phiR",
                        "label",
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset_provenance(
        manifest_path=split_manifest,
        data_root=data_root,
        task="classification",
        counts=counts,
        data_config={
            "point_bin_size_mm": 15.0,
            "coordinate_scale_mm": 1_000.0,
            "max_points": 512,
        },
        input_kind="sequence",
        representation_config=representation_config,
    )


def test_strict_energybench_on_synthetic_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = _temporary_dataset_provenance(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic immutable checkpoint fixture\n")
    metadata = {
        "dataset_provenance": provenance,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": file_sha256(checkpoint),
        },
        "native_metrics": {
            "loss": 0.25,
            "auc": 0.95,
            "accuracy": 0.9,
            "token_coverage_mean": 0.999,
            "token_coverage_minimum": 0.875,
            "token_truncated_events": 4,
            "token_truncated_event_fraction": 4.0 / 1_200.0,
        },
    }
    bundle = classification_bundle(
        **_synthetic_evaluation_columns(),
        metadata=metadata,
    )

    from energybench.config import load_manifest as load_energybench_manifest

    def load_frozen_manifest_without_plots(path: Any) -> dict[str, Any]:
        config = load_energybench_manifest(path)
        config["runtime"]["make_plots"] = False
        return config

    monkeypatch.setattr(
        "energybench.config.load_manifest",
        load_frozen_manifest_without_plots,
    )
    output_dir = tmp_path / "strict-output"
    metrics = evaluate_classification_bundle(
        bundle,
        architecture_id="transformer_001_sampled_hits_coordinate_mlp",
        provenance=provenance,
        output_dir=output_dir,
    )

    report = json.loads(
        (
            output_dir
            / "test_evaluation"
            / "energybench"
            / ".energybench"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert report["quality"]["strict"] is True
    assert report["quality"]["warnings"] == []
    assert report["quality"]["errors"] == []
    assert report["classification"]["status"] == "ok"
    matched_auc = report["classification"]["aggregates"]["matched_auc_macro"]
    assert matched_auc is not None
    assert np.isfinite(matched_auc)
    assert metrics["energy_matched_auc"] == pytest.approx(matched_auc)
    test_metrics = json.loads(
        (output_dir / "test_evaluation" / "test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert test_metrics["token_coverage_mean"] == pytest.approx(0.999)
    assert test_metrics["token_coverage_minimum"] == pytest.approx(0.875)
    assert test_metrics["token_truncated_events"] == 4
    assert test_metrics["token_truncated_event_fraction"] == pytest.approx(
        4.0 / 1_200.0
    )
    assert report["input"]["path"] == str(
        (output_dir / "test_evaluation" / "test_predictions.npz").resolve()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_classification_bundle(
            bundle,
            architecture_id="transformer_001_sampled_hits_coordinate_mlp",
            provenance=provenance,
            output_dir=output_dir,
        )


def test_transformer_checkpoint_identity_rejects_mismatch(tmp_path: Path) -> None:
    architecture_id = "transformer_002_voxel_coordinate_mlp"
    architecture = ARCHITECTURES[architecture_id]
    model = build_model(architecture_id, "classification")
    assert architecture.tokenization is not None
    provenance = _temporary_dataset_provenance(
        tmp_path,
        representation_config=architecture.tokenization.to_dict(),
    )
    assert provenance["schema_version"] == 2
    assert provenance["representation_config"] == architecture.tokenization.to_dict()
    assert provenance["input_policy"]["fields"] == ["tX", "tY", "tZ", "tR"]
    assert provenance["tokenization_source"]["sha256"] == file_sha256(
        provenance["tokenization_source"]["path"]
    )
    assert provenance["evaluation_adapter"]["sha256"] == file_sha256(
        provenance["evaluation_adapter"]["path"]
    )
    checkpoint = {
        "schema_version": 2,
        "kind": "best",
        "task": "classification",
        "architecture_id": architecture_id,
        "model_name": architecture.model_name,
        "input_kind": architecture.input_kind,
        "model_config": dict(architecture.model),
        "model_source": {
            "path": str(Path(inspect.getsourcefile(model.__class__)).resolve()),
            "sha256": file_sha256(inspect.getsourcefile(model.__class__)),
        },
        "training_source": {
            "path": str(Path(inspect.getsourcefile(train_model)).resolve()),
            "sha256": file_sha256(inspect.getsourcefile(train_model)),
        },
        "training_config": architecture.training.to_dict(),
        "dataset_provenance": provenance,
        "epoch": 1,
        "score": 0.75,
        "best_score": 0.75,
        "history": [{"epoch": 1}],
        "selection_split": "validation",
        "selection_metric": "energy_matched_auc",
        "model_state_dict": model.state_dict(),
    }
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = _load_checkpoint(
        model,
        checkpoint_path,
        task="classification",
        model_id=architecture_id,
        model_config=dict(architecture.model),
        training_config=architecture.training.to_dict(),
        provenance=provenance,
    )
    assert loaded["epoch"] == 1
    assert loaded["_checkpoint_file_sha256"] == file_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="provenance"):
        _load_checkpoint(
            model,
            checkpoint_path,
            task="classification",
            model_id=architecture_id,
            model_config=dict(architecture.model),
            training_config=architecture.training.to_dict(),
            provenance={"manifest_content_sha256": "different"},
        )
    mismatched_tokenization = json.loads(json.dumps(provenance))
    mismatched_tokenization["representation_config"]["voxel_size_mm"] = 61.0
    with pytest.raises(ValueError, match="provenance"):
        _load_checkpoint(
            model,
            checkpoint_path,
            task="classification",
            model_id=architecture_id,
            model_config=dict(architecture.model),
            training_config=architecture.training.to_dict(),
            provenance=mismatched_tokenization,
        )
    mismatched_evaluation_adapter = json.loads(json.dumps(provenance))
    mismatched_evaluation_adapter["evaluation_adapter"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        _load_checkpoint(
            model,
            checkpoint_path,
            task="classification",
            model_id=architecture_id,
            model_config=dict(architecture.model),
            training_config=architecture.training.to_dict(),
            provenance=mismatched_evaluation_adapter,
        )
    with pytest.raises(ValueError, match="architecture"):
        _load_checkpoint(
            build_model("transformer_003_voxel_fourier_xyz", "classification"),
            checkpoint_path,
            task="classification",
            model_id="transformer_003_voxel_fourier_xyz",
            model_config=dict(
                ARCHITECTURES["transformer_003_voxel_fourier_xyz"].model
            ),
            training_config=ARCHITECTURES[
                "transformer_003_voxel_fourier_xyz"
            ].training.to_dict(),
            provenance=provenance,
        )
    malformed = dict(checkpoint)
    malformed["selection_metric"] = "auc"
    malformed_path = tmp_path / "malformed.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="selection metric"):
        _load_checkpoint(
            model,
            malformed_path,
            task="classification",
            model_id=architecture_id,
            model_config=dict(architecture.model),
            training_config=architecture.training.to_dict(),
            provenance=provenance,
        )


def test_epoch_boundary_resume_is_exact_and_repairs_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EpochDataset:
        epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    class Loader:
        dataset = EpochDataset()

    model = nn.Linear(1, 1)
    model.architecture_id = "resume-test"
    model.model_name = "Linear"
    model.input_kind = "sequence"
    config = TrainingConfig(
        batch_size=1,
        epochs=3,
        learning_rate=1.0e-3,
        early_stopping_patience=5,
        use_amp=False,
        device="cpu",
    )
    provenance = {
        "manifest_content_sha256": "resume-fixture",
        "evaluation_protocol": {
            "manifest_sha256": file_sha256(EVALUATION_MANIFEST),
            "evaluator_code_sha256": energybench_code_sha256(),
        },
    }
    model_config = {"in_features": 1, "out_features": 1}
    calls = 0

    def epoch_result(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            zero_loss = sum(parameter.square().sum() for parameter in args[0].parameters()) * 0.0
            zero_loss.backward()
            optimizer.step()
        target = np.asarray([0.0, 1.0], dtype=np.float32)
        prediction = np.asarray([-0.2, 0.2], dtype=np.float32)
        empty_string = np.empty(0, dtype=np.str_)
        return (
            0.2,
            {"auc": 1.0, "accuracy": 1.0, "signal_fraction": 0.5, "events": 2},
            target,
            prediction,
            empty_string,
            empty_string,
            np.asarray([1_000.0, 1_000.0]),
            empty_string,
            empty_string,
        )

    monkeypatch.setattr(training_module, "_epoch", epoch_result)
    monkeypatch.setattr(training_module, "matched_validation_auc", lambda *args, **kwargs: 0.75)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_model(
            model,
            Loader(),
            Loader(),
            "classification",
            config,
            tmp_path,
            model_config=model_config,
            provenance=provenance,
        )
    committed = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert committed["epoch"] == 1
    (tmp_path / "history.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        train_model(
            model,
            Loader(),
            Loader(),
            "classification",
            config,
            tmp_path,
            model_config=model_config,
            provenance={
                "manifest_content_sha256": "wrong",
                "evaluation_protocol": provenance["evaluation_protocol"],
            },
            resume=True,
        )

    calls = -100
    history = train_model(
        model,
        Loader(),
        Loader(),
        "classification",
        config,
        tmp_path,
        model_config=model_config,
        provenance=provenance,
        resume=True,
    )
    assert [item["epoch"] for item in history] == [1, 2, 3]
    final = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert final["epoch"] == 3
    assert final["best_epoch"] == 1
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8")) == history


def test_epoch_records_token_coverage_without_exposing_it_to_model() -> None:
    class SizedDataset:
        def __len__(self) -> int:
            return 3

    class Loader:
        dataset = SizedDataset()

        def __iter__(self):
            yield {
                "inputs": {"logit": torch.tensor([[-1.0], [1.0]])},
                "target": torch.tensor([0.0, 1.0], dtype=torch.float32),
                "event_id": ["event-0", "event-1"],
                "category": ["Bi214", "2nu"],
                "token_coverage": torch.tensor([1.0, 0.8], dtype=torch.float32),
            }
            yield {
                "inputs": {"logit": torch.tensor([[0.5]])},
                "target": torch.tensor([1.0], dtype=torch.float32),
                "event_id": ["event-2"],
                "category": ["2nu"],
                "token_coverage": torch.tensor([0.5], dtype=torch.float32),
            }

    class MappingLogit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_input_keys: list[tuple[str, ...]] = []

        def forward(self, inputs):
            self.seen_input_keys.append(tuple(sorted(inputs)))
            return inputs["logit"][:, 0]

    model = MappingLogit()
    result = training_module._epoch(
        model,
        Loader(),
        task="classification",
        device=torch.device("cpu"),
        optimizer=None,
        gradient_clip_norm=1.0,
    )
    metrics = result[1]

    assert metrics["token_coverage_mean"] == pytest.approx((1.0 + 0.8 + 0.5) / 3.0)
    assert metrics["token_coverage_minimum"] == pytest.approx(0.5)
    assert metrics["token_truncated_events"] == 2
    assert metrics["token_truncated_event_fraction"] == pytest.approx(2.0 / 3.0)
    assert model.seen_input_keys == [("logit",), ("logit",)]
