from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from supernemobench import DataConfig, TrainingConfig, prepare_dataset, train_model
from supernemobench.config import (
    ARCHITECTURES,
    REQUIRED_FIELDS,
    SOURCE_SPECS,
    TRANSFORMER_ARCHITECTURE_IDS,
)
from supernemobench.evaluation import assert_same_provenance, classification_bundle
from supernemobench.models import build_model, registered_models
from supernemobench.training import _restore_rng_state


def _write_source(path: Path, raw_label: str, *, events: int = 20) -> None:
    hits_per_event = 6
    event_id = np.repeat(np.arange(events, dtype=np.int64), hits_per_event)
    hit = np.tile(np.arange(hits_per_event, dtype=np.float32), events)
    event = np.repeat(np.arange(events, dtype=np.float32), hits_per_event)
    arrays: dict[str, np.ndarray] = {
        "ev_no": event_id,
        "E1": 900.0 + event,
        "E2": 1000.0 + event,
        "tX": 46.0 * hit,
        "tY": 22.0 * (hit % 3),
        "tZ": 30.0 * (hit % 2),
        "tR": 2.0 + hit,
        "dY": np.full_like(hit, 10.0),
        "dZ": np.full_like(hit, 20.0),
        "theta": np.full_like(hit, 45.0),
        "phiS": np.full_like(hit, 30.0),
        "phiR": np.full_like(hit, 35.0),
        "label": np.asarray([raw_label.encode("utf-8")] * len(hit)),
    }
    self_fields = set(arrays)
    if self_fields != set(REQUIRED_FIELDS):
        raise AssertionError((self_fields, set(REQUIRED_FIELDS)))
    with h5py.File(path, "w") as handle:
        for name, values in arrays.items():
            handle.create_dataset(name, data=values)


def _make_data_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for spec in SOURCE_SPECS:
        _write_source(root / spec.file_name, spec.raw_label)


def _loader_signature(loader: object) -> list[tuple[str, float, float]]:
    signature = []
    for batch in loader:
        signature.extend(
            zip(
                [str(value) for value in batch["event_id"]],
                [float(value) for value in batch["target"].numpy()],
                [float(value) for value in batch["energy"].numpy()],
            )
        )
    return signature


class SuperNEMOBenchIntegrationTests(unittest.TestCase):
    def test_frozen_transformer_training_protocol(self) -> None:
        for identifier in TRANSFORMER_ARCHITECTURE_IDS:
            training = ARCHITECTURES[identifier].training
            self.assertEqual(training.batch_size, 64)
            self.assertEqual(training.epochs, 50)
            self.assertEqual(training.learning_rate, 5.0e-4)
            self.assertEqual(training.weight_decay, 1.0e-4)
            self.assertEqual(training.gradient_clip_norm, 1.0)
            self.assertEqual(training.early_stopping_patience, 5)
            self.assertEqual(training.num_workers, 8)
            self.assertEqual(training.seed, 42)
            self.assertTrue(training.use_amp)

    def test_rng_restore_converts_saved_cuda_states_to_cpu_bytes(self) -> None:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().to(dtype=torch.int64),
            "torch_cuda": [torch.arange(32, dtype=torch.int64)],
        }
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.set_rng_state_all"
        ) as restore_cuda:
            _restore_rng_state(state)
        restored = restore_cuda.call_args.args[0]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].device.type, "cpu")
        self.assertEqual(restored[0].dtype, torch.uint8)
        self.assertTrue(restored[0].is_contiguous())

    def test_registry_exposes_six_local_transformers_without_wing_architectures(self) -> None:
        self.assertTrue(set(TRANSFORMER_ARCHITECTURE_IDS).issubset(registered_models()))
        for identifier in TRANSFORMER_ARCHITECTURE_IDS:
            model = build_model(identifier, "classification")
            self.assertEqual(model.architecture_id, identifier)
            self.assertEqual(model.input_kind, "sequence")
            self.assertEqual(model.task, "classification")

    def test_six_representations_reuse_exact_manifest_and_event_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            manifest = root / "manifest" / "split_manifest.json"
            _make_data_root(data_root)
            config = DataConfig(
                data_root=data_root,
                manifest_path=manifest,
                split_block_events=2,
                scan_chunk_rows=31,
            )
            prepared = []
            signatures = []
            for identifier in TRANSFORMER_ARCHITECTURE_IDS:
                architecture = ARCHITECTURES[identifier]
                item = prepare_dataset(
                    task="classification",
                    input_kind="sequence",
                    data_config=config,
                    tokenization_config=architecture.tokenization,
                    batch_size=4,
                    num_workers=0,
                )
                prepared.append(item)
                split_signatures = {
                    "train": _loader_signature(item.train_loader),
                    "validation": _loader_signature(item.validation_loader),
                    "test": _loader_signature(item.test_loader),
                }
                signatures.append(split_signatures)
                batch = next(iter(item.train_loader))
                self.assertEqual(batch["inputs"]["mask"].dtype, torch.bool)
                all_categories = {
                    "2nu" if target == 1.0 else "Bi214"
                    for _, target, _ in split_signatures["train"]
                }
                self.assertEqual(all_categories, {"2nu", "Bi214"})
                for values in split_signatures.values():
                    for event_id, _, energy in values:
                        event_number = int(event_id.rsplit("::", 1)[1])
                        self.assertEqual(energy, 1900.0 + 2.0 * event_number)
            reference = prepared[0]
            for index, item in enumerate(prepared[1:], start=1):
                self.assertEqual(item.counts, reference.counts)
                self.assertEqual(item.manifest_path, reference.manifest_path)
                self.assertEqual(signatures[index], signatures[0])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for split in ("train", "validation", "test"):
                classification = payload["counts"]["classification"][split]
                self.assertGreater(classification["by_category"]["2nu"], 0)
                self.assertGreater(classification["by_category"]["Bi214"], 0)

    def test_multiworker_loader_covers_every_event_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            manifest = root / "manifest" / "split_manifest.json"
            _make_data_root(data_root)
            architecture = ARCHITECTURES[TRANSFORMER_ARCHITECTURE_IDS[0]]
            config = DataConfig(
                data_root=data_root,
                manifest_path=manifest,
                split_block_events=2,
                scan_chunk_rows=31,
            )
            single = prepare_dataset(
                task="classification",
                input_kind="sequence",
                data_config=config,
                tokenization_config=architecture.tokenization,
                batch_size=4,
                num_workers=0,
            )
            multiple = prepare_dataset(
                task="classification",
                input_kind="sequence",
                data_config=config,
                tokenization_config=architecture.tokenization,
                batch_size=4,
                num_workers=2,
            )
            for split in ("train", "validation", "test"):
                single_values = _loader_signature(getattr(single, f"{split}_loader"))
                multiple_values = _loader_signature(
                    getattr(multiple, f"{split}_loader")
                )
                self.assertEqual(len(multiple_values), len(single_values))
                self.assertEqual(sorted(multiple_values), sorted(single_values))
                self.assertEqual(
                    len({event_id for event_id, _, _ in multiple_values}),
                    len(multiple_values),
                )

    def test_one_epoch_uses_wing_train_model_and_writes_recoverable_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            _make_data_root(data_root)
            architecture = ARCHITECTURES[TRANSFORMER_ARCHITECTURE_IDS[0]]
            prepared = prepare_dataset(
                task="classification",
                input_kind="sequence",
                data_config=DataConfig(
                    data_root=data_root,
                    manifest_path=root / "manifest" / "split_manifest.json",
                    split_block_events=2,
                    scan_chunk_rows=31,
                ),
                tokenization_config=architecture.tokenization,
                batch_size=8,
                num_workers=0,
            )
            model = build_model(TRANSFORMER_ARCHITECTURE_IDS[0], "classification")
            training = TrainingConfig(
                batch_size=8,
                epochs=1,
                learning_rate=5.0e-4,
                early_stopping_patience=1,
                seed=42,
                num_workers=0,
                device="cpu",
                deterministic=True,
                use_amp=False,
            )
            provenance = {
                "evaluation_protocol": {
                    "manifest_sha256": "0" * 64,
                    "evaluator_code_sha256": "1" * 64,
                }
            }
            output = root / "outputs"
            with patch(
                "supernemobench.training.matched_validation_auc", return_value=0.5
            ):
                history = train_model(
                    model,
                    prepared.train_loader,
                    prepared.validation_loader,
                    task="classification",
                    config=training,
                    output_dir=output,
                    model_config=dict(architecture.model),
                    provenance=provenance,
                )
            self.assertEqual(len(history), 1)
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            self.assertTrue((output / "history.json").is_file())
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["selection_split"], "validation")
            self.assertEqual(checkpoint["selection_metric"], "energy_matched_auc")
            self.assertEqual(checkpoint["score"], 0.5)

            changed_provenance = dict(provenance)
            changed_provenance["evaluation_protocol"] = {
                "manifest_sha256": "2" * 64,
                "evaluator_code_sha256": "1" * 64,
            }
            with self.assertRaises(ValueError):
                train_model(
                    build_model(TRANSFORMER_ARCHITECTURE_IDS[0], "classification"),
                    prepared.train_loader,
                    prepared.validation_loader,
                    task="classification",
                    config=training,
                    output_dir=output,
                    model_config=dict(architecture.model),
                    provenance=changed_provenance,
                    resume=True,
                )

    def test_provenance_mismatch_is_rejected(self) -> None:
        expected = {"manifest_content_sha256": "a" * 64}
        assert_same_provenance(expected, expected, context="test")
        with self.assertRaises(ValueError):
            assert_same_provenance(
                {"manifest_content_sha256": "b" * 64}, expected, context="test"
            )

    def test_prediction_ordering_is_rejected(self) -> None:
        class FakeBundle:
            def __init__(self, columns: dict[str, np.ndarray], metadata: dict) -> None:
                self.columns = columns
                self.metadata = metadata

        package = types.ModuleType("energybench")
        data_module = types.ModuleType("energybench.data")
        data_module.PredictionBundle = FakeBundle
        event_ids = np.asarray(["SuperNEMO::2nubb::2", "SuperNEMO::Bi214::4"])
        digest = hashlib.sha256()
        for event_id in event_ids:
            digest.update(f"{event_id}\n".encode("utf-8"))
        metadata = {
            "dataset_provenance": {
                "prediction_order_sha256": {"test": digest.hexdigest()}
            }
        }
        arguments = {
            "event_id": event_ids,
            "label": np.asarray([1, 0]),
            "category": np.asarray(["2nu", "Bi214"]),
            "score": np.asarray([0.8, -0.2]),
            "energy_condition": np.asarray([2000.0, 2200.0]),
            "group_id": event_ids,
            "split": np.asarray(["test", "test"]),
            "metadata": metadata,
        }
        with patch.dict(
            sys.modules,
            {"energybench": package, "energybench.data": data_module},
        ), patch("supernemobench.evaluation._ensure_energybench_api"):
            classification_bundle(**arguments)
            reordered = dict(arguments)
            reordered["event_id"] = event_ids[::-1]
            with self.assertRaises(ValueError):
                classification_bundle(**reordered)


if __name__ == "__main__":
    unittest.main()
