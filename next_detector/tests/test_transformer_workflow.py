"""Synthetic end-to-end checks for EnergyBench plus the NEXT Transformer."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (
    PROJECT_ROOT / "evalutaions_workflow",
    PROJECT_ROOT / "next_detector",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from next_transformer import (  # noqa: E402
    NEXTTokenBuilder,
    NEXTTransformerClassifier,
    TokenizationConfig,
)
from simple_energybench import (  # noqa: E402
    TrainingConfig,
    prepare_dataset,
    train_model,
)


def _write_hdf5(path: Path, *, label: int, file_offset: int) -> None:
    """Write a small file using the real `/MC/hits/table` field layout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    events_per_file = 10
    hits_per_event = 6
    dtype = np.dtype(
        [
            ("index", "<i8"),
            ("values_block_0", "<i8", (1,)),
            ("values_block_1", "<f4", (4,)),
            ("values_block_2", "S6", (1,)),
        ]
    )
    rows = np.zeros(events_per_file * hits_per_event, dtype=dtype)
    raw_label = b"Signal" if label == 1 else b"Bkg"
    cursor = 0
    for event_index in range(events_per_file):
        event_id = file_offset + event_index
        hit_energy = np.float32(0.405 + 0.0004 * event_index + 0.0001 * label)
        for hit_index in range(hits_per_event):
            rows[cursor]["index"] = cursor
            rows[cursor]["values_block_0"][0] = event_id
            rows[cursor]["values_block_1"] = (
                20.0 * hit_index + label,
                10.0 * event_index + hit_index,
                5.0 * label + 2.0 * hit_index,
                hit_energy,
            )
            rows[cursor]["values_block_2"][0] = raw_label
            cursor += 1

    with h5py.File(path, "w") as handle:
        handle.create_dataset("MC/hits/table", data=rows)


def _make_dataset(root: Path) -> None:
    for file_index in range(4):
        _write_hdf5(
            root / "0nubb_part_1" / f"signal_{file_index}.h5",
            label=1,
            file_offset=10_000 + 100 * file_index,
        )
        _write_hdf5(
            root / "Bi_part_1" / f"background_{file_index}.h5",
            label=0,
            file_offset=20_000 + 100 * file_index,
        )


class TransformerWorkflowTests(unittest.TestCase):
    def test_raw_loader_forward_passes_and_training_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "NEXT"
            _make_dataset(data_root)
            manifest_path = root / "event_split.json"

            prepared_by_tokenization = {}
            for tokenization in ("sampled_hits", "voxel"):
                builder = NEXTTokenBuilder(
                    TokenizationConfig(
                        tokenization=tokenization,
                        max_tokens=8,
                        voxel_size=15.0,
                        coordinate_scale=1000.0,
                        center_coordinates=True,
                        seed=42,
                    )
                )
                prepared = prepare_dataset(
                    data_root,
                    batch_size=8,
                    mode="classification",
                    seed=42,
                    num_workers=0,
                    manifest_path=manifest_path,
                    input_builder=builder,
                    verbose=False,
                )
                prepared_by_tokenization[tokenization] = prepared
                batch = next(iter(prepared.train_loader))
                self.assertEqual(tuple(batch["inputs"]["coords"].shape[1:]), (8, 3))
                self.assertEqual(tuple(batch["inputs"]["features"].shape[1:]), (8, 2))
                self.assertEqual(tuple(batch["inputs"]["mask"].shape[1:]), (8,))

                for position_encoding in (
                    "coordinate_mlp",
                    "fourier_xyz",
                    "rope",
                ):
                    # RoPE needs head_dim = d_model // nhead >= 6, so it
                    # cannot share the other encodings' d_model=16
                    # (head_dim=4) tiny config.
                    if position_encoding == "rope":
                        model = NEXTTransformerClassifier(
                            position_encoding=position_encoding,
                            feature_dim=2,
                            d_model=24,
                            nhead=4,
                            num_layers=1,
                            dim_feedforward=32,
                            dropout=0.0,
                            rope_base=10.0,
                        )
                    else:
                        model = NEXTTransformerClassifier(
                            position_encoding=position_encoding,
                            feature_dim=2,
                            d_model=16,
                            nhead=4,
                            num_layers=1,
                            dim_feedforward=32,
                            dropout=0.0,
                            num_frequencies=2,
                        )
                    logits = model(batch["inputs"])
                    self.assertEqual(tuple(logits.shape), (len(batch["label"]),))
                    self.assertTrue(torch.isfinite(logits).all())

            self.assertEqual(
                prepared_by_tokenization["sampled_hits"].counts,
                prepared_by_tokenization["voxel"].counts,
            )

            training_data = prepared_by_tokenization["sampled_hits"]
            training_model = NEXTTransformerClassifier(
                position_encoding="coordinate_mlp",
                feature_dim=2,
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
                num_frequencies=2,
            )
            history = train_model(
                training_model,
                training_data.train_loader,
                training_data.validation_loader,
                config=TrainingConfig(
                    batch_size=8,
                    epochs=1,
                    learning_rate=5.0e-4,
                    early_stopping_patience=1,
                    seed=42,
                    deterministic=True,
                    use_amp=False,
                    device="cpu",
                    num_workers=0,
                ),
                task="classification",
                output_dir=root / "training",
            )
            self.assertEqual(history["epochs_completed"], 1)
            self.assertEqual(history["best_epoch"], 1)
            self.assertTrue(np.isfinite(history["best_metric"]))
            self.assertTrue((root / "training" / "best_model.pt").is_file())
            self.assertTrue((root / "training" / "last_model.pt").is_file())
            self.assertTrue((root / "training" / "history.json").is_file())

            # Rotary attention replaces the built-in TransformerEncoder
            # with a hand-rolled one, so its own optimizer/backward pass
            # needs a dedicated training smoke check.
            rope_model = NEXTTransformerClassifier(
                position_encoding="rope",
                feature_dim=2,
                d_model=24,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
                rope_base=10.0,
            )
            rope_history = train_model(
                rope_model,
                training_data.train_loader,
                training_data.validation_loader,
                config=TrainingConfig(
                    batch_size=8,
                    epochs=1,
                    learning_rate=5.0e-4,
                    early_stopping_patience=1,
                    seed=42,
                    deterministic=True,
                    use_amp=False,
                    device="cpu",
                    num_workers=0,
                ),
                task="classification",
                output_dir=root / "training_rope",
            )
            self.assertEqual(rope_history["epochs_completed"], 1)
            self.assertEqual(rope_history["best_epoch"], 1)
            self.assertTrue(np.isfinite(rope_history["best_metric"]))
            self.assertTrue(
                (root / "training_rope" / "best_model.pt").is_file()
            )
            self.assertTrue(
                (root / "training_rope" / "last_model.pt").is_file()
            )
            self.assertTrue(
                (root / "training_rope" / "history.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
