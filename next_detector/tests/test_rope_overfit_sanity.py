"""Tiny-scale sanity check: can rotary attention learn from position?

This is the cheapest proxy for "will real training plausibly work"
without touching real data or real compute budget: train for a
handful of epochs on a small, obviously position-separable synthetic
dataset on CPU, in seconds, and confirm the model actually learns.

The tokenizer centers each event's coordinates (subtracts the event's
own centroid), so any signal placed only in an event's absolute
position would be erased before it ever reaches the model. This test
instead encodes the label in each event's *shape*: label 1 events are
stretched along X, label 0 events are stretched along Y.

Per-hit energy follows the same profile (low to high, by hit index)
in both classes, so the *set* of content features is identical
between classes -- a permutation-invariant model that ignores
position cannot solve this. But energy must still vary meaningfully
from hit to hit within an event: rotary attention's output is a
softmax-weighted average of Value vectors, and a weighted average of
*identical* vectors equals that vector regardless of the weights.
With uniform per-hit energy, attention's output is mathematically
forced to ignore position entirely (independent of rope_base), no
matter how correct the rotation math is -- this is what an earlier
version of this test got wrong. Only with heterogeneous per-token
content can the position-dependent attention weights actually change
the pooled output, which is what lets the model solve this by
attending differently depending on which axis carries the spread.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

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
    set_seed,
    train_model,
)


def _write_position_separable_hdf5(
    path: Path,
    *,
    label: int,
    file_offset: int,
    events_per_file: int,
    hits_per_event: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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

    rng = np.random.default_rng(seed=file_offset)

    cursor = 0
    for event_index in range(events_per_file):
        event_id = file_offset + event_index
        for hit_index in range(hits_per_event):
            spread = 40.0 * hit_index
            jitter = rng.normal(scale=2.0, size=3)

            if label == 1:
                x, y, z = spread + jitter[0], jitter[1], jitter[2]
            else:
                x, y, z = jitter[0], spread + jitter[1], jitter[2]

            # Energy rises with hit_index, same profile in both
            # classes -- so the *set* of energy fractions per event
            # is identical between classes (no label leakage from
            # content alone), but tokens within an event are
            # meaningfully distinguishable from one another (so
            # attention's output can actually depend on which tokens
            # get weighted highest, rather than every token carrying
            # the same Value vector).
            energy_profile = 0.1 + 0.3 * (
                hit_index / (hits_per_event - 1)
            )
            hit_energy = np.float32(
                energy_profile + 0.01 * rng.normal()
            )

            rows[cursor]["index"] = cursor
            rows[cursor]["values_block_0"][0] = event_id
            rows[cursor]["values_block_1"] = (x, y, z, hit_energy)
            rows[cursor]["values_block_2"][0] = raw_label
            cursor += 1

    with h5py.File(path, "w") as handle:
        handle.create_dataset("MC/hits/table", data=rows)


def _make_position_separable_dataset(
    root: Path,
    *,
    events_per_class: int,
    hits_per_event: int,
) -> None:
    _write_position_separable_hdf5(
        root / "0nubb_part_1" / "signal.h5",
        label=1,
        file_offset=10_000,
        events_per_file=events_per_class,
        hits_per_event=hits_per_event,
    )
    _write_position_separable_hdf5(
        root / "Bi_part_1" / "background.h5",
        label=0,
        file_offset=20_000,
        events_per_file=events_per_class,
        hits_per_event=hits_per_event,
    )


class RopeOverfitSanityTests(unittest.TestCase):
    def test_rope_model_learns_a_position_only_signal(self) -> None:
        set_seed(42)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "NEXT"
            _make_position_separable_dataset(
                data_root,
                events_per_class=20,
                hits_per_event=6,
            )
            manifest_path = root / "event_split.json"

            builder = NEXTTokenBuilder(
                TokenizationConfig(
                    tokenization="sampled_hits",
                    max_tokens=6,
                    coordinate_scale=1000.0,
                    center_coordinates=True,
                    seed=42,
                )
            )
            prepared = prepare_dataset(
                data_root,
                batch_size=16,
                mode="classification",
                seed=42,
                num_workers=0,
                manifest_path=manifest_path,
                input_builder=builder,
                verbose=False,
            )

            model = NEXTTransformerClassifier(
                position_encoding="rope",
                feature_dim=2,
                d_model=24,
                nhead=4,
                num_layers=2,
                dim_feedforward=48,
                dropout=0.0,
                rope_base=10.0,
            )

            history = train_model(
                model,
                prepared.train_loader,
                prepared.validation_loader,
                config=TrainingConfig(
                    batch_size=16,
                    epochs=40,
                    learning_rate=3.0e-3,
                    early_stopping_patience=40,
                    seed=42,
                    deterministic=True,
                    use_amp=False,
                    device="cpu",
                    num_workers=0,
                ),
                task="classification",
                output_dir=root / "training",
            )

            # best_metric is validation AUC for classification (higher is
            # better). A model that genuinely uses position should nearly
            # perfectly separate this small, clearly shape-separable
            # dataset.
            self.assertGreater(history["best_metric"], 0.9)


if __name__ == "__main__":
    unittest.main()
