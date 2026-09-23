"""Synthetic equivalence tests for the NEXT token cache."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from simple_energybench import prepare_dataset
from next_transformer import (
    NEXTTokenBuilder,
    NEXTTransformerClassifier,
    TokenizationConfig,
    build_token_cache,
    prepare_cached_dataset,
    validate_token_cache,
)
from next_transformer.cache import _CachedBatchCollator


def _write_hdf5(
    path: Path,
    *,
    label: int,
    event_offset: int,
    hit_counts: tuple[int, ...],
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
    rows = np.zeros(sum(hit_counts), dtype=dtype)
    raw_label = b"Signal" if label == 1 else b"Bkg"
    cursor = 0
    for event_index, number_of_hits in enumerate(hit_counts):
        event_id = event_offset + event_index * 7
        # Keep the total energy in the physical NEXT range while making each
        # hit and event deterministic and distinguishable.
        total_energy = np.float32(2.44 + 0.001 * event_index + 0.0002 * label)
        hit_energy = np.float32(total_energy / number_of_hits)
        for hit_index in range(number_of_hits):
            rows[cursor]["index"] = cursor
            rows[cursor]["values_block_0"][0] = event_id
            rows[cursor]["values_block_1"] = (
                20.0 * hit_index + 0.5 * event_index,
                -11.0 * hit_index + float(label),
                7.0 * hit_index - float(event_index),
                hit_energy,
            )
            rows[cursor]["values_block_2"][0] = raw_label
            cursor += 1
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("MC/hits/table", data=rows)
        dataset.attrs["values_block_0_kind"] = np.bytes_(
            "(lp0\nVevent_id\np1\na."
        )
        dataset.attrs["values_block_1_kind"] = np.bytes_(
            "(lp0\nVx\np1\naVy\np2\naVz\np3\naVenergy\np4\na."
        )
        dataset.attrs["values_block_2_kind"] = np.bytes_(
            "(lp0\nVlabel\np1\na."
        )


def _make_dataset(root: Path) -> None:
    # Multiple differently sized files make the event-count split create
    # boundary slices.  Events cover fewer than, equal to, and more than the
    # tests' four-token ceiling.
    patterns = ((2, 4, 7), (3, 6, 2, 5), (8, 4, 3, 6, 2))
    for file_index, hit_counts in enumerate(patterns):
        _write_hdf5(
            root / "0nubb_part_1" / f"signal_{file_index}.h5",
            label=1,
            event_offset=10_000 + file_index * 1_000,
            hit_counts=hit_counts,
        )
        _write_hdf5(
            root / "Bi_part_1" / f"background_{file_index}.h5",
            label=0,
            event_offset=20_000 + file_index * 1_000,
            hit_counts=tuple(reversed(hit_counts)),
        )


def _events(loader) -> list[dict]:
    rows: list[dict] = []
    for batch in loader:
        batch_size = len(batch["event_id"])
        for index in range(batch_size):
            rows.append(
                {
                    "inputs": {
                        key: value[index].detach().cpu().numpy().copy()
                        for key, value in batch["inputs"].items()
                    },
                    "label": float(batch["label"][index]),
                    "energy": float(batch["energy"][index]),
                    "event_id": str(batch["event_id"][index]),
                    "category": str(batch["category"][index]),
                    "group_id": str(batch["group_id"][index]),
                    "split": str(batch["split"][index]),
                    "sample_weight": float(batch["sample_weight"][index]),
                    "projection_coverage": float(
                        batch["projection_coverage"][index]
                    ),
                }
            )
    return rows


def _collator_sample(valid_tokens: int, *, max_tokens: int = 4) -> dict:
    coords = np.zeros((max_tokens, 3), dtype=np.float32)
    features = np.zeros((max_tokens, 2), dtype=np.float32)
    mask = np.zeros(max_tokens, dtype=np.bool_)
    coords[:valid_tokens] = np.arange(valid_tokens * 3, dtype=np.float32).reshape(
        valid_tokens, 3
    )
    features[:valid_tokens] = np.arange(
        valid_tokens * 2, dtype=np.float32
    ).reshape(valid_tokens, 2)
    mask[:valid_tokens] = True
    return {
        "inputs": {"coords": coords, "features": features, "mask": mask},
        "label": np.float32(valid_tokens % 2),
        "energy": np.float64(2.45),
        "event_id": f"NEXT::synthetic::{valid_tokens}",
        "category": "0nubb" if valid_tokens % 2 else "Bi214",
        "group_id": "synthetic.h5",
        "split": "train",
        "sample_weight": np.float32(1.0),
        "projection_coverage": np.float32(1.0),
    }


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.cache_root = self.root / "cache"
        self.manifest_path = self.root / "event_split.json"
        _make_dataset(self.data_root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, tokenization: str) -> TokenizationConfig:
        return TokenizationConfig(
            tokenization=tokenization,
            max_tokens=4,
            voxel_size=5.0,
            coordinate_scale=1000.0,
            center_coordinates=True,
            voxel_truncation="occupancy",
            seed=42,
        )

    def _prepare_raw(self, config: TokenizationConfig):
        return prepare_dataset(
            self.data_root,
            batch_size=3,
            mode="classification",
            split_fractions=(0.8, 0.1, 0.1),
            seed=42,
            num_workers=0,
            manifest_path=self.manifest_path,
            verbose=False,
            input_builder=NEXTTokenBuilder(config),
        )

    def _build(self, config: TokenizationConfig, workers: int = 0):
        self._prepare_raw(config)
        return build_token_cache(
            self.data_root,
            self.manifest_path,
            self.cache_root,
            config,
            num_workers=workers,
        )

    def assert_events_equal(self, raw: list[dict], cached: list[dict]) -> None:
        self.assertEqual([row["event_id"] for row in raw], [row["event_id"] for row in cached])
        for expected, actual in zip(raw, cached):
            for key in (
                "label",
                "energy",
                "event_id",
                "category",
                "group_id",
                "split",
                "sample_weight",
                "projection_coverage",
            ):
                self.assertEqual(expected[key], actual[key], (key, expected, actual))
            for key in ("coords", "features", "mask"):
                np.testing.assert_array_equal(
                    expected["inputs"][key], actual["inputs"][key]
                )

    def test_raw_and_cached_events_match_for_both_tokenizers(self) -> None:
        for tokenization in ("sampled_hits", "voxel"):
            with self.subTest(tokenization=tokenization):
                config = self._config(tokenization)
                raw = self._prepare_raw(config)
                cache_dir = build_token_cache(
                    self.data_root,
                    self.manifest_path,
                    self.cache_root,
                    config,
                    num_workers=0,
                )
                report = validate_token_cache(
                    cache_dir, self.data_root, self.manifest_path, config
                )
                cached = prepare_cached_dataset(
                    cache_dir,
                    batch_size=3,
                    num_workers=0,
                    seed=42,
                    pin_memory=False,
                )
                self.assertEqual(raw.counts, cached.counts)
                self.assertEqual(report["counts"], cached.counts)
                for split in ("train", "validation", "test"):
                    self.assert_events_equal(
                        _events(getattr(raw, f"{split}_loader")),
                        _events(getattr(cached, f"{split}_loader")),
                    )

    def test_training_epoch_slice_order_matches_raw_loader(self) -> None:
        config = self._config("sampled_hits")
        raw = self._prepare_raw(config)
        cache_dir = build_token_cache(
            self.data_root,
            self.manifest_path,
            self.cache_root,
            config,
            num_workers=0,
        )
        cached = prepare_cached_dataset(
            cache_dir,
            batch_size=2,
            num_workers=0,
            seed=42,
            pin_memory=False,
        )
        orders: list[list[str]] = []
        for epoch in (0, 1):
            raw.train_loader.dataset.set_epoch(epoch)
            cached.train_loader.dataset.set_epoch(epoch)
            raw_ids = [row["event_id"] for row in _events(raw.train_loader)]
            cached_ids = [row["event_id"] for row in _events(cached.train_loader)]
            self.assertEqual(raw_ids, cached_ids)
            orders.append(raw_ids)
        self.assertNotEqual(orders[0], orders[1])

    def test_cached_batch_produces_identical_transformer_logits(self) -> None:
        config = self._config("sampled_hits")
        raw = self._prepare_raw(config)
        cache_dir = build_token_cache(
            self.data_root,
            self.manifest_path,
            self.cache_root,
            config,
            num_workers=0,
        )
        cached = prepare_cached_dataset(
            cache_dir,
            batch_size=3,
            num_workers=0,
            seed=42,
            pin_memory=False,
        )
        raw_batch = next(iter(raw.validation_loader))["inputs"]
        cached_batch = next(iter(cached.validation_loader))["inputs"]
        model = NEXTTransformerClassifier(
            position_encoding="coordinate_mlp",
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        ).eval()
        with torch.inference_mode():
            raw_logits = model(raw_batch)
            cached_logits = model(cached_batch)
        torch.testing.assert_close(raw_logits, cached_logits, rtol=0.0, atol=0.0)

    def test_trimmed_voxel_batch_preserves_tokens_and_logits(self) -> None:
        samples = [_collator_sample(2), _collator_sample(3)]
        untrimmed = _CachedBatchCollator()(samples)
        trimmed = _CachedBatchCollator(trim_padding=True)(samples)

        self.assertEqual(tuple(untrimmed["inputs"]["mask"].shape), (2, 4))
        self.assertEqual(tuple(trimmed["inputs"]["mask"].shape), (2, 3))
        for name in ("coords", "features", "mask"):
            self.assertTrue(trimmed["inputs"][name].is_contiguous())
            torch.testing.assert_close(
                trimmed["inputs"][name],
                untrimmed["inputs"][name][:, :3],
                rtol=0.0,
                atol=0.0,
            )

        model = NEXTTransformerClassifier(
            position_encoding="coordinate_mlp",
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        ).eval()
        with torch.inference_mode():
            untrimmed_logits = model(untrimmed["inputs"])
            trimmed_logits = model(trimmed["inputs"])
        torch.testing.assert_close(
            untrimmed_logits,
            trimmed_logits,
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_compact_training_batches_keep_test_metadata(self) -> None:
        config = self._config("voxel")
        cache_dir = self._build(config)
        cached = prepare_cached_dataset(
            cache_dir,
            batch_size=2,
            num_workers=0,
            seed=42,
            pin_memory=False,
            trim_padding=True,
            compact_training_batches=True,
        )

        training_keys = {"inputs", "label", "sample_weight"}
        self.assertEqual(set(next(iter(cached.train_loader))), training_keys)
        self.assertEqual(set(next(iter(cached.validation_loader))), training_keys)

        test_batch = next(iter(cached.test_loader))
        self.assertEqual(
            set(test_batch),
            {
                "inputs",
                "label",
                "energy",
                "event_id",
                "category",
                "group_id",
                "split",
                "sample_weight",
                "projection_coverage",
            },
        )
        self.assertLessEqual(test_batch["inputs"]["mask"].shape[1], 4)

    def test_sampled_hits_remain_at_configured_token_ceiling(self) -> None:
        config = self._config("sampled_hits")
        cache_dir = self._build(config)
        cached = prepare_cached_dataset(
            cache_dir,
            batch_size=2,
            num_workers=0,
            seed=42,
            pin_memory=False,
            trim_padding=False,
            compact_training_batches=True,
        )
        batch = next(iter(cached.train_loader))
        self.assertEqual(tuple(batch["inputs"]["coords"].shape[1:]), (4, 3))
        self.assertEqual(tuple(batch["inputs"]["features"].shape[1:]), (4, 2))
        self.assertEqual(batch["inputs"]["mask"].shape[1], 4)

    def test_parallel_builder_and_multiworker_loader_cover_all_events(self) -> None:
        config = self._config("voxel")
        raw = self._prepare_raw(config)
        cache_dir = build_token_cache(
            self.data_root,
            self.manifest_path,
            self.cache_root,
            config,
            num_workers=2,
        )
        cached = prepare_cached_dataset(
            cache_dir,
            batch_size=2,
            num_workers=2,
            seed=42,
            pin_memory=False,
        )
        raw_ids = {
            row["event_id"]
            for split in ("train", "validation", "test")
            for row in _events(getattr(raw, f"{split}_loader"))
        }
        cached_ids = {
            row["event_id"]
            for split in ("train", "validation", "test")
            for row in _events(getattr(cached, f"{split}_loader"))
        }
        self.assertEqual(raw_ids, cached_ids)
        self.assertEqual(len(cached_ids), raw.counts["total"])

    def test_optimized_multiworker_loaders_cover_every_event(self) -> None:
        config = self._config("voxel")
        cache_dir = self._build(config, workers=2)
        cached = prepare_cached_dataset(
            cache_dir,
            batch_size=2,
            num_workers=2,
            seed=42,
            pin_memory=False,
            trim_padding=True,
            compact_training_batches=True,
        )
        train_events = sum(len(batch["label"]) for batch in cached.train_loader)
        validation_events = sum(
            len(batch["label"]) for batch in cached.validation_loader
        )
        test_ids = {
            event_id
            for batch in cached.test_loader
            for event_id in batch["event_id"]
        }
        self.assertEqual(train_events, cached.counts["train"])
        self.assertEqual(validation_events, cached.counts["validation"])
        self.assertEqual(len(test_ids), cached.counts["test"])

    def test_cache_rejects_missing_success_and_changed_configuration(self) -> None:
        config = self._config("sampled_hits")
        cache_dir = self._build(config)
        marker = cache_dir / "_SUCCESS"
        marker.unlink()
        with self.assertRaisesRegex(ValueError, "_SUCCESS"):
            prepare_cached_dataset(cache_dir, num_workers=0, pin_memory=False)
        marker.write_text("complete\n", encoding="utf-8")
        changed = TokenizationConfig(
            **{**config.to_dict(), "coordinate_scale": 500.0}
        )
        with self.assertRaisesRegex(ValueError, "mismatch|settings"):
            validate_token_cache(
                cache_dir, self.data_root, self.manifest_path, changed
            )

    def test_completed_cache_build_is_idempotent(self) -> None:
        config = self._config("sampled_hits")
        first = self._build(config)
        second = build_token_cache(
            self.data_root,
            self.manifest_path,
            self.cache_root,
            config,
            num_workers=0,
        )
        self.assertEqual(first, second)

    def test_builder_rejects_source_file_changed_after_split(self) -> None:
        config = self._config("sampled_hits")
        self._prepare_raw(config)
        source = next((self.data_root / "0nubb_part_1").glob("*.h5"))
        with source.open("ab") as file:
            file.write(b"changed-after-manifest")
        with self.assertRaisesRegex(ValueError, "changed after split"):
            build_token_cache(
                self.data_root,
                self.manifest_path,
                self.cache_root,
                config,
                num_workers=0,
            )

    def test_incomplete_build_resumes_missing_slice(self) -> None:
        config = self._config("sampled_hits")
        completed = self._build(config)
        building = completed.with_name(completed.name + ".building")
        completed.replace(building)
        (building / "_SUCCESS").unlink()
        (building / "cache_manifest.json").unlink()

        state_path = building / "build_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        removed_key = state["completed"].pop()
        build_spec = json.loads(
            (building / "build_spec.json").read_text(encoding="utf-8")
        )
        split, slice_index = removed_key.split(":")
        removed_events = int(
            build_spec["slices"][split][int(slice_index)]["event_count"]
        )
        state["completed_events"] -= removed_events
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        resumed = build_token_cache(
            self.data_root,
            self.manifest_path,
            self.cache_root,
            config,
            num_workers=0,
            resume=True,
        )
        self.assertEqual(resumed, completed)
        report = validate_token_cache(
            resumed, self.data_root, self.manifest_path, config
        )
        self.assertEqual(report["counts"]["total"], 24)


if __name__ == "__main__":
    unittest.main()
