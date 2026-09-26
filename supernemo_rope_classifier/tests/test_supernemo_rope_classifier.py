"""Synthetic-data checks for the SuperNEMO RoPE classification benchmark.

Covers the published-split reproduction, the event index and streaming
dataset (against fabricated files in the real SuperNEMO column layout), the
model wiring for all three position encodings, and the end-to-end
train/evaluate contract with the unchanged EnergyBench. CPU-only, no real data.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _source_root in (
    PROJECT_ROOT / "evalutaions_workflow",
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "supernemo_rope_classifier",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from simple_energybench import (  # noqa: E402
    EvaluationConfig,
    TrainingConfig,
    evaluate_classification,
    train_model,
)

from supernemo_rope_transformer import (  # noqa: E402
    DEFAULT_SUPERNEMO_ROPE_BASE,
    SuperNEMOTrackerTokenizationConfig,
    build_supernemo_transformer,
    rope_frequency_table,
    coarsest_unambiguous_extents,
    tokenize_tracker_event,
)
from supernemorope_bench import (  # noqa: E402
    DataConfig,
    EnergyWindowLoader,
    SuperNEMOEventDataset,
    block_assignments,
    build_split,
    collate_events,
    prepare_dataset,
    scan_event_offsets,
    split_counts,
    verify_published_counts,
)
from supernemorope_bench.config import SOURCE_SPECS  # noqa: E402

SCRIPTS_DIR = PROJECT_ROOT / "supernemo_rope_classifier" / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "run_supernemo_rope_classification.py"

TOKENIZATION_KWARGS = {
    "sampled_hits": {"max_tokens": 16},
    "voxel": {"max_tokens": 16, "voxel_size_mm": 60.0},
    "summary_features": {"max_tokens": 6},
}
EXPECTED_FEATURE_DIM = {"sampled_hits": 4, "voxel": 4, "summary_features": 6}

# head_dim = 24 // 4 = 6 is the smallest legal RoPE head dimension.
TINY_MODEL = {"d_model": 24, "nhead": 4, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0}


def _tokenization(name: str) -> SuperNEMOTrackerTokenizationConfig:
    return SuperNEMOTrackerTokenizationConfig(tokenization=name, **TOKENIZATION_KWARGS[name])


def _write_source(
    path: Path,
    *,
    label: str,
    n_events: int,
    energy_kev: tuple[float, float],
    seed: int,
    hits_range: tuple[int, int] = (6, 14),
    corrupt_first_energy_row: bool = False,
) -> dict[str, np.ndarray]:
    """Write a file in the real SuperNEMO layout; return the arrays for checks.

    ``energy_kev`` bounds the event's total ``E1 + E2``; every event's energy is
    constant across its hit rows, and roughly a tenth of ``tR`` values are NaN.
    """

    rng = np.random.default_rng(seed)
    hits = rng.integers(hits_range[0], hits_range[1], size=n_events)
    rows = int(hits.sum())
    ev_no = np.repeat(np.arange(n_events, dtype=np.int64), hits)
    total = rng.uniform(energy_kev[0], energy_kev[1], size=n_events)
    fraction = rng.uniform(0.3, 0.7, size=n_events)
    per_event = {
        "E1": (total * fraction).astype(np.float32),
        "E2": (total * (1.0 - fraction)).astype(np.float32),
        "dY": rng.normal(0, 100, n_events).astype(np.float32),
        "dZ": rng.normal(0, 100, n_events).astype(np.float32),
        "theta": rng.uniform(0, 180, n_events).astype(np.float32),
        "phiS": rng.uniform(0, 180, n_events).astype(np.float32),
        "phiR": rng.uniform(0, 180, n_events).astype(np.float32),
    }
    columns = {name: np.repeat(values, hits) for name, values in per_event.items()}
    columns["tX"] = rng.uniform(-405, 405, rows).astype(np.float32)
    columns["tY"] = rng.uniform(-2400, 2400, rows).astype(np.float32)
    columns["tZ"] = rng.uniform(-1400, 1400, rows).astype(np.float32)
    radius = rng.uniform(0.0, 23.5, rows).astype(np.float32)
    radius[rng.random(rows) < 0.1] = np.nan
    columns["tR"] = radius
    if corrupt_first_energy_row:
        columns["E1"][1] += np.float32(50.0)  # breaks "constant within event 0"

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ev_no", data=ev_no)
        for name, values in columns.items():
            handle.create_dataset(name, data=values)
        handle.create_dataset(
            "label", data=np.array([label] * rows, dtype=object), dtype=h5py.string_dtype()
        )
    return {"ev_no": ev_no, "hits": hits, **columns}


def _make_dataset(
    root: Path,
    *,
    n_events: int = 1600,
    two_nu_energy: tuple[float, float] = (1500.0, 1600.0),
    bi_energy: tuple[float, float] = (1500.0, 1600.0),
    hits_range: tuple[int, int] = (6, 14),
) -> dict[str, dict[str, np.ndarray]]:
    """Fabricate both classified sources; the two classes share an energy window."""

    return {
        "2nubb": _write_source(
            root / "data_2nubb_merged.h5",
            label="2nubb",
            n_events=n_events,
            energy_kev=two_nu_energy,
            seed=1,
            hits_range=hits_range,
        ),
        "Bi214": _write_source(
            root / "data_Bi214_merged.h5",
            label="Bi214",
            n_events=n_events,
            energy_kev=bi_energy,
            seed=2,
            hits_range=hits_range,
        ),
    }


def _small_config(root: Path, **overrides) -> DataConfig:
    return DataConfig(data_root=root, split_block_events=50, **overrides)


def _all_events(loader) -> list[dict]:
    """Flatten a loader into per-event records of its collated batches."""

    events = []
    for batch in loader:
        for index, event_id in enumerate(batch["event_id"]):
            events.append(
                {
                    "event_id": event_id,
                    "category": batch["category"][index],
                    "label": float(batch["label"][index]),
                    "energy": float(batch["energy"][index]),
                    "n_tokens": int(batch["inputs"]["mask"][index].sum()),
                    "coords": batch["inputs"]["coords"][index].clone(),
                    "features": batch["inputs"]["features"][index].clone(),
                }
            )
    return events


class SplitTests(unittest.TestCase):
    # Event counts of the two classified sources in the released files.
    PUBLISHED_EVENT_COUNTS = {"2nubb": 3_284_116, "Bi214": 2_634_002}

    def test_reproduces_published_split_from_event_counts(self) -> None:
        split = build_split(self.PUBLISHED_EVENT_COUNTS, DataConfig())
        counts = split_counts(split)
        verify_published_counts(counts)  # raises unless every count matches
        # Slice counts pinned against the published split manifest (220/124/130).
        self.assertEqual(
            {name: len(items) for name, items in split.items()},
            {"train": 220, "validation": 124, "test": 130},
        )

    def test_verify_published_counts_rejects_a_different_split(self) -> None:
        split = build_split(self.PUBLISHED_EVENT_COUNTS, DataConfig(seed=7))
        with self.assertRaisesRegex(ValueError, "published"):
            verify_published_counts(split_counts(split))

    def test_blocks_partition_every_event_exactly_once(self) -> None:
        config = DataConfig(split_block_events=64)
        for event_count in (1, 63, 64, 65, 1000, 4097):
            allocation = block_assignments("Bi214", event_count, config)
            covered = np.zeros(event_count, dtype=np.int64)
            for items in allocation.values():
                for item in items:
                    covered[item.event_start : item.event_stop] += 1
            self.assertTrue(np.all(covered == 1), f"event_count={event_count}")

    def test_split_is_deterministic_and_seed_sensitive(self) -> None:
        counts = {"2nubb": 5000, "Bi214": 4000}
        first = build_split(counts, DataConfig(split_block_events=100))
        again = build_split(counts, DataConfig(split_block_events=100))
        other = build_split(counts, DataConfig(split_block_events=100, seed=43))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_source_position_decorrelates_sources(self) -> None:
        config = DataConfig(split_block_events=10)
        def ranges(source_key: str) -> dict[str, list[tuple[int, int]]]:
            allocation = block_assignments(source_key, 500, config)
            return {
                split: [(item.event_start, item.event_stop) for item in items]
                for split, items in allocation.items()
            }

        # Same size and seed, different source position -> different deal.
        self.assertNotEqual(ranges("2nubb"), ranges("Bi214"))

    def test_cap_keeps_exact_size_class_ratio_and_a_subset(self) -> None:
        counts = {"2nubb": 5000, "Bi214": 4000}
        full = build_split(counts, DataConfig(split_block_events=100))
        capped = build_split(counts, DataConfig(split_block_events=100, max_train_events=900))
        self.assertEqual(sum(item.event_count for item in capped["train"]), 900)
        by_category = split_counts(capped)["train"]
        full_by_category = split_counts(full)["train"]
        ratio_full = full_by_category["2nu"] / sum(full_by_category.values())
        ratio_capped = by_category["2nu"] / sum(by_category.values())
        self.assertAlmostEqual(ratio_capped, ratio_full, delta=0.01)
        allowed = {
            (item.source_key, event)
            for item in full["train"]
            for event in range(item.event_start, item.event_stop)
        }
        for item in capped["train"]:
            for event in range(item.event_start, item.event_stop):
                self.assertIn((item.source_key, event), allowed)
        # Validation and test are untouched by a train cap.
        self.assertEqual(capped["validation"], full["validation"])
        self.assertEqual(capped["test"], full["test"])

    def test_cap_larger_than_split_is_a_no_op(self) -> None:
        counts = {"2nubb": 500, "Bi214": 400}
        full = build_split(counts, DataConfig(split_block_events=20))
        capped = build_split(counts, DataConfig(split_block_events=20, max_test_events=10**9))
        self.assertEqual(full, capped)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(split_fractions=(0.5, 0.5, 0.5))
        with self.assertRaises(ValueError):
            DataConfig(max_train_events=0)
        self.assertTrue(DataConfig().is_published_split)
        self.assertFalse(DataConfig(max_train_events=10).is_published_split)
        self.assertFalse(DataConfig(seed=1).is_published_split)


class OffsetTests(unittest.TestCase):
    def _write_ev_no(self, path: Path, ev_no: list[int]) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ev_no", data=np.asarray(ev_no, dtype=np.int64))

    def test_offsets_delimit_contiguous_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.h5"
            self._write_ev_no(path, [0, 0, 0, 1, 2, 2])
            np.testing.assert_array_equal(scan_event_offsets(path), [0, 3, 4, 6])

    def test_gap_or_nonzero_start_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.h5"
            for bad in ([0, 0, 2, 2], [1, 1, 2], [0, 1, 0]):
                self._write_ev_no(path, bad)
                with self.assertRaisesRegex(ValueError, "0..N-1"):
                    scan_event_offsets(path)


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.coords = rng.uniform(-1500, 1500, (40, 3)).astype(np.float32)
        self.radius = rng.uniform(0, 23, 40).astype(np.float32)
        self.radius[::7] = np.nan

    def test_feature_dim_shapes_and_coverage(self) -> None:
        for name, feature_dim in EXPECTED_FEATURE_DIM.items():
            config = _tokenization(name)
            tokens, coverage = tokenize_tracker_event(
                self.coords, self.radius, sampling_key="k", config=config
            )
            self.assertEqual(config.feature_dim, feature_dim)
            self.assertEqual(tokens["features"].shape[1], feature_dim)
            self.assertEqual(tokens["coords"].shape[1], 3)
            self.assertLessEqual(len(tokens["coords"]), config.max_tokens)
            self.assertTrue(0.0 < coverage <= 1.0)
            self.assertTrue(np.isfinite(tokens["coords"]).all())
            self.assertTrue(np.isfinite(tokens["features"]).all())

    def test_hit_order_is_not_a_feature(self) -> None:
        permutation = np.random.default_rng(9).permutation(len(self.coords))
        for name in EXPECTED_FEATURE_DIM:
            config = _tokenization(name)
            a, _ = tokenize_tracker_event(self.coords, self.radius, sampling_key="k", config=config)
            b, _ = tokenize_tracker_event(
                self.coords[permutation], self.radius[permutation], sampling_key="k", config=config
            )
            np.testing.assert_allclose(a["coords"], b["coords"], atol=1e-6)
            np.testing.assert_allclose(a["features"], b["features"], atol=1e-6)

    def test_summary_tokens_keep_every_hit(self) -> None:
        _, coverage = tokenize_tracker_event(
            self.coords, self.radius, sampling_key="k", config=_tokenization("summary_features")
        )
        self.assertEqual(coverage, 1.0)

    def test_centered_coordinates_are_translation_invariant(self) -> None:
        config = _tokenization("voxel")
        shift = np.array([500.0, -800.0, 120.0], dtype=np.float32)
        a, _ = tokenize_tracker_event(self.coords, self.radius, sampling_key="k", config=config)
        b, _ = tokenize_tracker_event(
            self.coords + shift, self.radius, sampling_key="k", config=config
        )
        np.testing.assert_allclose(a["coords"], b["coords"], atol=5e-3)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.truth = _make_dataset(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _prepared(self, tokenization: str = "voxel", *, num_workers: int = 0, **config):
        return prepare_dataset(
            data_config=_small_config(self.root, **config),
            tokenization_config=_tokenization(tokenization),
            batch_size=16,
            num_workers=num_workers,
        )

    def test_each_event_appears_exactly_once_and_splits_are_disjoint(self) -> None:
        for num_workers in (0, 2):
            with self.subTest(num_workers=num_workers):
                prepared = self._prepared(num_workers=num_workers)
                seen: dict[str, set[str]] = {}
                for name, loader in (
                    ("train", prepared.train_loader),
                    ("validation", prepared.validation_loader),
                    ("test", prepared.test_loader),
                ):
                    ids = [event["event_id"] for event in _all_events(loader)]
                    self.assertEqual(len(ids), len(set(ids)), name)
                    self.assertEqual(len(ids), sum(prepared.counts[name].values()), name)
                    seen[name] = set(ids)
                self.assertFalse(seen["train"] & seen["validation"])
                self.assertFalse(seen["train"] & seen["test"])
                self.assertFalse(seen["validation"] & seen["test"])
                self.assertEqual(
                    sum(len(ids) for ids in seen.values()), 2 * len(self.truth["2nubb"]["hits"])
                )

    def test_training_events_come_from_their_own_source_file(self) -> None:
        # Regression: a lazily-evaluated loop variable once made every source's
        # stream open the last source's file, silently mixing up the data.
        prepared = self._prepared()
        truth_energy = {
            source: (arrays["E1"][np.r_[0, np.cumsum(arrays["hits"])[:-1]]].astype(np.float64)
                     + arrays["E2"][np.r_[0, np.cumsum(arrays["hits"])[:-1]]].astype(np.float64))
            / 1000.0
            for source, arrays in self.truth.items()
        }
        expected_label = {"2nubb": 1.0, "Bi214": 0.0}
        for loader in (prepared.train_loader, prepared.validation_loader, prepared.test_loader):
            for event in _all_events(loader):
                _, source, ordinal = event["event_id"].split("::")
                self.assertAlmostEqual(event["energy"], truth_energy[source][int(ordinal)], places=9)
                self.assertEqual(event["label"], expected_label[source])
                self.assertEqual(event["category"], "2nu" if source == "2nubb" else "Bi214")

    def test_training_batches_mix_the_classes(self) -> None:
        prepared = self._prepared()
        labels = np.array([event["label"] for event in _all_events(prepared.train_loader)])
        overall = labels.mean()
        self.assertTrue(0.4 < overall < 0.6)
        first_batches = labels[:64]
        self.assertTrue(0.25 < first_batches.mean() < 0.75)

    def test_epoch_changes_order_not_content_and_is_reproducible(self) -> None:
        prepared = self._prepared()
        dataset = prepared.train_loader.dataset

        def ids_for(epoch: int) -> list[str]:
            dataset.set_epoch(epoch)
            return [event["event_id"] for event in _all_events(prepared.train_loader)]

        epoch0, epoch1, epoch0_again = ids_for(0), ids_for(1), ids_for(0)
        self.assertEqual(epoch0, epoch0_again)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(sorted(epoch0), sorted(epoch1))

    def test_evaluation_order_is_fixed(self) -> None:
        prepared = self._prepared()
        first = [event["event_id"] for event in _all_events(prepared.test_loader)]
        second = [event["event_id"] for event in _all_events(prepared.test_loader)]
        self.assertEqual(first, second)

    def test_batch_contract_matches_energybench(self) -> None:
        prepared = self._prepared("summary_features")
        batch = next(iter(prepared.train_loader))
        coords, features, mask = (batch["inputs"][k] for k in ("coords", "features", "mask"))
        self.assertEqual(coords.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(features.shape[-1], 6)
        self.assertEqual(coords.shape[:2], mask.shape)
        self.assertTrue(mask.any(dim=1).all())
        self.assertTrue(torch.all(coords[~mask] == 0))  # padding is zeroed
        self.assertTrue(set(batch["label"].tolist()) <= {0.0, 1.0})
        self.assertEqual(batch["energy"].dtype, torch.float64)
        self.assertTrue(bool(((batch["energy"] > 1.4) & (batch["energy"] < 1.7)).all()))  # MeV
        for key in ("event_id", "group_id", "category", "split"):
            self.assertEqual(len(batch[key]), len(batch["label"]))
        self.assertTrue(bool(((batch["projection_coverage"] > 0) & (batch["projection_coverage"] <= 1)).all()))

    def test_tokens_do_not_depend_on_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp)
            # Same geometry (same seeds), completely different energies.
            _make_dataset(other, two_nu_energy=(300.0, 400.0), bi_energy=(2500.0, 2800.0))
            with_first = self._prepared().test_loader
            with_other = prepare_dataset(
                data_config=_small_config(other),
                tokenization_config=_tokenization("voxel"),
                batch_size=16,
                num_workers=0,
            ).test_loader
            first = {event["event_id"]: event for event in _all_events(with_first)}
            second = {event["event_id"]: event for event in _all_events(with_other)}
            self.assertEqual(first.keys(), second.keys())
            self.assertNotEqual(
                [round(e["energy"], 6) for e in first.values()],
                [round(e["energy"], 6) for e in second.values()],
            )
            for event_id, event in first.items():
                torch.testing.assert_close(event["coords"], second[event_id]["coords"])
                torch.testing.assert_close(event["features"], second[event_id]["features"])

    def test_caps_shrink_the_loaders(self) -> None:
        prepared = self._prepared(max_train_events=100, max_test_events=40)
        self.assertEqual(sum(prepared.counts["train"].values()), 100)
        self.assertEqual(sum(prepared.counts["test"].values()), 40)
        self.assertEqual(len(_all_events(prepared.train_loader)), 100)

    def test_corrupt_energy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)
            _make_dataset(bad)
            _write_source(
                bad / "data_Bi214_merged.h5",
                label="Bi214",
                n_events=1600,
                energy_kev=(1500.0, 1600.0),
                seed=2,
                corrupt_first_energy_row=True,
            )
            prepared = prepare_dataset(
                data_config=_small_config(bad),
                tokenization_config=_tokenization("voxel"),
                batch_size=16,
                num_workers=0,
            )
            with self.assertRaisesRegex(ValueError, "varies within an event"):
                # Event 0 sits in one of the splits; drain them all.
                for loader in (prepared.train_loader, prepared.validation_loader, prepared.test_loader):
                    _all_events(loader)

    def test_missing_file_and_wrong_label_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "SUPERNEMO_ROPE_DATA_DIR"):
                prepare_dataset(
                    data_config=_small_config(empty),
                    tokenization_config=_tokenization("voxel"),
                    batch_size=8,
                    num_workers=0,
                )
            _write_source(
                empty / "data_2nubb_merged.h5", label="Bi214", n_events=200,
                energy_kev=(1500.0, 1600.0), seed=1,
            )
            _write_source(
                empty / "data_Bi214_merged.h5", label="Bi214", n_events=200,
                energy_kev=(1500.0, 1600.0), seed=2,
            )
            with self.assertRaisesRegex(ValueError, "expected label"):
                prepare_dataset(
                    data_config=_small_config(empty),
                    tokenization_config=_tokenization("voxel"),
                    batch_size=8,
                    num_workers=0,
                )

    def test_collate_pads_to_the_longest_event(self) -> None:
        def sample(count: int) -> dict:
            return {
                "inputs": {
                    "coords": np.ones((count, 3), dtype=np.float32),
                    "features": np.ones((count, 4), dtype=np.float32),
                },
                "label": np.float32(1),
                "energy": np.float64(1.5),
                "event_id": f"e{count}",
                "group_id": f"e{count}",
                "category": "2nu",
                "split": "test",
                "projection_coverage": np.float32(1.0),
            }

        batch = collate_events([sample(2), sample(5)])
        self.assertEqual(tuple(batch["inputs"]["mask"].shape), (2, 5))
        self.assertEqual(batch["inputs"]["mask"].sum(dim=1).tolist(), [2, 5])
        with self.assertRaises(ValueError):
            collate_events([])

    def test_dataset_rejects_bad_arguments(self) -> None:
        with self.assertRaises(TypeError):
            SuperNEMOEventDataset(
                data_root=self.root, slices=[], offsets={}, tokenization_config=None,
                split="train", shuffle=True, seed=42, block_events=10,
            )


class EnergyGridTests(unittest.TestCase):
    """EnergyBench rejects events above 3 MeV; the wrapper must set them aside."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # Overlap in 2900-3000 keV; Bi214 spills past the 3000 keV grid edge.
        cls.truth = _make_dataset(
            cls.root, n_events=3000, two_nu_energy=(2900.0, 3000.0), bi_energy=(2900.0, 3100.0)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _prepared(self):
        return prepare_dataset(
            data_config=_small_config(self.root),
            tokenization_config=_tokenization("sampled_hits"),
            batch_size=32,
            num_workers=0,
        )

    def test_wrapper_keeps_only_in_grid_events_and_counts_the_rest(self) -> None:
        prepared = self._prepared()
        raw = _all_events(prepared.test_loader)
        expected_dropped = sum(1 for event in raw if event["energy"] > 3.0)
        self.assertGreater(expected_dropped, 0)  # the fixture really spills over
        wrapped = EnergyWindowLoader(prepared.test_loader, max_energy_mev=3.0)
        kept = _all_events(wrapped)
        self.assertEqual(len(kept), len(raw) - expected_dropped)
        self.assertTrue(all(event["energy"] <= 3.0 for event in kept))
        self.assertEqual(wrapped.dropped, {"Bi214": expected_dropped})
        self.assertIs(wrapped.dataset, prepared.test_loader.dataset)
        # A second pass reports the same counts rather than accumulating.
        _all_events(wrapped)
        self.assertEqual(wrapped.dropped, {"Bi214": expected_dropped})

    def test_wrapped_batches_stay_consistent(self) -> None:
        wrapped = EnergyWindowLoader(self._prepared().test_loader, max_energy_mev=3.0)
        for batch in wrapped:
            size = len(batch["label"])
            self.assertEqual(batch["inputs"]["coords"].shape[0], size)
            self.assertEqual(batch["inputs"]["mask"].shape[0], size)
            for key in ("event_id", "group_id", "category", "split"):
                self.assertEqual(len(batch[key]), size)
            self.assertEqual(batch["energy"].shape[0], size)
            self.assertEqual(batch["projection_coverage"].shape[0], size)
            self.assertTrue(bool(batch["inputs"]["mask"].any(dim=1).all()))

    def test_energybench_rejects_the_raw_loader_but_accepts_the_wrapped_one(self) -> None:
        prepared = self._prepared()
        torch.manual_seed(0)
        model = build_supernemo_transformer(_tokenization("sampled_hits"), "rope", **TINY_MODEL)
        configuration = EvaluationConfig(min_per_class=3, min_per_bin=3)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, r"canonical EnergyBench range"):
                evaluate_classification(
                    model, prepared.test_loader, device="cpu",
                    output_dir=Path(out) / "raw", config=configuration,
                )
            wrapped = EnergyWindowLoader(prepared.test_loader, max_energy_mev=3.0)
            metrics = evaluate_classification(
                model, wrapped, device="cpu", output_dir=Path(out) / "wrapped", config=configuration,
            )
            self.assertEqual(
                metrics["n_events"],
                sum(prepared.counts["test"].values()) - sum(wrapped.dropped.values()),
            )
            self.assertIsNotNone(metrics["matched_auc"])

    def test_invalid_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnergyWindowLoader(self._prepared().test_loader, max_energy_mev=0.0)


class ModelTests(unittest.TestCase):
    def _batch(self, feature_dim: int, batch: int = 3, tokens: int = 7) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(0)
        mask = torch.ones(batch, tokens, dtype=torch.bool)
        mask[0, 4:] = False  # one padded event
        return {
            "coords": torch.randn(batch, tokens, 3, generator=generator) * 0.4,
            "features": torch.randn(batch, tokens, feature_dim, generator=generator),
            "mask": mask,
        }

    def test_all_tokenizations_and_encodings_produce_finite_logits(self) -> None:
        for name, feature_dim in EXPECTED_FEATURE_DIM.items():
            for encoding in ("rope", "coordinate_mlp", "fourier_xyz"):
                with self.subTest(tokenization=name, encoding=encoding):
                    model = build_supernemo_transformer(
                        _tokenization(name), encoding, **TINY_MODEL
                    ).eval()
                    self.assertEqual(model.feature_dim, feature_dim)
                    logits = model(self._batch(feature_dim))
                    self.assertEqual(tuple(logits.shape), (3,))
                    self.assertTrue(torch.isfinite(logits).all())

    def test_rope_is_translation_invariant_and_additive_encoders_are_not(self) -> None:
        inputs = self._batch(4)
        shifted = {**inputs, "coords": inputs["coords"] + torch.tensor([0.31, -0.22, 0.17])}
        torch.manual_seed(0)
        rope = build_supernemo_transformer(_tokenization("voxel"), "rope", **TINY_MODEL).eval()
        torch.testing.assert_close(rope(inputs), rope(shifted), atol=1e-4, rtol=1e-4)
        torch.manual_seed(0)
        mlp = build_supernemo_transformer(
            _tokenization("voxel"), "coordinate_mlp", **TINY_MODEL
        ).eval()
        self.assertGreater(float((mlp(inputs) - mlp(shifted)).abs().max()), 1e-4)

    def test_rope_uses_the_relative_geometry(self) -> None:
        # Distinct geometry must change the output: RoPE is not ignoring coords.
        inputs = self._batch(4)
        moved = {**inputs, "coords": inputs["coords"] * 2.5}
        torch.manual_seed(0)
        rope = build_supernemo_transformer(_tokenization("voxel"), "rope", **TINY_MODEL).eval()
        self.assertGreater(float((rope(inputs) - rope(moved)).abs().max()), 1e-4)

    def test_rope_has_no_additive_position_encoder(self) -> None:
        rope = build_supernemo_transformer(_tokenization("voxel"), "rope", **TINY_MODEL)
        mlp = build_supernemo_transformer(_tokenization("voxel"), "coordinate_mlp", **TINY_MODEL)
        self.assertIsNone(rope.position_encoder)
        self.assertIsNotNone(mlp.position_encoder)

    def test_config_and_metadata_are_recorded(self) -> None:
        model = build_supernemo_transformer(
            _tokenization("summary_features"), "rope", rope_base=4.0, **TINY_MODEL
        )
        config = model.config_dict()
        self.assertEqual(config["position_encoding"], "rope")
        self.assertEqual(config["rope_base"], 4.0)
        self.assertEqual(config["feature_dim"], 6)
        self.assertEqual(model.architecture_metadata["dataset"], "SuperNEMO")
        self.assertEqual(
            model.architecture_metadata["tokenization"]["tokenization"], "summary_features"
        )

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_supernemo_transformer(_tokenization("voxel"), "sinusoidal")
        with self.assertRaises(TypeError):
            build_supernemo_transformer("voxel", "rope")
        with self.assertRaises(ValueError):
            build_supernemo_transformer(_tokenization("voxel"), "rope", rope_base=1.0)
        with self.assertRaises(ValueError):  # head_dim = 4 < 6
            build_supernemo_transformer(
                _tokenization("voxel"), "rope", d_model=16, nhead=4, num_layers=1,
                dim_feedforward=16,
            )

    def test_frequency_table_for_the_default_head(self) -> None:
        table = rope_frequency_table(head_dim=16, rope_base=8.0)
        axes = [row["axis"] for row in table]
        self.assertEqual(axes, ["x"] * 3 + ["y"] * 3 + ["z"] * 2)  # 8 pairs -> [3, 3, 2]
        x_theta = [row["theta"] for row in table if row["axis"] == "x"]
        z_theta = [row["theta"] for row in table if row["axis"] == "z"]
        np.testing.assert_allclose(x_theta, [8.0, 8.0 ** (2 / 3), 8.0 ** (1 / 3)])
        np.testing.assert_allclose(z_theta, [8.0, 8.0 ** 0.5])
        for row in table:
            self.assertAlmostEqual(row["wavelength"], 2 * math.pi / row["theta"])
            self.assertAlmostEqual(row["unambiguous_extent"], math.pi / row["theta"])

    def test_default_base_limits_match_the_documented_numbers(self) -> None:
        self.assertEqual(DEFAULT_SUPERNEMO_ROPE_BASE, 8.0)
        limits = coarsest_unambiguous_extents(head_dim=16, rope_base=DEFAULT_SUPERNEMO_ROPE_BASE)
        self.assertAlmostEqual(limits["z"], math.pi / 8.0 ** 0.5, places=6)  # 1.11
        self.assertAlmostEqual(limits["y"], math.pi / 8.0 ** (1 / 3), places=6)  # 1.57
        self.assertAlmostEqual(limits["x"], limits["y"], places=6)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _make_dataset(cls.root, n_events=3000)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_train_and_evaluate_with_unchanged_energybench(self) -> None:
        prepared = prepare_dataset(
            data_config=_small_config(self.root, max_train_events=400, max_validation_events=200),
            tokenization_config=_tokenization("voxel"),
            batch_size=32,
            num_workers=0,
        )
        torch.manual_seed(0)
        model = build_supernemo_transformer(_tokenization("voxel"), "rope", **TINY_MODEL)
        with tempfile.TemporaryDirectory() as out:
            out_dir = Path(out)
            history = train_model(
                model,
                prepared.train_loader,
                prepared.validation_loader,
                TrainingConfig(
                    batch_size=32, epochs=1, early_stopping_patience=1, seed=42,
                    deterministic=True, use_amp=False, device="cpu", num_workers=0,
                ),
                "classification",
                out_dir / "training",
            )
            self.assertEqual(history["epochs_completed"], 1)
            self.assertTrue(np.isfinite(history["best_metric"]))
            self.assertEqual(history["model"]["class_name"], "NEXTTransformerClassifier")
            self.assertEqual(history["model"]["config"]["position_encoding"], "rope")
            self.assertEqual(history["model"]["architecture"]["dataset"], "SuperNEMO")
            for name in ("best_model.pt", "last_model.pt", "history.json"):
                self.assertTrue((out_dir / "training" / name).is_file(), name)

            # The two classes share a 100 keV window (20 bins of 5 keV), so the
            # energy-matched AUC is evaluable with a small per-bin minimum.
            metrics = evaluate_classification(
                model,
                prepared.test_loader,
                device="cpu",
                output_dir=out_dir / "evaluation",
                config=EvaluationConfig(min_per_class=3, min_per_bin=3),
            )
            self.assertEqual(metrics["n_events"], sum(prepared.counts["test"].values()))
            self.assertIsNotNone(metrics["matched_auc"])
            self.assertTrue(0.0 <= metrics["matched_auc"] <= 1.0)
            self.assertTrue(0.0 <= metrics["auc"] <= 1.0)
            for name in ("metrics.json", "results.csv", "predictions.npz"):
                self.assertTrue((out_dir / "evaluation" / name).is_file(), name)
            with np.load(out_dir / "evaluation" / "predictions.npz") as arrays:
                self.assertEqual(set(np.unique(arrays["label"])), {0, 1})
                self.assertTrue(np.all((arrays["energy"] > 1.49) & (arrays["energy"] < 1.61)))

    def test_all_position_encodings_train_a_step(self) -> None:
        prepared = prepare_dataset(
            data_config=_small_config(self.root, max_train_events=64, max_validation_events=64),
            tokenization_config=_tokenization("sampled_hits"),
            batch_size=32,
            num_workers=0,
        )
        for encoding in ("rope", "coordinate_mlp", "fourier_xyz"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as out:
                model = build_supernemo_transformer(
                    _tokenization("sampled_hits"), encoding, **TINY_MODEL
                )
                history = train_model(
                    model, prepared.train_loader, prepared.validation_loader,
                    TrainingConfig(
                        batch_size=32, epochs=1, early_stopping_patience=1, seed=42,
                        deterministic=True, use_amp=False, device="cpu", num_workers=0,
                    ),
                    "classification", Path(out) / "training",
                )
                self.assertTrue(np.isfinite(history["best_metric"]))


class RunnerTests(unittest.TestCase):
    """Drive the env-var runner end to end on a fabricated 4096-event-block dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "data"
        # ~10 blocks of the (fixed) 4096-event split unit per source.
        _make_dataset(cls.root, n_events=40_960, hits_range=(4, 6))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _load_runner(self):
        spec = importlib.util.spec_from_file_location("supernemo_rope_runner", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_runner_writes_the_documented_tree_and_skips_completed_runs(self) -> None:
        output_root = Path(self._tmp.name) / "results"
        environment = {
            "SUPERNEMO_ROPE_DATA_DIR": str(self.root),
            "SUPERNEMO_ROPE_OUTPUT_ROOT": str(output_root),
            "SUPERNEMO_ROPE_TOKENIZATION": "sampled_hits",
            "SUPERNEMO_ROPE_POSITION_ENCODING": "rope",
            "SUPERNEMO_ROPE_BASE": "4.0",
            "SUPERNEMO_ROPE_EPOCHS": "1",
            "SUPERNEMO_ROPE_BATCH_SIZE": "32",
            "SUPERNEMO_ROPE_DEVICE": "cpu",
            "SUPERNEMO_ROPE_USE_AMP": "0",
            "SUPERNEMO_ROPE_MAX_TRAIN_EVENTS": "256",
            "SUPERNEMO_ROPE_MAX_VALIDATION_EVENTS": "128",
            "SUPERNEMO_ROPE_MAX_TEST_EVENTS": "256",
        }
        with mock.patch.dict(os.environ, environment):
            runner = self._load_runner()
            self.assertEqual(runner.main(), 0)
            run_dir = output_root / "classification__sampled_hits__rope"
            for relative in (
                "run_config.json",
                "run_summary.json",
                "training/best_model.pt",
                "training/history.json",
                "evaluation/metrics.json",
                "evaluation/predictions.npz",
            ):
                self.assertTrue((run_dir / relative).is_file(), relative)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(summary["position_encoding"], "rope")
            self.assertEqual(summary["rope_base"], 4.0)
            self.assertEqual(summary["feature_dim"], 4)
            self.assertEqual((summary["n_train"], summary["n_val"], summary["n_test"]), (256, 128, 256))
            self.assertFalse(summary["published_split"])
            self.assertEqual(summary["energy_grid_max_mev"], 3.0)
            self.assertEqual(summary["test_events_above_energy_grid"], 0)
            self.assertEqual(len(config["rope_frequency_table"]), 8)
            self.assertEqual(config["model"]["rope_base"], 4.0)

            before = (run_dir / "run_summary.json").stat().st_mtime_ns
            self.assertEqual(runner.main(), 0)  # already complete -> skipped
            self.assertEqual((run_dir / "run_summary.json").stat().st_mtime_ns, before)

    def test_runner_rejects_unknown_choices(self) -> None:
        for name, value in (
            ("SUPERNEMO_ROPE_TOKENIZATION", "raw_patches"),
            ("SUPERNEMO_ROPE_POSITION_ENCODING", "alibi"),
        ):
            with self.subTest(name=name), mock.patch.dict(
                os.environ,
                {"SUPERNEMO_ROPE_DATA_DIR": str(self.root), name: value},
            ):
                with self.assertRaises(ValueError):
                    self._load_runner().main()


class CollateTests(unittest.TestCase):
    def _load_collate(self):
        spec = importlib.util.spec_from_file_location(
            "supernemo_rope_collate", SCRIPTS_DIR / "collate_results.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_run(self, root: Path, run_id: str, tokenization: str, encoding: str, *,
                   matched: float | None, complete: bool = True, published: bool = True) -> None:
        run_dir = root / run_id
        (run_dir / "evaluation").mkdir(parents=True)
        (run_dir / "run_summary.json").write_text(json.dumps({
            "run_id": run_id, "tokenization": tokenization, "position_encoding": encoding,
            "rope_base": 8.0 if encoding == "rope" else None, "published_split": published,
            "test_matched_auc": matched, "test_inclusive_auc": 0.9, "epochs_completed": 3,
            "parameter_count": 1000,
        }))
        if complete:
            (run_dir / "evaluation" / "metrics.json").write_text("{}")

    def test_ranking_csv_and_require_all(self) -> None:
        collate = self._load_collate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "classification__voxel__rope", "voxel", "rope", matched=0.71)
            self._write_run(root, "classification__sampled_hits__rope", "sampled_hits", "rope", matched=0.69)
            self._write_run(root, "classification__voxel__coordinate_mlp", "voxel", "coordinate_mlp", matched=0.72)
            self._write_run(root, "classification__summary_features__rope_base16", "summary_features", "rope", matched=0.5, published=False)
            self._write_run(root, "classification__summary_features__rope", "summary_features", "rope", matched=0.9, complete=False)

            rows = collate._load_rows(root)
            self.assertEqual(len(rows), 4)  # the truncated run is skipped
            ranking = collate._ranking_markdown(rows)
            official_section = ranking.split("## Official matrix")[1].split("## Controls")[0]
            self.assertIn("voxel", official_section)
            self.assertIn("sampled_hits", official_section)
            self.assertNotIn("coordinate_mlp", official_section)
            self.assertIn("coordinate_mlp", ranking.split("## Controls")[1])
            self.assertIn("Reduced-scale runs present", ranking)
            self.assertLess(official_section.index("voxel"), official_section.index("sampled_hits"))

            argv = ["collate_results.py", "--output-root", str(root)]
            with mock.patch.object(sys, "argv", argv + ["--require-all"]):
                self.assertEqual(collate.main(), 1)  # summary_features/rope is truncated
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(collate.main(), 0)
            csv_text = (root / "supernemo_rope_classification_results.csv").read_text()
            self.assertTrue(csv_text.splitlines()[0].startswith("run_id,tokenization,position_encoding"))
            self.assertTrue((root / "supernemo_rope_ranking.md").is_file())


    def test_sbatch_script_is_valid_bash(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS_DIR / "run_supernemo_rope_classification_array.sbatch")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
