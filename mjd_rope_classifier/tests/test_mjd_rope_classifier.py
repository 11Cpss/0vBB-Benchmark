"""Synthetic-data checks for the MJD RoPE classification benchmark.

Covers the model (all three tokenizations under ``rope``, parity with
``mjd_transformer.model.MJDTransformer`` for the two additive encodings, RoPE
config guards) and the data layer (seeded cache build/reuse against a
fabricated MJD-schema HDF5 fixture, and the shared training/evaluation
contract). CPU-only, no real data.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _source_root in (
    PROJECT_ROOT / "mjd_detector",
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "mjd_rope_classifier",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from mjd_transformer.model import MJDTransformer  # noqa: E402
from mjd_transformer.tokenization import TokenizationConfig  # noqa: E402
from mjdbench.config import DataConfig, TrainingConfig  # noqa: E402
from mjdbench.training import evaluate_model, set_seed, train_model  # noqa: E402
from next_transformer.rotary_attention import RotaryTransformerEncoder  # noqa: E402

from mjd_rope_transformer import (  # noqa: E402
    DEFAULT_MJD_ROPE_BASE,
    MJDRopeTransformer,
)
from mjdrope_bench.data import (  # noqa: E402
    SubsetSizes,
    build_classification_cache,
    load_cached_loaders,
)

WF_LEN = 200
PATCH_SIZE = 20
TOKEN_COUNT = WF_LEN // PATCH_SIZE  # 10, matches TOKEN_COUNT

TOKENIZATIONS = ("raw_patches", "segment_summary", "pulse_entities")

# head_dim = 24 // 4 = 6, the minimum legal RoPE head_dim (even and >= 6).
TINY_MODEL = {
    "d_model": 24,
    "nhead": 4,
    "num_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
}


def _tokenization_config(name: str) -> TokenizationConfig:
    if name == "raw_patches":
        return TokenizationConfig(tokenization="raw_patches", patch_size=PATCH_SIZE)
    if name == "segment_summary":
        return TokenizationConfig(
            tokenization="segment_summary", token_count=TOKEN_COUNT
        )
    return TokenizationConfig(
        tokenization="pulse_entities",
        token_count=TOKEN_COUNT,
        uniform_entity_fraction=0.5,
        entity_context_size=3,
    )


def _model(tokenization: str, encoding: str, **overrides):
    settings = dict(TINY_MODEL)
    settings.setdefault("rope_base", 2.0)
    settings.update(overrides)
    return MJDRopeTransformer(
        task="classification",
        position_encoding=encoding,
        tokenization_config=_tokenization_config(tokenization),
        **settings,
    )


def _write_mjd_hdf5(path: Path, *, n_events: int, seed: int) -> None:
    """Fabricate a tiny MJD-schema HDF5 shard (real field names)."""

    rng = np.random.default_rng(seed)
    sample_axis = np.arange(WF_LEN, dtype=np.float64)
    waveforms = np.zeros((n_events, WF_LEN), dtype=np.float64)
    clean = np.arange(n_events) % 3 != 0  # ~67% clean, some structure

    for row in range(n_events):
        baseline = 0.05 * rng.standard_normal(WF_LEN)
        rise = 20.0 if clean[row] else 60.0  # cleaner pulses rise faster
        pulse = 1.0 / (1.0 + np.exp(-(sample_axis - WF_LEN // 2) / rise))
        waveforms[row] = baseline + pulse

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw_waveform", data=waveforms)
        handle.create_dataset(
            "energy_label", data=rng.uniform(0.0, 3000.0, size=n_events)
        )
        handle.create_dataset("psd_label_low_avse", data=clean)
        handle.create_dataset("psd_label_high_avse", data=np.ones(n_events, bool))
        handle.create_dataset("psd_label_dcr", data=np.ones(n_events, bool))
        handle.create_dataset("psd_label_lq", data=np.ones(n_events, bool))
        handle.create_dataset("id", data=np.arange(n_events, dtype=np.int64))
        handle.create_dataset(
            "run_number", data=np.full(n_events, 1000, dtype=np.int64)
        )
        handle.create_dataset("detector", data=np.zeros(n_events, dtype=np.int64))
        handle.create_dataset("tp0", data=np.zeros(n_events, dtype=np.int64))


class MJDRopeModelTests(unittest.TestCase):
    def test_all_three_tokenizations_rope_forward_and_backward(self) -> None:
        waveform = torch.rand(4, 1, WF_LEN)
        for tokenization in TOKENIZATIONS:
            with self.subTest(tokenization=tokenization):
                model = _model(tokenization, "rope")
                output = model(waveform)
                self.assertEqual(tuple(output.shape), (4,))
                self.assertTrue(torch.isfinite(output).all())

                output.sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"{name} has no gradient")
                    self.assertTrue(
                        torch.isfinite(parameter.grad).all(),
                        f"{name} has non-finite gradient",
                    )

    def test_mjd_parity_for_additive_encodings(self) -> None:
        """The non-RoPE cells must be identical to MJDTransformer."""

        waveform = torch.rand(3, 1, WF_LEN)
        for tokenization in TOKENIZATIONS:
            for encoding in ("coordinate_mlp", "fourier_coordinates"):
                with self.subTest(tokenization=tokenization, encoding=encoding):
                    tokenization_config = _tokenization_config(tokenization)

                    set_seed(42, True)
                    ours = MJDRopeTransformer(
                        task="classification",
                        position_encoding=encoding,
                        tokenization_config=tokenization_config,
                        **TINY_MODEL,
                    ).eval()

                    set_seed(42, True)
                    theirs = MJDTransformer(
                        task="classification",
                        position_encoding=encoding,
                        tokenization_config=tokenization_config,
                        **TINY_MODEL,
                    ).eval()

                    self.assertEqual(
                        list(ours.state_dict().keys()),
                        list(theirs.state_dict().keys()),
                    )
                    self.assertEqual(
                        sum(p.numel() for p in ours.parameters()),
                        sum(p.numel() for p in theirs.parameters()),
                    )
                    with torch.no_grad():
                        self.assertTrue(
                            torch.allclose(ours(waveform), theirs(waveform))
                        )

    def test_rope_branch_structure(self) -> None:
        model = _model("segment_summary", "rope")
        self.assertIsNone(model.position_encoder)
        self.assertIsInstance(model.transformer, RotaryTransformerEncoder)
        self.assertFalse(
            any("position_encoder" in name for name, _ in model.named_parameters())
        )
        config = model.config_dict()
        self.assertEqual(config["attention"], "rotary")
        self.assertEqual(config["rope_base"], 2.0)
        self.assertIn("time", config["rope_thetas_per_axis"])

    def test_default_rope_base(self) -> None:
        self.assertEqual(DEFAULT_MJD_ROPE_BASE, 2.0)
        model = _model("segment_summary", "rope", rope_base=DEFAULT_MJD_ROPE_BASE)
        output = model(torch.rand(2, 1, WF_LEN))
        self.assertTrue(torch.isfinite(output).all())

    def test_rope_config_guards(self) -> None:
        # head_dim = 24 // 8 = 3 -> odd and < 6
        with self.assertRaises(ValueError):
            _model("segment_summary", "rope", nhead=8)
        with self.assertRaises(ValueError):
            _model("segment_summary", "nonsense")
        with self.assertRaises(ValueError):
            MJDRopeTransformer(
                task="not_a_task",
                position_encoding="rope",
                tokenization_config=_tokenization_config("segment_summary"),
                **TINY_MODEL,
            )

    def test_shared_training_and_evaluation_contract(self) -> None:
        generator = torch.Generator().manual_seed(42)
        waveform = torch.randn(12, 1, WF_LEN, generator=generator)
        clean = torch.arange(12) % 2 == 0
        waveform[clean, :, WF_LEN // 2 :] += 0.75

        class _Synthetic(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return int(waveform.shape[0])

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {"inputs": waveform[index], "clean": clean[index]}

        data = _Synthetic()
        train_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(
            data, batch_size=4, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)
        model = _model("segment_summary", "rope")
        config = TrainingConfig(
            batch_size=4,
            epochs=1,
            learning_rate=1.0e-3,
            early_stopping_patience=1,
            num_workers=0,
            device="cpu",
            seed=42,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            history = train_model(
                model,
                train_loader,
                validation_loader,
                task="classification",
                config=config,
                output_dir=output,
            )
            metrics = evaluate_model(
                model,
                test_loader,
                task="classification",
                device="cpu",
                output_dir=output,
            )
            self.assertEqual(len(history), 1)
            self.assertEqual(metrics["events"], len(data))
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "predictions.npz").is_file())


class MJDRopeDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.data_dir = self.root / "MJD"
        _write_mjd_hdf5(self.data_dir / "MJD_Train_0.hdf5", n_events=60, seed=1)
        _write_mjd_hdf5(self.data_dir / "MJD_Test_0.hdf5", n_events=40, seed=2)
        self.cache_dir = self.root / "cache"

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_build_and_load_cache(self) -> None:
        sizes = SubsetSizes(n_train=20, n_val=5, n_test=15, seed=7)
        cache_path = build_classification_cache(
            data_root=self.data_dir,
            cache_dir=self.cache_dir,
            sizes=sizes,
            data_config=DataConfig(
                data_root=self.data_dir, validation_fraction=0.5, seed=7
            ),
        )
        self.assertTrue(cache_path.is_file())

        loaders = load_cached_loaders(cache_path, batch_size=4)
        self.assertEqual(loaders["counts"], {"train": 20, "validation": 5, "test": 15})

        batch = next(iter(loaders["train_loader"]))
        self.assertEqual(tuple(batch["inputs"].shape[1:]), (1, WF_LEN))
        self.assertEqual(batch["clean"].dtype, torch.float32)

        # Amplitude normalization (mjdbench's, reused unchanged): every
        # waveform's max absolute value is <= 1 (up to fp slop).
        train_wave = torch.load(cache_path, weights_only=False)["train"]["waveform"]
        self.assertTrue(
            bool((train_wave.abs().amax(dim=1) <= 1.0 + 1.0e-5).all().item())
        )

    def test_cache_is_reused_not_rebuilt(self) -> None:
        sizes = SubsetSizes(n_train=10, n_val=5, n_test=10, seed=1)
        first = build_classification_cache(
            data_root=self.data_dir, cache_dir=self.cache_dir, sizes=sizes
        )
        first_mtime = first.stat().st_mtime_ns
        second = build_classification_cache(
            data_root=self.data_dir, cache_dir=self.cache_dir, sizes=sizes
        )
        self.assertEqual(first, second)
        self.assertEqual(first_mtime, second.stat().st_mtime_ns)

    def test_oversized_subset_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_classification_cache(
                data_root=self.data_dir,
                cache_dir=self.cache_dir,
                sizes=SubsetSizes(n_train=1000, n_val=1, n_test=1),
            )


if __name__ == "__main__":
    unittest.main()
