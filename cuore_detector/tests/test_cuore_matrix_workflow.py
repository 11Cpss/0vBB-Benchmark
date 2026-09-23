"""Synthetic end-to-end checks for the CUORE 3 x 3 Δt regression matrix.

Fabricates tiny CUORE-schema HDF5 files, then exercises all nine
(tokenization x positional encoding) cells, the data contract, the RoPE branch,
and one full train + evaluate round trip. CPU-only, no real data.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _source_root in (
    PROJECT_ROOT / "evalutaions_workflow",
    PROJECT_ROOT / "mjd_detector",
    PROJECT_ROOT / "next_detector",
    PROJECT_ROOT / "cuore_detector",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from mjd_transformer.model import MJDTransformer  # noqa: E402
from mjd_transformer.tokenization import TokenizationConfig  # noqa: E402
from next_transformer.rotary_attention import (  # noqa: E402
    RotaryTransformerEncoder,
)
from simple_energybench import (  # noqa: E402
    EvaluationConfig,
    TrainingConfig,
    set_seed,
    train_model,
)

from cuore_transformer import CuoreWaveformTransformer  # noqa: E402
from cuorebench import (  # noqa: E402
    CuoreDataConfig,
    evaluate_cuore_regression,
    prepare_cuore_regression_data,
    scale_target,
    unscale_target,
)

WF_LEN = 1000
PATCH_SIZE = 20
TOKEN_COUNT = 50
NUM_TOKENS = WF_LEN // PATCH_SIZE  # 50, matches TOKEN_COUNT

TOKENIZATIONS = ("raw_patches", "segment_summary", "pulse_entities")
ENCODINGS = ("coordinate_mlp", "fourier_coordinates", "rope")

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
        entity_context_size=9,
    )


def _write_cuore_hdf5(
    path: Path,
    *,
    n_events: int,
    with_labels: bool,
    max_pulses: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sample_axis = np.arange(WF_LEN, dtype=np.float64)

    waveforms = np.zeros((n_events, WF_LEN), dtype=np.float64)
    n_pulses = np.ones(n_events, dtype=np.int64)
    dt_between = np.full((n_events, max_pulses - 1), np.nan, dtype=np.float64)
    peak_times = np.full((n_events, max_pulses), np.nan, dtype=np.float64)
    tail_elevated = np.zeros(n_events, dtype=bool)
    event_id = np.zeros(n_events, dtype=np.int64)
    labels = np.zeros(n_events, dtype=np.int64)

    main_center = 0.31 * WF_LEN
    for row in range(n_events):
        baseline = 0.15 + 0.02 * rng.standard_normal(WF_LEN)
        main = 0.8 * np.exp(-0.5 * ((sample_axis - main_center) / 12.0) ** 2)
        trace = baseline + main

        if row % 10 < 7:  # ~70% pile-up
            gap = float(rng.uniform(60.0, 450.0))
            trace = trace + 0.6 * np.exp(
                -0.5 * ((sample_axis - (main_center + gap)) / 12.0) ** 2
            )
            n_pulses[row] = 2
            dt_between[row, 0] = gap
            peak_times[row, 0] = main_center
            peak_times[row, 1] = main_center + gap
            tail_elevated[row] = row % 5 == 0
            event_id[row] = 7007 + row
            labels[row] = 1
        else:
            peak_times[row, 0] = main_center
            event_id[row] = row

        waveforms[row] = trace

    offset = waveforms[:, : WF_LEN // 4].mean(axis=1, keepdims=True)
    maximum = waveforms.max(axis=1, keepdims=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Waveform", data=waveforms)
        handle.create_dataset("normalizationOffset", data=offset)
        handle.create_dataset("normalizationScale", data=maximum - offset)
        handle.create_dataset("normalizationMaximum", data=maximum)
        handle.create_dataset("eventId", data=event_id)
        if with_labels:
            handle.create_dataset("Labels", data=labels)
        finder = handle.create_group("pulseFinder")
        finder.create_dataset("nPulses", data=n_pulses)
        finder.create_dataset("dtBetweenPeaks", data=dt_between)
        finder.create_dataset("peakTimes", data=peak_times)
        finder.create_dataset("tailElevated", data=tail_elevated)


class CuoreMatrixWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.data_dir = self.root / "CUORE"
        _write_cuore_hdf5(
            self.data_dir / "cuoreTraining.h5",
            n_events=60,
            with_labels=True,
            max_pulses=4,
            seed=1,
        )
        _write_cuore_hdf5(
            self.data_dir / "cuoreTest.h5",
            n_events=24,
            with_labels=False,
            max_pulses=3,
            seed=2,
        )

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _data_config(self, **overrides) -> CuoreDataConfig:
        settings = {
            "data_dir": str(self.data_dir),
            "waveform_length": WF_LEN,
            "val_fraction": 0.25,
            "dt_scale": 3000.0,
        }
        settings.update(overrides)
        return CuoreDataConfig(**settings)

    def _model(self, tokenization: str, encoding: str, **overrides):
        settings = dict(TINY_MODEL)
        settings.update(overrides)
        return CuoreWaveformTransformer(
            position_encoding=encoding,
            tokenization_config=_tokenization_config(tokenization),
            rope_base=8.0,
            **settings,
        )

    # ------------------------------------------------------------------ data

    def test_target_scaling_roundtrip(self) -> None:
        config = self._data_config()
        values = np.array([0.0, 150.0, 3000.0, 6800.0])
        self.assertTrue(
            np.allclose(values, unscale_target(scale_target(values, config), config))
        )

    def test_data_contract(self) -> None:
        config = self._data_config()
        train_loader, val_loader, test_loader, meta = prepare_cuore_regression_data(
            config, TrainingConfig(batch_size=8, num_workers=0)
        )

        low, high = meta["scaled_target_range"]
        self.assertGreaterEqual(low, 0.0)
        self.assertLess(high, 3.0)
        self.assertEqual(meta["n_test"], 24)
        self.assertEqual(meta["n_train"] + meta["n_val"], 60)
        self.assertEqual(meta["waveform_length"], WF_LEN)

        batch = next(iter(train_loader))
        self.assertIsInstance(batch["inputs"], torch.Tensor)
        self.assertEqual(batch["inputs"].shape[1:], (1, WF_LEN))
        self.assertTrue(batch["inputs"].is_floating_point())
        self.assertEqual(batch["energy"].ndim, 1)
        self.assertTrue(torch.isfinite(batch["energy"]).all())
        self.assertIn("sample_weight", batch)
        self.assertTrue(
            torch.allclose(
                batch["sample_weight"], torch.ones_like(batch["sample_weight"])
            )
        )
        self.assertTrue(set(batch["category"]) <= {"pileup", "clean"})

        clean_targets = [
            float(sample["energy"])
            for split in (train_loader, val_loader, test_loader)
            for sample in split.dataset
            if sample["category"] == "clean"
        ]
        self.assertTrue(clean_targets)
        self.assertTrue(all(value == 0.0 for value in clean_targets))

    def test_sample_weight_propagates(self) -> None:
        config = self._data_config(tailelevated_weight=0.3, zero_dt_weight=0.5)
        train_loader, _, test_loader, meta = prepare_cuore_regression_data(
            config, TrainingConfig(batch_size=64, num_workers=0)
        )
        weights = np.asarray(
            [float(sample["sample_weight"]) for sample in train_loader.dataset]
        )
        self.assertGreater(weights.min(), 0.0)
        self.assertLess(weights.min(), 1.0)
        self.assertGreater(
            meta["splits"]["train"]["sample_weight_stats"]["downweighted"], 0
        )
        # test weights stay unweighted
        test_weights = np.asarray(
            [float(sample["sample_weight"]) for sample in test_loader.dataset]
        )
        self.assertTrue(np.allclose(test_weights, 1.0))

    def test_raw_patches_divisibility_guard(self) -> None:
        config = self._data_config()
        _, _, _, _ = prepare_cuore_regression_data(
            config, TrainingConfig(batch_size=8, num_workers=0)
        )
        # 1000 % 32 != 0 -> the tokenizer must reject it
        model = self._model("raw_patches", "coordinate_mlp")
        model.tokenizer.patch_size = 32
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 1, WF_LEN))

    # ----------------------------------------------------------------- model

    def test_all_nine_cells_forward_and_backward(self) -> None:
        waveform = torch.rand(4, 1, WF_LEN)
        for tokenization in TOKENIZATIONS:
            for encoding in ENCODINGS:
                with self.subTest(tokenization=tokenization, encoding=encoding):
                    model = self._model(tokenization, encoding)
                    output = model(waveform)
                    self.assertEqual(tuple(output.shape), (4,))
                    self.assertTrue(torch.isfinite(output).all())

                    output.sum().backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(
                            parameter.grad, f"{name} has no gradient"
                        )
                        self.assertTrue(
                            torch.isfinite(parameter.grad).all(),
                            f"{name} has non-finite gradient",
                        )

    def test_mjd_parity_for_additive_encodings(self) -> None:
        """The 6 non-RoPE cells must be identical to MJDTransformer."""

        waveform = torch.rand(3, 1, WF_LEN)
        for tokenization in TOKENIZATIONS:
            for encoding in ("coordinate_mlp", "fourier_coordinates"):
                with self.subTest(tokenization=tokenization, encoding=encoding):
                    tokenization_config = _tokenization_config(tokenization)

                    set_seed(42, True)
                    ours = CuoreWaveformTransformer(
                        position_encoding=encoding,
                        tokenization_config=tokenization_config,
                        **TINY_MODEL,
                    ).eval()

                    set_seed(42, True)
                    theirs = MJDTransformer(
                        task="regression",
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
        model = self._model("segment_summary", "rope")
        self.assertIsNone(model.position_encoder)
        self.assertIsInstance(model.transformer, RotaryTransformerEncoder)
        self.assertFalse(
            any("position_encoder" in name for name, _ in model.named_parameters())
        )
        config = model.config_dict()
        self.assertEqual(config["attention"], "rotary")
        self.assertEqual(config["rope_base"], 8.0)
        self.assertIn("time", config["rope_thetas_per_axis"])

    def test_rope_time_axis_only_flag(self) -> None:
        model = self._model("segment_summary", "rope", **{})
        baseline = model(torch.rand(2, 1, WF_LEN))

        diagnostic = CuoreWaveformTransformer(
            position_encoding="rope",
            tokenization_config=_tokenization_config("segment_summary"),
            rope_base=8.0,
            rope_time_axis_only=True,
            **TINY_MODEL,
        )
        self.assertTrue(diagnostic.config_dict()["rope_time_axis_only"])
        output = diagnostic(torch.rand(2, 1, WF_LEN))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(baseline).all())

    def test_rope_config_guards(self) -> None:
        # head_dim = 24 // 8 = 3 -> odd and < 6
        with self.assertRaises(ValueError):
            self._model("segment_summary", "rope", nhead=8)
        with self.assertRaises(ValueError):
            self._model("segment_summary", "rope", **{})._validate_configuration(
                position_encoding="rope",
                d_model=24,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
                num_frequencies=6,
                rope_base=1.0,
            )
        with self.assertRaises(ValueError):
            self._model("segment_summary", "nonsense")

    # ------------------------------------------------------------ end to end

    def test_end_to_end_train_and_evaluate(self) -> None:
        config = self._data_config()
        train_loader, val_loader, test_loader, _ = prepare_cuore_regression_data(
            config, TrainingConfig(batch_size=8, num_workers=0)
        )
        model = self._model("segment_summary", "rope")

        history = train_model(
            model,
            train_loader,
            val_loader,
            TrainingConfig(
                batch_size=8,
                epochs=1,
                early_stopping_patience=1,
                use_amp=False,
                device="cpu",
                num_workers=0,
                deterministic=True,
            ),
            "regression",
            output_dir=self.root / "training",
        )
        self.assertEqual(history["epochs_completed"], 1)
        self.assertTrue(np.isfinite(history["best_metric"]))

        relaxed = EvaluationConfig(
            min_per_class=1,
            min_valid_bins=1,
            support_trim_quantile=0.0,
            min_coverage=0.0,
            score_bins=5,
            min_per_bin=1,
            performance_bins=3,
        )
        result = evaluate_cuore_regression(
            model,
            test_loader,
            output_dir=self.root / "evaluation",
            data_config=config,
            evaluation_config=relaxed,
            device="cpu",
        )

        for name in (
            "metrics.json",
            "predictions.npz",
            "cuore_regression_ms.json",
            "cuore_dt_scatter.png",
        ):
            self.assertTrue((self.root / "evaluation" / name).is_file(), name)

        report = result["ms"]
        self.assertTrue(np.isfinite(report["overall_ms"]["rmse_ms"]))
        self.assertTrue(np.isfinite(report["pileup_ms"]["rmse_ms"]))
        self.assertAlmostEqual(
            report["overall_ms"]["rmse_ms"],
            result["scaled"]["rmse"] * config.dt_scale,
            places=3,
        )
        # Pinned as expected behaviour, not a bug: every clean target is 0, so
        # the total sum of squares is 0 and R^2 is undefined.
        self.assertTrue(math.isnan(report["clean_ms"]["r2"]))


if __name__ == "__main__":
    unittest.main()
