from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from exobench import TrainingConfig, evaluate_model, train_model
from exo_transformer import (
    CoordinateMLPEncoding,
    EXOTransformerClassifier,
    FourierCoordinateEncoding,
    PulseEntityTokenizer,
    RawSensorPatchTokenizer,
    SensorRegionSummaryTokenizer,
    TokenizationConfig,
)


class _SyntheticEXO(Dataset):
    def __init__(self, events: int = 8) -> None:
        generator = torch.Generator().manual_seed(42)
        self.waveforms = torch.randn(events, 226, 300, generator=generator)
        self.labels = torch.arange(events, dtype=torch.int64) % 2
        self.waveforms[self.labels == 1, 0:38, 220:240] += 0.75

    def __len__(self) -> int:
        return int(self.waveforms.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.waveforms[index],
            "label": self.labels[index],
        }


def _small_model(tokenization: str, encoding: str) -> EXOTransformerClassifier:
    return EXOTransformerClassifier(
        tokenization_config=TokenizationConfig(
            tokenization=tokenization,
            channel_regions=1,
            time_regions=2,
            uniform_entity_fraction=0.5,
            entity_context_size=3,
        ),
        position_encoding=encoding,
        # RoPE needs at least one channel pair for each of six coordinates.
        d_model=24,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        num_frequencies=2,
    )


class EXOTransformerTests(unittest.TestCase):
    def test_official_rope_configuration_is_frozen(self) -> None:
        model = EXOTransformerClassifier(
            tokenization_config=TokenizationConfig(
                tokenization="segment_summary"
            ),
            position_encoding="rope",
        )
        config = model.config_dict()
        self.assertEqual(config["rotary_pair_counts"], [2, 2, 1, 1, 1, 1])
        self.assertAlmostEqual(config["rope_base"], math.pi / 2.0)
        self.assertIsNone(model.position_encoder)

    def test_official_tokenizer_shapes(self) -> None:
        waveform = torch.randn(2, 226, 300)
        expected_features = {
            RawSensorPatchTokenizer: 150,
            SensorRegionSummaryTokenizer: 4,
            PulseEntityTokenizer: 2,
        }
        for tokenizer_type, feature_dim in expected_features.items():
            tokenizer = tokenizer_type()
            output = tokenizer(waveform)
            self.assertEqual(tuple(output["coords"].shape), (2, 504, 6))
            self.assertEqual(
                tuple(output["features"].shape),
                (2, 504, feature_dim),
            )
            self.assertEqual(tuple(output["mask"].shape), (2, 504))
            self.assertEqual(output["mask"].dtype, torch.bool)
            self.assertTrue(bool(output["mask"].all().item()))

    def test_raw_patch_preserves_real_values_and_pads_region_width(self) -> None:
        waveform = torch.arange(226 * 300, dtype=torch.float32).reshape(1, 226, 300)
        output = RawSensorPatchTokenizer()(waveform)
        first_expected = waveform[0, 0:6, 0:25].reshape(-1)
        torch.testing.assert_close(output["features"][0, 0], first_expected)

        # Positive-U region 3 is the first five-channel region. Its first time
        # token follows 3 * 12 earlier tokens and has one padded zero channel.
        token_index = 3 * 12
        expected_real = waveform[0, 18:23, 0:25].reshape(-1)
        torch.testing.assert_close(
            output["features"][0, token_index, : expected_real.numel()],
            expected_real,
        )
        torch.testing.assert_close(
            output["features"][0, token_index, expected_real.numel() :],
            torch.zeros(25),
        )

    def test_summary_excludes_internal_padding(self) -> None:
        waveform = torch.ones(1, 226, 300)
        output = SensorRegionSummaryTokenizer()(waveform)
        self.assertTrue(
            bool(torch.allclose(output["features"][..., 0], torch.ones(1, 504)))
        )
        self.assertTrue(
            bool(torch.allclose(output["features"][..., 1], torch.ones(1, 504)))
        )
        self.assertTrue(
            bool(torch.allclose(output["features"][..., 2], torch.ones(1, 504)))
        )
        self.assertTrue(
            bool(torch.allclose(output["features"][..., 3], torch.zeros(1, 504)))
        )

    def test_sensor_coordinates_encode_side_and_family(self) -> None:
        output = SensorRegionSummaryTokenizer()(torch.zeros(1, 226, 300))
        coordinates = output["coords"][0]
        block_size = 84
        expected = (
            (1.0, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 1.0, 0.0)),
            (-1.0, (1.0, 0.0, 0.0)),
            (-1.0, (0.0, 1.0, 0.0)),
            (1.0, (0.0, 0.0, 1.0)),
            (-1.0, (0.0, 0.0, 1.0)),
        )
        for block_index, (side, family) in enumerate(expected):
            block = coordinates[
                block_index * block_size : (block_index + 1) * block_size
            ]
            torch.testing.assert_close(
                block[:, 2],
                torch.full((block_size,), side),
            )
            torch.testing.assert_close(
                block[:, 3:],
                torch.tensor(family).expand(block_size, -1),
            )

    def test_both_position_encoders_embed_all_504_tokens(self) -> None:
        coordinates = SensorRegionSummaryTokenizer()(
            torch.zeros(2, 226, 300)
        )["coords"]
        for encoder in (
            CoordinateMLPEncoding(coordinate_dim=6, d_model=8),
            FourierCoordinateEncoding(
                coordinate_dim=6,
                d_model=8,
                num_frequencies=2,
            ),
        ):
            output = encoder(coordinates)
            self.assertEqual(tuple(output.shape), (2, 504, 8))
            self.assertTrue(bool(torch.isfinite(output).all().item()))

    def test_pulse_entities_are_deterministic_and_block_balanced(self) -> None:
        waveform = torch.zeros(2, 226, 300)
        waveform[0, 10, 140] = 5.0
        waveform[1, 180, 220] = -4.0
        tokenizer = PulseEntityTokenizer()
        first = tokenizer(waveform)
        second = tokenizer(waveform)
        for name in ("coords", "features", "mask"):
            torch.testing.assert_close(first[name], second[name])
        self.assertTrue(bool((first["features"][0, :, 0] == 5.0).any().item()))
        self.assertTrue(bool((first["features"][1, :, 0] == -4.0).any().item()))
        self.assertEqual(tuple(first["features"].shape), (2, 504, 2))
        for event in range(2):
            for block_index in range(6):
                block = first["coords"][
                    event,
                    block_index * 84 : (block_index + 1) * 84,
                ]
                locations = block[:, :2]
                self.assertEqual(torch.unique(locations, dim=0).shape[0], 84)

    def test_all_nine_model_combinations(self) -> None:
        waveform = torch.randn(1, 226, 300)
        for tokenization in ("raw_patches", "segment_summary", "pulse_entities"):
            for encoding in ("coordinate_mlp", "fourier_coordinates", "rope"):
                model = _small_model(tokenization, encoding)
                output = model(waveform)
                self.assertEqual(tuple(output.shape), (1,))
                self.assertTrue(bool(torch.isfinite(output).all().item()))
                output.sum().backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )

    def test_invalid_inputs_and_configuration(self) -> None:
        tokenizer = SensorRegionSummaryTokenizer()
        with self.assertRaises(ValueError):
            tokenizer(torch.zeros(1, 225, 300))
        with self.assertRaises(TypeError):
            tokenizer(torch.zeros(1, 226, 300, dtype=torch.int16))
        invalid = torch.zeros(1, 226, 300)
        invalid[0, 0, 0] = torch.nan
        with self.assertRaises(ValueError):
            tokenizer(invalid)
        with self.assertRaises(ValueError):
            TokenizationConfig(time_regions=7)

    def test_shared_training_and_evaluation_contract(self) -> None:
        data = _SyntheticEXO()
        train_loader = DataLoader(data, batch_size=4, shuffle=True)
        validation_loader = DataLoader(data, batch_size=4, shuffle=False)
        test_loader = DataLoader(data, batch_size=4, shuffle=False)
        # Exercise the complete EXOBench checkpoint/evaluation contract on
        # the new attention-internal positional-encoding path.
        model = _small_model("segment_summary", "rope")
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
                config=config,
                output_dir=output,
            )
            metrics = evaluate_model(
                model,
                test_loader,
                device="cpu",
                output_dir=output,
            )
            self.assertEqual(len(history), 1)
            self.assertEqual(metrics["events"], len(data))
            for name in ("best.pt", "history.json", "metrics.json", "predictions.npz"):
                self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
