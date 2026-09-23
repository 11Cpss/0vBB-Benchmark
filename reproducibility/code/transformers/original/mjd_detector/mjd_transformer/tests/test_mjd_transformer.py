from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    # Name used on the lab computer.
    from mjdbench import TrainingConfig, evaluate_model, train_model
except ModuleNotFoundError:
    # Name of the read-only local audit copy.
    from mjd_partner_workflow import TrainingConfig, evaluate_model, train_model
from mjd_transformer import (
    MJDTransformer,
    PulseEntityTokenizer,
    RawPatchTokenizer,
    SegmentSummaryTokenizer,
    TokenizationConfig,
)


class _SyntheticMJD(Dataset):
    def __init__(self, events: int = 12, samples: int = 32) -> None:
        generator = torch.Generator().manual_seed(42)
        waveform = torch.randn(events, 1, samples, generator=generator)
        clean = torch.arange(events) % 2 == 0
        waveform[clean, :, samples // 2 :] += 0.75
        self.waveform = waveform
        self.clean = clean
        self.energy = waveform.abs().sum(dim=(1, 2))

    def __len__(self) -> int:
        return int(self.waveform.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.waveform[index],
            "clean": self.clean[index],
            "energy": self.energy[index],
        }


def _small_model(
    task: str,
    position_encoding: str,
    tokenization: str = "segment_summary",
) -> MJDTransformer:
    token_config = {
        "segment_summary": TokenizationConfig(
            tokenization="segment_summary",
            token_count=8,
        ),
        "raw_patches": TokenizationConfig(
            tokenization="raw_patches",
            patch_size=4,
        ),
        "pulse_entities": TokenizationConfig(
            tokenization="pulse_entities",
            token_count=8,
            uniform_entity_fraction=0.5,
            entity_context_size=3,
        ),
    }[tokenization]
    return MJDTransformer(
        task=task,
        tokenization_config=token_config,
        position_encoding=position_encoding,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_frequencies=2,
    )


class MJDTransformerTests(unittest.TestCase):
    def test_segment_tokenizer_shapes_and_determinism(self) -> None:
        waveform = torch.linspace(-1.0, 1.0, 64).reshape(2, 1, 32)
        tokenizer = SegmentSummaryTokenizer(token_count=8)
        first = tokenizer(waveform)
        second = tokenizer(waveform)
        self.assertEqual(tuple(first["coords"].shape), (2, 8, 3))
        self.assertEqual(tuple(first["features"].shape), (2, 8, 2))
        self.assertEqual(tuple(first["mask"].shape), (2, 8))
        self.assertEqual(first["mask"].dtype, torch.bool)
        for name in ("coords", "features", "mask"):
            torch.testing.assert_close(first[name], second[name])

    def test_raw_patch_tokenizer_preserves_samples(self) -> None:
        waveform = torch.linspace(-1.0, 1.0, 64).reshape(2, 1, 32)
        tokenizer = RawPatchTokenizer(patch_size=4)
        output = tokenizer(waveform)
        self.assertEqual(tuple(output["coords"].shape), (2, 8, 3))
        self.assertEqual(tuple(output["features"].shape), (2, 8, 4))
        self.assertEqual(tuple(output["mask"].shape), (2, 8))
        expected = waveform.squeeze(1).reshape(2, 8, 4)
        torch.testing.assert_close(output["features"], expected)

    def test_pulse_entity_tokenizer_is_deterministic(self) -> None:
        waveform = torch.zeros(2, 1, 32)
        waveform[0, 0, 13] = 5.0
        waveform[1, 0, 22] = -4.0
        tokenizer = PulseEntityTokenizer(
            token_count=8,
            uniform_fraction=0.5,
            context_size=3,
        )
        first = tokenizer(waveform)
        second = tokenizer(waveform)
        self.assertEqual(tuple(first["coords"].shape), (2, 8, 3))
        self.assertEqual(tuple(first["features"].shape), (2, 8, 2))
        self.assertEqual(tuple(first["mask"].shape), (2, 8))
        for name in ("coords", "features", "mask"):
            torch.testing.assert_close(first[name], second[name])
        self.assertTrue(bool((first["features"][0, :, 0] == 5.0).any().item()))
        self.assertTrue(bool((first["features"][1, :, 0] == -4.0).any().item()))

    def test_both_tasks_and_position_encodings(self) -> None:
        waveform = torch.randn(3, 1, 32)
        for tokenization in (
            "segment_summary",
            "raw_patches",
            "pulse_entities",
        ):
            for task in ("classification", "regression"):
                for encoding in ("coordinate_mlp", "fourier_coordinates"):
                    model = _small_model(task, encoding, tokenization)
                    output = model(waveform)
                    self.assertEqual(tuple(output.shape), (3,))
                    self.assertTrue(bool(torch.isfinite(output).all().item()))
                    output.sum().backward()
                    self.assertTrue(
                        any(
                            parameter.grad is not None
                            for parameter in model.parameters()
                        )
                    )

    def test_shared_training_and_evaluation_contract(self) -> None:
        data = _SyntheticMJD()
        train_loader = DataLoader(data, batch_size=4, shuffle=True)
        validation_loader = DataLoader(data, batch_size=4, shuffle=False)
        test_loader = DataLoader(data, batch_size=4, shuffle=False)
        model = _small_model("classification", "coordinate_mlp")
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
            self.assertTrue((output / "history.json").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "predictions.npz").is_file())


if __name__ == "__main__":
    unittest.main()
