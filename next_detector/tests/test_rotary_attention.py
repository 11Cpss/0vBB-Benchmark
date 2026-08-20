"""Unit tests for rotary positional encoding (RoPE), in isolation.

These tests exercise the rotation math and the custom attention/encoder
modules directly, without touching HDF5 data, the tokenizer, or the
full NEXTTransformerClassifier. They are meant to be the cheapest,
fastest layer of confidence before any real training run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (
    PROJECT_ROOT / "next_detector",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from next_transformer.rotary_attention import (  # noqa: E402
    RotaryPositionAngles,
    RotarySelfAttention,
    RotaryTransformerEncoder,
    _axis_pair_counts,
    apply_rotary_embedding,
    rotate_half,
)


class RotationMathTests(unittest.TestCase):
    """Pure math checks, run in float64 for tight tolerances."""

    def test_axis_pair_counts_are_balanced_and_sum_correctly(self) -> None:
        for pairs_total, expected in (
            (3, [1, 1, 1]),
            (8, [3, 3, 2]),
            (9, [3, 3, 3]),
            (15, [5, 5, 5]),
        ):
            counts = _axis_pair_counts(pairs_total)
            self.assertEqual(sum(counts), pairs_total)
            self.assertEqual(counts, expected)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_rotate_half_matches_manual_definition(self) -> None:
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]],
            dtype=torch.float64,
        )
        expected = torch.tensor(
            [[-3.0, -4.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(rotate_half(x), expected)

    def test_apply_rotary_embedding_preserves_norm(self) -> None:
        generator = torch.Generator().manual_seed(0)
        head_dim = 8
        half_dim = head_dim // 2

        x = torch.randn(
            4, 5, head_dim, generator=generator, dtype=torch.float64
        )

        # apply_rotary_embedding only forms a valid 2D rotation when
        # cos/sin repeat the same angle in both halves of head_dim
        # (this is what RotaryPositionAngles.forward always produces).
        angle_half = torch.rand(
            4, 5, half_dim, generator=generator, dtype=torch.float64
        ) * 6.0
        angle = torch.cat([angle_half, angle_half], dim=-1)

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        rotated = apply_rotary_embedding(x, cos, sin)

        torch.testing.assert_close(
            rotated.norm(dim=-1),
            x.norm(dim=-1),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_same_position_reduces_to_unrotated_dot_product(self) -> None:
        generator = torch.Generator().manual_seed(1)
        head_dim = 8
        half_dim = head_dim // 2

        q = torch.randn(
            2, head_dim, generator=generator, dtype=torch.float64
        )
        k = torch.randn(
            2, head_dim, generator=generator, dtype=torch.float64
        )

        # Same requirement as above: a valid rotation needs cos/sin
        # duplicated across both halves, not independent per channel.
        angle_half = torch.rand(
            2, half_dim, generator=generator, dtype=torch.float64
        ) * 6.0
        angle = torch.cat([angle_half, angle_half], dim=-1)

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        rotated_q = apply_rotary_embedding(q, cos, sin)
        rotated_k = apply_rotary_embedding(k, cos, sin)

        rotated_dot = (rotated_q * rotated_k).sum(dim=-1)
        plain_dot = (q * k).sum(dim=-1)

        torch.testing.assert_close(
            rotated_dot,
            plain_dot,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_translation_invariance_of_the_attention_score(self) -> None:
        """dot(rotate(q,p_m), rotate(k,p_n)) depends only on p_m - p_n."""

        rotary_angles = RotaryPositionAngles(
            head_dim=8,
            rope_base=10.0,
        ).double()

        generator = torch.Generator().manual_seed(2)
        batch, tokens = 3, 2

        q = torch.randn(
            batch, tokens, 8, generator=generator, dtype=torch.float64
        )
        k = torch.randn(
            batch, tokens, 8, generator=generator, dtype=torch.float64
        )
        coords = torch.randn(
            batch, tokens, 3, generator=generator, dtype=torch.float64
        )
        shift = torch.randn(
            batch, 1, 3, generator=generator, dtype=torch.float64
        )

        def score(coordinates: torch.Tensor) -> torch.Tensor:
            cos, sin = rotary_angles(coordinates)
            rotated_q = apply_rotary_embedding(q, cos, sin)
            rotated_k = apply_rotary_embedding(k, cos, sin)
            return (rotated_q * rotated_k).sum(dim=-1)

        original_score = score(coords)
        shifted_score = score(coords + shift)

        torch.testing.assert_close(
            original_score,
            shifted_score,
            atol=1e-9,
            rtol=1e-9,
        )

    def test_gradient_flows_through_rotation(self) -> None:
        head_dim = 6
        x = torch.randn(
            2, 3, head_dim, dtype=torch.float64, requires_grad=True
        )
        angle = torch.rand(2, 3, head_dim, dtype=torch.float64)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda tensor: apply_rotary_embedding(tensor, cos, sin),
                (x,),
            )
        )

    def test_angles_are_deterministic_across_calls(self) -> None:
        rotary_angles = RotaryPositionAngles(head_dim=8, rope_base=10.0)
        coordinates = torch.randn(2, 4, 3)

        cos_first, sin_first = rotary_angles(coordinates)
        cos_second, sin_second = rotary_angles(coordinates)

        torch.testing.assert_close(cos_first, cos_second)
        torch.testing.assert_close(sin_first, sin_second)

    def test_rejects_head_dim_below_minimum(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 6"):
            RotaryPositionAngles(head_dim=4, rope_base=10.0)

    def test_rejects_odd_head_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            RotaryPositionAngles(head_dim=7, rope_base=10.0)

    def test_rejects_non_positive_rope_base(self) -> None:
        with self.assertRaises(ValueError):
            RotaryPositionAngles(head_dim=8, rope_base=1.0)


class RotarySelfAttentionTests(unittest.TestCase):
    def test_forward_shape_and_finiteness(self) -> None:
        attention = RotarySelfAttention(
            d_model=24, nhead=4, dropout=0.0, rope_base=10.0
        )
        x = torch.randn(3, 6, 24)
        coords = torch.randn(3, 6, 3)

        output = attention(x, coords)

        self.assertEqual(tuple(output.shape), (3, 6, 24))
        self.assertTrue(torch.isfinite(output).all())

    def test_masking_ignores_padded_key_content(self) -> None:
        attention = RotarySelfAttention(
            d_model=24, nhead=4, dropout=0.0, rope_base=10.0
        )
        attention.eval()

        torch.manual_seed(0)
        x = torch.randn(1, 4, 24)
        coords = torch.randn(1, 4, 3)
        mask = torch.tensor([[False, False, True, True]])

        x_variant = x.clone()
        x_variant[0, 2:] = torch.randn(2, 24)
        coords_variant = coords.clone()
        coords_variant[0, 2:] = torch.randn(2, 3)

        with torch.no_grad():
            output = attention(x, coords, key_padding_mask=mask)
            output_variant = attention(
                x_variant, coords_variant, key_padding_mask=mask
            )

        torch.testing.assert_close(
            output[0, :2],
            output_variant[0, :2],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_output_changes_when_coordinates_change(self) -> None:
        attention = RotarySelfAttention(
            d_model=24, nhead=4, dropout=0.0, rope_base=10.0
        )
        attention.eval()

        torch.manual_seed(0)
        x = torch.randn(2, 5, 24)
        coords_a = torch.randn(2, 5, 3)
        coords_b = torch.randn(2, 5, 3)

        with torch.no_grad():
            output_a = attention(x, coords_a)
            output_b = attention(x, coords_b)

        self.assertFalse(torch.allclose(output_a, output_b))

    def test_parameter_count_matches_multihead_attention(self) -> None:
        d_model, nhead = 24, 4

        custom = RotarySelfAttention(d_model=d_model, nhead=nhead)
        builtin = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        custom_params = sum(p.numel() for p in custom.parameters())
        builtin_params = sum(p.numel() for p in builtin.parameters())

        self.assertEqual(custom_params, builtin_params)

    def test_rejects_d_model_not_divisible_by_nhead(self) -> None:
        with self.assertRaises(ValueError):
            RotarySelfAttention(d_model=17, nhead=4)


class RotaryTransformerEncoderTests(unittest.TestCase):
    def test_multi_layer_stack_shape_and_finiteness(self) -> None:
        encoder = RotaryTransformerEncoder(
            d_model=24,
            nhead=4,
            num_layers=2,
            dim_feedforward=32,
            dropout=0.0,
            rope_base=10.0,
        )
        x = torch.randn(2, 7, 24)
        coords = torch.randn(2, 7, 3)
        mask = torch.zeros(2, 7, dtype=torch.bool)

        output = encoder(x, coords, key_padding_mask=mask)

        self.assertEqual(tuple(output.shape), (2, 7, 24))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
