from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from exo_transformer import (
    RotaryPositionAngles,
    RotarySelfAttention,
    apply_rotary_embedding,
    axis_pair_counts,
    rotate_half,
)


class EXORotaryMathTests(unittest.TestCase):
    def test_official_pair_assignment(self) -> None:
        self.assertEqual(axis_pair_counts(8, 6), (2, 2, 1, 1, 1, 1))

    def test_every_coordinate_controls_an_angle(self) -> None:
        angles = RotaryPositionAngles(
            head_dim=16,
            coordinate_dim=6,
            rope_base=math.pi / 2.0,
        )
        zero = torch.zeros(1, 1, 6)
        base_cosine, base_sine = angles(zero)
        for axis in range(6):
            coordinates = zero.clone()
            coordinates[..., axis] = 1.0
            cosine, sine = angles(coordinates)
            self.assertFalse(
                torch.equal(cosine, base_cosine) and torch.equal(sine, base_sine)
            )

    def test_zero_coordinates_are_identity(self) -> None:
        angles = RotaryPositionAngles(head_dim=16, coordinate_dim=6)
        cosine, sine = angles(torch.zeros(2, 5, 6))
        value = torch.randn(2, 5, 16)
        torch.testing.assert_close(
            apply_rotary_embedding(value, cosine, sine), value
        )

    def test_rotation_preserves_norm(self) -> None:
        angles = RotaryPositionAngles(head_dim=16, coordinate_dim=6).double()
        value = torch.randn(2, 5, 16, dtype=torch.float64)
        coordinates = torch.randn(2, 5, 6, dtype=torch.float64)
        cosine, sine = angles(coordinates)
        rotated = apply_rotary_embedding(value, cosine, sine)
        torch.testing.assert_close(
            rotated.norm(dim=-1), value.norm(dim=-1), atol=1e-10, rtol=1e-10
        )

    def test_common_translation_preserves_pairwise_score(self) -> None:
        angles = RotaryPositionAngles(head_dim=16, coordinate_dim=6).double()
        query = torch.randn(3, 2, 16, dtype=torch.float64)
        key = torch.randn(3, 2, 16, dtype=torch.float64)
        coordinates = torch.randn(3, 2, 6, dtype=torch.float64)
        shift = torch.randn(3, 1, 6, dtype=torch.float64)

        def pairwise_score(position: torch.Tensor) -> torch.Tensor:
            cosine, sine = angles(position)
            rotated_query = apply_rotary_embedding(query, cosine, sine)
            rotated_key = apply_rotary_embedding(key, cosine, sine)
            return torch.matmul(rotated_query, rotated_key.transpose(-2, -1))

        torch.testing.assert_close(
            pairwise_score(coordinates),
            pairwise_score(coordinates + shift),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_rotate_half_definition(self) -> None:
        value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        torch.testing.assert_close(rotate_half(value), expected)


class EXORotaryAttentionTests(unittest.TestCase):
    @staticmethod
    def _copy_multihead_weights(
        source: nn.MultiheadAttention,
        destination: RotarySelfAttention,
    ) -> None:
        with torch.no_grad():
            query, key, value = source.in_proj_weight.chunk(3, dim=0)
            query_bias, key_bias, value_bias = source.in_proj_bias.chunk(3, dim=0)
            destination.q_proj.weight.copy_(query)
            destination.k_proj.weight.copy_(key)
            destination.v_proj.weight.copy_(value)
            destination.q_proj.bias.copy_(query_bias)
            destination.k_proj.bias.copy_(key_bias)
            destination.v_proj.bias.copy_(value_bias)
            destination.out_proj.weight.copy_(source.out_proj.weight)
            destination.out_proj.bias.copy_(source.out_proj.bias)

    def test_zero_coordinate_attention_matches_pytorch(self) -> None:
        torch.manual_seed(42)
        builtin = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)
        rotary = RotarySelfAttention(
            d_model=64,
            nhead=4,
            coordinate_dim=6,
            dropout=0.0,
        )
        self._copy_multihead_weights(builtin, rotary)
        value = torch.randn(2, 7, 64)
        coordinates = torch.zeros(2, 7, 6)
        mask = torch.tensor(
            [[False, False, False, False, False, True, True]] * 2
        )
        expected, _ = builtin(
            value, value, value, key_padding_mask=mask, need_weights=False
        )
        actual = rotary(value, coordinates, key_padding_mask=mask)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_masked_tokens_cannot_change_valid_outputs(self) -> None:
        torch.manual_seed(7)
        attention = RotarySelfAttention(
            d_model=64,
            nhead=4,
            coordinate_dim=6,
            dropout=0.0,
        ).eval()
        value = torch.randn(1, 5, 64)
        coordinates = torch.randn(1, 5, 6)
        mask = torch.tensor([[False, False, False, True, True]])
        changed_value = value.clone()
        changed_coordinates = coordinates.clone()
        changed_value[:, 3:] = 1.0e4
        changed_coordinates[:, 3:] = -1.0e4
        with torch.no_grad():
            original = attention(value, coordinates, mask)
            changed = attention(changed_value, changed_coordinates, mask)
        torch.testing.assert_close(
            original[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6
        )

    def test_parameter_count_matches_multihead_attention(self) -> None:
        rotary = RotarySelfAttention(d_model=64, nhead=4, coordinate_dim=6)
        builtin = nn.MultiheadAttention(64, 4, batch_first=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in rotary.parameters()),
            sum(parameter.numel() for parameter in builtin.parameters()),
        )

    def test_rejects_too_few_pairs_for_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "each coordinate"):
            RotaryPositionAngles(head_dim=8, coordinate_dim=6)


if __name__ == "__main__":
    unittest.main()
