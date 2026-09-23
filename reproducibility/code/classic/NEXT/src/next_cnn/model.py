"""CNN architectures for NEXT orthographic projections."""

from __future__ import annotations

import math
from typing import Any, Dict

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised through the CLI message
    raise RuntimeError(
        "NEXT CNN requires GPU PyTorch; install "
        "`requirements/next-cnn-cu128.txt` in the GPU environment"
    ) from exc


class SimpleNextCNN(nn.Module):
    """Classify a three-channel ``XY/XZ/YZ`` event image with one logit."""

    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        if int(base_channels) != base_channels or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        channels = int(base_channels)
        self.base_channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.classifier = nn.Linear(channels * 2, 1)

    def forward(self, images: "torch.Tensor") -> "torch.Tensor":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        features = self.features(images).flatten(start_dim=1)
        return self.classifier(features).squeeze(1)

    def config_dict(self) -> Dict[str, Any]:
        return {"base_channels": self.base_channels}


class SimpleNextEnergyRegressor(nn.Module):
    """Regress one standardized energy value from the v1 CNN features."""

    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        if int(base_channels) != base_channels or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        channels = int(base_channels)
        self.base_channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.regressor = nn.Linear(channels * 2, 1)

    def forward(self, images: "torch.Tensor") -> "torch.Tensor":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        features = self.features(images).flatten(start_dim=1)
        return self.regressor(features).squeeze(1)

    def config_dict(self) -> Dict[str, Any]:
        return {"base_channels": self.base_channels}


class GlobalEnergySkipCNN(nn.Module):
    """Fuse CNN topology with an explicit, energy-preserving global skip.

    ``global_center`` and ``global_scale`` define the standardized per-view
    image sums. The output is their mean plus a learned residual correction.
    Classification uses normalized images, so this skip is zero at full
    projection coverage; regression uses training-only energy statistics, so
    it is a physical energy baseline that the CNN can correct when coverage is
    incomplete.
    """

    def __init__(
        self,
        base_channels: int = 8,
        input_scale: float = 100.0,
        global_center: float = 1.0,
        global_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if int(base_channels) != base_channels or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        values = (float(input_scale), float(global_center), float(global_scale))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("global energy configuration must be finite")
        if input_scale <= 0.0:
            raise ValueError("input_scale must be positive")
        if global_scale <= 0.0:
            raise ValueError("global_scale must be positive")

        channels = int(base_channels)
        self.base_channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.correction_head = nn.Linear(channels * 2 + 3, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        self.register_buffer(
            "global_input_scale",
            torch.tensor(float(input_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "global_center",
            torch.tensor(float(global_center), dtype=torch.float32),
        )
        self.register_buffer(
            "global_scale",
            torch.tensor(float(global_scale), dtype=torch.float32),
        )

    def global_energy_features(
        self,
        images: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return three standardized view sums, always accumulated in FP32."""

        view_sums = images.float().sum(dim=(-2, -1), dtype=torch.float32)
        view_energies = view_sums / self.global_input_scale
        return (view_energies - self.global_center) / self.global_scale

    def forward(self, images: "torch.Tensor") -> "torch.Tensor":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        topology = self.features(images).flatten(start_dim=1).float()
        global_energy = self.global_energy_features(images)
        fused = torch.cat((topology, global_energy), dim=1)
        correction = self.correction_head(fused).squeeze(1)
        return global_energy.mean(dim=1) + correction

    def config_dict(self) -> Dict[str, Any]:
        return {
            "base_channels": self.base_channels,
            "input_scale": float(self.global_input_scale.detach().cpu()),
            "global_center": float(self.global_center.detach().cpu()),
            "global_scale": float(self.global_scale.detach().cpu()),
        }


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides ``channels``."""

    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an optional projected residual path."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            _group_count(output_channels),
            output_channels,
        )
        self.activation = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            _group_count(output_channels),
            output_channels,
        )
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(output_channels),
                    output_channels,
                ),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        residual = self.skip(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class ResidualSpatialNextCNN(nn.Module):
    """Deep residual CNN that retains spatial and global track geometry.

    The same scalar-output architecture is used for CNN-003 classification and
    standardized energy regression.  Unlike the two-convolution baselines, it
    keeps a 4x4 feature grid and explicitly exposes per-view mass, centroid,
    and spatial-scale features to the output head.
    """

    GLOBAL_FEATURES = 15
    INPUT_SIZE = 128
    ENCODED_SIZE = 16

    def __init__(
        self,
        base_channels: int = 16,
        pooled_size: int = 4,
        head_features: int = 256,
    ) -> None:
        super().__init__()
        integer_values = {
            "base_channels": base_channels,
            "pooled_size": pooled_size,
            "head_features": head_features,
        }
        for name, value in integer_values.items():
            if int(value) != value or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        if self.ENCODED_SIZE % int(pooled_size) != 0:
            raise ValueError(
                "pooled_size must be a divisor of %d" % self.ENCODED_SIZE
            )

        channels = int(base_channels)
        self.base_channels = channels
        self.pooled_size = int(pooled_size)
        self.head_features = int(head_features)
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            _ResidualBlock(channels, channels),
            _ResidualBlock(channels, channels),
            _ResidualBlock(channels, channels * 2, stride=2),
            _ResidualBlock(channels * 2, channels * 2),
            _ResidualBlock(channels * 2, channels * 4, stride=2),
            _ResidualBlock(channels * 4, channels * 4),
            _ResidualBlock(channels * 4, channels * 8, stride=2),
            _ResidualBlock(channels * 8, channels * 8),
        )
        pool_factor = self.ENCODED_SIZE // self.pooled_size
        self.spatial_pool = nn.AvgPool2d(
            kernel_size=pool_factor,
            stride=pool_factor,
        )
        flattened_features = (
            channels * 8 * self.pooled_size * self.pooled_size
        )
        self.head = nn.Sequential(
            nn.Linear(
                flattened_features + self.GLOBAL_FEATURES,
                self.head_features,
            ),
            nn.SiLU(inplace=True),
            nn.Linear(self.head_features, 1),
        )

    @staticmethod
    def global_geometry_features(images: "torch.Tensor") -> "torch.Tensor":
        """Return stable per-view mass, centroid, and spatial-scale features."""

        weights = images.float().clamp_min(0.0)
        height, width = images.shape[-2:]
        mass = weights.sum(dim=(-2, -1), dtype=torch.float32)
        safe_mass = mass.clamp_min(torch.finfo(torch.float32).eps)
        x_axis = torch.linspace(
            -1.0,
            1.0,
            width,
            device=images.device,
            dtype=torch.float32,
        ).view(1, 1, 1, width)
        y_axis = torch.linspace(
            -1.0,
            1.0,
            height,
            device=images.device,
            dtype=torch.float32,
        ).view(1, 1, height, 1)
        center_x = (weights * x_axis).sum(dim=(-2, -1)) / safe_mass
        center_y = (weights * y_axis).sum(dim=(-2, -1)) / safe_mass
        variance_x = (
            weights * (x_axis - center_x[:, :, None, None]).square()
        ).sum(dim=(-2, -1)) / safe_mass
        variance_y = (
            weights * (y_axis - center_y[:, :, None, None]).square()
        ).sum(dim=(-2, -1)) / safe_mass
        mass_scale = math.log1p(float(height * width))
        log_mass = torch.log1p(mass) / mass_scale
        return torch.cat(
            (log_mass, center_x, center_y, variance_x, variance_y),
            dim=1,
        )

    def forward(self, images: "torch.Tensor") -> "torch.Tensor":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        if images.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
            raise ValueError(
                "ResidualSpatialNextCNN requires 128x128 input images"
            )
        spatial = self.features(self.stem(images))
        spatial = self.spatial_pool(spatial).flatten(start_dim=1).float()
        global_geometry = self.global_geometry_features(images)
        fused = torch.cat((spatial, global_geometry), dim=1)
        return self.head(fused).squeeze(1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "base_channels": self.base_channels,
            "pooled_size": self.pooled_size,
            "head_features": self.head_features,
        }


class ResidualSpatialEnergyRegressor(ResidualSpatialNextCNN):
    """Regress deposited energy with a residual CNN and physical-energy skip.

    The convolutional core intentionally matches :class:`ResidualSpatialNextCNN`,
    but the task contract is regression-only.  Inputs are unnormalised,
    energy-weighted projections.  Their mean per-view sum is an
    energy-preserving baseline; the CNN head learns only a standardized
    residual for projection loss and detector-dependent spatial effects.
    """

    def __init__(
        self,
        base_channels: int = 4,
        pooled_size: int = 1,
        head_features: int = 32,
        input_scale: float = 100.0,
        energy_mean: float = 0.0,
        energy_std: float = 1.0,
    ) -> None:
        values = (float(input_scale), float(energy_mean), float(energy_std))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("energy-regression configuration must be finite")
        if input_scale <= 0.0:
            raise ValueError("input_scale must be positive")
        if energy_std <= 0.0:
            raise ValueError("energy_std must be positive")
        super().__init__(
            base_channels=base_channels,
            pooled_size=pooled_size,
            head_features=head_features,
        )
        # Keep regression checkpoint keys and semantics distinct from the
        # classification model even though both use the same CNN core.
        self.regressor = self.head
        del self.head
        self.input_scale = float(input_scale)
        self.register_buffer(
            "energy_mean", torch.tensor(float(energy_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "energy_std", torch.tensor(float(energy_std), dtype=torch.float32)
        )

    def observed_energy(self, images: "torch.Tensor") -> "torch.Tensor":
        """Return the energy-preserving mean of the three projection sums."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        per_view = images.float().sum(dim=(-2, -1)) / self.input_scale
        return per_view.mean(dim=1)

    def forward(self, images: "torch.Tensor") -> "torch.Tensor":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        if images.shape[-2:] != (self.INPUT_SIZE, self.INPUT_SIZE):
            raise ValueError(
                "ResidualSpatialEnergyRegressor requires 128x128 input images"
            )
        spatial = self.features(self.stem(images))
        spatial = self.spatial_pool(spatial).flatten(start_dim=1).float()
        global_geometry = self.global_geometry_features(images)
        fused = torch.cat((spatial, global_geometry), dim=1)
        residual = self.regressor(fused).squeeze(1)
        baseline = (self.observed_energy(images) - self.energy_mean) / self.energy_std
        return baseline + residual

    def config_dict(self) -> Dict[str, Any]:
        return {
            "base_channels": self.base_channels,
            "pooled_size": self.pooled_size,
            "head_features": self.head_features,
            "input_scale": self.input_scale,
            "energy_mean": float(self.energy_mean.detach().cpu()),
            "energy_std": float(self.energy_std.detach().cpu()),
        }


class MultiTaskNextCNN(nn.Module):
    """Joint NEXT classifier and summed-deposited-energy regressor."""

    def __init__(
        self,
        base_channels: int = 8,
        energy_mean: float = 2.45,
        energy_std: float = 0.01,
    ) -> None:
        super().__init__()
        if int(base_channels) != base_channels or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        if not float(energy_std) > 0.0:
            raise ValueError("energy_std must be positive")
        channels = int(base_channels)
        self.base_channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.classifier = nn.Linear(channels * 2, 1)
        self.energy_regressor = nn.Linear(channels * 2, 1)
        self.register_buffer(
            "energy_mean",
            torch.tensor(float(energy_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "energy_std",
            torch.tensor(float(energy_std), dtype=torch.float32),
        )

    def encode(self, images: "torch.Tensor") -> "torch.Tensor":
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        return self.features(images).flatten(start_dim=1)

    def forward(
        self,
        images: "torch.Tensor",
        return_parts: bool = False,
    ) -> Any:
        features = self.encode(images)
        logits = self.classifier(features).squeeze(1)
        energy_standardized = self.energy_regressor(features).squeeze(1)
        energy_pred = self.energy_mean + self.energy_std * energy_standardized
        if return_parts:
            return logits, {
                "energy_pred": energy_pred,
                "energy_standardized": energy_standardized,
                "features": features,
            }
        return logits

    def config_dict(self) -> Dict[str, Any]:
        return {
            "base_channels": self.base_channels,
            "energy_mean": float(self.energy_mean.detach().cpu()),
            "energy_std": float(self.energy_std.detach().cpu()),
        }


__all__ = [
    "GlobalEnergySkipCNN",
    "MultiTaskNextCNN",
    "ResidualSpatialEnergyRegressor",
    "ResidualSpatialNextCNN",
    "SimpleNextCNN",
    "SimpleNextEnergyRegressor",
]
