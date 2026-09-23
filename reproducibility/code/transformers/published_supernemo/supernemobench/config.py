"""Single-source configuration for the SuperNEMO workflow."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .tokenization import SuperNEMOTrackerTokenizationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/klz/Data/zeronu_benchmark/SuperNEMO")
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SEED = 42
TRAIN_EVENT_ORDER = "proportional_without_replacement"

SPLIT_NAMES = ("train", "validation", "test")
TASKS = ("classification", "energy")
EVENT_ID_FIELD = "ev_no"
CATEGORY_FIELD = "label"
RADIUS_FIELD = "tR"
ENERGY_FIELDS = ("E1", "E2")
ENERGY_UNIT = "keV"
ENERGY_TARGET = "E1 + E2"
COORDINATE_FIELDS = ("tX", "tY", "tZ")
PROJECTION_PLANES = (("tY", "tX"), ("tZ", "tX"), ("tZ", "tY"))
POINT_FEATURES = ("occupancy_fraction", "log1p_count")
EVENT_CONSTANT_FIELDS = (
    *ENERGY_FIELDS,
    "dY",
    "dZ",
    "theta",
    "phiS",
    "phiR",
)
REQUIRED_FIELDS = (
    EVENT_ID_FIELD,
    *ENERGY_FIELDS,
    *COORDINATE_FIELDS,
    RADIUS_FIELD,
    "dY",
    "dZ",
    "theta",
    "phiS",
    "phiR",
    CATEGORY_FIELD,
)


@dataclass(frozen=True)
class SourceSpec:
    """Canonical identity and task eligibility for one raw HDF5 source."""

    source_key: str
    file_name: str
    raw_label: str
    category: str
    classification_label: int | None


SOURCE_SPECS = (
    SourceSpec("0nubb", "data_0nubb_merged.h5", "0nubb", "0nubb", None),
    SourceSpec("2nubb", "data_2nubb_merged.h5", "2nubb", "2nu", 1),
    SourceSpec("Bi214", "data_Bi214_merged.h5", "Bi214", "Bi214", 0),
    SourceSpec("Tl208", "data_Tl208_merged.h5", "Tl208", "Tl208", None),
)
SOURCE_BY_KEY = {spec.source_key: spec for spec in SOURCE_SPECS}
SOURCE_BY_RAW_LABEL = {spec.raw_label: spec for spec in SOURCE_SPECS}
CLASSIFICATION_LABELS = {
    spec.category: spec.classification_label
    for spec in SOURCE_SPECS
    if spec.classification_label is not None
}
if len(CLASSIFICATION_LABELS) != 2 or set(CLASSIFICATION_LABELS.values()) != {0, 1}:
    raise RuntimeError("SuperNEMO requires one signal and one background label")
NUM_CLASSES = len(CLASSIFICATION_LABELS)


@dataclass(frozen=True)
class DataConfig:
    """Raw-data, split-manifest, and topology-representation settings."""

    data_root: Path = DEFAULT_DATA_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = DEFAULT_SEED
    split_block_events: int = 4096
    scan_chunk_rows: int = 262_144
    projection_grid_size: int = 128
    projection_bin_size_mm: float = 44.0
    projection_origin_mm: tuple[float, float, float] = (-2816.0, -2816.0, -2816.0)
    projection_input_scale: float = 1.0
    point_bin_size_mm: float = 15.0
    coordinate_scale_mm: float = 1000.0
    max_points: int = 512
    train_event_order: Literal["proportional_without_replacement"] = TRAIN_EVENT_ORDER
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).expanduser())
        fractions = tuple(float(item) for item in self.split_fractions)
        if len(fractions) != 3 or any(item <= 0.0 for item in fractions):
            raise ValueError("split_fractions must contain three positive values")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("split_fractions must sum to one")
        object.__setattr__(self, "split_fractions", fractions)
        for name in (
            "split_block_events",
            "scan_chunk_rows",
            "projection_grid_size",
            "max_points",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if int(self.num_classes) != NUM_CLASSES:
            raise ValueError(f"SuperNEMO classification requires num_classes={NUM_CLASSES}")
        if self.train_event_order != TRAIN_EVENT_ORDER:
            raise ValueError(
                f"SuperNEMO requires train_event_order={TRAIN_EVENT_ORDER!r}"
            )
        for name in (
            "projection_bin_size_mm",
            "projection_input_scale",
            "point_bin_size_mm",
            "coordinate_scale_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.projection_origin_mm) != 3 or any(
            not math.isfinite(float(item)) for item in self.projection_origin_mm
        ):
            raise ValueError("projection_origin_mm must contain three finite values")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["manifest_path"] = str(self.manifest_path)
        payload["split_fractions"] = list(self.split_fractions)
        payload["projection_origin_mm"] = list(self.projection_origin_mm)
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    """MJD architecture defaults, kept unchanged for each reused model."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0
    seed: int = DEFAULT_SEED
    num_workers: int = 0
    device: str = "auto"
    deterministic: bool = False
    use_amp: bool = True
    amp_precision: Literal["auto", "float16", "bfloat16"] = "auto"
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["cosine"] = "cosine"
    classification_loss: Literal["bce_with_logits"] = "bce_with_logits"
    energy_loss: Literal["mse"] = "mse"

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "early_stopping_patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_workers) < 0 or int(self.seed) < 0:
            raise ValueError("num_workers and seed must be non-negative")
        for name in ("learning_rate", "gradient_clip_norm"):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "early_stopping_min_delta"):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.amp_precision not in {"auto", "float16", "bfloat16"}:
            raise ValueError("invalid amp_precision")
        if self.optimizer != "adamw" or self.scheduler != "cosine":
            raise ValueError("the frozen optimizer/scheduler are AdamW and cosine")
        if self.classification_loss != "bce_with_logits" or self.energy_loss != "mse":
            raise ValueError("the frozen task losses are BCE-with-logits and MSE")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_device(self, device: str | None) -> "TrainingConfig":
        if device is None:
            return self
        payload = self.to_dict()
        payload["device"] = str(device)
        return TrainingConfig(**payload)


@dataclass(frozen=True)
class ArchitectureConfig:
    """Local model identity and frozen model/training settings."""

    architecture_id: str
    model_name: str
    input_kind: Literal["projection2d", "graph", "sequence"]
    model: Mapping[str, Any]
    training: TrainingConfig
    tasks: tuple[Literal["classification", "energy"], ...] = TASKS
    tokenization: SuperNEMOTrackerTokenizationConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the complete architecture record in JSON-ready form."""

        return {
            "architecture_id": self.architecture_id,
            "model_name": self.model_name,
            "input_kind": self.input_kind,
            "model": dict(self.model),
            "training": self.training.to_dict(),
            "tasks": list(self.tasks),
            "tokenization": (
                None if self.tokenization is None else self.tokenization.to_dict()
            ),
        }


_COMMON_TRAINING = {
    "epochs": 50,
    "weight_decay": 1.0e-4,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 12,
    "early_stopping_min_delta": 0.0,
    "seed": DEFAULT_SEED,
    "num_workers": 0,
    "deterministic": False,
    "use_amp": True,
    "amp_precision": "auto",
}

TRANSFORMER_ARCHITECTURE_IDS = (
    "transformer_001_sampled_hits_coordinate_mlp",
    "transformer_002_voxel_coordinate_mlp",
    "transformer_003_voxel_fourier_xyz",
    "transformer_004_sampled_hits_fourier_xyz",
    "transformer_005_summary_features_coordinate_mlp",
    "transformer_006_summary_features_fourier_xyz",
)

_SAMPLED_HITS_TOKENIZATION = SuperNEMOTrackerTokenizationConfig(
    tokenization="sampled_hits"
)
_VOXEL_TOKENIZATION = SuperNEMOTrackerTokenizationConfig(tokenization="voxel")
_SUMMARY_FEATURES_TOKENIZATION = SuperNEMOTrackerTokenizationConfig(
    tokenization="summary_features"
)

ARCHITECTURES: dict[str, ArchitectureConfig] = {
    "cnn_004_multiview_late_fusion": ArchitectureConfig(
        architecture_id="cnn_004_multiview_late_fusion",
        model_name="MultiViewLateFusionCNN",
        input_kind="projection2d",
        model={
            "base_channels": 16,
            "stage_blocks": [2, 2, 2, 2],
            "fusion_features": 256,
            "dropout": 0.1,
        },
        training=TrainingConfig(
            batch_size=16,
            learning_rate=1.0e-3,
            **_COMMON_TRAINING,
        ),
    ),
    "gnn_001_static_gine": ArchitectureConfig(
        architecture_id="gnn_001_static_gine",
        model_name="StaticGINEClassifier",
        input_kind="graph",
        model={
            "feature_dim": 2,
            "hidden_dim": 128,
            "num_layers": 5,
            "k": 12,
            "classifier_dim": 160,
            "dropout": 0.10,
            "train_eps": True,
        },
        training=TrainingConfig(
            batch_size=16,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
    ),
    "seq_001_bigru": ArchitectureConfig(
        architecture_id="seq_001_bigru",
        model_name="HilbertBiGRUClassifier",
        input_kind="sequence",
        model={
            "feature_dim": 2,
            "embedding_dim": 96,
            "hidden_dim": 128,
            "num_layers": 2,
            "hilbert_bits": 10,
            "classifier_dim": 256,
            "dropout": 0.10,
        },
        training=TrainingConfig(
            batch_size=16,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
    ),
    "ssm_001_pointmamba": ArchitectureConfig(
        architecture_id="ssm_001_pointmamba",
        model_name="PointMambaLiteClassifier",
        input_kind="sequence",
        model={
            "feature_dim": 2,
            "model_dim": 128,
            "inner_dim": 192,
            "state_dim": 16,
            "dt_rank": 16,
            "num_layers": 3,
            "conv_kernel": 4,
            "hilbert_bits": 10,
            "scan_chunk_size": 32,
            "classifier_dim": 160,
            "dropout": 0.10,
        },
        training=TrainingConfig(
            batch_size=4,
            learning_rate=3.0e-4,
            **_COMMON_TRAINING,
        ),
    ),
    "transformer_001_sampled_hits_coordinate_mlp": ArchitectureConfig(
        architecture_id="transformer_001_sampled_hits_coordinate_mlp",
        model_name="TrackerHitTransformer",
        input_kind="sequence",
        model={
            "position_encoding": "coordinate_mlp",
            "feature_dim": 4,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "num_frequencies": 6,
            "pooling": "masked_mean",
        },
        training=TrainingConfig(
            batch_size=64,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
        tasks=("classification",),
        tokenization=_SAMPLED_HITS_TOKENIZATION,
    ),
    "transformer_002_voxel_coordinate_mlp": ArchitectureConfig(
        architecture_id="transformer_002_voxel_coordinate_mlp",
        model_name="TrackerHitTransformer",
        input_kind="sequence",
        model={
            "position_encoding": "coordinate_mlp",
            "feature_dim": 4,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "num_frequencies": 6,
            "pooling": "masked_mean",
        },
        training=TrainingConfig(
            batch_size=64,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
        tasks=("classification",),
        tokenization=_VOXEL_TOKENIZATION,
    ),
    "transformer_003_voxel_fourier_xyz": ArchitectureConfig(
        architecture_id="transformer_003_voxel_fourier_xyz",
        model_name="TrackerHitTransformer",
        input_kind="sequence",
        model={
            "position_encoding": "fourier_xyz",
            "feature_dim": 4,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "num_frequencies": 6,
            "pooling": "masked_mean",
        },
        training=TrainingConfig(
            batch_size=64,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
        tasks=("classification",),
        tokenization=_VOXEL_TOKENIZATION,
    ),
    "transformer_004_sampled_hits_fourier_xyz": ArchitectureConfig(
        architecture_id="transformer_004_sampled_hits_fourier_xyz",
        model_name="TrackerHitTransformer",
        input_kind="sequence",
        model={
            "position_encoding": "fourier_xyz",
            "feature_dim": 4,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "num_frequencies": 6,
            "pooling": "masked_mean",
        },
        training=TrainingConfig(
            batch_size=64,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
        tasks=("classification",),
        tokenization=_SAMPLED_HITS_TOKENIZATION,
    ),
    "transformer_005_summary_features_coordinate_mlp": ArchitectureConfig(
        architecture_id="transformer_005_summary_features_coordinate_mlp",
        model_name="TrackerHitTransformer",
        input_kind="sequence",
        model={
            "position_encoding": "coordinate_mlp",
            "feature_dim": 6,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "num_frequencies": 6,
            "pooling": "masked_mean",
        },
        training=TrainingConfig(
            batch_size=64,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
        tasks=("classification",),
        tokenization=_SUMMARY_FEATURES_TOKENIZATION,
    ),
    "transformer_006_summary_features_fourier_xyz": ArchitectureConfig(
        architecture_id="transformer_006_summary_features_fourier_xyz",
        model_name="TrackerHitTransformer",
        input_kind="sequence",
        model={
            "position_encoding": "fourier_xyz",
            "feature_dim": 6,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "num_frequencies": 6,
            "pooling": "masked_mean",
        },
        training=TrainingConfig(
            batch_size=64,
            learning_rate=5.0e-4,
            **_COMMON_TRAINING,
        ),
        tasks=("classification",),
        tokenization=_SUMMARY_FEATURES_TOKENIZATION,
    ),
}
for _architecture in ARCHITECTURES.values():
    if not _architecture.tasks or any(task not in TASKS for task in _architecture.tasks):
        raise RuntimeError(f"{_architecture.architecture_id} has invalid task support")
    if _architecture.tokenization is None:
        expected_feature_dim = len(POINT_FEATURES)
    else:
        expected_feature_dim = _architecture.tokenization.feature_dim
        if (
            _architecture.model_name != "TrackerHitTransformer"
            or _architecture.input_kind != "sequence"
            or _architecture.tasks != ("classification",)
        ):
            raise RuntimeError(
                f"{_architecture.architecture_id} has invalid Transformer task/input settings"
            )
    if _architecture.input_kind in {"graph", "sequence"} and int(
        _architecture.model["feature_dim"]
    ) != expected_feature_dim:
        raise RuntimeError(
            f"{_architecture.architecture_id} feature_dim disagrees with point features"
        )

_configured_transformers = tuple(
    identifier
    for identifier, configured in ARCHITECTURES.items()
    if configured.tokenization is not None
)
if _configured_transformers != TRANSFORMER_ARCHITECTURE_IDS:
    raise RuntimeError("configured Transformer architecture IDs are not the frozen six")


def source_for_raw_label(raw_label: str) -> SourceSpec:
    """Resolve a validated raw label without fuzzy matching or fallback."""

    try:
        return SOURCE_BY_RAW_LABEL[str(raw_label)]
    except KeyError as error:
        raise ValueError(f"unknown SuperNEMO raw category: {raw_label!r}") from error


def classification_label(raw_label: str) -> int | None:
    """Return the sole centralized binary mapping; other categories are excluded."""

    return source_for_raw_label(raw_label).classification_label


__all__ = [
    "ARCHITECTURES",
    "CLASSIFICATION_LABELS",
    "CATEGORY_FIELD",
    "COORDINATE_FIELDS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SEED",
    "DataConfig",
    "ENERGY_FIELDS",
    "ENERGY_TARGET",
    "ENERGY_UNIT",
    "EVENT_ID_FIELD",
    "EVENT_CONSTANT_FIELDS",
    "NUM_CLASSES",
    "POINT_FEATURES",
    "PROJECTION_PLANES",
    "PROJECT_ROOT",
    "REQUIRED_FIELDS",
    "RADIUS_FIELD",
    "SOURCE_BY_KEY",
    "SOURCE_SPECS",
    "SPLIT_NAMES",
    "TASKS",
    "TRANSFORMER_ARCHITECTURE_IDS",
    "TRAIN_EVENT_ORDER",
    "TrainingConfig",
    "classification_label",
    "source_for_raw_label",
]
