"""Central registry for alternative NEXT classification architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Type


@dataclass(frozen=True)
class ModelSpec:
    """Describe how one checkpoint-declared architecture is constructed."""

    architecture_id: str
    model_name: str
    input_kind: str
    model_class: Type[Any]


def _specs() -> Dict[str, ModelSpec]:
    # Imports are local so data utilities and CLI help remain usable even when
    # an optional model implementation has an import-time problem.
    from .models.cnn import (
        Dense3DResidualCNN,
        MultiScaleProjectionCNN,
        MultiViewLateFusionCNN,
    )
    from .models.point_graph import (
        CNNGNNHybridClassifier,
        DeepSetsClassifier,
        EGNNClassifier,
        GravNetClassifier,
        ParticleNetLiteClassifier,
        PointNetPPClassifier,
        StaticGINEClassifier,
    )
    from .models.classic_topology import TopologyBoostedTreeClassifier
    from .models.graph_topology import (
        DimeNetLiteClassifier,
        PersistencePersLayClassifier,
    )
    from .models.mixer_sparse import (
        ProjectionMLPMixerClassifier,
        RigidKPConvClassifier,
        SubmanifoldSparseResNetClassifier,
    )
    from .models.point_sequence import (
        HilbertBiGRUClassifier,
        HilbertTCNClassifier,
        PointMLPClassifier,
        PointMambaLiteClassifier,
    )

    definitions = (
        (
            "cnn_004_multiview_late_fusion",
            "MultiViewLateFusionCNN",
            "projection2d",
            MultiViewLateFusionCNN,
        ),
        (
            "cnn_005_multiscale_projection",
            "MultiScaleProjectionCNN",
            "multiscale2d",
            MultiScaleProjectionCNN,
        ),
        (
            "cnn_006_dense_3d_resnet",
            "Dense3DResidualCNN",
            "dense3d",
            Dense3DResidualCNN,
        ),
        (
            "point_001_deepsets",
            "DeepSetsClassifier",
            "points",
            DeepSetsClassifier,
        ),
        (
            "point_002_pointnetpp",
            "PointNetPPClassifier",
            "points",
            PointNetPPClassifier,
        ),
        (
            "gnn_001_static_gine",
            "StaticGINEClassifier",
            "graph",
            StaticGINEClassifier,
        ),
        (
            "gnn_002_particlenet_edgeconv",
            "ParticleNetLiteClassifier",
            "graph",
            ParticleNetLiteClassifier,
        ),
        (
            "gnn_003_egnn",
            "EGNNClassifier",
            "graph",
            EGNNClassifier,
        ),
        (
            "gnn_004_gravnet",
            "GravNetClassifier",
            "graph",
            GravNetClassifier,
        ),
        (
            "hybrid_001_cnn_gnn",
            "CNNGNNHybridClassifier",
            "hybrid",
            CNNGNNHybridClassifier,
        ),
        (
            "classic_001_topology_xgboost",
            "TopologyBoostedTreeClassifier",
            "topology",
            TopologyBoostedTreeClassifier,
        ),
        (
            "point_003_pointmlp",
            "PointMLPClassifier",
            "points",
            PointMLPClassifier,
        ),
        (
            "seq_001_bigru",
            "HilbertBiGRUClassifier",
            "sequence",
            HilbertBiGRUClassifier,
        ),
        (
            "seq_002_dilated_tcn",
            "HilbertTCNClassifier",
            "sequence",
            HilbertTCNClassifier,
        ),
        (
            "mixer_001_projection_mlp_mixer",
            "ProjectionMLPMixerClassifier",
            "projection2d",
            ProjectionMLPMixerClassifier,
        ),
        (
            "gnn_005_dimenet_lite",
            "DimeNetLiteClassifier",
            "graph",
            DimeNetLiteClassifier,
        ),
        (
            "point_004_rigid_kpconv",
            "RigidKPConvClassifier",
            "points",
            RigidKPConvClassifier,
        ),
        (
            "topo_001_persistence_perslay",
            "PersistencePersLayClassifier",
            "topology",
            PersistencePersLayClassifier,
        ),
        (
            "ssm_001_pointmamba",
            "PointMambaLiteClassifier",
            "sequence",
            PointMambaLiteClassifier,
        ),
        (
            "sparse_001_submanifold_resnet",
            "SubmanifoldSparseResNetClassifier",
            "sparse3d",
            SubmanifoldSparseResNetClassifier,
        ),
    )
    return {
        architecture_id: ModelSpec(
            architecture_id=architecture_id,
            model_name=model_name,
            input_kind=input_kind,
            model_class=model_class,
        )
        for architecture_id, model_name, input_kind, model_class in definitions
    }


def registered_architectures() -> Tuple[str, ...]:
    """Return stable architecture identifiers accepted by the runner."""

    return tuple(_specs())


def get_model_spec(identifier: str) -> ModelSpec:
    """Resolve either an architecture ID or its checkpoint model name."""

    requested = str(identifier).strip()
    specs = _specs()
    if requested in specs:
        return specs[requested]
    for spec in specs.values():
        if requested == spec.model_name:
            return spec
    raise KeyError(
        "unknown alternative NEXT architecture %r; expected one of: %s"
        % (requested, ", ".join(specs))
    )


def build_model(
    identifier: str,
    model_config: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate one registered model from checkpoint-safe keyword args."""

    spec = get_model_spec(identifier)
    config = {} if model_config is None else dict(model_config)
    return spec.model_class(**config)


__all__ = [
    "ModelSpec",
    "build_model",
    "get_model_spec",
    "registered_architectures",
]
