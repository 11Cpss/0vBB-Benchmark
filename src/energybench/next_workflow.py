"""A small, guided CLI workflow for the repository's NEXT CNN models.

The generic EnergyBench commands remain the source of truth.  This module only
fills in information that is already recorded in a trusted NEXT checkpoint and
then composes the existing adapter and evaluator APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .adapters import run_adapter
from .config import apply_overrides, load_manifest
from .data import PredictionBundle, resolve_schema, save_bundle
from .evaluation import run_evaluation
from .utils import slugify


NEXT_CLASSIFICATION_MANIFEST = "next_0nubb_vs_bi214.yaml"
NEXT_REGRESSION_MANIFEST = "next_energy_regression.yaml"
NEXT_MULTITASK_MANIFEST = "next_0nubb_vs_bi214_multitask.yaml"


def find_project_root(start: Optional[Any] = None) -> Optional[Path]:
    """Find the checkout that contains both ``pyproject.toml`` and manifests."""

    location = (
        Path(start).expanduser().resolve()
        if start
        else Path(__file__).resolve()
    )
    if location.is_file():
        location = location.parent
    for candidate in (location,) + tuple(location.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "manifests").is_dir()
        ):
            return candidate
    return None


def checkpoint_defaults(checkpoint: Any) -> Dict[str, Any]:
    """Read only the small set of workflow defaults stored in a checkpoint.

    NEXT checkpoints are PyTorch pickle containers and must therefore be
    trusted.  The model adapter has the same trust boundary when it later loads
    the state dict for inference.
    """

    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("checkpoint does not exist: %s" % source)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "NEXT evaluation requires PyTorch in the active environment"
        ) from exc
    try:
        payload = torch.load(
            str(source), map_location="cpu", weights_only=False
        )
    except TypeError:
        # ``weights_only`` was added after older supported PyTorch releases.
        payload = torch.load(str(source), map_location="cpu")
    except Exception as exc:
        raise ValueError("could not read NEXT checkpoint %s: %s" % (source, exc))
    if not isinstance(payload, Mapping):
        raise ValueError("NEXT checkpoint root must be a mapping: %s" % source)

    training = payload.get("training_config", {})
    training = training if isinstance(training, Mapping) else {}
    data_config = training.get("data", {})
    data_config = data_config if isinstance(data_config, Mapping) else {}
    data_root = data_config.get("root")
    model_name = str(payload.get("model_name") or "unknown")
    model_suffix = str(payload.get("model_suffix") or "")
    declared_task = str(payload.get("task") or "").strip().lower()
    if "multitask" in model_name.lower():
        task = "multitask"
    elif declared_task == "energy_regression":
        task = "regression"
    else:
        task = "classification"
    return {
        "checkpoint": source,
        "data_root": (
            None if data_root is None or data_root == "" else str(data_root)
        ),
        "model_name": model_name,
        "model_suffix": model_suffix,
        "epoch": payload.get("epoch"),
        "task": task,
        "multitask": task == "multitask",
    }


def default_model_id(checkpoint: Any) -> str:
    """Derive a readable run ID without claiming extra model semantics."""

    name = Path(checkpoint).expanduser().stem
    if name.lower().startswith("nextcnn_"):
        name = name[len("NEXTCNN_") :]
    name = name.replace("_", "-").lower()
    return slugify(name, fallback="next-model")


def next_available_directory(base: Any) -> Path:
    """Return ``base`` or the first ``base_runN`` path that does not exist."""

    path = Path(base).expanduser().resolve()
    if not path.exists():
        return path
    for number in range(2, 10000):
        candidate = path.parent / ("%s_run%d" % (path.name, number))
        if not candidate.exists():
            return candidate
    raise FileExistsError("could not find a free run directory beside %s" % path)


def _manifest_path(task: str, explicit: Optional[Any] = None) -> Path:
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("manifest does not exist: %s" % source)
        return source
    root = find_project_root()
    if root is None:
        raise FileNotFoundError(
            "could not locate the built-in NEXT manifest; pass --manifest PATH"
        )
    manifest_names = {
        "classification": NEXT_CLASSIFICATION_MANIFEST,
        "regression": NEXT_REGRESSION_MANIFEST,
        "multitask": NEXT_MULTITASK_MANIFEST,
    }
    try:
        name = manifest_names[str(task)]
    except KeyError as exc:
        raise ValueError("unsupported NEXT task %r" % task) from exc
    source = root / "manifests" / name
    if not source.is_file():
        raise FileNotFoundError(
            "built-in NEXT manifest is missing: %s; pass --manifest PATH" % source
        )
    return source


def select_next_manifest(
    bundle: PredictionBundle, explicit: Optional[Any] = None
) -> Path:
    """Select the task contract from actual exported prediction roles."""

    if explicit:
        return _manifest_path("classification", explicit)
    schema = resolve_schema(bundle)
    classification = bool(schema.get("score"))
    regression = bool(schema.get("energy_true") and schema.get("energy_pred"))
    if classification and regression:
        task = "multitask"
    elif regression:
        task = "regression"
    else:
        task = "classification"
    return _manifest_path(task)


def _resolved_output_root(
    output_root: Optional[Any], model_id: str, project_root: Optional[Path]
) -> Path:
    if output_root:
        destination = Path(output_root).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(
                "output directory already exists: %s; choose a new path" % destination
            )
        return destination
    parent = (
        project_root / "04_evaluations"
        if project_root is not None
        else Path.cwd() / "04_evaluations"
    )
    return next_available_directory(parent / slugify(model_id, "next-model"))


def _positive_integer(value: Any, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError("%s must be an integer" % name)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be an integer" % name) from exc
    minimum = 0 if allow_zero else 1
    try:
        exact = float(value) == float(number)
    except (TypeError, ValueError):
        exact = False
    if not exact or number < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return number


def run_next_evaluation(
    checkpoint: Any,
    data: Optional[Any] = None,
    output_root: Optional[Any] = None,
    model_id: Optional[str] = None,
    manifest: Optional[Any] = None,
    device: str = "cuda:0",
    batch_size: int = 32,
    num_workers: int = 0,
    split: str = "test",
    max_files_per_class: int = 0,
    no_plots: bool = False,
    dry_run: bool = False,
    print_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Predict and strictly evaluate one trusted NEXT checkpoint."""

    batch_size = _positive_integer(batch_size, "batch_size")
    num_workers = _positive_integer(num_workers, "num_workers", allow_zero=True)
    max_files_per_class = _positive_integer(
        max_files_per_class, "max_files_per_class", allow_zero=True
    )
    split = str(split).strip()
    if not split:
        raise ValueError("split cannot be empty")

    defaults = checkpoint_defaults(checkpoint)
    checkpoint_path = Path(defaults["checkpoint"])
    data_value = data if data is not None else defaults.get("data_root")
    if not data_value:
        raise ValueError(
            "checkpoint does not record a dataset root; pass --data PATH"
        )
    data_path = Path(data_value).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(
            "NEXT dataset directory does not exist: %s; pass --data PATH"
            % data_path
        )

    resolved_model_id = str(model_id or default_model_id(checkpoint_path)).strip()
    if not resolved_model_id:
        raise ValueError("model_id cannot be empty")
    checkpoint_task = str(defaults.get("task", "classification"))
    planned_manifest = _manifest_path(checkpoint_task, manifest)
    project_root = find_project_root()
    run_root = _resolved_output_root(output_root, resolved_model_id, project_root)
    suffix = slugify(split, "test")
    predictions_path = run_root / ("predictions_%s.npz" % suffix)
    evaluation_dir = run_root / ("evaluation_%s" % suffix)

    print_fn("")
    print_fn("EnergyBench NEXT 一键评测")
    print_fn("=" * 52)
    print_fn("模型      : %s" % checkpoint_path)
    print_fn(
        "类型      : %s%s"
        % (
            defaults.get("model_name", "unknown"),
            " (%s)" % checkpoint_task,
        )
    )
    if defaults.get("epoch") is not None:
        print_fn("checkpoint: epoch %s" % defaults["epoch"])
    print_fn("数据      : %s" % data_path)
    print_fn("任务配置  : %s" % planned_manifest)
    print_fn("模型 ID   : %s" % resolved_model_id)
    print_fn("设备/批量 : %s / %d" % (device, batch_size))
    print_fn("输出目录  : %s" % run_root)
    print_fn("评测模式  : strict（不会覆盖已有结果）")

    base_result = {
        "dry_run": bool(dry_run),
        "checkpoint": checkpoint_path,
        "data": data_path,
        "manifest": planned_manifest,
        "model_id": resolved_model_id,
        "output_root": run_root,
        "predictions_path": predictions_path,
        "evaluation_dir": evaluation_dir,
        "checkpoint_info": defaults,
    }
    if dry_run:
        return base_result

    print_fn("")
    print_fn("[1/3] 正在运行模型推理并导出逐事件预测…")
    bundle = run_adapter(
        "next_cnn.adapter:predict",
        str(checkpoint_path),
        str(data_path),
        {
            "batch_size": batch_size,
            "device": str(device),
            "num_workers": num_workers,
            "split": split,
            "max_files_per_class": max_files_per_class,
        },
    )
    selected_manifest = select_next_manifest(bundle, manifest)
    if selected_manifest != planned_manifest:
        print_fn(
            "      导出列显示任务类型不同，已切换配置: %s"
            % selected_manifest
        )
    schema = resolve_schema(bundle)
    classification_ready = bool(
        schema.get("score") and (schema.get("label") or schema.get("category"))
    )
    regression_ready = bool(schema.get("energy_true") and schema.get("energy_pred"))
    print_fn(
        "      %d events | classification: %s | regression: %s"
        % (
            bundle.n_events,
            "ready" if classification_ready else "not available",
            "ready" if regression_ready else "not available",
        )
    )
    save_bundle(bundle, predictions_path)

    print_fn("[2/3] 输入检查通过，正在执行 strict evaluation…")
    config = apply_overrides(
        load_manifest(selected_manifest),
        {
            "model_id": resolved_model_id,
            "dataset.split": split,
            "runtime.make_plots": False if no_plots else None,
        },
    )
    report = run_evaluation(
        bundle,
        config,
        evaluation_dir,
        strict=True,
        allow_existing=False,
    )
    print_fn("[3/3] 评测完成。")

    result = dict(base_result)
    result.update(
        {
            "dry_run": False,
            "manifest": selected_manifest,
            "config": config,
            "report": report,
            "preflight": {
                "n_events": bundle.n_events,
                "resolved_columns": schema,
                "classification_ready": classification_ready,
                "regression_ready": regression_ready,
            },
        }
    )
    return result


__all__ = [
    "checkpoint_defaults",
    "default_model_id",
    "find_project_root",
    "next_available_directory",
    "run_next_evaluation",
    "select_next_manifest",
]
