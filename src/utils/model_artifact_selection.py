from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.model_summary import metrics_by_split


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _metric_value(metrics: Mapping[str, Any], split: str, key: str) -> float:
    section = metrics.get(split)
    if not isinstance(section, Mapping):
        return float("nan")
    aliases = {
        "auc": ("auc", "roc_auc"),
        "rmse": ("rmse",),
        "mae": ("mae",),
        "f1": ("f1",),
        "accuracy": ("accuracy",),
    }
    for candidate in aliases.get(key, (key,)):
        value = _safe_float(section.get(candidate))
        if not math.isnan(value):
            return value
    return float("nan")


def _summary_score(summary_path: Path) -> tuple[float, ...]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return (-1.0, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not metrics and any(key in payload for key in ("train_metrics", "val_metrics", "test_metrics")):
        metrics = metrics_by_split(
            {
                split: payload.get(f"{split}_metrics", {})
                for split in ("train", "val", "test")
            }
        )

    test_auc = _metric_value(metrics, "test", "auc")
    val_auc = _metric_value(metrics, "val", "auc")
    test_f1 = _metric_value(metrics, "test", "f1")
    val_f1 = _metric_value(metrics, "val", "f1")
    test_accuracy = _metric_value(metrics, "test", "accuracy")
    val_accuracy = _metric_value(metrics, "val", "accuracy")
    test_rmse = _metric_value(metrics, "test", "rmse")
    val_rmse = _metric_value(metrics, "val", "rmse")
    test_mae = _metric_value(metrics, "test", "mae")
    val_mae = _metric_value(metrics, "val", "mae")

    classification_available = any(
        not math.isnan(value)
        for value in (test_auc, val_auc, test_f1, val_f1, test_accuracy, val_accuracy)
    )
    regression_available = any(not math.isnan(value) for value in (test_rmse, val_rmse, test_mae, val_mae))

    if classification_available:
        return (
            2.0,
            0.0 if math.isnan(test_auc) else test_auc,
            0.0 if math.isnan(val_auc) else val_auc,
            0.0 if math.isnan(test_f1) else test_f1,
            0.0 if math.isnan(val_f1) else val_f1,
            0.0 if math.isnan(test_accuracy) else test_accuracy,
            0.0 if math.isnan(val_accuracy) else val_accuracy,
        )
    if regression_available:
        return (
            1.0,
            float("-inf") if math.isnan(test_rmse) else -test_rmse,
            float("-inf") if math.isnan(val_rmse) else -val_rmse,
            float("-inf") if math.isnan(test_mae) else -test_mae,
            float("-inf") if math.isnan(val_mae) else -val_mae,
            0.0,
            0.0,
        )
    return (0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))


def _version_rank(version: str, version_priority: Sequence[str]) -> int:
    try:
        return list(version_priority).index(version)
    except ValueError:
        return len(tuple(version_priority))


def _family_prefix_from_reference_dir(directory: Path) -> str | None:
    name = directory.name
    if "_" not in name:
        return None
    prefix, version = name.rsplit("_", 1)
    if not version.startswith("v"):
        return None
    return prefix


def resolve_best_versioned_model_file(
    reference: str | Path,
    *,
    expected_filename: str | None = None,
    version_priority: Sequence[str] = (),
) -> Path:
    reference_path = Path(reference).expanduser()
    if reference_path.is_dir() or expected_filename is not None:
        base_dir = reference_path
        model_filename = expected_filename
    else:
        base_dir = reference_path.parent
        model_filename = reference_path.name

    if model_filename is None:
        raise ValueError("expected_filename is required when reference is a directory")

    family_prefix = _family_prefix_from_reference_dir(base_dir)
    if family_prefix is None:
        model_path = base_dir / model_filename if base_dir.is_dir() else reference_path
        return model_path

    parent = base_dir.parent
    if not parent.exists():
        return base_dir / model_filename

    candidates: list[tuple[tuple[float, ...], int, str, Path]] = []
    for sibling in parent.glob(f"{family_prefix}_v*"):
        if not sibling.is_dir():
            continue
        model_path = sibling / model_filename
        if not model_path.exists():
            continue
        summary_path = sibling / "summary.json"
        score = _summary_score(summary_path)
        _, version = sibling.name.rsplit("_", 1)
        candidates.append((score, -_version_rank(version, version_priority), sibling.name, model_path))

    if not candidates:
        return base_dir / model_filename

    candidates.sort(reverse=True)
    return candidates[0][3]
