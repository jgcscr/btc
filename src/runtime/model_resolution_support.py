from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def model_suffix_candidates(
    horizon: float,
    *,
    normalize_horizon_value: Callable[[float], float],
) -> List[str]:
    normalized = normalize_horizon_value(horizon)
    candidates: List[str] = []
    if normalized < 1.0:
        minutes = int(round(normalized * 60))
        candidates.append(f"{minutes}m")
        candidates.append(f"{normalized:g}h")
    else:
        if float(normalized).is_integer():
            candidates.append(f"{int(normalized)}h")
        else:
            candidates.append(f"{normalized:g}h")

    if normalized < 1.0 and "1h" not in candidates:
        candidates.append("1h")

    unique: List[str] = []
    for suffix in candidates:
        if suffix not in unique:
            unique.append(suffix)
    return unique


def model_paths_for_horizon(
    horizon: float,
    *,
    format_horizon_label: Callable[[float], str],
    normalize_horizon_value: Callable[[float], float],
    model_root: Path,
    model_version_priority: Sequence[str],
    dir_version_overrides: Mapping[str, Sequence[str]],
    resolve_best_versioned_model_file_fn: Callable[..., Path],
    stderr_write: Callable[[str], None],
) -> tuple[Path, Path]:
    suffixes = model_suffix_candidates(horizon, normalize_horizon_value=normalize_horizon_value)
    label = format_horizon_label(horizon)
    fallback: tuple[Path, Path] | None = None

    for suffix_idx, suffix in enumerate(suffixes):
        reg_path = resolve_best_versioned_model_file_fn(
            model_root / f"xgb_ret{suffix}_v1",
            expected_filename=f"xgb_ret{suffix}_model.json",
            version_priority=tuple(model_version_priority),
        )
        dir_path = resolve_best_versioned_model_file_fn(
            model_root / f"xgb_dir{suffix}_v1",
            expected_filename=f"xgb_dir{suffix}_model.json",
            version_priority=tuple(dir_version_overrides.get(suffix, tuple(model_version_priority))),
        )

        if fallback is None:
            fallback = (reg_path, dir_path)

        if reg_path.exists() and dir_path.exists():
            if suffix_idx > 0:
                stderr_write(f"Info: using {suffix} model artifacts for {label} horizon fallback.\n")
            return reg_path, dir_path

    if fallback is not None and len(suffixes) > 1:
        stderr_write(
            f"Warning: dedicated model artifacts for {label} horizon are missing; using {suffixes[-1]} fallback paths.\n"
        )
        return fallback

    if fallback is None:
        raise RuntimeError(f"Unable to resolve model paths for {label} horizon.")
    return fallback


def prepare_base_direction_configs(
    *,
    config_json_path: str | None,
    weight_spec: str | None,
    dir_lstm_path: str | None,
    dir_bilstm_path: str | None,
    dir_gru_path: str | None,
    dir_cnn_lstm_path: str | None,
    dir_cnn_bilstm_path: str | None,
    dir_garch_lstm_path: str | None,
    dir_transformer_path: str | None,
    default_dir_models_1h: Sequence[Any],
    resolve_direction_model_configs_fn: Callable[..., list[Any]],
) -> List[Any]:
    overrides = {
        "lstm": dir_lstm_path,
        "bilstm": dir_bilstm_path,
        "gru": dir_gru_path,
        "cnn_lstm": dir_cnn_lstm_path,
        "cnn_bilstm": dir_cnn_bilstm_path,
        "garch_lstm": dir_garch_lstm_path,
        "transformer": dir_transformer_path,
    }
    return resolve_direction_model_configs_fn(
        default_dir_models_1h,
        config_json_path=config_json_path,
        weight_spec=weight_spec,
        path_overrides=overrides,
    )


def direction_configs_for_horizon(
    base_configs: Sequence[Any],
    *,
    dir_model_path: str,
    horizon: float,
    horizon_label: str,
    normalize_horizon_value: Callable[[float], float],
    default_transformer_model_dir_by_suffix: Mapping[str, str | None],
    model_root: Path,
    model_version_priority: Sequence[str],
    clone_direction_model_configs_fn: Callable[[Sequence[Any]], list[Any]],
    apply_path_overrides_fn: Callable[[list[Any], Mapping[str, str]], None],
    log_direction_model_configs_fn: Callable[..., None],
    direction_configs_to_weight_map_fn: Callable[[Sequence[Any]], Dict[str, float]],
    registry_model_exists_fn: Callable[[str], bool],
    env_get: Callable[[str], str | None] = os.getenv,
) -> tuple[List[Any], Dict[str, float]]:
    def explicit_transformer_path(suffix: str) -> Optional[str]:
        path = default_transformer_model_dir_by_suffix.get(suffix)
        if not path:
            return None
        if path.startswith("models:/"):
            parts = path.split("/")
            model_name = parts[1] if len(parts) > 1 else ""
            if model_name and registry_model_exists_fn(model_name):
                return path
            return None
        path_obj = Path(path).expanduser()
        return str(path_obj) if path_obj.exists() else None

    def sequence_model_overrides() -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        suffixes = model_suffix_candidates(horizon, normalize_horizon_value=normalize_horizon_value)
        seq_types = (
            "lstm",
            "bilstm",
            "gru",
            "cnn_lstm",
            "cnn_bilstm",
            "garch_lstm",
            "transformer",
            "transformer_large",
        )
        for model_type in seq_types:
            for suffix in suffixes:
                if model_type == "transformer":
                    explicit_path = explicit_transformer_path(suffix)
                    if explicit_path:
                        overrides[model_type] = explicit_path
                        break
                prefix = f"{model_type}_dir{suffix}"
                if model_type == "transformer_large":
                    prefix = f"transformer_dir{suffix}_large"
                for version in model_version_priority:
                    candidate = model_root / f"{prefix}_{version}"
                    if candidate.exists():
                        overrides[model_type] = str(candidate)
                        break
                if model_type in overrides:
                    break
                if model_type == "transformer" and horizon >= 1.0 and suffix.endswith("h"):
                    use_registry = str(env_get("USE_MLFLOW_REGISTRY") or "").lower() in {"1", "true", "yes"}
                    if use_registry:
                        model_name = f"transformer_dir{suffix}"
                        if registry_model_exists_fn(model_name):
                            overrides[model_type] = f"models:/{model_name}/latest"
                            break
        return overrides

    def lgbm_model_path() -> Optional[str]:
        suffixes = model_suffix_candidates(horizon, normalize_horizon_value=normalize_horizon_value)
        for suffix in suffixes:
            for version in model_version_priority:
                model_dir = model_root / f"lgbm_dir{suffix}_{version}"
                model_path = model_dir / f"lgbm_dir{suffix}_model.joblib"
                if model_path.exists():
                    return str(model_path)
        return None

    configs = clone_direction_model_configs_fn(base_configs)
    overrides = {"xgb": dir_model_path}
    overrides.update(sequence_model_overrides())
    apply_path_overrides_fn(configs, overrides)
    lgbm_path = lgbm_model_path()
    if lgbm_path and not any(entry.get("type") == "lgbm" for entry in configs):
        configs.append(
            {
                "name": "lgbm",
                "type": "lgbm",
                "path": lgbm_path,
                "weight": 1.0,
            }
        )
    log_direction_model_configs_fn(configs, label=f"[run_refresh_and_predict] direction models ({horizon_label})")
    weight_map = direction_configs_to_weight_map_fn(configs)
    return configs, weight_map