from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from src.trading.ensembles import parse_weight_spec

DirectionModelConfig = Dict[str, Any]


def load_dir_model_config_json(path: str | Path) -> List[DirectionModelConfig]:
    """Load a structured direction-model config from disk."""

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Direction model config not found: {path_obj}")

    try:
        payload = json.loads(path_obj.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - surface via tests
        raise ValueError(f"Failed to parse direction model config JSON at {path_obj}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Direction model config JSON must contain a top-level list of entries.")

    return [dict(entry) for entry in payload]


def _normalize_entry(raw: Mapping[str, Any]) -> DirectionModelConfig | None:
    if raw.get("enabled") is False:
        return None

    model_type = str(raw.get("type", "")).strip().lower()
    if not model_type:
        raise ValueError("Direction model entry is missing a 'type' field.")

    name = str(raw.get("name") or model_type).strip().lower()
    path_value = raw.get("path")
    if path_value is None or str(path_value).strip() == "":
        raise ValueError(f"Direction model '{name}' is missing a path.")

    try:
        weight_value = float(raw.get("weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Direction model '{name}' has an invalid weight.") from exc

    entry: DirectionModelConfig = {
        "name": name,
        "type": model_type,
        "path": str(path_value),
        "weight": weight_value,
    }

    # Preserve optional friendly label if present.
    if "label" in raw:
        entry["label"] = str(raw["label"])

    if "optional" in raw:
        entry["optional"] = bool(raw["optional"])

    return entry


def prepare_direction_model_configs(
    default_configs: Sequence[Mapping[str, Any]],
    *,
    config_entries: Sequence[Mapping[str, Any]] | None = None,
    weight_spec: Optional[str] = None,
) -> List[DirectionModelConfig]:
    """Return a normalized direction-model config list.

    ``config_entries`` can be supplied from a JSON file to override the
    defaults. ``weight_spec`` accepts the legacy "xgb:1,lstm:2" syntax and
    is applied on top of either the defaults or the supplied config list.
    """

    source = config_entries if config_entries is not None else default_configs
    entries: List[DirectionModelConfig] = []
    for raw in deepcopy(list(source)):
        entry = _normalize_entry(raw)
        if entry is None:
            continue
        entries.append(entry)

    if not entries:
        raise ValueError("At least one direction model must be configured.")

    apply_weight_overrides(entries, weight_spec)
    _ensure_unique_names(entries)
    return entries


def apply_weight_overrides(
    configs: Sequence[DirectionModelConfig],
    weight_spec: Optional[str],
) -> None:
    if not weight_spec:
        return
    overrides = parse_weight_spec(weight_spec)
    if not overrides:
        return

    valid_keys: set[str] = set()
    for entry in configs:
        valid_keys.add(str(entry["name"]))
        valid_keys.add(str(entry["type"]))
    unknown = sorted(key for key in overrides.keys() if key not in valid_keys)
    if unknown:
        raise ValueError(
            "Unknown direction-model weight overrides: " + ", ".join(unknown),
        )

    for entry in configs:
        name = entry["name"]
        key = name if name in overrides else entry["type"]
        if key in overrides:
            entry["weight"] = float(overrides[key])


def _ensure_unique_names(configs: Sequence[DirectionModelConfig]) -> None:
    seen: set[str] = set()
    for entry in configs:
        name = entry["name"]
        if name in seen:
            raise ValueError(f"Duplicate direction model name '{name}' detected.")
        seen.add(name)


def clone_direction_model_configs(configs: Sequence[DirectionModelConfig]) -> List[DirectionModelConfig]:
    return deepcopy([dict(entry) for entry in configs])


def apply_path_overrides(
    configs: Sequence[DirectionModelConfig],
    overrides: Mapping[str, Optional[str]] | None,
) -> None:
    if not overrides:
        return
    lowered: Dict[str, str] = {
        key.lower(): str(value)
        for key, value in overrides.items()
        if value not in (None, "")
    }
    if not lowered:
        return

    for entry in configs:
        override = lowered.get(entry["name"]) or lowered.get(entry["type"])
        if override:
            entry["path"] = override


def direction_configs_to_weight_map(configs: Sequence[DirectionModelConfig]) -> Dict[str, float]:
    return {entry["name"]: float(entry.get("weight", 1.0)) for entry in configs}


def resolve_direction_model_configs(
    default_configs: Sequence[Mapping[str, Any]],
    *,
    config_entries: Optional[Sequence[Mapping[str, Any]]] = None,
    config_json_path: str | Path | None = None,
    weight_spec: Optional[str] = None,
    path_overrides: Mapping[str, Optional[str]] | None = None,
) -> List[DirectionModelConfig]:
    """Load, normalize, and override direction-model configs for CLI usage."""

    entries = config_entries
    if config_json_path:
        entries = load_dir_model_config_json(config_json_path)

    configs = prepare_direction_model_configs(
        default_configs,
        config_entries=entries,
        weight_spec=weight_spec,
    )
    apply_path_overrides(configs, path_overrides)
    return configs


def log_direction_model_configs(
    configs: Sequence[DirectionModelConfig],
    *,
    label: Optional[str] = None,
) -> str:
    payload = json.dumps(list(configs), indent=2)
    if label:
        message = f"{label}:\n{payload}"
    else:
        message = payload
    print(message)
    return message


__all__ = [
    "DirectionModelConfig",
    "apply_path_overrides",
    "apply_weight_overrides",
    "clone_direction_model_configs",
    "direction_configs_to_weight_map",
    "log_direction_model_configs",
    "resolve_direction_model_configs",
    "load_dir_model_config_json",
    "prepare_direction_model_configs",
]
