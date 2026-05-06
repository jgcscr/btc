from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {str(key): value for key, value in base.items()}
    for key, value in override.items():
        normalized_key = str(key)
        existing = merged.get(normalized_key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[normalized_key] = deep_merge_mappings(existing, value)
        else:
            merged[normalized_key] = value
    return merged


def load_composed_yaml(path: Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if resolved in seen:
        raise ValueError(f"Detected recursive config inheritance at {resolved}")
    seen.add(resolved)

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a mapping: {resolved}")

    payload_dict = {str(key): value for key, value in payload.items()}
    extends_value = payload_dict.pop("extends", None)
    if extends_value is None:
        return payload_dict

    if isinstance(extends_value, str):
        parent_refs = [extends_value]
    elif isinstance(extends_value, list) and all(isinstance(item, str) for item in extends_value):
        parent_refs = list(extends_value)
    else:
        raise ValueError(f"Config 'extends' must be a string or list of strings: {resolved}")

    merged: dict[str, Any] = {}
    for parent_ref in parent_refs:
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        merged = deep_merge_mappings(merged, load_composed_yaml(parent_path, _seen=seen))
    return deep_merge_mappings(merged, payload_dict)