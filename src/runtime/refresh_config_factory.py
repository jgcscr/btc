from __future__ import annotations

from typing import Any, Callable, Sequence

from src.runtime.config_normalization_support import normalize_config_value


def build_refresh_config_value_normalizer(
    *,
    default_targets: Sequence[float],
    config_int_fields: Sequence[str],
    config_float_fields: Sequence[str],
    config_bool_fields: Sequence[str],
    config_path_fields: Sequence[str],
    config_allowed_keys: Sequence[str],
    regimes: Sequence[str],
    bool_env: Callable[[str], bool],
    parse_targets: Callable[[str], list[float]],
    normalize_horizon_value: Callable[[Any], float],
    normalize_horizon_float_map: Callable[..., dict[float, float]],
    normalize_horizon_regime_float_map: Callable[..., dict[float, dict[str, float]]],
    stderr_write: Callable[[str], None],
) -> Callable[[str, Any], Any]:
    def _normalize(name: str, value: Any) -> Any:
        return normalize_config_value(
            name,
            value,
            default_targets=default_targets,
            config_int_fields=config_int_fields,
            config_float_fields=config_float_fields,
            config_bool_fields=config_bool_fields,
            config_path_fields=config_path_fields,
            config_allowed_keys=config_allowed_keys,
            regimes=regimes,
            bool_env=bool_env,
            parse_targets=parse_targets,
            normalize_horizon_value=normalize_horizon_value,
            normalize_horizon_float_map=normalize_horizon_float_map,
            normalize_horizon_regime_float_map=normalize_horizon_regime_float_map,
            stderr_write=stderr_write,
        )

    return _normalize