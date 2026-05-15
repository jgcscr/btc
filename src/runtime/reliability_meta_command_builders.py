from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

ALLOWED_META_COMPONENTS = (
    "transformer",
    "transformer_large",
    "lstm",
    "bilstm",
    "gru",
    "cnn_lstm",
    "cnn_bilstm",
    "garch_lstm",
    "xgb",
    "lgbm",
    "regime_logit",
)


def _coerce_cli_columns(raw: Any) -> List[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(value) for value in raw]


def resolve_meta_component_weight_spec(
    *,
    search_cfg: Mapping[str, Any],
    load_json: Callable[[Path], Any],
    extract_audit_weight_spec: Callable[..., str | None],
    requested_components: Sequence[str] | None = None,
) -> tuple[str | None, Path | None, str | None]:
    meta_component_weight_audit_path = search_cfg.get("meta_component_weights_from_audit_path")
    if not meta_component_weight_audit_path:
        return None, None, None
    audit_path = Path(str(meta_component_weight_audit_path))
    if not audit_path.exists():
        return None, audit_path, None
    try:
        allowed_components = tuple(
            str(component)
            for component in (requested_components or ALLOWED_META_COMPONENTS)
            if str(component) in ALLOWED_META_COMPONENTS
        )
        if not allowed_components:
            allowed_components = ALLOWED_META_COMPONENTS
        return (
            extract_audit_weight_spec(load_json(audit_path), allowed_components=allowed_components),
            audit_path,
            None,
        )
    except Exception as exc:
        return None, audit_path, str(exc)


def build_meta_ensemble_command(
    *,
    python: str,
    output_csv: Path,
    config_path: Path,
    search_cfg: Mapping[str, Any],
    component_frame_path: Path | None = None,
    component_columns: Sequence[str] | None = None,
    extra_feature_columns: Sequence[str] | None = None,
    component_weight_spec: str | None = None,
) -> List[str]:
    cmd = [
        python,
        "-m",
        "src.scripts.train_meta_ensemble",
        "--output-csv",
        str(output_csv),
        "--config-path",
        str(config_path),
        "--weight-threshold",
        str(search_cfg.get("meta_weight_threshold", 0.5)),
        "--signal-mode",
        str(search_cfg.get("meta_signal_mode", "gate_only")),
    ]
    if component_frame_path is not None:
        cmd.extend(["--component-frame-csv", str(component_frame_path)])
        for column in _coerce_cli_columns(component_columns):
            cmd.extend(["--component-column", column])
        for column in _coerce_cli_columns(extra_feature_columns):
            cmd.extend(["--extra-feature-column", column])
    if component_weight_spec:
        cmd.extend(["--component-weight-spec", component_weight_spec])
    return cmd