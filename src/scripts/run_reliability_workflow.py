from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.runtime.reliability_workflow_common import (
    StepResult,
    emit_step_event as _emit_step_event,
    load_json as _load_json,
    load_yaml as _load_yaml,
    run_with_step_event_sink as _run_with_step_event_sink,
)
from src.runtime.reliability_config_support import (
    build_audit_weighted_runtime_config as _build_audit_weighted_runtime_config,
    extract_audit_weight_spec as _extract_audit_weight_spec,
    format_horizon_label as _format_horizon_label,
    format_weight_spec as _format_weight_spec,
    join_horizons as _join_horizons,
    load_prediction_targets as _load_prediction_targets,
    parse_weight_spec as _parse_weight_spec,
)
from src.runtime.reliability_candidate_config_support import (
    resolve_direction_output_shadow_horizons as _resolve_direction_output_shadow_horizons,
    write_direction_output_shadow_config as _write_direction_output_shadow_config,
    write_upstream_direction_candidate_config as _write_upstream_direction_candidate_config,
    derive_trade_decision_regime_midband_candidate as _derive_trade_decision_regime_midband_candidate_impl,
    write_trade_decision_midband_candidate_config as _write_trade_decision_midband_candidate_config_impl,
)
from src.runtime.reliability_champion_support import (
    build_champion_gate_alignment_check as _build_champion_gate_alignment_check,
    extract_trade_decision_reference_source as _extract_trade_decision_reference_source,
    resolve_effective_champion_gate as _resolve_effective_champion_gate,
    resolve_trade_decision_model_path_for_variant as _resolve_trade_decision_model_path_for_variant,
)
from src.runtime.reliability_command_builders import (
    build_calibration_robustness_command as _build_calibration_robustness_command,
    build_directional_objectives_command as _build_directional_objectives_command,
    build_rolling_ab_command as _build_rolling_ab_command,
)
from src.runtime.reliability_model_shift_guard_support import (
    apply_trade_decision_model_shift_guard as _apply_trade_decision_model_shift_guard_impl,
    build_trade_decision_model_shift_guard as _build_trade_decision_model_shift_guard_impl,
)
from src.runtime.reliability_quality_support import (
    write_calibrated_quality_input as _write_calibrated_quality_input_impl,
    write_meta_component_frame as _write_meta_component_frame,
)
from src.runtime.reliability_selection_guard_support import (
    augment_selection_guard_candidate_floors as _augment_selection_guard_candidate_floors_impl,
    dedupe_selection_calibration_guard_rules as _dedupe_selection_calibration_guard_rules_impl,
    load_reusable_selection_calibration_guard_rules as _load_reusable_selection_calibration_guard_rules,
    normalize_selection_calibration_guard_rules as _normalize_selection_calibration_guard_rules,
)
from src.runtime.reliability_regime_shadow_support import (
    build_regime_abs_ret_pred_floor_shadow as _build_regime_abs_ret_pred_floor_shadow,
    build_regime_max_p_up_shadow as _build_regime_max_p_up_shadow,
)
from src.runtime.reliability_shadow_variant_support import (
    REFERENCE_FEATURE_ABLATION_THRESHOLD_VARIANT_PREFIX,
    REFERENCE_FEATURE_ABLATION_VARIANT,
    format_threshold_variant_name as _format_threshold_variant_name,
    format_reference_feature_ablation_threshold_variant_name as _format_reference_feature_ablation_threshold_variant_name,
    format_reference_feature_ablation_selection_guard_variant_name as _format_reference_feature_ablation_selection_guard_variant_name,
    format_reference_feature_ablation_abs_ret_pred_variant_name as _format_reference_feature_ablation_abs_ret_pred_variant_name,
    format_reference_feature_ablation_neutral_p_up_cap_variant_name as _format_reference_feature_ablation_neutral_p_up_cap_variant_name,
    shadow_variant_uses_reference_feature_ablation_model as _shadow_variant_uses_reference_feature_ablation_model,
    is_supported_official_shadow_variant as _is_supported_official_shadow_variant,
    official_shadow_overlap_triggered_trade_diag_path as _official_shadow_overlap_triggered_trade_diag_path,
)
from src.scripts import run_refresh_and_predict as rrp


def _annotate_monitoring_with_regime(monitoring_path: Path, regime_payload: Dict[str, Any]) -> None:
    if not monitoring_path.exists():
        return
    current = json.loads(monitoring_path.read_text(encoding="utf-8"))
    if not isinstance(current, dict):
        return
    current["regime"] = regime_payload
    monitoring_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _derive_trade_decision_regime_midband_candidate(
    *,
    candidate_path: Path,
    recent_window_rows: int,
    signal_col: str,
    p_col: str,
    ret_pred_col: str,
    return_col: str,
    regime_col: str,
    volatility_col: str = "volatility_realized_24h",
    min_regime_rows: int,
    require_overall_regime_negative: bool,
    band_step: float = 0.01,
) -> Dict[str, Any]:
    return _derive_trade_decision_regime_midband_candidate_impl(
        candidate_path=candidate_path,
        recent_window_rows=recent_window_rows,
        signal_col=signal_col,
        p_col=p_col,
        ret_pred_col=ret_pred_col,
        return_col=return_col,
        regime_col=regime_col,
        volatility_col=volatility_col,
        min_regime_rows=min_regime_rows,
        require_overall_regime_negative=require_overall_regime_negative,
        band_step=band_step,
        derive_recent_triggered_regime_volatility_rule=_derive_recent_triggered_regime_volatility_rule,
    )


def _write_trade_decision_midband_candidate_config(
    *,
    base_config_path: Path,
    candidate_path: Path,
    output_path: Path,
    meta_output_path: Path,
    recent_window_rows: int = 288,
    signal_col: str = "signal_ensemble",
    p_col: str = "p_up",
    ret_pred_col: str = "ret_pred",
    return_col: str = "ret_ensemble_net",
    regime_col: str = "regime_state",
    volatility_col: str = "volatility_realized_24h",
    min_regime_rows: int = 2,
    require_overall_regime_negative: bool = True,
    apply_to_paper_live: bool = False,
) -> Dict[str, Any]:
    return _write_trade_decision_midband_candidate_config_impl(
        base_config_path=base_config_path,
        candidate_path=candidate_path,
        output_path=output_path,
        meta_output_path=meta_output_path,
        recent_window_rows=recent_window_rows,
        signal_col=signal_col,
        p_col=p_col,
        ret_pred_col=ret_pred_col,
        return_col=return_col,
        regime_col=regime_col,
        volatility_col=volatility_col,
        min_regime_rows=min_regime_rows,
        require_overall_regime_negative=require_overall_regime_negative,
        apply_to_paper_live=apply_to_paper_live,
        derive_recent_triggered_regime_volatility_rule=_derive_recent_triggered_regime_volatility_rule,
    )


def _count_npz_rows(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        required = ("y_train", "y_val", "y_test")
        if not all(k in data.files for k in required):
            raise KeyError(f"Dataset {npz_path} missing keys {required}")
        return int(len(data["y_train"]) + len(data["y_val"]) + len(data["y_test"]))


def _count_npz_rows_from_x(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        required = ("X_train", "X_val", "X_test")
        if not all(k in data.files for k in required):
            raise KeyError(f"Dataset {npz_path} missing keys {required}")
        return int(len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]))


def _compute_feasible_walkforward(
    n_samples: int,
    *,
    folds: int,
    train_size: int,
    val_size: int,
    test_size: int,
    gap: int,
    purge: int,
    embargo: int,
) -> tuple[int, int, int, int]:
    overhead = 2 * int(gap) + 2 * int(purge) + int(embargo)
    available = n_samples - overhead
    if available < 60:
        test = max(10, available // 5)
        val = max(10, available // 4)
        train = max(20, available - val - test)
    else:
        test = max(20, min(int(test_size), max(20, available // 5)))
        val = max(20, min(int(val_size), max(20, available // 4)))
        train = max(30, min(int(train_size), available - val - test))

    while train + val + test + overhead > n_samples and train > 20:
        train -= 5
    while train + val + test + overhead > n_samples and val > 10:
        val -= 5
    while train + val + test + overhead > n_samples and test > 10:
        test -= 5

    base = train + val + test + overhead
    if base > n_samples:
        raise ValueError("Insufficient samples for even minimal walkforward setup")

    max_splits = 1 + max(0, (n_samples - base) // max(test, 1))
    resolved_folds = max(1, min(int(folds), int(max_splits)))
    return resolved_folds, train, val, test


def _run_step(
    name: str,
    cmd: List[str],
    log_path: Path,
    dry_run: bool,
    *,
    allowed_returncodes: Sequence[int] | None = None,
) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(part) for part in cmd)
    allowed = set(int(code) for code in (allowed_returncodes or [0]))
    _emit_step_event(
        name,
        "started",
        {"command": list(cmd), "log_path": str(log_path), "dry_run": bool(dry_run)},
    )
    if dry_run:
        log_path.write_text(f"[dry-run] {rendered}\n", encoding="utf-8")
        print(f"[dry-run] {name}: {rendered}")
        _emit_step_event(name, "completed", {"returncode": 0, "dry_run": True, "log_path": str(log_path)})
        return StepResult(name=name, command=cmd, returncode=0, log_path=log_path)

    print(f"\n>>> {name}")
    print(rendered)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode not in allowed:
        _emit_step_event(name, "failed", {"returncode": process.returncode, "log_path": str(log_path)})
        raise RuntimeError(f"Step '{name}' failed (exit={process.returncode}). See {log_path}")
    if process.returncode != 0:
        print(
            f"Warning: step '{name}' returned non-zero exit {process.returncode} but is configured as allowed.",
            file=sys.stderr,
        )
    _emit_step_event(name, "completed", {"returncode": process.returncode, "log_path": str(log_path)})
    return StepResult(name=name, command=cmd, returncode=process.returncode, log_path=log_path)


def _walkforward_depth_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    folds_obj = payload.get("folds", [])
    folds: List[Dict[str, Any]] = folds_obj if isinstance(folds_obj, list) else []
    n_folds = int(len(folds))
    test_rows = int(
        np.nansum(
            np.asarray(
                [float(f.get("n_test", 0.0) or 0.0) for f in folds if isinstance(f, dict)],
                dtype=float,
            )
        )
    )
    trade_count_total = int(payload.get("trade_count_total", 0) or 0)
    return {
        "path": str(path),
        "n_folds": n_folds,
        "test_rows_total": test_rows,
        "trade_count_total": trade_count_total,
    }


def _apply_joint_tuning_to_thresholds(
    thresholds_path: Path,
    *,
    joint_tuning_payload: Dict[str, Any],
    horizon: int,
) -> bool:
    if not thresholds_path.exists():
        return False
    best_obj = joint_tuning_payload.get("best")
    if not isinstance(best_obj, dict):
        return False

    p_up_min = best_obj.get("p_up_min")
    ret_min = best_obj.get("ret_min")
    if p_up_min is None or ret_min is None:
        return False

    payload = _load_json(thresholds_path)
    horizons_obj = payload.get("horizons")
    if not isinstance(horizons_obj, dict):
        return False

    key = str(int(horizon))
    current_obj = horizons_obj.get(key)
    if not isinstance(current_obj, dict):
        return False

    updated = dict(current_obj)
    updated["p_up_min"] = float(p_up_min)
    updated["ret_min"] = float(ret_min)
    updated["joint_tuning_applied"] = True
    updated["joint_tuning_source"] = "overlap_slice"
    updated["joint_tuning_best"] = {
        "p_up_min": float(p_up_min),
        "ret_min": float(ret_min),
        "direction_threshold": float(best_obj.get("direction_threshold", 0.5) or 0.5),
        "selection_value": float(best_obj.get("selection_value", best_obj.get("cum_ret", 0.0)) or 0.0),
        "economics_score": float(best_obj.get("economics_score", 0.0) or 0.0),
    }
    horizons_obj[key] = updated
    thresholds_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _xgb_dataset_for_horizon(horizon: int) -> str:
    return (
        "artifacts/datasets/btc_features_1h_splits.npz"
        if horizon == 1
        else "artifacts/datasets/btc_features_multi_horizon_splits.npz"
    )


def _direction_dataset_for_horizon(horizon: int) -> str:
    return (
        "artifacts/datasets/btc_features_1h_direction_splits.npz"
        if horizon == 1
        else "artifacts/datasets/btc_features_multi_horizon_splits.npz"
    )


def _load_numeric_returns(path: Path, col: str) -> np.ndarray:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}")
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)


def _paired_diff_stats(
    *,
    baseline_path: Path,
    candidate_path: Path,
    baseline_col: str,
    candidate_col: str,
) -> Dict[str, Any]:
    baseline = _load_numeric_returns(baseline_path, baseline_col)
    candidate = _load_numeric_returns(candidate_path, candidate_col)
    n = int(min(candidate.size, baseline.size))
    if n <= 0:
        return {
            "paired_rows": 0,
            "non_zero_paired_rows": 0,
            "pairwise_diff_std": float("nan"),
        }

    diff = candidate[-n:] - baseline[-n:]
    non_zero = int(np.count_nonzero(np.abs(diff) > 1e-12))
    std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    return {
        "paired_rows": int(n),
        "non_zero_paired_rows": non_zero,
        "pairwise_diff_std": std,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _calibration_label_from_value(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "1h"
    if text.endswith("h"):
        return text
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if numeric.is_integer():
        return f"{int(numeric)}h"
    normalized = f"{numeric:.6f}".rstrip("0").rstrip(".")
    return f"{normalized}h"


def _write_calibrated_quality_input(
    *,
    source_path: Path,
    calibration_path: Path,
    output_path: Path,
    regime_col: str = "regime_state",
) -> Dict[str, Any]:
    return _write_calibrated_quality_input_impl(
        source_path=source_path,
        calibration_path=calibration_path,
        output_path=output_path,
        regime_col=regime_col,
        safe_float=_safe_float,
        calibration_label_from_value=_calibration_label_from_value,
        resolve_trade_probability_for_horizon=rrp._resolve_trade_probability_for_horizon,
    )


def _override_cli_arg(args: Sequence[str], flag: str, value: str) -> List[str]:
    updated = list(args)
    for index, item in enumerate(updated[:-1]):
        if item == flag:
            updated[index + 1] = value
            return updated
    updated.extend([flag, value])
    return updated


def _load_policy_frame(
    path: Path,
    *,
    signal_col: str = "signal_ensemble",
    return_col: str = "ret_ensemble_net",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"ts", signal_col, return_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[["ts", signal_col, return_col]].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out[signal_col] = pd.to_numeric(out[signal_col], errors="coerce").fillna(0.0)
    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
    out = out.dropna(subset=["ts", return_col]).sort_values("ts")
    return out.reset_index(drop=True)


def _policy_trade_stats(frame: pd.DataFrame, *, signal_col: str, return_col: str) -> Dict[str, Any]:
    signal = pd.to_numeric(frame[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0)
    active = returns[signal > 0.0]
    trade_count = int(active.shape[0])
    return {
        "row_count": int(frame.shape[0]),
        "trade_count": trade_count,
        "net_return_total": float(active.sum()) if trade_count else 0.0,
        "hit_rate": float((active > 0.0).mean()) if trade_count else None,
    }


def _compute_recent_policy_slice_metrics(
    *,
    baseline_path: Path,
    candidate_path: Path,
    recent_window_rows: int,
    signal_col: str = "signal_ensemble",
    return_col: str = "ret_ensemble_net",
) -> Dict[str, Any]:
    baseline = _load_policy_frame(baseline_path, signal_col=signal_col, return_col=return_col).rename(
        columns={signal_col: "signal_baseline", return_col: "ret_baseline"}
    )
    candidate = _load_policy_frame(candidate_path, signal_col=signal_col, return_col=return_col).rename(
        columns={signal_col: "signal_candidate", return_col: "ret_candidate"}
    )
    merged = baseline.merge(candidate, on="ts", how="inner")
    if merged.empty:
        return {
            "row_count": 0,
            "baseline": {"row_count": 0, "trade_count": 0, "net_return_total": 0.0, "hit_rate": None},
            "candidate": {"row_count": 0, "trade_count": 0, "net_return_total": 0.0, "hit_rate": None},
            "delta_net_return": 0.0,
        }
    recent = merged.tail(max(int(recent_window_rows), 1)).copy()
    baseline_stats = _policy_trade_stats(recent, signal_col="signal_baseline", return_col="ret_baseline")
    candidate_stats = _policy_trade_stats(recent, signal_col="signal_candidate", return_col="ret_candidate")
    return {
        "row_count": int(recent.shape[0]),
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "delta_net_return": float(candidate_stats["net_return_total"] - baseline_stats["net_return_total"]),
    }


def _extract_recent_calibration_payload(
    calibration_payload: Dict[str, Any],
    *,
    horizon_key: str,
) -> Dict[str, Any]:
    horizon_payloads = calibration_payload.get("horizons", {}) if isinstance(calibration_payload, dict) else {}
    calibration_summary = horizon_payloads.get(horizon_key, {}) if isinstance(horizon_payloads, dict) else {}
    if not isinstance(calibration_summary, dict):
        calibration_summary = {}
    return {
        "horizon": horizon_key,
        "promotion_hardening": calibration_summary.get("promotion_hardening"),
        "recent": calibration_summary.get("recent"),
        "baseline": calibration_summary.get("baseline"),
        "ece_drift": calibration_summary.get("ece_drift"),
        "recent_diagnostics": calibration_summary.get("recent_diagnostics"),
    }


def _derive_recent_triggered_regime_volatility_rule(
    *,
    candidate_path: Path,
    recent_window_rows: int,
    signal_col: str,
    return_col: str,
    regime_col: str,
    volatility_col: str,
    min_regime_rows: int,
    require_overall_regime_negative: bool,
) -> Dict[str, Any]:
    if not candidate_path.exists():
        return {
            "enabled": False,
            "reason": "candidate_not_found",
            "candidate_path": str(candidate_path),
            "selected_regimes": [],
            "min_volatility": None,
        }

    df = pd.read_csv(candidate_path)
    required = {signal_col, return_col, regime_col, volatility_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {
            "enabled": False,
            "reason": "missing_required_columns",
            "candidate_path": str(candidate_path),
            "missing_columns": missing,
            "selected_regimes": [],
            "min_volatility": None,
        }

    working = df.copy()
    if "ts" in working.columns:
        working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
        working = working.sort_values("ts")
    recent = working.tail(max(int(recent_window_rows), 1)).copy()
    signal = pd.to_numeric(recent[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(recent[return_col], errors="coerce").fillna(0.0)
    regimes = recent[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    volatility = pd.to_numeric(recent[volatility_col], errors="coerce")
    active = recent.loc[signal > 0.0].copy()
    if active.empty:
        return {
            "enabled": False,
            "reason": "no_recent_active_trades",
            "candidate_path": str(candidate_path),
            "recent_rows": int(recent.shape[0]),
            "active_trade_rows": 0,
            "selected_regimes": [],
            "min_volatility": None,
        }

    active["_return"] = returns.loc[active.index]
    active["_regime"] = regimes.loc[active.index]
    active["_volatility"] = volatility.loc[active.index]
    active_grouped = (
        active.groupby("_regime", dropna=False)["_return"]
        .agg(
            [("row_count", "size"), ("net_return_total", "sum"), ("net_return_mean", "mean")]
        )
        .reset_index()
    )
    active_regime_summary = [
        {
            "regime_state": str(row["_regime"]),
            "row_count": int(row["row_count"]),
            "net_return_total": float(row["net_return_total"]),
            "net_return_mean": float(row["net_return_mean"]),
        }
        for _, row in active_grouped.iterrows()
    ]
    active_regime_map = {
        str(item["regime_state"]): item
        for item in active_regime_summary
    }
    negative = active.loc[active["_return"] < 0.0].copy()
    regime_summary: List[Dict[str, Any]] = []
    selected_regimes: List[str] = []
    min_volatility: float | None = None
    if not negative.empty:
        grouped = (
            negative.groupby("_regime", dropna=False)["_return"]
            .agg([("row_count", "size"), ("net_return_total", "sum"), ("net_return_mean", "mean")])
            .reset_index()
            .sort_values(["net_return_total", "net_return_mean", "row_count"], ascending=[True, True, False])
        )
        regime_summary = [
            {
                "regime_state": str(row["_regime"]),
                "row_count": int(row["row_count"]),
                "net_return_total": float(row["net_return_total"]),
                "net_return_mean": float(row["net_return_mean"]),
            }
            for _, row in grouped.iterrows()
        ]
        selected_regimes = [
            str(item["regime_state"])
            for item in regime_summary
            if int(item["row_count"]) >= int(min_regime_rows)
            and float(item["net_return_total"]) < 0.0
            and float(item["net_return_mean"]) < 0.0
            and (
                not require_overall_regime_negative
                or (
                    str(item["regime_state"]) in active_regime_map
                    and int(active_regime_map[str(item["regime_state"])].get("row_count", 0)) >= int(min_regime_rows)
                    and float(active_regime_map[str(item["regime_state"])].get("net_return_total", 0.0)) < 0.0
                    and float(active_regime_map[str(item["regime_state"])].get("net_return_mean", 0.0)) < 0.0
                )
            )
        ]
        threshold_source = negative.loc[negative["_regime"].isin(selected_regimes)].copy()
        if not threshold_source.empty:
            valid_volatility = pd.to_numeric(threshold_source["_volatility"], errors="coerce").dropna()
            if not valid_volatility.empty:
                min_volatility = float(valid_volatility.median())

    return {
        "enabled": bool(selected_regimes and min_volatility is not None and np.isfinite(min_volatility)),
        "reason": "ready" if (selected_regimes and min_volatility is not None and np.isfinite(min_volatility)) else "no_harmful_recent_regime_slice",
        "candidate_path": str(candidate_path),
        "recent_rows": int(recent.shape[0]),
        "active_trade_rows": int(active.shape[0]),
        "negative_trade_rows": int(negative.shape[0]),
        "selected_regimes": selected_regimes,
        "min_volatility": min_volatility,
        "regime_col": regime_col,
        "volatility_col": volatility_col,
        "min_regime_rows": int(min_regime_rows),
        "require_overall_regime_negative": bool(require_overall_regime_negative),
        "active_regime_summary": active_regime_summary,
        "negative_regime_summary": regime_summary,
    }


def _dedupe_selection_calibration_guard_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _dedupe_selection_calibration_guard_rules_impl(rules, safe_float=_safe_float)


def _augment_selection_guard_candidate_floors(
    *,
    base_floors: List[float],
    reference_rules: List[Dict[str, Any]],
    step: float,
    lower_steps: int,
    upper_steps: int,
) -> List[float]:
    return _augment_selection_guard_candidate_floors_impl(
        base_floors=base_floors,
        reference_rules=reference_rules,
        step=step,
        lower_steps=lower_steps,
        upper_steps=upper_steps,
        safe_float=_safe_float,
    )


def _summarize_selection_guard_recent_distribution(
    *,
    candidate_path: Path,
    recent_window_rows: int,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
) -> Dict[str, Any]:
    if not candidate_path.exists():
        return {"available": False, "reason": "candidate_not_found", "candidate_path": str(candidate_path)}
    if candidate_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(candidate_path)
    else:
        df = pd.read_csv(candidate_path)
    required = {signal_col, return_col, regime_col, p_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {
            "available": False,
            "reason": "missing_required_columns",
            "candidate_path": str(candidate_path),
            "missing_columns": missing,
        }

    working = df.copy()
    if "ts" in working.columns:
        working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
        working = working.sort_values("ts")
    recent_n = min(max(int(recent_window_rows), 1), max(len(working) // 2, 1))
    recent = working.iloc[-recent_n:].copy()
    signal = pd.to_numeric(recent[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(recent[return_col], errors="coerce").fillna(0.0)
    probabilities = pd.to_numeric(recent[p_col], errors="coerce")
    regimes = recent[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    active_mask = (signal > 0.0) & probabilities.notna()

    def _quantile_summary(values: pd.Series) -> Dict[str, Any]:
        valid = pd.to_numeric(values, errors="coerce").dropna()
        if valid.empty:
            return {
                "count": 0,
                "min": None,
                "q10": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q90": None,
                "max": None,
            }
        quantiles = valid.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        return {
            "count": int(valid.shape[0]),
            "min": float(valid.min()),
            "q10": float(quantiles.loc[0.1]),
            "q25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "q75": float(quantiles.loc[0.75]),
            "q90": float(quantiles.loc[0.9]),
            "max": float(valid.max()),
        }

    active_by_regime: List[Dict[str, Any]] = []
    active_regimes = sorted({str(value) for value in regimes.loc[active_mask].unique() if str(value)})
    for regime_state in active_regimes:
        regime_mask = active_mask & (regimes == regime_state)
        regime_probs = probabilities.loc[regime_mask]
        regime_returns = returns.loc[regime_mask]
        active_by_regime.append(
            {
                "regime_state": regime_state,
                "trade_count": int(regime_mask.sum()),
                "net_return_total": float(regime_returns.sum()) if bool(regime_mask.any()) else 0.0,
                "p_up": _quantile_summary(regime_probs),
            }
        )

    return {
        "available": True,
        "candidate_path": str(candidate_path),
        "recent_window_rows": int(recent_window_rows),
        "resolved_recent_rows": int(recent_n),
        "active_trade_count": int(active_mask.sum()),
        "active_net_return_total": float(returns.loc[active_mask].sum()) if bool(active_mask.any()) else 0.0,
        "overall_p_up": _quantile_summary(probabilities.loc[active_mask]),
        "active_by_regime": active_by_regime,
    }


def _build_selection_calibration_guard_distribution_shift(
    *,
    current_candidate_path: Path,
    source_candidate_path: Path | None,
    recent_window_rows: int,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    reference_rules: List[Dict[str, Any]],
    source_run_id: str | None,
) -> Dict[str, Any]:
    current_summary = _summarize_selection_guard_recent_distribution(
        candidate_path=current_candidate_path,
        recent_window_rows=recent_window_rows,
        signal_col=signal_col,
        return_col=return_col,
        regime_col=regime_col,
        p_col=p_col,
    )
    source_summary = (
        _summarize_selection_guard_recent_distribution(
            candidate_path=source_candidate_path,
            recent_window_rows=recent_window_rows,
            signal_col=signal_col,
            return_col=return_col,
            regime_col=regime_col,
            p_col=p_col,
        )
        if source_candidate_path is not None
        else {"available": False, "reason": "source_candidate_path_missing"}
    )

    current_regime_map = {
        str(item.get("regime_state")): item
        for item in current_summary.get("active_by_regime", [])
        if isinstance(item, dict)
    } if isinstance(current_summary, dict) else {}
    source_regime_map = {
        str(item.get("regime_state")): item
        for item in source_summary.get("active_by_regime", [])
        if isinstance(item, dict)
    } if isinstance(source_summary, dict) else {}

    rule_impacts: List[Dict[str, Any]] = []
    for rule in reference_rules:
        regime_state = str(rule.get("regime_state", "")).strip().lower()
        min_p_up = _safe_float(rule.get("min_p_up"), default=float("nan"))
        current_regime = current_regime_map.get(regime_state, {})
        source_regime = source_regime_map.get(regime_state, {})
        current_q = current_regime.get("p_up", {}) if isinstance(current_regime, dict) else {}
        source_q = source_regime.get("p_up", {}) if isinstance(source_regime, dict) else {}
        current_trade_count = int(current_regime.get("trade_count", 0)) if isinstance(current_regime, dict) else 0
        source_trade_count = int(source_regime.get("trade_count", 0)) if isinstance(source_regime, dict) else 0
        current_full_block = None
        source_full_block = None
        current_max = current_q.get("max") if isinstance(current_q, dict) else None
        source_max = source_q.get("max") if isinstance(source_q, dict) else None
        if np.isfinite(min_p_up) and current_trade_count > 0 and current_max is not None:
            current_full_block = bool(float(current_max) < float(min_p_up))
        if np.isfinite(min_p_up) and source_trade_count > 0 and source_max is not None:
            source_full_block = bool(float(source_max) < float(min_p_up))
        rule_impacts.append(
            {
                "regime_state": regime_state,
                "min_p_up": None if not np.isfinite(min_p_up) else float(min_p_up),
                "current_trade_count": current_trade_count,
                "source_trade_count": source_trade_count,
                "current_recent_p_up": current_q,
                "source_recent_p_up": source_q,
                "current_full_block": current_full_block,
                "source_full_block": source_full_block,
            }
        )

    return {
        "available": True,
        "source_run_id": source_run_id,
        "reference_rules": reference_rules,
        "current": current_summary,
        "source": source_summary,
        "rule_impacts": rule_impacts,
    }


def _summarize_trade_decision_stage_distribution(
    *,
    candidate_path: Path,
    model_path: Path,
    trade_policy_cfg: Dict[str, Any],
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    if not candidate_path.exists():
        return {"available": False, "reason": "candidate_not_found", "candidate_path": str(candidate_path)}
    if not model_path.exists():
        return {
            "available": False,
            "reason": "trade_decision_model_not_found",
            "candidate_path": str(candidate_path),
            "model_path": str(model_path),
        }
    if candidate_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(candidate_path)
    else:
        df = pd.read_csv(candidate_path)
    if df.empty:
        return {
            "available": False,
            "reason": "candidate_empty",
            "candidate_path": str(candidate_path),
            "model_path": str(model_path),
        }

    resolved_policy_cfg = dict(trade_policy_cfg or {})
    resolved_policy_cfg["enabled"] = True
    resolved_policy_cfg["model_path"] = str(model_path)
    policy = rrp._resolve_trade_decision_policy(resolved_policy_cfg)
    if not bool(policy.get("enabled", False)):
        return {
            "available": False,
            "reason": "trade_policy_disabled",
            "candidate_path": str(candidate_path),
            "model_path": str(model_path),
        }

    require_alignment = bool(policy.get("require_direction_ret_alignment", True))
    enforce_envelope = bool(policy.get("enforce_positive_oof_envelope", False))
    block_no_positive_bin = bool(policy.get("block_when_no_positive_oof_bin", True))
    allow_raw_fallback = bool(policy.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False))
    replace_threshold_rule = bool(policy.get("replace_threshold_rule", True))
    min_expected_net = float(policy.get("min_expected_net", 0.0))
    min_edge_over_fee = float(policy.get("min_edge_over_fee", 0.0))
    midband_veto_cfg = policy.get("midband_veto") if isinstance(policy.get("midband_veto"), Mapping) else {}

    stage_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        p_up = _safe_float(row.get("p_up", row.get("p_up_meta", 0.5)), default=0.5)
        ret_pred = _safe_float(row.get("ret_pred", 0.0), default=0.0)
        inferred_dir = 1 if ret_pred >= 0.0 else 0
        signal_dir_only = int(_safe_float(row.get("signal_dir_only", inferred_dir), default=float(inferred_dir)))
        regime_state = str(row.get("regime_state", rrp.REGIME_NEUTRAL)).strip().lower()
        result = {
            "p_up": p_up,
            "ret_pred": ret_pred,
            "expected_value": _safe_float(row.get("expected_value", p_up * ret_pred), default=p_up * ret_pred),
            "signal_dir_only": signal_dir_only,
            "signal_ensemble": int(_safe_float(row.get("signal_ensemble", 0.0), default=0.0)),
            "incumbent_signal_reference": _safe_float(row.get("incumbent_signal_reference", 0.0), default=0.0),
            "candidate_only_reference": _safe_float(row.get("candidate_only_reference", 0.0), default=0.0),
            "candidate_incumbent_disagreement": _safe_float(
                row.get("candidate_incumbent_disagreement", 0.0),
                default=0.0,
            ),
            "trade_action": str(row.get("trade_action", "hold")),
            "volatility": {
                "snapshot": {
                    "volatility_realized_24h": _safe_float(row.get("volatility_realized_24h", 0.0), default=0.0),
                    "volatility_ewm_24h": _safe_float(row.get("volatility_ewm_24h", 0.0), default=0.0),
                    "volatility_garch_like": _safe_float(row.get("volatility_garch_like", 0.0), default=0.0),
                }
            },
        }
        payload = rrp._apply_trade_decision_model(
            result=result,
            regime_state=regime_state,
                horizon_label=str(row.get("horizon", "1h")),
            residual_std=0.0,
            policy=policy,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        trade_probability = _safe_float(payload.get("trade_probability"), default=0.0)
        threshold = _safe_float(payload.get("threshold"), default=1.0)
        expected_net = _safe_float(payload.get("expected_net"), default=float("-inf"))
        edge_over_fee = _safe_float(payload.get("edge_over_fee"), default=float("-inf"))
        threshold_pass = trade_probability >= threshold
        expected_net_valid_pass = bool(payload.get("expected_net_valid", False))
        expected_net_pass = expected_net_valid_pass and expected_net >= min_expected_net
        edge_over_fee_pass = edge_over_fee >= min_edge_over_fee
        direction_alignment_pass = (not require_alignment) or bool(payload.get("direction_ret_aligned", True))
        positive_envelope_pass = True
        envelope_obj = payload.get("positive_oof_envelope", {})
        envelope = envelope_obj if isinstance(envelope_obj, dict) else {}
        envelope_available = bool(envelope.get("available", False))
        if enforce_envelope and envelope_available:
            has_positive_bin = bool(envelope.get("has_positive_bin", False))
            in_positive_bin = bool(envelope.get("in_positive_bin", False))
            matched_populated_bin = bool(envelope.get("matched_populated_bin", False))
            matched_positive_bin = bool(envelope.get("matched_positive_bin", False))
            envelope_mode = str(
                payload.get("positive_oof_envelope_mode", policy.get("positive_oof_envelope_mode", "strict_positive_bin"))
            ).lower()
            if envelope_mode == "populated_bin_sign" and matched_populated_bin and (not matched_positive_bin):
                positive_envelope_pass = False
            elif envelope_mode == "populated_bin_sign" and (not matched_populated_bin):
                positive_envelope_pass = False
            elif has_positive_bin and (not in_positive_bin):
                positive_envelope_pass = False
            elif (not has_positive_bin) and block_no_positive_bin:
                positive_envelope_pass = bool(payload.get("raw_ev_fallback_pass", False)) if allow_raw_fallback else False

        policy_midband_pass = True
        if replace_threshold_rule and bool(midband_veto_cfg.get("enabled", False)):
            p_up_low = float(midband_veto_cfg.get("p_up_low", 0.55))
            p_up_high = float(midband_veto_cfg.get("p_up_high", 0.60))
            high_inclusive = bool(midband_veto_cfg.get("high_inclusive", False))
            regime_filters = [
                str(value).strip().lower()
                for value in (
                    midband_veto_cfg.get("regime_states", [])
                    if isinstance(midband_veto_cfg.get("regime_states", []), list)
                    else []
                )
                if str(value).strip()
            ]
            abs_ret_pred = abs(ret_pred)
            in_band = (p_up >= p_up_low) and ((p_up <= p_up_high) if high_inclusive else (p_up < p_up_high))
            if regime_filters and regime_state not in regime_filters:
                in_band = False
            min_abs_ret_pred = midband_veto_cfg.get("min_abs_ret_pred")
            max_abs_ret_pred = midband_veto_cfg.get("max_abs_ret_pred")
            if in_band and min_abs_ret_pred is not None and abs_ret_pred < float(min_abs_ret_pred):
                in_band = False
            if in_band and max_abs_ret_pred is not None and abs_ret_pred >= float(max_abs_ret_pred):
                in_band = False
            if in_band:
                policy_midband_pass = False

        stage_rows.append(
            {
                "regime_state": regime_state,
                "p_up": p_up,
                "trade_probability": trade_probability,
                "threshold_pass": threshold_pass,
                "expected_net_valid_pass": expected_net_valid_pass,
                "expected_net_pass": expected_net_pass,
                "edge_over_fee_pass": edge_over_fee_pass,
                "direction_alignment_pass": direction_alignment_pass,
                "positive_envelope_pass": positive_envelope_pass,
                "policy_midband_pass": policy_midband_pass,
                "triggered": bool(payload.get("triggered", False)),
            }
        )

    stage_df = pd.DataFrame(stage_rows)

    def _stage_snapshot(name: str, mask: pd.Series) -> Dict[str, Any]:
        valid_mask = mask.fillna(False)
        active = stage_df.loc[valid_mask]
        regime_counts = active["regime_state"].value_counts().to_dict() if not active.empty else {}
        return {
            "stage": name,
            "count": int(valid_mask.sum()),
            "share": float(valid_mask.sum() / max(len(stage_df), 1)),
            "regime_counts": {str(key): int(value) for key, value in regime_counts.items()},
        }

    stage_sequence = [
        ("all_rows", None),
        ("threshold_pass", "threshold_pass"),
        ("expected_net_valid_pass", "expected_net_valid_pass"),
        ("expected_net_pass", "expected_net_pass"),
        ("edge_over_fee_pass", "edge_over_fee_pass"),
        ("direction_alignment_pass", "direction_alignment_pass"),
        ("positive_envelope_pass", "positive_envelope_pass"),
        ("policy_midband_pass", "policy_midband_pass"),
        ("triggered", "triggered"),
    ]
    cumulative_mask = pd.Series(True, index=stage_df.index, dtype=bool)
    stages: List[Dict[str, Any]] = []
    for stage_name, column_name in stage_sequence:
        if column_name is not None:
            cumulative_mask = cumulative_mask & stage_df[column_name].fillna(False).astype(bool)
        stages.append(_stage_snapshot(stage_name, cumulative_mask.copy()))

    stage_map = {str(item.get("stage")): item for item in stages}
    dominant_drop: Dict[str, Any] | None = None
    previous_count = len(stage_df)
    for stage_name, _ in stage_sequence[1:]:
        current_count = int(stage_map.get(stage_name, {}).get("count", 0))
        dropped = previous_count - current_count
        candidate = {"stage": stage_name, "dropped_rows": int(dropped), "previous_count": int(previous_count), "count": int(current_count)}
        if dominant_drop is None or int(candidate["dropped_rows"]) > int(dominant_drop.get("dropped_rows", -1)):
            dominant_drop = candidate
        previous_count = current_count

    return {
        "available": True,
        "candidate_path": str(candidate_path),
        "model_path": str(model_path),
        "rows": int(len(stage_df)),
        "stages": stages,
        "dominant_drop": dominant_drop,
        "policy": {
            "threshold": float(policy.get("threshold", 0.55)),
            "require_direction_ret_alignment": require_alignment,
            "enforce_positive_oof_envelope": enforce_envelope,
            "positive_oof_envelope_mode": str(policy.get("positive_oof_envelope_mode", "strict_positive_bin")),
            "replace_threshold_rule": replace_threshold_rule,
            "midband_veto": {
                "enabled": bool(midband_veto_cfg.get("enabled", False)),
                "p_up_low": float(midband_veto_cfg.get("p_up_low", 0.55)),
                "p_up_high": float(midband_veto_cfg.get("p_up_high", 0.60)),
                "high_inclusive": bool(midband_veto_cfg.get("high_inclusive", False)),
                "min_abs_ret_pred": (
                    float(midband_veto_cfg.get("min_abs_ret_pred"))
                    if midband_veto_cfg.get("min_abs_ret_pred") is not None
                    else None
                ),
                "max_abs_ret_pred": (
                    float(midband_veto_cfg.get("max_abs_ret_pred"))
                    if midband_veto_cfg.get("max_abs_ret_pred") is not None
                    else None
                ),
                "regime_states": [
                    str(value).strip().lower()
                    for value in (
                        midband_veto_cfg.get("regime_states", [])
                        if isinstance(midband_veto_cfg.get("regime_states", []), list)
                        else []
                    )
                    if str(value).strip()
                ],
            },
        },
    }


def _load_trade_decision_rule_occurrences(diagnostics_path: Path | None) -> Dict[str, int]:
    if diagnostics_path is None or not diagnostics_path.exists():
        return {}
    payload = _load_json(diagnostics_path)
    rules = payload.get("rules", {}) if isinstance(payload, dict) else {}
    if not isinstance(rules, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in rules.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def _build_trade_decision_distribution_shift(
    *,
    current_candidate_path: Path,
    current_model_path: Path,
    source_candidate_path: Path | None,
    source_model_path: Path | None,
    trade_policy_cfg: Dict[str, Any],
    fee_bps: float,
    slippage_bps: float,
    current_diagnostics_path: Path | None,
    source_diagnostics_path: Path | None,
    source_run_id: str | None,
) -> Dict[str, Any]:
    current_summary = _summarize_trade_decision_stage_distribution(
        candidate_path=current_candidate_path,
        model_path=current_model_path,
        trade_policy_cfg=trade_policy_cfg,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    source_summary = (
        _summarize_trade_decision_stage_distribution(
            candidate_path=source_candidate_path,
            model_path=source_model_path,
            trade_policy_cfg=trade_policy_cfg,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        if source_candidate_path is not None and source_model_path is not None
        else {"available": False, "reason": "source_candidate_or_model_missing"}
    )
    current_rules = _load_trade_decision_rule_occurrences(current_diagnostics_path)
    source_rules = _load_trade_decision_rule_occurrences(source_diagnostics_path)

    current_stage_map = {
        str(item.get("stage")): item
        for item in current_summary.get("stages", [])
        if isinstance(item, dict)
    } if isinstance(current_summary, dict) else {}
    source_stage_map = {
        str(item.get("stage")): item
        for item in source_summary.get("stages", [])
        if isinstance(item, dict)
    } if isinstance(source_summary, dict) else {}
    stage_names = []
    for name in [*current_stage_map.keys(), *source_stage_map.keys()]:
        if name not in stage_names:
            stage_names.append(name)

    stage_deltas: List[Dict[str, Any]] = []
    dominant_stage_drop: Dict[str, Any] | None = None
    for stage_name in stage_names:
        current_count = int(current_stage_map.get(stage_name, {}).get("count", 0))
        source_count = int(source_stage_map.get(stage_name, {}).get("count", 0))
        current_share = _safe_float(current_stage_map.get(stage_name, {}).get("share"), default=0.0)
        source_share = _safe_float(source_stage_map.get(stage_name, {}).get("share"), default=0.0)
        delta = {
            "stage": stage_name,
            "current_count": current_count,
            "source_count": source_count,
            "count_delta": int(current_count - source_count),
            "current_share": float(current_share),
            "source_share": float(source_share),
            "share_delta": float(current_share - source_share),
        }
        stage_deltas.append(delta)
        if dominant_stage_drop is None or int(delta["count_delta"]) < int(dominant_stage_drop.get("count_delta", 0)):
            dominant_stage_drop = delta

    rule_names = []
    for name in [*current_rules.keys(), *source_rules.keys()]:
        if name not in rule_names:
            rule_names.append(name)
    rule_occurrence_deltas: List[Dict[str, Any]] = []
    dominant_rule_increase: Dict[str, Any] | None = None
    for rule_name in rule_names:
        current_count = int(current_rules.get(rule_name, 0))
        source_count = int(source_rules.get(rule_name, 0))
        delta = {
            "rule": rule_name,
            "current_count": current_count,
            "source_count": source_count,
            "count_delta": int(current_count - source_count),
        }
        rule_occurrence_deltas.append(delta)
        if dominant_rule_increase is None or int(delta["count_delta"]) > int(dominant_rule_increase.get("count_delta", 0)):
            dominant_rule_increase = delta

    return {
        "available": bool(current_summary.get("available", False)),
        "source_run_id": source_run_id,
        "current": current_summary,
        "source": source_summary,
        "stage_deltas": stage_deltas,
        "dominant_stage_drop": dominant_stage_drop,
        "rule_occurrence_deltas": rule_occurrence_deltas,
        "dominant_rule_increase": dominant_rule_increase,
    }


def _materialize_trade_decision_feature_frame(
    *,
    candidate_path: Path,
    model_payload: Dict[str, Any],
) -> pd.DataFrame:
    from src.utils.component_diversity_support import build_component_feature_frame

    if candidate_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(candidate_path)
    else:
        df = pd.read_csv(candidate_path)
    feature_columns = [str(value) for value in model_payload.get("feature_columns", [])]
    working = df.copy()
    ret_pred_series = pd.to_numeric(working.get("ret_pred", 0.0), errors="coerce").fillna(0.0)
    p_up_series = pd.to_numeric(working.get("p_up", working.get("p_up_meta", 0.5)), errors="coerce").fillna(0.5)
    raw_p_up_series = pd.to_numeric(working.get("raw_p_up", p_up_series), errors="coerce").fillna(p_up_series)
    close_source = working.get("close")
    if close_source is None:
        close_source = pd.Series(0.0, index=working.index, dtype=float)
    close_series = pd.to_numeric(close_source, errors="coerce").fillna(0.0)
    projected_source = working.get("projected_price")
    if projected_source is None:
        projected_source = close_series
    projected_price_series = pd.to_numeric(projected_source, errors="coerce").fillna(close_series)
    direction_series = working.get("direction_next", pd.Series(index=working.index, dtype=object)).map(
        lambda value: str(value).strip().lower() if pd.notna(value) else "neutral"
    )
    regime_series = working.get("regime_state", pd.Series(index=working.index, dtype=object)).map(
        lambda value: str(value).strip().lower() if pd.notna(value) else "missing"
    )

    ret_side = pd.Series("neutral", index=working.index, dtype=object)
    ret_side.loc[ret_pred_series > 0.0] = "up"
    ret_side.loc[ret_pred_series < 0.0] = "down"
    projected_side = pd.Series("neutral", index=working.index, dtype=object)
    valid_prices = (close_series > 0.0) & (projected_price_series > 0.0)
    projected_side.loc[valid_prices & (projected_price_series > close_series)] = "up"
    projected_side.loc[valid_prices & (projected_price_series < close_series)] = "down"
    raw_side = pd.Series("neutral", index=working.index, dtype=object)
    raw_side.loc[raw_p_up_series >= 0.52] = "up"
    raw_side.loc[raw_p_up_series <= 0.48] = "down"
    resolved_side = pd.Series("neutral", index=working.index, dtype=object)
    resolved_side.loc[p_up_series >= 0.52] = "up"
    resolved_side.loc[p_up_series <= 0.48] = "down"
    component_columns = [
        str(column)
        for column in working.columns
        if str(column).startswith("p_up_") and str(column) not in {"p_up_meta", "p_up_gate"}
    ]
    component_features = build_component_feature_frame(working, component_columns)
    for column in feature_columns:
        if column == "expected_value_proxy":
            working[column] = p_up_series * ret_pred_series
        elif column == "abs_ret_pred":
            working[column] = ret_pred_series.abs()
        elif column == "raw_p_up":
            working[column] = raw_p_up_series
        elif column == "raw_calibrated_probability_gap":
            working[column] = p_up_series - raw_p_up_series
        elif column == "probability_alignment_gap":
            working[column] = (p_up_series - raw_p_up_series).abs()
        elif column == "raw_p_up_ret_mismatch":
            working[column] = ((raw_side != "neutral") & (ret_side != "neutral") & (raw_side != ret_side)).astype(float)
        elif column == "p_up_ret_mismatch":
            working[column] = ((resolved_side != "neutral") & (ret_side != "neutral") & (resolved_side != ret_side)).astype(float)
        elif column == "raw_p_up_direction_mismatch":
            working[column] = ((raw_side != "neutral") & (direction_series != "neutral") & (raw_side != direction_series)).astype(float)
        elif column == "p_up_direction_mismatch":
            working[column] = ((resolved_side != "neutral") & (direction_series != "neutral") & (resolved_side != direction_series)).astype(float)
        elif column == "ret_projected_price_consensus":
            working[column] = ((ret_side == projected_side) & (ret_side != "neutral")).astype(float)
        elif column == "probability_calibration_guard_applied":
            working[column] = pd.to_numeric(working.get(column, 0.0), errors="coerce").fillna(0.0)
        elif column == "probability_calibration_used_regime_key":
            working[column] = pd.to_numeric(working.get(column, 0.0), errors="coerce").fillna(0.0)
        elif column == "regime_is_trend":
            working[column] = (regime_series == "trend_ignition").astype(float)
        elif column == "regime_is_neutral":
            working[column] = (regime_series == "neutral").astype(float)
        elif column == "regime_is_chop":
            working[column] = (regime_series == "chop").astype(float)
        elif column in component_features.columns:
            working[column] = pd.to_numeric(component_features[column], errors="coerce").fillna(0.0)
        else:
            source = working[column] if column in working.columns else pd.Series(0.0, index=working.index, dtype=float)
            working[column] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    return working


def _build_trade_decision_model_shift(
    *,
    current_candidate_path: Path,
    current_model_path: Path,
    current_feature_meta_path: Path | None,
    source_candidate_path: Path | None,
    source_model_path: Path | None,
    source_feature_meta_path: Path | None,
    source_run_id: str | None,
) -> Dict[str, Any]:
    if not current_candidate_path.exists():
        return {"available": False, "reason": "current_candidate_not_found", "candidate_path": str(current_candidate_path)}
    if not current_model_path.exists():
        return {"available": False, "reason": "current_model_not_found", "model_path": str(current_model_path)}
    if source_candidate_path is None or not source_candidate_path.exists():
        return {"available": False, "reason": "source_candidate_not_found", "source_run_id": source_run_id}
    if source_model_path is None or not source_model_path.exists():
        return {"available": False, "reason": "source_model_not_found", "source_run_id": source_run_id}

    current_model_payload = _load_json(current_model_path)
    source_model_payload = _load_json(source_model_path)
    current_df = _materialize_trade_decision_feature_frame(
        candidate_path=current_candidate_path,
        model_payload=current_model_payload,
    )
    source_df = _materialize_trade_decision_feature_frame(
        candidate_path=source_candidate_path,
        model_payload=source_model_payload,
    )

    current_features = [str(value) for value in current_model_payload.get("feature_columns", [])]
    source_features = [str(value) for value in source_model_payload.get("feature_columns", [])]
    feature_names: List[str] = []
    for name in [*current_features, *source_features]:
        if name not in feature_names:
            feature_names.append(name)

    source_coef_map = {
        str(name): float(value)
        for name, value in zip(source_features, source_model_payload.get("coefficients", []))
    }
    current_coef_map = {
        str(name): float(value)
        for name, value in zip(current_features, current_model_payload.get("coefficients", []))
    }

    coefficient_deltas: List[Dict[str, Any]] = [
        {
            "feature": "__intercept__",
            "source_coef": float(source_model_payload.get("intercept", 0.0)),
            "current_coef": float(current_model_payload.get("intercept", 0.0)),
            "coef_delta": float(current_model_payload.get("intercept", 0.0)) - float(source_model_payload.get("intercept", 0.0)),
        }
    ]
    contribution_deltas: List[Dict[str, Any]] = []
    feature_shift_effects: List[Dict[str, Any]] = []
    for feature_name in feature_names:
        source_coef = float(source_coef_map.get(feature_name, 0.0))
        current_coef = float(current_coef_map.get(feature_name, 0.0))
        source_mean = float(pd.to_numeric(source_df.get(feature_name, 0.0), errors="coerce").fillna(0.0).mean())
        current_mean = float(pd.to_numeric(current_df.get(feature_name, 0.0), errors="coerce").fillna(0.0).mean())
        coefficient_deltas.append(
            {
                "feature": feature_name,
                "source_coef": source_coef,
                "current_coef": current_coef,
                "coef_delta": current_coef - source_coef,
            }
        )
        contribution_deltas.append(
            {
                "feature": feature_name,
                "source_coef": source_coef,
                "current_coef": current_coef,
                "source_mean": source_mean,
                "current_mean": current_mean,
                "contrib_delta_on_current_mean": (current_coef - source_coef) * current_mean,
            }
        )
        feature_shift_effects.append(
            {
                "feature": feature_name,
                "current_coef": current_coef,
                "source_mean": source_mean,
                "current_mean": current_mean,
                "feature_shift_effect_under_current_coef": current_coef * (current_mean - source_mean),
            }
        )

    def _sigmoid(value: float) -> float:
        clipped = max(min(float(value), 60.0), -60.0)
        return float(1.0 / (1.0 + np.exp(-clipped)))

    current_threshold = float(current_model_payload.get("threshold", 0.55))
    source_threshold = float(source_model_payload.get("threshold", 0.55))
    current_rows = current_df.copy()
    current_logit_under_source = np.full(len(current_rows), float(source_model_payload.get("intercept", 0.0)), dtype=float)
    current_logit_under_current = np.full(len(current_rows), float(current_model_payload.get("intercept", 0.0)), dtype=float)
    for feature_name in feature_names:
        values = pd.to_numeric(current_rows.get(feature_name, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        current_logit_under_source += float(source_coef_map.get(feature_name, 0.0)) * values
        current_logit_under_current += float(current_coef_map.get(feature_name, 0.0)) * values
    current_prob_under_source = pd.Series([_sigmoid(value) for value in current_logit_under_source], index=current_rows.index)
    current_prob_under_current = pd.Series([_sigmoid(value) for value in current_logit_under_current], index=current_rows.index)
    source_not_current_mask = (current_prob_under_source >= source_threshold) & (current_prob_under_current < current_threshold)
    current_not_source_mask = (current_prob_under_current >= current_threshold) & (current_prob_under_source < source_threshold)
    source_not_current_rows = current_rows.loc[source_not_current_mask].copy()

    current_feature_meta = _load_json(current_feature_meta_path) if current_feature_meta_path is not None and current_feature_meta_path.exists() else {}
    source_feature_meta = _load_json(source_feature_meta_path) if source_feature_meta_path is not None and source_feature_meta_path.exists() else {}
    current_reference_meta = current_feature_meta.get("incumbent_reference", {}) if isinstance(current_feature_meta, dict) else {}
    source_reference_meta = source_feature_meta.get("incumbent_reference", {}) if isinstance(source_feature_meta, dict) else {}

    return {
        "available": True,
        "source_run_id": source_run_id,
        "current_candidate_path": str(current_candidate_path),
        "source_candidate_path": str(source_candidate_path),
        "current_model_path": str(current_model_path),
        "source_model_path": str(source_model_path),
        "reference_sources": {
            "current": {
                "source": current_reference_meta.get("source"),
                "rows_with_reference": int(_safe_float(current_reference_meta.get("rows_with_reference"), default=0.0)),
                "candidate_only_rows": int(_safe_float(current_reference_meta.get("candidate_only_rows"), default=0.0)),
                "disagreement_rows": int(_safe_float(current_reference_meta.get("disagreement_rows"), default=0.0)),
            },
            "source": {
                "source": source_reference_meta.get("source"),
                "rows_with_reference": int(_safe_float(source_reference_meta.get("rows_with_reference"), default=0.0)),
                "candidate_only_rows": int(_safe_float(source_reference_meta.get("candidate_only_rows"), default=0.0)),
                "disagreement_rows": int(_safe_float(source_reference_meta.get("disagreement_rows"), default=0.0)),
            },
        },
        "top_coefficient_deltas": sorted(
            coefficient_deltas,
            key=lambda item: abs(float(item.get("coef_delta", 0.0))),
            reverse=True,
        )[:10],
        "top_contribution_deltas_on_current_mean": sorted(
            contribution_deltas,
            key=lambda item: abs(float(item.get("contrib_delta_on_current_mean", 0.0))),
            reverse=True,
        )[:10],
        "top_feature_shift_effects_under_current_coef": sorted(
            feature_shift_effects,
            key=lambda item: abs(float(item.get("feature_shift_effect_under_current_coef", 0.0))),
            reverse=True,
        )[:10],
        "counterfactual_threshold_pass": {
            "current_rows_under_source_model": int((current_prob_under_source >= source_threshold).sum()),
            "current_rows_under_current_model": int((current_prob_under_current >= current_threshold).sum()),
            "source_threshold": source_threshold,
            "current_threshold": current_threshold,
            "source_not_current_count": int(source_not_current_mask.sum()),
            "current_not_source_count": int(current_not_source_mask.sum()),
            "source_not_current_regime_counts": {
                str(key): int(value)
                for key, value in source_not_current_rows.get("regime_state", pd.Series(dtype=object)).astype(str).value_counts().to_dict().items()
            },
            "source_not_current_feature_means": {
                field: float(pd.to_numeric(source_not_current_rows.get(field, 0.0), errors="coerce").fillna(0.0).mean())
                for field in [
                    "p_up",
                    "ret_pred",
                    "incumbent_signal_reference",
                    "candidate_only_reference",
                    "candidate_incumbent_disagreement",
                ]
            },
        },
    }


def _build_trade_decision_ablation_comparison(
    *,
    candidate_path: Path,
    base_model_path: Path,
    ablation_model_path: Path,
    trade_policy_cfg: Dict[str, Any],
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    base_summary = _summarize_trade_decision_stage_distribution(
        candidate_path=candidate_path,
        model_path=base_model_path,
        trade_policy_cfg=trade_policy_cfg,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    ablation_summary = _summarize_trade_decision_stage_distribution(
        candidate_path=candidate_path,
        model_path=ablation_model_path,
        trade_policy_cfg=trade_policy_cfg,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    if not bool(base_summary.get("available", False)) or not bool(ablation_summary.get("available", False)):
        return {
            "available": False,
            "reason": "base_or_ablation_summary_unavailable",
            "base": base_summary,
            "ablation": ablation_summary,
        }

    def _stage_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            str(item.get("stage")): item
            for item in summary.get("stages", [])
            if isinstance(item, dict)
        }

    base_stage_map = _stage_map(base_summary)
    ablation_stage_map = _stage_map(ablation_summary)
    stage_names = []
    for name in [*base_stage_map.keys(), *ablation_stage_map.keys()]:
        if name not in stage_names:
            stage_names.append(name)

    stage_deltas: List[Dict[str, Any]] = []
    for stage_name in stage_names:
        base_stage = base_stage_map.get(stage_name, {})
        ablation_stage = ablation_stage_map.get(stage_name, {})
        stage_deltas.append(
            {
                "stage": stage_name,
                "base_count": int(base_stage.get("count", 0)),
                "ablation_count": int(ablation_stage.get("count", 0)),
                "count_delta": int(ablation_stage.get("count", 0)) - int(base_stage.get("count", 0)),
                "base_regime_counts": base_stage.get("regime_counts", {}),
                "ablation_regime_counts": ablation_stage.get("regime_counts", {}),
            }
        )

    base_model_payload = _load_json(base_model_path) if base_model_path.exists() else {}
    ablation_model_payload = _load_json(ablation_model_path) if ablation_model_path.exists() else {}
    return {
        "available": True,
        "candidate_path": str(candidate_path),
        "base_model_path": str(base_model_path),
        "ablation_model_path": str(ablation_model_path),
        "base": base_summary,
        "ablation": ablation_summary,
        "stage_deltas": stage_deltas,
        "base_reference_feature_controls": base_model_payload.get("reference_feature_controls", {}),
        "ablation_reference_feature_controls": ablation_model_payload.get("reference_feature_controls", {}),
    }


def _build_trade_decision_model_shift_guard(
    *,
    model_shift_payload: Dict[str, Any] | None,
    guard_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return _build_trade_decision_model_shift_guard_impl(
        model_shift_payload=model_shift_payload,
        guard_cfg=guard_cfg,
        safe_float=_safe_float,
    )


def _apply_trade_decision_model_shift_guard(
    *,
    summary_dir: Path,
    promotion_gate_payload: Dict[str, Any] | None,
    trade_decision_cfg: Dict[str, Any],
    model_shift_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return _apply_trade_decision_model_shift_guard_impl(
        summary_dir=summary_dir,
        promotion_gate_payload=promotion_gate_payload,
        trade_decision_cfg=trade_decision_cfg,
        model_shift_payload=model_shift_payload,
        safe_float=_safe_float,
    )


def _evaluate_selection_calibration_guard_rule_viability(
    *,
    candidate_path: Path,
    rules: List[Dict[str, Any]],
    recent_window_rows: int,
    baseline_window_rows: int,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    y_col: str,
    min_selection_rows: int,
    adaptive_selection_cfg: Dict[str, Any],
    min_candidate_trades: int,
) -> Dict[str, Any]:
    if not candidate_path.exists():
        return {
            "enabled": False,
            "reason": "candidate_not_found",
            "candidate_path": str(candidate_path),
            "rules": rules,
        }
    if candidate_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(candidate_path)
    else:
        df = pd.read_csv(candidate_path)
    required = {signal_col, return_col, regime_col, p_col, y_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {
            "enabled": False,
            "reason": "missing_required_columns",
            "candidate_path": str(candidate_path),
            "missing_columns": missing,
            "rules": rules,
        }

    working = df.copy()
    if "ts" in working.columns:
        working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
        working = working.sort_values("ts")
    recent_n = min(max(int(recent_window_rows), 1), max(len(working) // 2, 1))
    recent = working.iloc[-recent_n:].copy()
    baseline_pool = working.iloc[:-recent_n]
    baseline_n = min(max(int(baseline_window_rows), 0), len(baseline_pool))
    baseline = baseline_pool.iloc[-baseline_n:].copy() if baseline_n > 0 else baseline_pool.iloc[0:0].copy()

    full_guarded, blocked_full = _apply_selection_calibration_guard_rules_to_frame(
        working,
        signal_col=signal_col,
        return_col=return_col,
        regime_col=regime_col,
        p_col=p_col,
        rules=rules,
    )
    recent_guarded, blocked_recent = _apply_selection_calibration_guard_rules_to_frame(
        recent,
        signal_col=signal_col,
        return_col=return_col,
        regime_col=regime_col,
        p_col=p_col,
        rules=rules,
    )
    baseline_guarded, _ = _apply_selection_calibration_guard_rules_to_frame(
        baseline,
        signal_col=signal_col,
        return_col=return_col,
        regime_col=regime_col,
        p_col=p_col,
        rules=rules,
    )

    recent_metrics = _selection_guard_active_metrics(
        recent_guarded,
        signal_col=signal_col,
        return_col=return_col,
        p_col=p_col,
        y_col=y_col,
    )
    baseline_metrics = _selection_guard_active_metrics(
        baseline_guarded,
        signal_col=signal_col,
        return_col=return_col,
        p_col=p_col,
        y_col=y_col,
    )
    full_signal = pd.to_numeric(full_guarded[signal_col], errors="coerce").fillna(0.0)
    full_returns = pd.to_numeric(full_guarded[return_col], errors="coerce").fillna(0.0)
    guarded_trade_count = int((full_signal > 0.0).sum())
    guarded_net_return_total = float(full_returns.loc[full_signal > 0.0].sum()) if guarded_trade_count > 0 else 0.0

    row_policy = _resolve_selection_row_policy(
        recent_selection_rows=int(recent_metrics.get("rows", 0)),
        baseline_selection_rows=int(baseline_metrics.get("rows", 0)),
        strict_min_selection_rows=int(min_selection_rows),
        adaptive_enabled=bool(adaptive_selection_cfg.get("enabled", False)),
        adaptive_min_floor=int(adaptive_selection_cfg.get("min_floor", 0)),
        adaptive_baseline_ratio=float(adaptive_selection_cfg.get("baseline_ratio", 0.0)),
        adaptive_max_shortfall=int(adaptive_selection_cfg.get("max_shortfall", 0)),
    )

    errors: List[str] = []
    if not bool(row_policy.get("effective_ok", False)):
        errors.append("recent_selection_rows_below_effective_min")
    if guarded_trade_count < int(min_candidate_trades):
        errors.append("guarded_trade_count_below_min_candidate_trades")

    blocked_recent_returns = pd.to_numeric(recent[return_col], errors="coerce").fillna(0.0)
    return {
        "enabled": not errors,
        "reason": "ready" if not errors else "guard_reuse_not_viable",
        "candidate_path": str(candidate_path),
        "rules": rules,
        "errors": errors,
        "recent_metrics": recent_metrics,
        "baseline_metrics": baseline_metrics,
        "row_policy": row_policy,
        "guarded_trade_count": guarded_trade_count,
        "guarded_net_return_total": guarded_net_return_total,
        "blocked_recent_rows": int(blocked_recent.sum()),
        "blocked_recent_net_return_total": float(blocked_recent_returns.loc[blocked_recent].sum()) if bool(blocked_recent.any()) else 0.0,
        "blocked_total_rows": int(blocked_full.sum()),
        "min_candidate_trades": int(min_candidate_trades),
        "resolved_recent_rows": int(recent_n),
        "resolved_baseline_rows": int(baseline_n),
    }


def _selection_guard_expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if y_true.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = max(int(y_true.size), 1)
    for index in range(len(edges) - 1):
        lo, hi = float(edges[index]), float(edges[index + 1])
        mask = (p >= lo) & (p < hi if index < len(edges) - 2 else p <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p[mask]))
        ece += (float(np.sum(mask)) / float(n)) * abs(acc - conf)
    return float(ece)


def _selection_guard_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _resolve_selection_row_policy(
    *,
    recent_selection_rows: int,
    baseline_selection_rows: int,
    strict_min_selection_rows: int,
    adaptive_enabled: bool,
    adaptive_min_floor: int,
    adaptive_baseline_ratio: float,
    adaptive_max_shortfall: int,
) -> Dict[str, Any]:
    strict_min_rows = max(int(strict_min_selection_rows), 0)
    baseline_rows = max(int(baseline_selection_rows), 0)
    recent_rows = max(int(recent_selection_rows), 0)
    min_floor = max(int(adaptive_min_floor), 0)
    max_shortfall = max(int(adaptive_max_shortfall), 0)
    baseline_ratio = max(float(adaptive_baseline_ratio), 0.0)
    baseline_ratio_rows = int(np.ceil(baseline_rows * baseline_ratio)) if baseline_rows > 0 else 0

    effective_min_rows = strict_min_rows
    if adaptive_enabled and strict_min_rows > 0:
        effective_min_rows = min(
            strict_min_rows,
            max(min_floor, baseline_ratio_rows, strict_min_rows - max_shortfall),
        )

    strict_ok = bool(recent_rows >= strict_min_rows)
    effective_ok = bool(recent_rows >= effective_min_rows)
    borderline_eligible = bool(adaptive_enabled and effective_ok and not strict_ok)
    return {
        "adaptive_enabled": bool(adaptive_enabled),
        "recent_selection_rows": recent_rows,
        "baseline_selection_rows": baseline_rows,
        "strict_min_selection_rows": strict_min_rows,
        "effective_min_selection_rows": effective_min_rows,
        "adaptive_min_floor": min_floor,
        "adaptive_baseline_ratio": baseline_ratio,
        "adaptive_baseline_ratio_rows": baseline_ratio_rows,
        "adaptive_max_shortfall": max_shortfall,
        "strict_ok": strict_ok,
        "effective_ok": effective_ok,
        "row_shortfall_vs_strict": max(strict_min_rows - recent_rows, 0),
        "row_shortfall_vs_effective": max(effective_min_rows - recent_rows, 0),
        "borderline_exception_eligible": borderline_eligible,
    }


def _selection_guard_active_metrics(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    return_col: str,
    p_col: str,
    y_col: str,
) -> Dict[str, Any]:
    signal = pd.to_numeric(frame[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(frame[return_col], errors="coerce").fillna(0.0)
    probabilities = pd.to_numeric(frame[p_col], errors="coerce")
    outcomes = pd.to_numeric(frame[y_col], errors="coerce")
    mask = (signal > 0.0) & probabilities.notna() & outcomes.notna()
    if not bool(mask.any()):
        return {"rows": 0, "auc": float("nan"), "ece_10": float("nan"), "net_return_total": 0.0}
    y = outcomes.loc[mask].to_numpy(dtype=float)
    p = np.clip(probabilities.loc[mask].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    active_returns = returns.loc[mask]
    return {
        "rows": int(mask.sum()),
        "auc": _selection_guard_auc(y.astype(int), p),
        "ece_10": _selection_guard_expected_calibration_error(y.astype(int), p, bins=10),
        "net_return_total": float(active_returns.sum()),
    }


def _apply_selection_calibration_guard_rules_to_frame(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    rules: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.Series]:
    working = frame.copy()
    signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    probabilities = pd.to_numeric(working[p_col], errors="coerce")
    regimes = working[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    blocked_mask = pd.Series(False, index=working.index)
    for rule in rules:
        regime_state = str(rule.get("regime_state", "")).strip().lower()
        min_p_up = _safe_float(rule.get("min_p_up"), default=float("nan"))
        if not regime_state or not np.isfinite(min_p_up):
            continue
        rule_mask = (signal > 0.0) & probabilities.notna() & (regimes == regime_state) & (probabilities < float(min_p_up))
        blocked_mask = blocked_mask | rule_mask
    working.loc[blocked_mask, signal_col] = 0.0
    working.loc[blocked_mask, return_col] = 0.0
    return working, blocked_mask


def _derive_selection_calibration_guard_rules(
    *,
    candidate_path: Path,
    recent_window_rows: int,
    baseline_window_rows: int,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    y_col: str,
    min_selection_rows: int,
    adaptive_selection_cfg: Dict[str, Any],
    floors: List[float],
    min_blocked_recent_rows: int,
    max_rules: int,
    min_recent_ece_improvement: float,
    min_ece_drift_improvement: float,
    max_recent_ece: float | None,
    max_ece_drift: float | None,
    min_recent_auc: float | None,
    require_blocked_recent_net_nonpositive: bool,
    max_blocked_recent_net_return_total: float | None,
    require_recent_net_nonnegative: bool,
    sparse_active_trade_cap: int,
    sparse_min_blocked_recent_rows: int,
    sparse_min_retained_recent_rows: int,
    sparse_allow_row_policy_override: bool,
    sparse_allow_missing_baseline: bool,
    sparse_use_observed_p_up_values: bool,
) -> Dict[str, Any]:
    if not candidate_path.exists():
        return {"enabled": False, "reason": "candidate_not_found", "rules": [], "steps": []}
    if candidate_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(candidate_path)
    else:
        df = pd.read_csv(candidate_path)
    required = {signal_col, return_col, regime_col, p_col, y_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {
            "enabled": False,
            "reason": "missing_required_columns",
            "missing_columns": missing,
            "rules": [],
            "steps": [],
        }
    working = df.copy()
    if "ts" in working.columns:
        working["ts"] = pd.to_datetime(working["ts"], utc=True, errors="coerce")
        working = working.sort_values("ts")
    recent_n = min(max(int(recent_window_rows), 1), max(len(working) // 2, 1))
    recent = working.iloc[-recent_n:].copy()
    baseline_pool = working.iloc[:-recent_n]
    baseline_n = min(max(int(baseline_window_rows), 0), len(baseline_pool))
    baseline = baseline_pool.iloc[-baseline_n:].copy() if baseline_n > 0 else baseline_pool.iloc[0:0].copy()

    current_recent = recent.copy()
    current_baseline = baseline.copy()
    selected_rules: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    adaptive_enabled = bool(adaptive_selection_cfg.get("enabled", False))
    adaptive_min_floor = int(adaptive_selection_cfg.get("min_floor", 0))
    adaptive_baseline_ratio = float(adaptive_selection_cfg.get("baseline_ratio", 0.0))
    adaptive_max_shortfall = int(adaptive_selection_cfg.get("max_shortfall", 0))
    candidate_floors = sorted({float(value) for value in floors if np.isfinite(float(value))})

    while len(selected_rules) < max(int(max_rules), 0) and candidate_floors:
        current_recent_metrics = _selection_guard_active_metrics(
            current_recent,
            signal_col=signal_col,
            return_col=return_col,
            p_col=p_col,
            y_col=y_col,
        )
        current_baseline_metrics = _selection_guard_active_metrics(
            current_baseline,
            signal_col=signal_col,
            return_col=return_col,
            p_col=p_col,
            y_col=y_col,
        )
        current_row_policy = _resolve_selection_row_policy(
            recent_selection_rows=int(current_recent_metrics.get("rows", 0)),
            baseline_selection_rows=int(current_baseline_metrics.get("rows", 0)),
            strict_min_selection_rows=int(min_selection_rows),
            adaptive_enabled=adaptive_enabled,
            adaptive_min_floor=adaptive_min_floor,
            adaptive_baseline_ratio=adaptive_baseline_ratio,
            adaptive_max_shortfall=adaptive_max_shortfall,
        )
        current_recent_auc = _safe_float(current_recent_metrics.get("auc"), default=float("nan"))
        current_recent_ece = _safe_float(current_recent_metrics.get("ece_10"), default=float("nan"))
        current_baseline_ece = _safe_float(current_baseline_metrics.get("ece_10"), default=float("nan"))
        current_drift = (
            float(current_recent_ece - current_baseline_ece)
            if np.isfinite(current_recent_ece) and np.isfinite(current_baseline_ece)
            else float("nan")
        )
        signal_recent = pd.to_numeric(current_recent[signal_col], errors="coerce").fillna(0.0)
        regimes_recent = current_recent[regime_col].map(
            lambda value: str(value).strip().lower() if pd.notna(value) else "missing"
        )
        probabilities_recent = pd.to_numeric(current_recent[p_col], errors="coerce")
        returns_recent = pd.to_numeric(current_recent[return_col], errors="coerce").fillna(0.0)
        active_regimes = sorted({str(value) for value in regimes_recent.loc[signal_recent > 0.0].unique() if str(value)})
        best_candidate: Dict[str, Any] | None = None
        sparse_mode = bool(
            sparse_allow_row_policy_override
            and int(current_recent_metrics.get("rows", 0)) > 0
            and int(current_recent_metrics.get("rows", 0)) <= max(int(sparse_active_trade_cap), 0)
            and not bool(current_row_policy.get("effective_ok", False))
        )

        for regime_state in active_regimes:
            active_regime_trade_count = int(
                ((signal_recent > 0.0) & (regimes_recent == regime_state) & probabilities_recent.notna()).sum()
            )
            regime_candidate_floors = list(candidate_floors)
            if sparse_mode and sparse_use_observed_p_up_values and active_regime_trade_count <= max(int(sparse_active_trade_cap), 0):
                observed_probs = pd.to_numeric(
                    probabilities_recent.loc[(signal_recent > 0.0) & (regimes_recent == regime_state)],
                    errors="coerce",
                ).dropna()
                regime_candidate_floors = sorted(
                    {
                        *{round(float(value), 6) for value in regime_candidate_floors if np.isfinite(float(value))},
                        *{
                            round(float(value) + 1e-6, 6)
                            for value in observed_probs.tolist()
                            if np.isfinite(float(value))
                        },
                    }
                )
            for min_p_up in regime_candidate_floors:
                if any(
                    str(rule.get("regime_state", "")).strip().lower() == regime_state
                    and abs(_safe_float(rule.get("min_p_up"), default=float("nan")) - float(min_p_up)) < 1e-9
                    for rule in selected_rules
                ):
                    continue
                blocked_recent_mask = (
                    (signal_recent > 0.0)
                    & (regimes_recent == regime_state)
                    & probabilities_recent.notna()
                    & (probabilities_recent < float(min_p_up))
                )
                blocked_recent_rows = int(blocked_recent_mask.sum())
                required_blocked_recent_rows = int(min_blocked_recent_rows)
                if sparse_mode and active_regime_trade_count <= max(int(sparse_active_trade_cap), 0):
                    required_blocked_recent_rows = min(
                        required_blocked_recent_rows,
                        max(int(sparse_min_blocked_recent_rows), 1),
                    )
                if blocked_recent_rows < required_blocked_recent_rows:
                    continue
                blocked_recent_net = float(returns_recent.loc[blocked_recent_mask].sum())
                if require_blocked_recent_net_nonpositive and blocked_recent_net > 0.0:
                    continue
                if (
                    max_blocked_recent_net_return_total is not None
                    and np.isfinite(float(max_blocked_recent_net_return_total))
                    and blocked_recent_net > float(max_blocked_recent_net_return_total)
                ):
                    continue

                trial_rule = {"regime_state": regime_state, "min_p_up": float(min_p_up)}
                trial_recent, _ = _apply_selection_calibration_guard_rules_to_frame(
                    current_recent,
                    signal_col=signal_col,
                    return_col=return_col,
                    regime_col=regime_col,
                    p_col=p_col,
                    rules=[trial_rule],
                )
                trial_baseline, _ = _apply_selection_calibration_guard_rules_to_frame(
                    current_baseline,
                    signal_col=signal_col,
                    return_col=return_col,
                    regime_col=regime_col,
                    p_col=p_col,
                    rules=[trial_rule],
                )
                trial_recent_metrics = _selection_guard_active_metrics(
                    trial_recent,
                    signal_col=signal_col,
                    return_col=return_col,
                    p_col=p_col,
                    y_col=y_col,
                )
                trial_baseline_metrics = _selection_guard_active_metrics(
                    trial_baseline,
                    signal_col=signal_col,
                    return_col=return_col,
                    p_col=p_col,
                    y_col=y_col,
                )
                row_policy = _resolve_selection_row_policy(
                    recent_selection_rows=int(trial_recent_metrics.get("rows", 0)),
                    baseline_selection_rows=int(trial_baseline_metrics.get("rows", 0)),
                    strict_min_selection_rows=int(min_selection_rows),
                    adaptive_enabled=adaptive_enabled,
                    adaptive_min_floor=adaptive_min_floor,
                    adaptive_baseline_ratio=adaptive_baseline_ratio,
                    adaptive_max_shortfall=adaptive_max_shortfall,
                )
                if not bool(row_policy.get("effective_ok", False)):
                    if not (
                        sparse_mode
                        and int(trial_recent_metrics.get("rows", 0)) >= max(int(sparse_min_retained_recent_rows), 1)
                    ):
                        continue

                trial_recent_rows = int(trial_recent_metrics.get("rows", 0))
                if sparse_mode and trial_recent_rows < max(int(sparse_min_retained_recent_rows), 1):
                    continue

                trial_recent_auc = _safe_float(trial_recent_metrics.get("auc"), default=float("nan"))
                trial_recent_ece = _safe_float(trial_recent_metrics.get("ece_10"), default=float("nan"))
                trial_baseline_ece = _safe_float(trial_baseline_metrics.get("ece_10"), default=float("nan"))
                if not np.isfinite(trial_recent_ece):
                    continue
                baseline_available = bool(np.isfinite(trial_baseline_ece))
                if not baseline_available and not (sparse_mode and sparse_allow_missing_baseline):
                    continue
                trial_drift = float(trial_recent_ece - trial_baseline_ece) if baseline_available else float("nan")
                ece_improvement = (
                    current_recent_ece - trial_recent_ece if np.isfinite(current_recent_ece) else float("nan")
                )
                auc_improvement = (
                    trial_recent_auc - current_recent_auc
                    if np.isfinite(trial_recent_auc) and np.isfinite(current_recent_auc)
                    else float("nan")
                )
                drift_improvement = current_drift - trial_drift if np.isfinite(current_drift) else float("nan")
                if require_recent_net_nonnegative and float(trial_recent_metrics.get("net_return_total", 0.0)) < 0.0:
                    continue

                recent_auc_ok = True
                if min_recent_auc is not None:
                    recent_auc_ok = bool(
                        np.isfinite(trial_recent_auc) and trial_recent_auc >= float(min_recent_auc)
                    )
                recent_ece_ok = True
                if max_recent_ece is not None:
                    recent_ece_ok = bool(
                        np.isfinite(trial_recent_ece) and trial_recent_ece <= float(max_recent_ece)
                    )
                ece_drift_ok = True
                if max_ece_drift is not None:
                    if baseline_available:
                        ece_drift_ok = bool(
                            np.isfinite(trial_drift) and trial_drift <= float(max_ece_drift)
                        )
                    else:
                        ece_drift_ok = bool(sparse_mode and sparse_allow_missing_baseline)
                formal_policy_pass = bool(recent_auc_ok and recent_ece_ok and ece_drift_ok)
                failed_formal_checks = int(sum(1 for check in (recent_auc_ok, recent_ece_ok, ece_drift_ok) if not check))

                meets_improvement_gate = True
                if not formal_policy_pass:
                    if np.isfinite(ece_improvement) and ece_improvement < float(min_recent_ece_improvement):
                        meets_improvement_gate = False
                    if baseline_available and np.isfinite(drift_improvement) and drift_improvement < float(min_ece_drift_improvement):
                        meets_improvement_gate = False
                if not meets_improvement_gate:
                    continue

                candidate = {
                    "rule": trial_rule,
                    "blocked_recent_rows": blocked_recent_rows,
                    "blocked_recent_net_return_total": blocked_recent_net,
                    "trial_recent_metrics": trial_recent_metrics,
                    "trial_baseline_metrics": trial_baseline_metrics,
                    "trial_ece_drift": trial_drift,
                    "trial_recent_auc": trial_recent_auc,
                    "ece_improvement": ece_improvement,
                    "auc_improvement": auc_improvement,
                    "ece_drift_improvement": drift_improvement,
                    "formal_policy_pass": formal_policy_pass,
                    "formal_policy_checks": {
                        "recent_auc_ok": recent_auc_ok,
                        "recent_ece_ok": recent_ece_ok,
                        "ece_drift_ok": ece_drift_ok,
                        "baseline_available": baseline_available,
                    },
                    "row_policy": row_policy,
                    "sparse_mode": sparse_mode,
                    "score": (
                        int(formal_policy_pass),
                        1 if sparse_mode else 0,
                        -failed_formal_checks,
                        _safe_float(ece_improvement, default=float("-inf")),
                        _safe_float(drift_improvement, default=float("-inf")),
                        _safe_float(auc_improvement, default=float("-inf")),
                        float(trial_recent_metrics.get("net_return_total", 0.0)),
                        int(trial_recent_metrics.get("rows", 0)),
                        -blocked_recent_rows,
                    ),
                }
                if best_candidate is None or candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate

        if best_candidate is None:
            break

        selected_rules.append(best_candidate["rule"])
        steps.append(best_candidate)
        current_recent, _ = _apply_selection_calibration_guard_rules_to_frame(
            current_recent,
            signal_col=signal_col,
            return_col=return_col,
            regime_col=regime_col,
            p_col=p_col,
            rules=[best_candidate["rule"]],
        )
        current_baseline, _ = _apply_selection_calibration_guard_rules_to_frame(
            current_baseline,
            signal_col=signal_col,
            return_col=return_col,
            regime_col=regime_col,
            p_col=p_col,
            rules=[best_candidate["rule"]],
        )

    return {
        "enabled": bool(selected_rules),
        "reason": "ready" if selected_rules else "no_derived_rules_met_thresholds",
        "rules": selected_rules,
        "steps": steps,
        "candidate_path": str(candidate_path),
        "recent_window_rows": int(recent_window_rows),
        "baseline_window_rows": int(baseline_window_rows),
        "resolved_recent_rows": int(recent_n),
        "resolved_baseline_rows": int(baseline_n),
    }


def _build_selection_calibration_guard_shadow(
    *,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    required = {signal_col, return_col, regime_col, p_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    working = df.copy()
    signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    probabilities = pd.to_numeric(working[p_col], errors="coerce")
    regimes = working[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")

    blocked_mask = pd.Series(False, index=working.index)
    rule_summaries: List[Dict[str, Any]] = []
    for rule in rules:
        regime_state = str(rule.get("regime_state", "")).strip().lower()
        min_p_up = float(rule.get("min_p_up", 0.0))
        rule_mask = (signal > 0.0) & (regimes == regime_state) & probabilities.notna() & (probabilities < min_p_up)
        blocked_mask = blocked_mask | rule_mask
        rule_summaries.append(
            {
                "regime_state": regime_state,
                "min_p_up": min_p_up,
                "blocked_rows": int(rule_mask.sum()),
                "blocked_net_return_total": float(returns.loc[rule_mask].sum()) if bool(rule_mask.any()) else 0.0,
            }
        )

    working.loc[blocked_mask, signal_col] = 0.0
    working.loc[blocked_mask, return_col] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        working.to_parquet(output_path, index=False)
    else:
        working.to_csv(output_path, index=False)

    final_signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    final_returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "trade_count": int((final_signal > 0.0).sum()),
        "net_return_total": float(final_returns.loc[final_signal > 0.0].sum()),
        "selection_calibration_guard": {
            "enabled": True,
            "regime_col": regime_col,
            "p_col": p_col,
            "blocked_rows": int(blocked_mask.sum()),
            "rules": rule_summaries,
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_regime_abs_ret_pred_floor_shadow(
    *,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    signal_col: str,
    return_col: str,
    regime_col: str,
    ret_pred_col: str,
    regime_state: str,
    min_abs_ret_pred: float,
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    required = {signal_col, return_col, regime_col, ret_pred_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    working = df.copy()
    signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    ret_pred = pd.to_numeric(working[ret_pred_col], errors="coerce").fillna(0.0).abs()
    regimes = working[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    normalized_regime_state = str(regime_state).strip().lower()
    blocked_mask = (signal > 0.0) & (regimes == normalized_regime_state) & (ret_pred < float(min_abs_ret_pred))

    working.loc[blocked_mask, signal_col] = 0.0
    working.loc[blocked_mask, return_col] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        working.to_parquet(output_path, index=False)
    else:
        working.to_csv(output_path, index=False)

    final_signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    final_returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "trade_count": int((final_signal > 0.0).sum()),
        "net_return_total": float(final_returns.loc[final_signal > 0.0].sum()),
        "neutral_abs_ret_pred_floor": {
            "enabled": True,
            "regime_col": regime_col,
            "ret_pred_col": ret_pred_col,
            "regime_state": normalized_regime_state,
            "min_abs_ret_pred": float(min_abs_ret_pred),
            "blocked_rows": int(blocked_mask.sum()),
            "blocked_net_return_total": float(returns.loc[blocked_mask].sum()) if bool(blocked_mask.any()) else 0.0,
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_regime_max_p_up_shadow(
    *,
    input_path: Path,
    output_path: Path,
    meta_path: Path,
    signal_col: str,
    return_col: str,
    regime_col: str,
    p_col: str,
    regime_state: str,
    max_p_up_exclusive: float,
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    required = {signal_col, return_col, regime_col, p_col}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    working = df.copy()
    signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    p_up = pd.to_numeric(working[p_col], errors="coerce")
    regimes = working[regime_col].map(lambda value: str(value).strip().lower() if pd.notna(value) else "missing")
    normalized_regime_state = str(regime_state).strip().lower()
    blocked_mask = (signal > 0.0) & (regimes == normalized_regime_state) & (p_up >= float(max_p_up_exclusive))

    working.loc[blocked_mask, signal_col] = 0.0
    working.loc[blocked_mask, return_col] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        working.to_parquet(output_path, index=False)
    else:
        working.to_csv(output_path, index=False)

    final_signal = pd.to_numeric(working[signal_col], errors="coerce").fillna(0.0)
    final_returns = pd.to_numeric(working[return_col], errors="coerce").fillna(0.0)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "trade_count": int((final_signal > 0.0).sum()),
        "net_return_total": float(final_returns.loc[final_signal > 0.0].sum()),
        "neutral_p_up_cap": {
            "enabled": True,
            "regime_col": regime_col,
            "p_col": p_col,
            "regime_state": normalized_regime_state,
            "max_p_up_exclusive": float(max_p_up_exclusive),
            "blocked_rows": int(blocked_mask.sum()),
            "blocked_net_return_total": float(returns.loc[blocked_mask].sum()) if bool(blocked_mask.any()) else 0.0,
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _extract_selection_scope_ranking_metrics(calibration_variant_recent: Dict[str, Any]) -> Dict[str, Any]:
    recent_diagnostics = (
        calibration_variant_recent.get("recent_diagnostics", {})
        if isinstance(calibration_variant_recent.get("recent_diagnostics", {}), dict)
        else {}
    )
    selection_scope = recent_diagnostics.get("selection_scope", {}) if isinstance(recent_diagnostics, dict) else {}
    if not isinstance(selection_scope, dict):
        selection_scope = {}
    recent = selection_scope.get("recent", {}) if isinstance(selection_scope.get("recent", {}), dict) else {}
    failed_checks = calibration_variant_recent.get("promotion_hardening", {}).get("failed_checks", [])
    if not isinstance(failed_checks, list):
        failed_checks = []
    return {
        "recent_rows": int(_safe_float(recent.get("rows"), default=0.0)),
        "recent_auc": _safe_float(recent.get("auc"), default=float("-inf")),
        "recent_ece": _safe_float(recent.get("ece_10"), default=1e9),
        "ece_drift": _safe_float(selection_scope.get("ece_drift"), default=1e9),
        "failed_check_count": len(failed_checks),
    }


def _deploy_promoted_reliability_artifacts(
    *,
    run_dir: Path,
    deploy_cfg: Dict[str, Any],
    thresholds_path: Path,
    platt_calibration_path: Path,
    trade_decision_model_path: Path,
    trade_decision_deploy_ready: bool,
    promoted_profile_path: Path,
    promoted_profile_meta_path: Path | None,
    candidate_quality_path: Path,
    promotion_gate_path: Path,
    calibration_robustness_path: Path,
    rolling_ab_report_path: Path,
    rolling_ab_md_path: Path,
    selection_guard_rule_path: Path | None,
    official_shadow_variant: str,
    champion_gate_resolution: Dict[str, Any],
) -> Dict[str, Any]:
    deployed_files: Dict[str, Dict[str, str]] = {}
    skipped_files: Dict[str, str] = {}

    def _copy_artifact(name: str, source: Path | None, target_value: Any, *, required: bool) -> None:
        if source is None:
            if required:
                raise FileNotFoundError(f"Missing required source for {name}")
            skipped_files[name] = "source_not_configured"
            return
        target = Path(str(target_value))
        if not source.exists():
            if required:
                raise FileNotFoundError(f"Missing required source for {name}: {source}")
            skipped_files[name] = f"source_missing:{source}"
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        deployed_files[name] = {"source": str(source), "target": str(target)}

    _copy_artifact(
        "thresholds_json",
        thresholds_path,
        deploy_cfg.get("thresholds_target", "artifacts/models/calibrated_thresholds_merged.json"),
        required=True,
    )
    _copy_artifact(
        "platt_calibration",
        platt_calibration_path,
        deploy_cfg.get("platt_calibration_target", "artifacts/models/platt_calibration.json"),
        required=True,
    )
    if trade_decision_deploy_ready:
        _copy_artifact(
            "trade_decision_model",
            trade_decision_model_path,
            deploy_cfg.get("trade_decision_model_target", "artifacts/models/trade_decision_model.json"),
            required=False,
        )
    else:
        skipped_files["trade_decision_model"] = "deploy_not_ready"

    _copy_artifact(
        "promoted_profile_csv",
        promoted_profile_path,
        deploy_cfg.get("promoted_profile_target", "artifacts/monitoring/labeled_backtest_1h.csv"),
        required=True,
    )
    _copy_artifact(
        "promoted_profile_meta",
        promoted_profile_meta_path,
        deploy_cfg.get("promoted_profile_meta_target", "artifacts/monitoring/labeled_backtest_1h_meta.json"),
        required=False,
    )
    _copy_artifact(
        "incumbent_profile_csv",
        promoted_profile_path,
        deploy_cfg.get("incumbent_profile_target", "artifacts/monitoring/labeled_backtest_1h_incumbent.csv"),
        required=True,
    )
    _copy_artifact(
        "incumbent_profile_meta",
        promoted_profile_meta_path,
        deploy_cfg.get("incumbent_profile_meta_target", "artifacts/monitoring/labeled_backtest_1h_incumbent_meta.json"),
        required=False,
    )
    _copy_artifact(
        "incumbent_quality_backtest",
        candidate_quality_path,
        deploy_cfg.get("incumbent_quality_backtest_target", "artifacts/monitoring/model_quality_incumbent_1h_backtest.json"),
        required=True,
    )
    _copy_artifact(
        "incumbent_quality",
        candidate_quality_path,
        deploy_cfg.get("incumbent_quality_target", "artifacts/monitoring/model_quality_incumbent_1h.json"),
        required=False,
    )
    _copy_artifact(
        "promotion_gate",
        promotion_gate_path,
        deploy_cfg.get("promotion_gate_target", "artifacts/monitoring/promotion_gate_1h.json"),
        required=False,
    )
    _copy_artifact(
        "calibration_robustness",
        calibration_robustness_path,
        deploy_cfg.get("calibration_robustness_target", "artifacts/monitoring/calibration_robustness.json"),
        required=False,
    )
    _copy_artifact(
        "rolling_ab_report",
        rolling_ab_report_path,
        deploy_cfg.get("rolling_ab_report_target", "artifacts/monitoring/rolling_ab_report.json"),
        required=False,
    )
    _copy_artifact(
        "rolling_ab_report_md",
        rolling_ab_md_path,
        deploy_cfg.get("rolling_ab_report_md_target", "artifacts/monitoring/rolling_ab_report.md"),
        required=False,
    )
    if str(official_shadow_variant or "none").strip().lower() == "selection_calibration_guard":
        _copy_artifact(
            "selection_guard_rule",
            selection_guard_rule_path,
            deploy_cfg.get(
                "selection_guard_rule_target",
                "artifacts/monitoring/selection_calibration_guard_rule_1h.json",
            ),
            required=False,
        )
    else:
        skipped_files["selection_guard_rule"] = "official_shadow_variant_not_selection_calibration_guard"

    manifest_path = Path(
        str(
            deploy_cfg.get(
                "manifest_target",
                "artifacts/monitoring/reliability_promotion_deploy_manifest.json",
            )
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_dir.name),
        "run_dir": str(run_dir),
        "official_shadow_variant": str(official_shadow_variant or "none"),
        "champion_gate_resolution": champion_gate_resolution,
        "trade_decision_deploy_ready": bool(trade_decision_deploy_ready),
        "deployed_files": deployed_files,
        "skipped_files": skipped_files,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest_payload


def _find_latest_profile_snapshot_run(
    *,
    run_root: Path,
    current_run_id: str,
    profile_id: str,
    snapshot_name: str = "live_predictions_snapshot.json",
) -> tuple[str, Path] | None:
    if not run_root.exists():
        return None

    candidates = sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    for run_path in candidates:
        run_id = run_path.name
        if run_id == current_run_id:
            continue

        summary_dir = run_path / "summary"
        manifest_path = summary_dir / "workflow_manifest.json"
        snapshot_path = summary_dir / snapshot_name
        if not manifest_path.exists() or not snapshot_path.exists():
            continue

        try:
            manifest_payload = _load_json(manifest_path)
        except Exception:
            continue

        profile = manifest_payload.get("profile") if isinstance(manifest_payload.get("profile"), dict) else {}
        if str(profile.get("id", "")).strip() != profile_id:
            continue
        return run_id, snapshot_path
    return None


def _find_latest_trusted_baseline_pack(
    *,
    run_root: Path,
    current_run_id: str,
) -> tuple[str, Path] | None:
    if not run_root.exists():
        return None

    candidates = sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    for run_path in candidates:
        run_id = run_path.name
        if run_id == current_run_id:
            continue
        pack_path = run_path / "summary" / "trusted_baseline_pack.json"
        if not pack_path.exists():
            continue
        try:
            payload = _load_json(pack_path)
        except Exception:
            continue
        if not bool(payload.get("edge_trustworthy", False)):
            continue
        return run_id, pack_path
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full reliability workflow: CV+Optuna, calibration, thresholds, ensemble, monitoring, paper-live.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reliability_workflow.default.yaml"),
        help="Workflow YAML config.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/reliability"),
        help="Root output directory for workflow artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip CV+Optuna searches (steps 1-2).",
    )
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step (step 3).")
    parser.add_argument("--skip-thresholds", action="store_true", help="Skip threshold search step (step 4).")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip meta-ensemble training step (step 5).")
    parser.add_argument("--skip-monitoring", action="store_true", help="Skip reliability trigger checks (step 6).")
    parser.add_argument(
        "--skip-quality-evals",
        action="store_true",
        help="Skip optional walk-forward, model-quality, hygiene, and promotion-gate checks.",
    )
    parser.add_argument(
        "--continue-on-promotion-fail",
        action="store_true",
        help=(
            "Continue workflow when promotion gate returns exit 3 (promote=false). "
            "Useful for keeping paper-live/shadow steps running while gate blocks promotion."
        ),
    )
    parser.add_argument("--skip-paper-live", action="store_true", help="Skip paper-live refresh run (step 7).")
    return parser.parse_args(argv)


def execute_reliability_workflow(
    args: argparse.Namespace,
    *,
    step_event_sink: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
) -> Dict[str, Any]:
    if step_event_sink is not None:
        return _run_with_step_event_sink(
            step_event_sink,
            lambda: execute_reliability_workflow(args),
        )

    config = _load_yaml(args.config)
    cv_cfg = config.get("cv", {})
    search_cfg = config.get("search", {})
    direction_output_shadow_cfg_obj = search_cfg.get("direction_output_shadow", {}) if isinstance(search_cfg, dict) else {}
    direction_output_shadow_cfg = direction_output_shadow_cfg_obj if isinstance(direction_output_shadow_cfg_obj, dict) else {}
    upstream_direction_candidate_cfg_obj = search_cfg.get("upstream_direction_candidate", {}) if isinstance(search_cfg, dict) else {}
    upstream_direction_candidate_cfg = (
        upstream_direction_candidate_cfg_obj if isinstance(upstream_direction_candidate_cfg_obj, dict) else {}
    )
    trade_decision_chop_suppression_candidate_cfg_obj = (
        search_cfg.get("trade_decision_chop_suppression_candidate", {}) if isinstance(search_cfg, dict) else {}
    )
    trade_decision_chop_suppression_candidate_cfg = (
        trade_decision_chop_suppression_candidate_cfg_obj
        if isinstance(trade_decision_chop_suppression_candidate_cfg_obj, dict)
        else {}
    )
    monitoring_cfg = config.get("monitoring", {})
    cadence_cfg = config.get("cadence", {})
    quality_cfg = config.get("quality", {})
    rolling_ab_cfg = quality_cfg.get("rolling_ab", {}) if isinstance(quality_cfg, dict) else {}
    tuning_cfg = quality_cfg.get("joint_threshold_tuning", {}) if isinstance(quality_cfg, dict) else {}
    calibration_cfg = quality_cfg.get("calibration_robustness", {}) if isinstance(quality_cfg, dict) else {}
    no_trade_cfg = quality_cfg.get("no_trade", {}) if isinstance(quality_cfg, dict) else {}
    leakage_cfg = quality_cfg.get("leakage_audit", {}) if isinstance(quality_cfg, dict) else {}
    cv_stress_cfg = quality_cfg.get("cv_stress_sweep", {}) if isinstance(quality_cfg, dict) else {}
    feature_rel_cfg = quality_cfg.get("feature_reliability", {}) if isinstance(quality_cfg, dict) else {}
    directional_objectives_cfg = quality_cfg.get("directional_objectives", {}) if isinstance(quality_cfg, dict) else {}
    champ_cfg = quality_cfg.get("champion_challenger", {}) if isinstance(quality_cfg, dict) else {}
    label_ablation_cfg = quality_cfg.get("label_ablation", {}) if isinstance(quality_cfg, dict) else {}
    trade_decision_cfg = quality_cfg.get("trade_decision_model", {}) if isinstance(quality_cfg, dict) else {}
    compare_cfg = quality_cfg.get("walkforward_model_compare", {}) if isinstance(quality_cfg, dict) else {}
    canonical_cfg = quality_cfg.get("canonical_direction_dataset", {}) if isinstance(quality_cfg, dict) else {}
    reconcile_cfg = quality_cfg.get("walkforward_labeled_reconciliation", {}) if isinstance(quality_cfg, dict) else {}
    overlap_pre_tuning_cfg = quality_cfg.get("overlap_pre_tuning", {}) if isinstance(quality_cfg, dict) else {}
    overlap_drift_guard_cfg = quality_cfg.get("overlap_feature_drift_guard", {}) if isinstance(quality_cfg, dict) else {}
    raw_snapshot_cfg = quality_cfg.get("raw_direction_feature_snapshot", {}) if isinstance(quality_cfg, dict) else {}
    trusted_baseline_pack_cfg = quality_cfg.get("trusted_baseline_pack", {}) if isinstance(quality_cfg, dict) else {}
    regime_weakness_cfg = quality_cfg.get("regime_weakness", {}) if isinstance(quality_cfg, dict) else {}
    profile_cfg_obj = config.get("profile", {}) if isinstance(config, dict) else {}
    profile_cfg = profile_cfg_obj if isinstance(profile_cfg_obj, dict) else {}
    run_profile_id = str(profile_cfg.get("id", "default_runtime")).strip() or "default_runtime"
    run_profile_name = str(profile_cfg.get("name", run_profile_id)).strip() or run_profile_id

    horizons = [int(v) for v in search_cfg.get("horizons", [1, 4, 8, 12])]
    paper_live_config_path = Path(str(search_cfg.get("paper_live_config", "configs/run_refresh_and_predict.default.yaml")))
    paper_live_targets = _load_prediction_targets(paper_live_config_path)
    calibration_horizons = sorted({float(v) for v in [*horizons, *paper_live_targets] if float(v) > 0})
    n_trials = int(search_cfg.get("n_trials", 25))
    timeout = search_cfg.get("timeout")

    cv_folds = int(cv_cfg.get("folds", 4))
    cv_train_size = int(cv_cfg.get("train_size", 2400))
    cv_val_size = int(cv_cfg.get("val_size", 400))
    cv_test_size = int(cv_cfg.get("test_size", 400))
    cv_gap = int(cv_cfg.get("gap", 24))
    cv_purge_size = int(cv_cfg.get("purge_size", 0))
    cv_embargo_size = int(cv_cfg.get("embargo_size", 0))
    cv_mode = str(cv_cfg.get("mode", "expanding"))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root / timestamp
    logs_dir = run_dir / "logs"
    model_dir = run_dir / "models"
    summary_dir = run_dir / "summary"
    for directory in (logs_dir, model_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results: List[StepResult] = []
    edge_trustworthy_for_paper_live = True
    overlap_as_primary = bool(reconcile_cfg.get("use_as_primary", True))
    overlap_pruning_allows_tuning = True
    overlap_feature_reliability_path = summary_dir / "overlap_feature_reliability.json"
    overlap_model_pruning_path = summary_dir / "overlap_model_pruning.json"
    overlap_feature_drift_guard_path = summary_dir / "overlap_feature_drift_guard.json"
    raw_feature_snapshot_path = summary_dir / "direction_features_raw.snapshot.csv"
    raw_feature_snapshot_meta = summary_dir / "direction_features_raw.snapshot_meta.json"
    raw_feature_overlap_snapshot_path = summary_dir / "direction_features_raw.labeled_overlap.csv"
    raw_feature_overlap_snapshot_meta = summary_dir / "direction_features_raw.labeled_overlap_meta.json"

    python = sys.executable

    if not args.skip_optuna:
        for horizon in horizons:
            direction_dataset = Path(_direction_dataset_for_horizon(horizon))
            optuna_cv_folds = cv_folds
            optuna_cv_train_size = cv_train_size
            optuna_cv_val_size = cv_val_size
            optuna_cv_test_size = cv_test_size
            if direction_dataset.exists() and not args.dry_run:
                try:
                    n_rows = _count_npz_rows_from_x(direction_dataset)
                    (
                        optuna_cv_folds,
                        optuna_cv_train_size,
                        optuna_cv_val_size,
                        optuna_cv_test_size,
                    ) = _compute_feasible_walkforward(
                        n_rows,
                        folds=cv_folds,
                        train_size=cv_train_size,
                        val_size=cv_val_size,
                        test_size=cv_test_size,
                        gap=cv_gap,
                        purge=cv_purge_size,
                        embargo=cv_embargo_size,
                    )
                except Exception as exc:
                    print(f"Warning: failed to auto-resize optuna CV windows: {exc}", file=sys.stderr)

            xgb_out = model_dir / f"xgb_dir{horizon}h_optuna"
            xgb_cmd = [
                python,
                "-m",
                "src.scripts.search_xgb_optuna",
                "--mode",
                "dir",
                "--horizon",
                str(horizon),
                "--dataset-path",
                _direction_dataset_for_horizon(horizon),
                "--output-dir",
                str(xgb_out),
                "--n-trials",
                str(n_trials),
                "--cv-folds",
                str(optuna_cv_folds),
                "--cv-train-size",
                str(optuna_cv_train_size),
                "--cv-val-size",
                str(optuna_cv_val_size),
                "--cv-test-size",
                str(optuna_cv_test_size),
                "--cv-gap",
                str(cv_gap),
                "--cv-purge-size",
                str(cv_purge_size),
                "--cv-embargo-size",
                str(cv_embargo_size),
                "--cv-mode",
                cv_mode,
            ]
            if timeout is not None:
                xgb_cmd.extend(["--timeout", str(timeout)])
            results.append(_run_step(f"xgb_dir_optuna_{horizon}h", xgb_cmd, logs_dir / f"xgb_dir_optuna_{horizon}h.log", args.dry_run))

            if horizon == 1:
                lstm_out = model_dir / "lstm_dir1h_optuna"
                lstm_cmd = [
                    python,
                    "-m",
                    "src.scripts.search_lstm_optuna",
                    "--horizon",
                    "1",
                    "--dataset-path",
                    _direction_dataset_for_horizon(1),
                    "--output-dir",
                    str(lstm_out),
                    "--n-trials",
                    str(n_trials),
                    "--cv-folds",
                    str(optuna_cv_folds),
                    "--cv-train-size",
                    str(optuna_cv_train_size),
                    "--cv-val-size",
                    str(optuna_cv_val_size),
                    "--cv-test-size",
                    str(optuna_cv_test_size),
                    "--cv-gap",
                    str(cv_gap),
                    "--cv-purge-size",
                    str(cv_purge_size),
                    "--cv-embargo-size",
                    str(cv_embargo_size),
                    "--cv-mode",
                    cv_mode,
                ]
                if timeout is not None:
                    lstm_cmd.extend(["--timeout", str(int(timeout))])
                results.append(_run_step("lstm_dir_optuna_1h", lstm_cmd, logs_dir / "lstm_dir_optuna_1h.log", args.dry_run))

                transformer_out = model_dir / "transformer_dir1h_optuna"
                transformer_cmd = [
                    python,
                    "-m",
                    "src.scripts.search_transformer_optuna",
                    "--horizon",
                    "1",
                    "--dataset-path",
                    _direction_dataset_for_horizon(1),
                    "--output-dir",
                    str(transformer_out),
                    "--n-trials",
                    str(n_trials),
                    "--cv-folds",
                    str(optuna_cv_folds),
                    "--cv-train-size",
                    str(optuna_cv_train_size),
                    "--cv-val-size",
                    str(optuna_cv_val_size),
                    "--cv-test-size",
                    str(optuna_cv_test_size),
                    "--cv-gap",
                    str(cv_gap),
                    "--cv-purge-size",
                    str(cv_purge_size),
                    "--cv-embargo-size",
                    str(cv_embargo_size),
                    "--cv-mode",
                    cv_mode,
                ]
                if timeout is not None:
                    transformer_cmd.extend(["--timeout", str(timeout)])
                results.append(_run_step("transformer_dir_optuna_1h", transformer_cmd, logs_dir / "transformer_dir_optuna_1h.log", args.dry_run))

    if not args.skip_calibration:
        calibr_cmd = [
            python,
            "-m",
            "src.scripts.train_platt_calibration",
            "--horizons",
            *[str(h) for h in calibration_horizons],
            "--output-path",
            str(summary_dir / "platt_calibration.json"),
            "--coverage-output-path",
            str(summary_dir / "platt_calibration_coverage.json"),
            "--method",
            str(calibration_cfg.get("method", "platt")),
        ]
        results.append(_run_step("platt_calibration", calibr_cmd, logs_dir / "platt_calibration.log", args.dry_run))

    thresholds_path = summary_dir / "calibrated_thresholds.json"
    trade_decision_model_path = summary_dir / "trade_decision_model.json"
    trade_decision_ablation_model_path = summary_dir / "trade_decision_model_reference_feature_ablation.json"
    trade_decision_deploy_ready = False
    trade_decision_model_shift_payload: Dict[str, Any] = {}
    joint_tuning_output_path = summary_dir / "joint_threshold_tuning.json"
    joint_tuning_accepted: bool | None = None
    last_deployable_thresholds_path = Path(
        str(
            tuning_cfg.get(
                "last_deployable_thresholds_path",
                "artifacts/monitoring/calibrated_thresholds_last_deployable.json",
            )
        )
    )
    use_last_deployable_on_joint_reject = bool(tuning_cfg.get("fallback_to_last_deployable_on_reject", True))
    used_last_deployable_thresholds = False
    if not args.skip_thresholds:
        threshold_objective = str(search_cfg.get("threshold_objective", "cumret_with_dd_constraint"))
        thresholds_cmd = [
            python,
            "-m",
            "src.scripts.search_ensemble_thresholds",
            "--targets",
            _join_horizons(calibration_horizons),
            "--objective",
            threshold_objective,
            "--max-dd",
            str(search_cfg.get("max_drawdown", -0.08)),
            "--min-trades",
            str(search_cfg.get("min_trades", 10)),
            "--output-dir",
            str(summary_dir / "threshold_search"),
            "--output",
            str(thresholds_path),
        ]
        if threshold_objective == "utility_stability":
            thresholds_cmd.extend(
                [
                    "--utility-dd-penalty",
                    str(float(search_cfg.get("utility_dd_penalty", 2.0))),
                    "--utility-std-penalty",
                    str(float(search_cfg.get("utility_std_penalty", 0.5))),
                    "--utility-turnover-penalty",
                    str(float(search_cfg.get("utility_turnover_penalty", 0.5))),
                ]
            )
        results.append(_run_step("threshold_search", thresholds_cmd, logs_dir / "threshold_search.log", args.dry_run))

    if not args.skip_ensemble:
        meta_component_frame_csv = search_cfg.get("meta_component_frame_csv")
        meta_component_columns = search_cfg.get("meta_component_columns")
        meta_component_frame_source = search_cfg.get("meta_component_frame_source")
        default_meta_inputs = [
            Path("artifacts/backtests/historical_1h_pup060_full_simplified/backtest_signals.csv"),
            Path("artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv"),
            Path("artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv"),
        ]
        component_frame_path = Path(str(meta_component_frame_csv)) if meta_component_frame_csv else None
        component_frame_source_path = Path(str(meta_component_frame_source)) if meta_component_frame_source else None
        component_frame_status: Dict[str, Any] | None = None
        if component_frame_path is not None and component_frame_source_path is not None and not args.dry_run:
            try:
                component_frame_status = _write_meta_component_frame(
                    source_path=component_frame_source_path,
                    output_path=component_frame_path,
                    requested_columns=meta_component_columns if isinstance(meta_component_columns, Sequence) and not isinstance(meta_component_columns, (str, bytes)) else None,
                )
                if not bool(component_frame_status.get("written", False)):
                    print(
                        "Warning: failed to derive meta component frame: "
                        f"{component_frame_status.get('reason', 'unknown')}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                component_frame_status = {
                    "written": False,
                    "reason": f"exception:{exc}",
                    "source": str(component_frame_source_path),
                    "output": str(component_frame_path),
                }
                print(f"Warning: failed to derive meta component frame: {exc}", file=sys.stderr)
        missing_meta_inputs = [str(path) for path in default_meta_inputs if not path.exists()]
        has_component_frame = component_frame_path is not None and component_frame_path.exists()
        if missing_meta_inputs and not has_component_frame:
            print(
                "Warning: skipping meta-ensemble training because required inputs are missing: "
                + ", ".join(missing_meta_inputs),
                file=sys.stderr,
            )
        else:
            meta_component_weight_spec = None
            meta_component_weight_audit_path = search_cfg.get("meta_component_weights_from_audit_path")
            if meta_component_weight_audit_path:
                audit_path = Path(str(meta_component_weight_audit_path))
                if audit_path.exists():
                    try:
                        meta_component_weight_spec = _extract_audit_weight_spec(
                            _load_json(audit_path),
                            allowed_components=(
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
                            ),
                        )
                    except Exception as exc:
                        print(f"Warning: failed to load meta component weights from audit {audit_path}: {exc}", file=sys.stderr)
            meta_cmd = [
                python,
                "-m",
                "src.scripts.train_meta_ensemble",
                "--output-csv",
                str(summary_dir / "backtest_signals_meta_ensemble.csv"),
                "--config-path",
                str(summary_dir / "meta_ensemble_config.json"),
                "--weight-threshold",
                str(search_cfg.get("meta_weight_threshold", 0.5)),
            ]
            if has_component_frame and component_frame_path is not None:
                meta_cmd.extend(["--component-frame-csv", str(component_frame_path)])
                if isinstance(meta_component_columns, Sequence) and not isinstance(meta_component_columns, (str, bytes)):
                    for column in meta_component_columns:
                        meta_cmd.extend(["--component-column", str(column)])
            if meta_component_weight_spec:
                meta_cmd.extend(["--component-weight-spec", meta_component_weight_spec])
            results.append(_run_step("meta_ensemble_train", meta_cmd, logs_dir / "meta_ensemble_train.log", args.dry_run))
            if component_frame_status is not None:
                (summary_dir / "meta_component_frame_status.json").write_text(
                    json.dumps(component_frame_status, indent=2),
                    encoding="utf-8",
                )

    if not args.skip_quality_evals and bool(quality_cfg.get("enabled", False)):
        quality_input = Path(quality_cfg.get("quality_input") or (summary_dir / "backtest_signals_meta_ensemble.csv"))
        labeled_snapshot_csv = summary_dir / "labeled_backtest.snapshot.csv"
        labeled_meta_output = summary_dir / "labeled_backtest_meta.json"
        labeled_snapshot_meta = summary_dir / "labeled_backtest_meta.snapshot.json"
        candidate_quality_input_cfg = quality_cfg.get("candidate_quality_input")
        candidate_quality_input = (
            Path(str(candidate_quality_input_cfg))
            if candidate_quality_input_cfg
            else (summary_dir / "backtest_signals_meta_ensemble.csv")
        )
        if not args.dry_run and not candidate_quality_input.exists():
            candidate_quality_input = quality_input
        candidate_gate_input = candidate_quality_input
        tuning_quality_input = candidate_quality_input
        official_shadow_variant = "none"
        selection_payload: Dict[str, Any] | None = None
        selected_shadow_path: Path | None = None
        selected_shadow_meta_path: Path | None = None
        selected_shadow_companion_path: Path | None = None
        quality_backtest_csv_raw = quality_cfg.get("backtest_csv")
        quality_backtest_csv: Path | None = None
        if quality_backtest_csv_raw is not None and str(quality_backtest_csv_raw).strip():
            quality_backtest_csv = Path(str(quality_backtest_csv_raw))
        default_latest_backtest = Path("artifacts/backtests/latest/backtest_signals.csv")
        quality_backtest_csv_is_auto = quality_backtest_csv_raw is None or str(quality_backtest_csv_raw).strip().lower() in {"", "auto"}
        quality_backtest_csv_points_to_missing_default = bool(
            quality_backtest_csv is not None
            and quality_backtest_csv == default_latest_backtest
            and not quality_backtest_csv.exists()
        )
        resolved_quality_backtest_csv = quality_backtest_csv
        if candidate_quality_input is not None and (
            quality_backtest_csv_is_auto or quality_backtest_csv_points_to_missing_default
        ):
            resolved_quality_backtest_csv = candidate_quality_input
        pinned_labeled_csv_cfg = quality_cfg.get("pinned_labeled_csv_path")
        pinned_labeled_meta_cfg = quality_cfg.get("pinned_labeled_meta_path")
        pinned_labeled_csv = (
            Path(str(pinned_labeled_csv_cfg))
            if str(pinned_labeled_csv_cfg or "").strip()
            else None
        )
        pinned_labeled_meta = (
            Path(str(pinned_labeled_meta_cfg))
            if str(pinned_labeled_meta_cfg or "").strip()
            else None
        )
        use_pinned_labeled_csv = pinned_labeled_csv is not None
        pinned_canonical_dataset_cfg = canonical_cfg.get("pinned_dataset_path")
        pinned_canonical_meta_cfg = canonical_cfg.get("pinned_meta_path")
        pinned_canonical_dataset = (
            Path(str(pinned_canonical_dataset_cfg))
            if str(pinned_canonical_dataset_cfg or "").strip()
            else None
        )
        pinned_canonical_meta = (
            Path(str(pinned_canonical_meta_cfg))
            if str(pinned_canonical_meta_cfg or "").strip()
            else None
        )
        use_pinned_canonical_dataset = pinned_canonical_dataset is not None

        if use_pinned_canonical_dataset and not args.dry_run:
            if not pinned_canonical_dataset.exists():
                raise FileNotFoundError(
                    f"Pinned canonical direction dataset not found: {pinned_canonical_dataset}"
                )
            if pinned_canonical_meta is not None and not pinned_canonical_meta.exists():
                raise FileNotFoundError(
                    f"Pinned canonical direction meta not found: {pinned_canonical_meta}"
                )
        if use_pinned_labeled_csv and not args.dry_run:
            if not pinned_labeled_csv.exists():
                raise FileNotFoundError(f"Pinned labeled backtest CSV not found: {pinned_labeled_csv}")
            if pinned_labeled_meta is not None and not pinned_labeled_meta.exists():
                raise FileNotFoundError(f"Pinned labeled backtest meta not found: {pinned_labeled_meta}")

        # Rebuild canonical hourly and direction datasets so walk-forward and audits use the latest expanded Binance history.
        if bool(canonical_cfg.get("enabled", True)) and not use_pinned_canonical_dataset:
            label_policy = str(canonical_cfg.get("labeling_scheme", "binary"))
            if bool(canonical_cfg.get("enforce_binary_label_policy", True)) and label_policy != "binary":
                raise ValueError(
                    f"quality.canonical_direction_dataset.labeling_scheme must be binary when enforce_binary_label_policy=true; got {label_policy}",
                )
            if bool(canonical_cfg.get("rebuild_hourly", True)):
                build_hourly_cmd = [
                    python,
                    "-m",
                    "src.scripts.build_training_dataset",
                    "--output-dir",
                    str(canonical_cfg.get("output_dir", "artifacts/datasets")),
                ]
                results.append(
                    _run_step(
                        "build_canonical_hourly_dataset",
                        build_hourly_cmd,
                        logs_dir / "build_canonical_hourly_dataset.log",
                        args.dry_run,
                    )
                )

            build_direction_cmd = [
                python,
                "-m",
                "src.scripts.build_training_dataset_direction",
                "--output-dir",
                str(canonical_cfg.get("output_dir", "artifacts/datasets")),
                "--threshold",
                str(float(canonical_cfg.get("threshold", 0.0))),
                "--labeling-scheme",
                label_policy,
                "--no-trade-abs-ret",
                str(float(canonical_cfg.get("no_trade_abs_ret", 0.0))),
                "--no-trade-vol-mult",
                str(float(canonical_cfg.get("no_trade_vol_mult", 0.0))),
                "--meta-path",
                str(canonical_cfg.get("meta_path", "artifacts/datasets/btc_features_1h_direction_meta.json")),
                "--tb-horizon-steps",
                str(int(canonical_cfg.get("tb_horizon_steps", 3))),
                "--tb-vol-window",
                str(int(canonical_cfg.get("tb_vol_window", 24))),
                "--tb-upper-mult",
                str(float(canonical_cfg.get("tb_upper_mult", 1.5))),
                "--tb-lower-mult",
                str(float(canonical_cfg.get("tb_lower_mult", 1.5))),
            ]
            results.append(
                _run_step(
                    "build_canonical_direction_dataset",
                    build_direction_cmd,
                    logs_dir / "build_canonical_direction_dataset.log",
                    args.dry_run,
                )
            )
        elif use_pinned_canonical_dataset and not args.dry_run:
            print(
                f"Using pinned canonical direction dataset: {pinned_canonical_dataset}",
                file=sys.stderr,
            )

        if use_pinned_labeled_csv and not args.dry_run:
            shutil.copyfile(pinned_labeled_csv, labeled_snapshot_csv)
            if pinned_labeled_meta is not None:
                shutil.copyfile(pinned_labeled_meta, labeled_meta_output)
                shutil.copyfile(pinned_labeled_meta, labeled_snapshot_meta)
            quality_input = labeled_snapshot_csv
            if not candidate_quality_input_cfg:
                candidate_quality_input = quality_input
            elif not candidate_quality_input.exists():
                candidate_quality_input = quality_input
            candidate_gate_input = candidate_quality_input
            print(f"Using pinned labeled backtest CSV: {pinned_labeled_csv}", file=sys.stderr)
        elif bool(quality_cfg.get("build_labeled_dataset", True)):
            labeled_cmd = [
                python,
                "-m",
                "src.scripts.build_labeled_backtest_from_history",
                "--output",
                str(quality_input),
                "--meta-output",
                str(labeled_meta_output),
                "--fold-size",
                str(int(quality_cfg.get("fold_size", 12))),
                "--lookback-rows",
                str(int(quality_cfg.get("lookback_rows", 2000))),
                "--lookback-hours",
                str(int(quality_cfg.get("lookback_hours", 0))),
                "--min-rows",
                str(int(quality_cfg.get("min_labeled_rows", 200))),
            ]
            if resolved_quality_backtest_csv is not None:
                labeled_cmd.extend(["--backtest-csv", str(resolved_quality_backtest_csv)])
            if bool(quality_cfg.get("prefer_backtest", True)):
                labeled_cmd.append("--prefer-backtest")
            else:
                labeled_cmd.append("--no-prefer-backtest")
            results.append(
                _run_step(
                    "build_labeled_dataset",
                    labeled_cmd,
                    logs_dir / "build_labeled_dataset.log",
                    args.dry_run,
                )
            )
            if not args.dry_run:
                (summary_dir / "quality_backtest_resolution.json").write_text(
                    json.dumps(
                        {
                            "configured_backtest_csv": str(quality_backtest_csv_raw) if quality_backtest_csv_raw is not None else None,
                            "candidate_quality_input": str(candidate_quality_input),
                            "resolved_backtest_csv": str(resolved_quality_backtest_csv) if resolved_quality_backtest_csv is not None else None,
                            "resolved_from_auto": bool(quality_backtest_csv_is_auto),
                            "resolved_from_missing_default_latest": bool(quality_backtest_csv_points_to_missing_default),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        if not args.dry_run and quality_input.exists() and quality_input != labeled_snapshot_csv:
            shutil.copyfile(quality_input, labeled_snapshot_csv)
            quality_input = labeled_snapshot_csv
            if not candidate_quality_input_cfg:
                candidate_quality_input = quality_input
            elif not candidate_quality_input.exists():
                candidate_quality_input = quality_input
            candidate_gate_input = candidate_quality_input
        if not args.dry_run and labeled_meta_output.exists() and labeled_meta_output != labeled_snapshot_meta:
            shutil.copyfile(labeled_meta_output, labeled_snapshot_meta)

        walkforward_horizon = int(quality_cfg.get("walkforward_horizon", 1))
        walkforward_dataset = Path(
            quality_cfg.get("walkforward_dataset") or _direction_dataset_for_horizon(walkforward_horizon)
        )
        walkforward_target = str(quality_cfg.get("walkforward_y_key") or "y")
        walkforward_output = summary_dir / "walkforward_validation.json"
        selected_model_kind = str(quality_cfg.get("walkforward_model_kind", "meta_stack"))
        labeled_overlap_dataset: Path | None = None
        labeled_overlap_meta: Path | None = None

        if bool(canonical_cfg.get("enabled", True)):
            canonical_output_dir = Path(str(canonical_cfg.get("output_dir", "artifacts/datasets")))
            canonical_dataset = (
                pinned_canonical_dataset
                if pinned_canonical_dataset is not None
                else (canonical_output_dir / "btc_features_1h_direction_splits.npz")
            )
            canonical_meta = (
                pinned_canonical_meta
                if pinned_canonical_dataset is not None
                else Path(str(canonical_cfg.get("meta_path", "artifacts/datasets/btc_features_1h_direction_meta.json")))
            )
            snapshot_dataset = summary_dir / "btc_features_1h_direction_splits.snapshot.npz"
            snapshot_meta = summary_dir / "btc_features_1h_direction_meta.snapshot.json"
            walkforward_dataset = snapshot_dataset

            if not args.dry_run and canonical_dataset.exists():
                shutil.copyfile(canonical_dataset, snapshot_dataset)
            if not args.dry_run and canonical_meta is not None and canonical_meta.exists():
                shutil.copyfile(canonical_meta, snapshot_meta)

            if (not args.dry_run) and walkforward_dataset.exists() and snapshot_meta.exists():
                try:
                    meta_payload = _load_json(snapshot_meta)
                    meta_rows = int(meta_payload.get("row_count", 0) or 0)
                    npz_rows = _count_npz_rows(walkforward_dataset)
                    consistency_payload = {
                        "dataset_path": str(walkforward_dataset),
                        "meta_path": str(snapshot_meta),
                        "meta_row_count": meta_rows,
                        "npz_row_count": npz_rows,
                        "rows_match": bool(meta_rows == npz_rows),
                    }
                    (summary_dir / "canonical_dataset_consistency.json").write_text(
                        json.dumps(consistency_payload, indent=2),
                        encoding="utf-8",
                    )
                    if meta_rows != npz_rows:
                        print(
                            "Warning: canonical dataset snapshot mismatch "
                            f"(meta rows={meta_rows}, npz rows={npz_rows}).",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(f"Warning: failed canonical dataset consistency check: {exc}", file=sys.stderr)

            if bool(raw_snapshot_cfg.get("enabled", True)) and (walkforward_dataset.exists() or args.dry_run):
                raw_snapshot_cmd = [
                    python,
                    "-m",
                    "src.scripts.export_direction_feature_snapshot",
                    "--dataset",
                    str(walkforward_dataset),
                    "--output",
                    str(raw_feature_snapshot_path),
                    "--meta-output",
                    str(raw_feature_snapshot_meta),
                ]
                results.append(
                    _run_step(
                        "direction_feature_snapshot",
                        raw_snapshot_cmd,
                        logs_dir / "direction_feature_snapshot.log",
                        args.dry_run,
                    )
                )

        if bool(reconcile_cfg.get("enabled", False)) and (quality_input.exists() or args.dry_run):
            labeled_overlap_dataset = summary_dir / "btc_features_1h_direction_splits.labeled_overlap.npz"
            labeled_overlap_meta = summary_dir / "walkforward_labeled_overlap_meta.json"
            overlap_cmd = [
                python,
                "-m",
                "src.scripts.slice_direction_dataset_by_timestamps",
                "--dataset",
                str(walkforward_dataset),
                "--labeled-csv",
                str(quality_input),
                "--ts-col",
                str(reconcile_cfg.get("ts_col", "ts")),
                "--min-rows",
                str(int(reconcile_cfg.get("min_rows", 120))),
                "--output-dataset",
                str(labeled_overlap_dataset),
                "--output-meta",
                str(labeled_overlap_meta),
            ]
            results.append(
                _run_step(
                    "walkforward_labeled_overlap_dataset",
                    overlap_cmd,
                    logs_dir / "walkforward_labeled_overlap_dataset.log",
                    args.dry_run,
                )
            )

            if bool(raw_snapshot_cfg.get("enabled", True)) and (labeled_overlap_dataset.exists() or args.dry_run):
                raw_overlap_cmd = [
                    python,
                    "-m",
                    "src.scripts.export_direction_feature_snapshot",
                    "--dataset",
                    str(labeled_overlap_dataset),
                    "--output",
                    str(raw_feature_overlap_snapshot_path),
                    "--meta-output",
                    str(raw_feature_overlap_snapshot_meta),
                ]
                results.append(
                    _run_step(
                        "direction_feature_overlap_snapshot",
                        raw_overlap_cmd,
                        logs_dir / "direction_feature_overlap_snapshot.log",
                        args.dry_run,
                    )
                )

        if (
            bool(overlap_drift_guard_cfg.get("enabled", False))
            and labeled_overlap_dataset is not None
            and (labeled_overlap_dataset.exists() or args.dry_run)
        ):
            baseline_pack_path_cfg = overlap_drift_guard_cfg.get("baseline_pack_path")
            baseline_pack_path = Path(str(baseline_pack_path_cfg)) if baseline_pack_path_cfg else None
            if baseline_pack_path is None and bool(overlap_drift_guard_cfg.get("auto_discover_latest", True)):
                latest_pack = _find_latest_trusted_baseline_pack(
                    run_root=args.run_root,
                    current_run_id=timestamp,
                )
                if latest_pack is not None:
                    _, baseline_pack_path = latest_pack
            if baseline_pack_path is not None and (baseline_pack_path.exists() or args.dry_run):
                guard_cmd = [
                    python,
                    "-m",
                    "src.scripts.analyze_overlap_feature_drift_guard",
                    "--baseline-pack",
                    str(baseline_pack_path),
                    "--current-overlap-dataset",
                    str(labeled_overlap_dataset),
                    "--tail-rows",
                    str(int(overlap_drift_guard_cfg.get("tail_rows", 24))),
                    "--warn-abs-train-std-shift",
                    str(float(overlap_drift_guard_cfg.get("warn_abs_train_std_shift", 1.5))),
                    "--fail-abs-train-std-shift",
                    str(float(overlap_drift_guard_cfg.get("fail_abs_train_std_shift", 2.5))),
                    "--min-failed-features",
                    str(int(overlap_drift_guard_cfg.get("min_failed_features", 2))),
                    "--output",
                    str(overlap_feature_drift_guard_path),
                ]
                for prefix in overlap_drift_guard_cfg.get("feature_prefixes", []):
                    if str(prefix).strip():
                        guard_cmd.extend(["--feature-prefix", str(prefix)])
                for feature_name in overlap_drift_guard_cfg.get("feature_names", []):
                    if str(feature_name).strip():
                        guard_cmd.extend(["--feature-name", str(feature_name)])
                results.append(
                    _run_step(
                        "overlap_feature_drift_guard",
                        guard_cmd,
                        logs_dir / "overlap_feature_drift_guard.log",
                        args.dry_run,
                    )
                )
                if overlap_feature_drift_guard_path.exists() and not args.dry_run:
                    guard_payload = _load_json(overlap_feature_drift_guard_path)
                    if bool(guard_payload.get("guard_failed", False)) and bool(
                        overlap_drift_guard_cfg.get("enforce_for_paper_live", True)
                    ):
                        edge_trustworthy_for_paper_live = False
                        print(
                            "Overlap feature drift guard failed; paper-live run will use conservative hold thresholds.",
                            file=sys.stderr,
                        )
            elif not args.dry_run:
                print(
                    "Warning: overlap feature drift guard enabled but no trusted baseline pack was resolved; skipping guard.",
                    file=sys.stderr,
                )

        if (
            bool(overlap_pre_tuning_cfg.get("enabled", True))
            and bool(overlap_pre_tuning_cfg.get("feature_reliability", {}).get("enabled", True))
            and labeled_overlap_dataset is not None
            and (labeled_overlap_dataset.exists() or args.dry_run)
        ):
            overlap_feature_cfg = overlap_pre_tuning_cfg.get("feature_reliability", {})
            overlap_feature_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_feature_reliability",
                "--input",
                str(labeled_overlap_dataset),
                "--baseline-window",
                str(int(overlap_feature_cfg.get("baseline_window", 80))),
                "--recent-window",
                str(int(overlap_feature_cfg.get("recent_window", 40))),
                "--min-score",
                str(float(overlap_feature_cfg.get("min_score", 0.55))),
                "--max-features",
                str(int(overlap_feature_cfg.get("max_features", 0))),
                "--output",
                str(overlap_feature_reliability_path),
            ]
            results.append(
                _run_step(
                    "overlap_feature_reliability",
                    overlap_feature_cmd,
                    logs_dir / "overlap_feature_reliability.log",
                    args.dry_run,
                )
            )

        if walkforward_dataset.exists() or args.dry_run:
            if bool(compare_cfg.get("enabled", True)):
                compare_output = summary_dir / "walkforward_model_compare.json"
                full_min_train = int(compare_cfg.get("min_train_size", 30))
                full_min_val = int(compare_cfg.get("min_val_size", 20))
                full_min_test = int(compare_cfg.get("min_test_size", 20))
                compare_cmd = [
                    python,
                    "-m",
                    "src.scripts.compare_walkforward_models",
                    "--dataset-path",
                    str(walkforward_dataset),
                    "--y-key",
                    walkforward_target,
                    "--folds",
                    str(int(compare_cfg.get("folds", cv_folds))),
                    "--train-size",
                    str(int(compare_cfg.get("train_size", cv_train_size))),
                    "--val-size",
                    str(int(compare_cfg.get("val_size", cv_val_size))),
                    "--test-size",
                    str(int(compare_cfg.get("test_size", cv_test_size))),
                    "--gap",
                    str(int(compare_cfg.get("gap", cv_gap))),
                    "--purge-size",
                    str(int(compare_cfg.get("purge_size", cv_purge_size))),
                    "--embargo-size",
                    str(int(compare_cfg.get("embargo_size", cv_embargo_size))),
                    "--mode",
                    str(compare_cfg.get("mode", cv_mode)),
                    "--min-train-size",
                    str(int(full_min_train)),
                    "--min-val-size",
                    str(int(full_min_val)),
                    "--min-test-size",
                    str(int(full_min_test)),
                    "--signal-threshold",
                    str(float(compare_cfg.get("signal_threshold", quality_cfg.get("walkforward_signal_threshold", 0.5)))),
                    "--fee-bps",
                    str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                    "--slippage-bps",
                    str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                    "--rolling-guard" if bool(compare_cfg.get("rolling_guard", True)) else "",
                    "--meta-margin",
                    str(float(compare_cfg.get("meta_margin", 0.0))),
                    "--meta-min-rolling-trades",
                    str(int(compare_cfg.get("meta_min_rolling_trades", 0))),
                    "--selection-policy",
                    str(compare_cfg.get("selection_policy", "incumbent_guarded")),
                    "--min-auc",
                    str(float(compare_cfg.get("min_auc", 0.5))),
                    "--output",
                    str(compare_output),
                ]
                compare_cmd = [part for part in compare_cmd if part != ""]
                results.append(
                    _run_step(
                        "walkforward_model_compare",
                        compare_cmd,
                        logs_dir / "walkforward_model_compare.log",
                        args.dry_run,
                    )
                )
                if compare_output.exists() and not args.dry_run:
                    compare_payload = _load_json(compare_output)
                    selected_model_kind = str(compare_payload.get("selected_model_kind", selected_model_kind))
                    selected_path = None
                    rows = compare_payload.get("rows", [])
                    if isinstance(rows, list):
                        for row in rows:
                            if not isinstance(row, dict):
                                continue
                            if str(row.get("model_kind")) == selected_model_kind:
                                selected_path = row.get("path")
                                break
                    if selected_path:
                        selected_file = Path(str(selected_path))
                        if selected_file.exists():
                            shutil.copyfile(selected_file, walkforward_output)

                if labeled_overlap_dataset is not None and (labeled_overlap_dataset.exists() or args.dry_run):
                    overlap_compare_output = summary_dir / "walkforward_model_compare_labeled_overlap.json"
                    overlap_compare_cfg_obj = reconcile_cfg.get("overlap_compare", {}) if isinstance(reconcile_cfg, dict) else {}
                    overlap_compare_cfg = overlap_compare_cfg_obj if isinstance(overlap_compare_cfg_obj, dict) else {}
                    overlap_folds = int(overlap_compare_cfg.get("folds", compare_cfg.get("folds", cv_folds)))
                    overlap_train_size = int(overlap_compare_cfg.get("train_size", compare_cfg.get("train_size", cv_train_size)))
                    overlap_val_size = int(overlap_compare_cfg.get("val_size", compare_cfg.get("val_size", cv_val_size)))
                    overlap_test_size = int(overlap_compare_cfg.get("test_size", compare_cfg.get("test_size", cv_test_size)))
                    overlap_gap = int(overlap_compare_cfg.get("gap", compare_cfg.get("gap", cv_gap)))
                    overlap_purge = int(overlap_compare_cfg.get("purge_size", compare_cfg.get("purge_size", cv_purge_size)))
                    overlap_embargo = int(overlap_compare_cfg.get("embargo_size", compare_cfg.get("embargo_size", cv_embargo_size)))
                    overlap_mode = str(overlap_compare_cfg.get("mode", compare_cfg.get("mode", cv_mode)))
                    overlap_min_train = int(overlap_compare_cfg.get("min_train_size", 20))
                    overlap_min_val = int(overlap_compare_cfg.get("min_val_size", 10))
                    overlap_min_test = int(overlap_compare_cfg.get("min_test_size", 10))
                    overlap_compare_cmd = [
                        python,
                        "-m",
                        "src.scripts.compare_walkforward_models",
                        "--dataset-path",
                        str(labeled_overlap_dataset),
                        "--y-key",
                        walkforward_target,
                        "--folds",
                        str(int(overlap_folds)),
                        "--train-size",
                        str(int(overlap_train_size)),
                        "--val-size",
                        str(int(overlap_val_size)),
                        "--test-size",
                        str(int(overlap_test_size)),
                        "--gap",
                        str(int(overlap_gap)),
                        "--purge-size",
                        str(int(overlap_purge)),
                        "--embargo-size",
                        str(int(overlap_embargo)),
                        "--mode",
                        str(overlap_mode),
                        "--min-train-size",
                        str(int(overlap_min_train)),
                        "--min-val-size",
                        str(int(overlap_min_val)),
                        "--min-test-size",
                        str(int(overlap_min_test)),
                        "--signal-threshold",
                        str(float(compare_cfg.get("signal_threshold", quality_cfg.get("walkforward_signal_threshold", 0.5)))),
                        "--fee-bps",
                        str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                        "--slippage-bps",
                        str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                        "--rolling-guard" if bool(compare_cfg.get("rolling_guard", True)) else "",
                        "--meta-margin",
                        str(float(compare_cfg.get("meta_margin", 0.0))),
                        "--meta-min-rolling-trades",
                        str(int(compare_cfg.get("meta_min_rolling_trades", 0))),
                        "--selection-policy",
                        str(overlap_compare_cfg.get("selection_policy", "best_cum_ret")),
                        "--min-auc",
                        str(float(overlap_compare_cfg.get("min_auc", compare_cfg.get("min_auc", 0.5)))),
                        "--output",
                        str(overlap_compare_output),
                    ]
                    overlap_compare_cmd = [part for part in overlap_compare_cmd if part != ""]
                    results.append(
                        _run_step(
                            "walkforward_model_compare_labeled_overlap",
                            overlap_compare_cmd,
                            logs_dir / "walkforward_model_compare_labeled_overlap.log",
                            args.dry_run,
                        )
                    )

                    if not args.dry_run and compare_output.exists() and overlap_compare_output.exists():
                        full_payload = _load_json(compare_output)
                        overlap_payload = _load_json(overlap_compare_output)

                        def _selected_row(payload: Dict[str, Any]) -> Dict[str, Any]:
                            selected_kind = str(payload.get("selected_model_kind", ""))
                            rows_obj = payload.get("rows", [])
                            if not isinstance(rows_obj, list):
                                return {}
                            for item in rows_obj:
                                if isinstance(item, dict) and str(item.get("model_kind", "")) == selected_kind:
                                    return item
                            return {}

                        def _row_by_model_kind(payload: Dict[str, Any], model_kind: str) -> Dict[str, Any]:
                            rows_obj = payload.get("rows", [])
                            if not isinstance(rows_obj, list):
                                return {}
                            for item in rows_obj:
                                if isinstance(item, dict) and str(item.get("model_kind", "")) == str(model_kind):
                                    return item
                            return {}

                        overlap_selection_cfg_obj = reconcile_cfg.get("overlap_model_selection", {}) if isinstance(reconcile_cfg, dict) else {}
                        overlap_selection_cfg = overlap_selection_cfg_obj if isinstance(overlap_selection_cfg_obj, dict) else {}
                        overlap_selection_enabled = bool(overlap_selection_cfg.get("enabled", False))
                        overlap_selection_primary = str(overlap_selection_cfg.get("primary_model", "xgb"))
                        overlap_selection_fallback = str(overlap_selection_cfg.get("fallback_model", "meta_stack"))
                        overlap_selection_min_ret_improvement = float(overlap_selection_cfg.get("min_ret_improvement", 0.0))
                        overlap_selection_only_when_primary_negative = bool(
                            overlap_selection_cfg.get("only_when_primary_negative", True)
                        )
                        overlap_selection_require_auc_non_worse = bool(
                            overlap_selection_cfg.get("require_fallback_auc_non_worse", False)
                        )
                        overlap_selection_min_fallback_trades = int(overlap_selection_cfg.get("min_fallback_trades", 0))

                        overlap_selection_override_payload: Dict[str, Any] | None = None
                        if overlap_selection_enabled:
                            primary_row = _row_by_model_kind(overlap_payload, overlap_selection_primary)
                            fallback_row = _row_by_model_kind(overlap_payload, overlap_selection_fallback)
                            if primary_row and fallback_row:
                                primary_ret = float(primary_row.get("cum_ret_net_total", 0.0) or 0.0)
                                fallback_ret = float(fallback_row.get("cum_ret_net_total", 0.0) or 0.0)
                                primary_auc = float(primary_row.get("auc_mean", float("nan")))
                                fallback_auc = float(fallback_row.get("auc_mean", float("nan")))
                                fallback_trades = int(fallback_row.get("trade_count_total", 0) or 0)

                                ret_improvement_ok = fallback_ret >= (primary_ret + overlap_selection_min_ret_improvement)
                                primary_negative_ok = (primary_ret < 0.0) if overlap_selection_only_when_primary_negative else True
                                auc_ok = (fallback_auc >= primary_auc) if overlap_selection_require_auc_non_worse else True
                                trades_ok = fallback_trades >= overlap_selection_min_fallback_trades
                                selected_override = bool(ret_improvement_ok and primary_negative_ok and auc_ok and trades_ok)

                                if selected_override:
                                    overlap_payload["selected_model_kind"] = overlap_selection_fallback

                                overlap_selection_override_payload = {
                                    "enabled": True,
                                    "primary_model": overlap_selection_primary,
                                    "fallback_model": overlap_selection_fallback,
                                    "selected_override": bool(selected_override),
                                    "constraints": {
                                        "min_ret_improvement": overlap_selection_min_ret_improvement,
                                        "only_when_primary_negative": overlap_selection_only_when_primary_negative,
                                        "require_fallback_auc_non_worse": overlap_selection_require_auc_non_worse,
                                        "min_fallback_trades": overlap_selection_min_fallback_trades,
                                    },
                                    "metrics": {
                                        "primary_ret": primary_ret,
                                        "fallback_ret": fallback_ret,
                                        "primary_auc": primary_auc,
                                        "fallback_auc": fallback_auc,
                                        "fallback_trade_count": fallback_trades,
                                    },
                                    "checks": {
                                        "ret_improvement_ok": bool(ret_improvement_ok),
                                        "primary_negative_ok": bool(primary_negative_ok),
                                        "auc_ok": bool(auc_ok),
                                        "trades_ok": bool(trades_ok),
                                    },
                                }
                            else:
                                overlap_selection_override_payload = {
                                    "enabled": True,
                                    "primary_model": overlap_selection_primary,
                                    "fallback_model": overlap_selection_fallback,
                                    "selected_override": False,
                                    "reason": "missing_model_rows",
                                }

                        full_row = _selected_row(full_payload)
                        overlap_row = _selected_row(overlap_payload)
                        overlap_rows_obj = overlap_payload.get("rows", [])
                        overlap_rows = overlap_rows_obj if isinstance(overlap_rows_obj, list) else []

                        overlap_pruning_cfg_obj = overlap_pre_tuning_cfg.get("model_pruning", {}) if isinstance(overlap_pre_tuning_cfg, dict) else {}
                        overlap_pruning_cfg = overlap_pruning_cfg_obj if isinstance(overlap_pruning_cfg_obj, dict) else {}
                        overlap_pruning_enabled = bool(overlap_pre_tuning_cfg.get("enabled", True)) and bool(
                            overlap_pruning_cfg.get("enabled", True)
                        )
                        overlap_min_model_cum_ret = float(overlap_pruning_cfg.get("min_cum_ret", 0.0))
                        overlap_min_model_trades = int(overlap_pruning_cfg.get("min_trade_count", 10))

                        pruned_rows: List[Dict[str, Any]] = []
                        rejected_rows: List[Dict[str, Any]] = []
                        for row in overlap_rows:
                            if not isinstance(row, dict):
                                continue
                            row_ret = float(row.get("cum_ret_net_total", float("nan")))
                            row_trades = int(row.get("trade_count_total", 0) or 0)
                            reasons: List[str] = []
                            if row_ret < overlap_min_model_cum_ret:
                                reasons.append("min_cum_ret")
                            if row_trades < overlap_min_model_trades:
                                reasons.append("min_trade_count")
                            if reasons:
                                rejected_rows.append(
                                    {
                                        "model_kind": row.get("model_kind"),
                                        "cum_ret_net_total": row_ret,
                                        "trade_count_total": row_trades,
                                        "reasons": reasons,
                                    }
                                )
                            else:
                                pruned_rows.append(row)

                        pruned_selected_model_kind = None
                        pruned_selected_row: Dict[str, Any] | None = None
                        if overlap_pruning_enabled and pruned_rows:
                            pruned_rows_sorted = sorted(
                                pruned_rows,
                                key=lambda r: (
                                    float(r.get("cum_ret_net_total", float("-inf"))),
                                    int(r.get("trade_count_total", 0) or 0),
                                    float(r.get("auc_mean", float("-inf"))),
                                ),
                                reverse=True,
                            )
                            pruned_selected_row = pruned_rows_sorted[0]
                            pruned_selected_model_kind = str(pruned_selected_row.get("model_kind", ""))
                            overlap_payload["selected_model_kind"] = pruned_selected_model_kind
                            overlap_row = pruned_selected_row
                        require_viable_for_tuning = bool(overlap_pruning_cfg.get("require_viable_model_for_tuning", True))
                        overlap_pruning_allows_tuning = (not overlap_pruning_enabled) or bool(pruned_rows) or (not require_viable_for_tuning)

                        overlap_pruning_payload = {
                            "enabled": bool(overlap_pruning_enabled),
                            "source_compare_path": str(overlap_compare_output),
                            "constraints": {
                                "min_cum_ret": overlap_min_model_cum_ret,
                                "min_trade_count": overlap_min_model_trades,
                                "require_viable_model_for_tuning": require_viable_for_tuning,
                            },
                            "selected_model_from_compare": overlap_payload.get("selected_model_kind"),
                            "pruned_selected_model": pruned_selected_model_kind,
                            "pruned_selected_row": pruned_selected_row,
                            "viable_rows": pruned_rows,
                            "rejected_rows": rejected_rows,
                            "allows_tuning": bool(overlap_pruning_allows_tuning),
                        }
                        overlap_model_pruning_path.write_text(
                            json.dumps(overlap_pruning_payload, indent=2),
                            encoding="utf-8",
                        )

                        full_ret = float(full_row.get("cum_ret_net_total", 0.0) or 0.0)
                        overlap_ret = float(overlap_row.get("cum_ret_net_total", 0.0) or 0.0)
                        full_depth = _walkforward_depth_metrics(Path(str(full_row.get("path", "")))) if full_row else {}
                        overlap_depth = _walkforward_depth_metrics(Path(str(overlap_row.get("path", "")))) if overlap_row else {}
                        same_sign = bool((full_ret == 0.0 and overlap_ret == 0.0) or (full_ret * overlap_ret > 0.0))
                        abs_gap = float(abs(full_ret - overlap_ret))
                        min_overlap = float(reconcile_cfg.get("min_overlap_cum_ret", -0.005))
                        require_non_negative_overlap = bool(reconcile_cfg.get("require_non_negative_overlap_selected", True))
                        overlap_non_negative_ok = (overlap_ret >= 0.0) if require_non_negative_overlap else True
                        max_abs_gap = float(reconcile_cfg.get("max_abs_cum_ret_gap", 0.03))
                        min_overlap_folds = int(reconcile_cfg.get("min_overlap_folds", 2))
                        min_overlap_test_rows = int(reconcile_cfg.get("min_overlap_test_rows", 40))
                        overlap_folds_ok = int(overlap_depth.get("n_folds", 0) or 0) >= min_overlap_folds
                        overlap_test_rows_ok = int(overlap_depth.get("test_rows_total", 0) or 0) >= min_overlap_test_rows
                        overlap_depth_ok = bool(overlap_folds_ok and overlap_test_rows_ok)
                        trustworthy = bool(
                            same_sign
                            and overlap_ret >= min_overlap
                            and overlap_non_negative_ok
                            and abs_gap <= max_abs_gap
                            and overlap_depth_ok
                        )
                        failed_checks = []
                        if not same_sign:
                            failed_checks.append("same_return_sign")
                        if overlap_ret < min_overlap:
                            failed_checks.append("min_overlap_cum_ret")
                        if not overlap_non_negative_ok:
                            failed_checks.append("overlap_selected_non_negative")
                        if abs_gap > max_abs_gap:
                            failed_checks.append("max_abs_cum_ret_gap")
                        if not overlap_folds_ok:
                            failed_checks.append("min_overlap_folds")
                        if not overlap_test_rows_ok:
                            failed_checks.append("min_overlap_test_rows")

                        reconciliation_payload = {
                            "full_compare_path": str(compare_output),
                            "labeled_overlap_compare_path": str(overlap_compare_output),
                            "labeled_overlap_meta_path": str(labeled_overlap_meta) if labeled_overlap_meta else None,
                            "overlap_model_pruning_path": str(overlap_model_pruning_path),
                            "overlap_model_selection_override": overlap_selection_override_payload,
                            "full_selected_model": full_payload.get("selected_model_kind"),
                            "overlap_selected_model": overlap_payload.get("selected_model_kind"),
                            "full_selected_row": full_row,
                            "overlap_selected_row": overlap_row,
                            "full_depth": full_depth,
                            "overlap_depth": overlap_depth,
                            "agreement": {
                                "same_return_sign": same_sign,
                                "abs_cum_ret_gap": abs_gap,
                                "min_overlap_cum_ret": min_overlap,
                                "require_non_negative_overlap_selected": bool(require_non_negative_overlap),
                                "overlap_non_negative_ok": bool(overlap_non_negative_ok),
                                "max_abs_cum_ret_gap": max_abs_gap,
                                "min_overlap_folds": int(min_overlap_folds),
                                "min_overlap_test_rows": int(min_overlap_test_rows),
                                "overlap_folds_ok": bool(overlap_folds_ok),
                                "overlap_test_rows_ok": bool(overlap_test_rows_ok),
                                "overlap_depth_ok": bool(overlap_depth_ok),
                                "edge_trustworthy": trustworthy,
                                "failed_checks": failed_checks,
                            },
                        }
                        (summary_dir / "walkforward_labeled_reconciliation.json").write_text(
                            json.dumps(reconciliation_payload, indent=2),
                            encoding="utf-8",
                        )
                        overlap_stability_output = summary_dir / "overlap_trust_stability.json"
                        full_selected_path = Path(str(full_row.get("path", ""))) if full_row else None
                        overlap_selected_path = Path(str(overlap_row.get("path", ""))) if overlap_row else None
                        if (
                            quality_input.exists()
                            and labeled_overlap_dataset is not None
                            and labeled_overlap_dataset.exists()
                            and full_selected_path is not None
                            and overlap_selected_path is not None
                            and full_selected_path.exists()
                            and overlap_selected_path.exists()
                        ):
                            overlap_feature_sources = [summary_dir / "backtest_signals_meta_ensemble_decision_aligned.csv"]
                            overlap_feature_sources.append(quality_input)
                            overlap_diag_cmd = [
                                python,
                                "-m",
                                "src.scripts.analyze_overlap_trust_stability",
                                "--full-walkforward",
                                str(full_selected_path),
                                "--overlap-walkforward",
                                str(overlap_selected_path),
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--labeled-csv",
                                str(quality_input),
                                "--feature-source",
                                str(overlap_feature_sources[0]),
                                "--feature-source",
                                str(overlap_feature_sources[1]),
                                "--ts-col",
                                str(reconcile_cfg.get("ts_col", "ts")),
                                "--return-col",
                                str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                "--signal-col",
                                str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                "--output",
                                str(overlap_stability_output),
                            ]
                            results.append(
                                _run_step(
                                    "overlap_trust_stability",
                                    overlap_diag_cmd,
                                    logs_dir / "overlap_trust_stability.log",
                                    args.dry_run,
                                )
                            )
                            reconciliation_payload["overlap_trust_stability_path"] = str(overlap_stability_output)
                            (summary_dir / "walkforward_labeled_reconciliation.json").write_text(
                                json.dumps(reconciliation_payload, indent=2),
                                encoding="utf-8",
                            )
                        edge_trustworthiness_payload = {
                            "edge_trustworthy": bool(trustworthy),
                            "source": "walkforward_labeled_reconciliation",
                            "reconciliation_path": str(summary_dir / "walkforward_labeled_reconciliation.json"),
                            "enforce_for_paper_live": bool(reconcile_cfg.get("enforce_for_paper_live", True)),
                        }
                        if overlap_feature_drift_guard_path.exists():
                            try:
                                drift_guard_payload = _load_json(overlap_feature_drift_guard_path)
                                edge_trustworthiness_payload["overlap_feature_drift_guard_path"] = str(
                                    overlap_feature_drift_guard_path
                                )
                                edge_trustworthiness_payload["overlap_feature_drift_guard_failed"] = bool(
                                    drift_guard_payload.get("guard_failed", False)
                                )
                            except Exception:
                                pass
                        (summary_dir / "edge_trustworthiness.json").write_text(
                            json.dumps(edge_trustworthiness_payload, indent=2),
                            encoding="utf-8",
                        )
                        if bool(reconcile_cfg.get("enforce_for_paper_live", True)) and not bool(trustworthy):
                            edge_trustworthy_for_paper_live = False

                        if (
                            bool(trusted_baseline_pack_cfg.get("enabled", True))
                            and bool(trustworthy)
                            and bool(trusted_baseline_pack_cfg.get("write_when_edge_trustworthy", True))
                        ):
                            baseline_pack_cmd = [
                                python,
                                "-m",
                                "src.scripts.create_trusted_baseline_pack",
                                "--run-id",
                                timestamp,
                                "--run-root",
                                str(args.run_root),
                                "--output",
                                str(summary_dir / "trusted_baseline_pack.json"),
                            ]
                            results.append(
                                _run_step(
                                    "trusted_baseline_pack",
                                    baseline_pack_cmd,
                                    logs_dir / "trusted_baseline_pack.log",
                                    args.dry_run,
                                )
                            )

                        if overlap_as_primary:
                            selected_model_kind = str(overlap_payload.get("selected_model_kind", selected_model_kind))
                            overlap_rows = overlap_payload.get("rows", [])
                            if isinstance(overlap_rows, list):
                                for row in overlap_rows:
                                    if not isinstance(row, dict):
                                        continue
                                    if str(row.get("model_kind")) != selected_model_kind:
                                        continue
                                    selected_path = row.get("path")
                                    if selected_path:
                                        selected_file = Path(str(selected_path))
                                        if selected_file.exists():
                                            shutil.copyfile(selected_file, walkforward_output)
                                    break
                            walkforward_dataset = labeled_overlap_dataset
            else:
                walkforward_cmd = [
                    python,
                    "-m",
                    "src.scripts.run_walkforward_validation",
                    "--dataset-path",
                    str(walkforward_dataset),
                    "--y-key",
                    walkforward_target,
                    "--folds",
                    str(cv_folds),
                    "--train-size",
                    str(cv_train_size),
                    "--val-size",
                    str(cv_val_size),
                    "--test-size",
                    str(cv_test_size),
                    "--gap",
                    str(cv_gap),
                    "--purge-size",
                    str(cv_purge_size),
                    "--embargo-size",
                    str(cv_embargo_size),
                    "--mode",
                    cv_mode,
                    "--model-kind",
                    selected_model_kind,
                    "--signal-threshold",
                    str(float(quality_cfg.get("walkforward_signal_threshold", 0.5))),
                    "--fee-bps",
                    str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                    "--slippage-bps",
                    str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                    "--output",
                    str(walkforward_output),
                ]
                results.append(_run_step("walkforward_validation", walkforward_cmd, logs_dir / "walkforward_validation.log", args.dry_run))

        if bool(label_ablation_cfg.get("enabled", False)):
            label_ablation_cmd = [
                python,
                "-m",
                "src.scripts.run_label_ablation",
                "--output-dir",
                str(summary_dir / "label_ablation"),
                "--threshold",
                str(float(label_ablation_cfg.get("threshold", 0.0))),
                "--walkforward-folds",
                str(int(label_ablation_cfg.get("folds", cv_folds))),
                "--walkforward-train-size",
                str(int(label_ablation_cfg.get("train_size", cv_train_size))),
                "--walkforward-val-size",
                str(int(label_ablation_cfg.get("val_size", cv_val_size))),
                "--walkforward-test-size",
                str(int(label_ablation_cfg.get("test_size", cv_test_size))),
                "--walkforward-gap",
                str(int(label_ablation_cfg.get("gap", cv_gap))),
                "--walkforward-purge",
                str(int(label_ablation_cfg.get("purge_size", cv_purge_size))),
                "--walkforward-embargo",
                str(int(label_ablation_cfg.get("embargo_size", cv_embargo_size))),
                "--walkforward-mode",
                str(label_ablation_cfg.get("mode", cv_mode)),
                "--model-kind",
                str(label_ablation_cfg.get("model_kind", selected_model_kind)),
                "--signal-threshold",
                str(float(label_ablation_cfg.get("signal_threshold", quality_cfg.get("walkforward_signal_threshold", 0.5)))),
                "--fee-bps",
                str(float(label_ablation_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                "--slippage-bps",
                str(float(label_ablation_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                "--economics-min-trades",
                str(float(label_ablation_cfg.get("economics_min_trades", quality_cfg.get("min_trade_count", 10)))),
                "--economics-min-cum-ret",
                str(float(label_ablation_cfg.get("economics_min_cum_ret", quality_cfg.get("min_net_return", 0.0)))),
                "--economics-turnover-penalty",
                str(float(label_ablation_cfg.get("economics_turnover_penalty", 0.002))),
                "--economics-downside-penalty",
                str(float(label_ablation_cfg.get("economics_downside_penalty", 2.0))),
            ]
            feature_rel_json = label_ablation_cfg.get("feature_reliability_json")
            if (
                not feature_rel_json
                and bool(overlap_pre_tuning_cfg.get("enabled", True))
                and bool(overlap_pre_tuning_cfg.get("use_overlap_feature_reliability_for_ablation", True))
                and overlap_feature_reliability_path.exists()
            ):
                feature_rel_json = overlap_feature_reliability_path
            if feature_rel_json:
                label_ablation_cmd.extend(["--feature-reliability-json", str(feature_rel_json)])
                label_ablation_cmd.extend(
                    [
                        "--feature-reliability-min-score",
                        str(
                            float(
                                label_ablation_cfg.get(
                                    "feature_reliability_min_score",
                                    overlap_pre_tuning_cfg.get("feature_reliability", {}).get("min_score", 0.55),
                                )
                            )
                        ),
                    ]
                )
            results.append(
                _run_step(
                    "label_ablation",
                    label_ablation_cmd,
                    logs_dir / "label_ablation.log",
                    args.dry_run,
                )
            )

        if bool(leakage_cfg.get("enabled", True)):
            leakage_cmd = [
                python,
                "-m",
                "src.scripts.audit_point_in_time_integrity",
                "--dataset-path",
                str(walkforward_dataset),
                "--y-key",
                str(leakage_cfg.get("y_key", walkforward_target)),
                "--leakage-corr-alert",
                str(float(leakage_cfg.get("corr_alert", 0.98))),
                "--output",
                str(summary_dir / "point_in_time_audit.json"),
            ]
            results.append(_run_step("point_in_time_audit", leakage_cmd, logs_dir / "point_in_time_audit.log", args.dry_run))

        if bool(cv_stress_cfg.get("enabled", True)):
            cv_stress_folds = int(cv_stress_cfg.get("folds", cv_folds))
            cv_stress_train = int(cv_stress_cfg.get("train_size", cv_train_size))
            cv_stress_val = int(cv_stress_cfg.get("val_size", cv_val_size))
            cv_stress_test = int(cv_stress_cfg.get("test_size", cv_test_size))
            if walkforward_dataset.exists() and not args.dry_run:
                try:
                    n_rows = _count_npz_rows_from_x(walkforward_dataset)
                    cv_stress_folds, cv_stress_train, cv_stress_val, cv_stress_test = _compute_feasible_walkforward(
                        n_rows,
                        folds=cv_stress_folds,
                        train_size=cv_stress_train,
                        val_size=cv_stress_val,
                        test_size=cv_stress_test,
                        gap=int(cv_stress_cfg.get("gap", cv_gap)),
                        purge=int(cv_purge_size),
                        embargo=int(cv_embargo_size),
                    )
                except Exception as exc:
                    print(f"Warning: failed to auto-resize cv stress windows: {exc}", file=sys.stderr)
                    cv_stress_folds = 0
            if cv_stress_folds < 2 and not args.dry_run:
                print(
                    "Skipping cv_stress_sweep: insufficient samples for at least 2 feasible folds.",
                    file=sys.stderr,
                )
            else:
                cv_stress_cmd = [
                python,
                "-m",
                "src.scripts.run_cv_stress_sweep",
                "--dataset-path",
                str(walkforward_dataset),
                "--y-key",
                str(cv_stress_cfg.get("y_key", walkforward_target)),
                "--folds",
                str(int(cv_stress_folds)),
                "--train-size",
                str(int(cv_stress_train)),
                "--val-size",
                str(int(cv_stress_val)),
                "--test-size",
                str(int(cv_stress_test)),
                "--gap",
                str(int(cv_stress_cfg.get("gap", cv_gap))),
                "--purge-list",
                str(cv_stress_cfg.get("purge_list", "0,12,24")),
                "--embargo-list",
                str(cv_stress_cfg.get("embargo_list", "0,12,24")),
                "--mode",
                str(cv_stress_cfg.get("mode", cv_mode)),
                "--output",
                str(summary_dir / "cv_stress_sweep.json"),
                ]
                results.append(_run_step("cv_stress_sweep", cv_stress_cmd, logs_dir / "cv_stress_sweep.log", args.dry_run))

        if candidate_quality_input.exists() or args.dry_run:
            quality_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_model_quality",
                "--input",
                str(candidate_quality_input),
                "--output",
                str(summary_dir / "model_quality_candidate.json"),
            ]
            results.append(_run_step("model_quality_eval", quality_cmd, logs_dir / "model_quality_eval.log", args.dry_run))

            if bool(trade_decision_cfg.get("enabled", True)):
                decision_feature_input = summary_dir / "backtest_signals_meta_ensemble_decision_features.csv"
                decision_feature_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_features_meta.json"
                enrich_candidate_cmd = [
                    python,
                    "-m",
                    "src.scripts.enrich_backtest_with_decision_features",
                    "--input",
                    str(candidate_quality_input),
                    "--output",
                    str(decision_feature_input),
                    "--meta-output",
                    str(decision_feature_meta_path),
                    "--auto-discover-sources",
                ]
                if quality_input.exists() or args.dry_run:
                    enrich_candidate_cmd.extend(["--feature-source", str(quality_input)])
                incumbent_backtest_for_features = quality_cfg.get("incumbent_backtest_csv")
                if incumbent_backtest_for_features:
                    enrich_candidate_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    enrich_candidate_cmd.extend(["--incumbent-reference-source", str(incumbent_backtest_for_features)])
                results.append(
                    _run_step(
                        "enrich_candidate_decision_features",
                        enrich_candidate_cmd,
                        logs_dir / "enrich_candidate_decision_features.log",
                        args.dry_run,
                    )
                )
                decision_model_input = decision_feature_input if (decision_feature_input.exists() or args.dry_run) else candidate_quality_input
                tuning_quality_input = decision_model_input

                decision_cmd = [
                    python,
                    "-m",
                    "src.scripts.train_trade_decision_model",
                    "--input",
                    str(decision_model_input),
                    "--target-col",
                    str(trade_decision_cfg.get("target_col", "ret_ensemble_net")),
                    "--signal-col",
                    str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                    "--threshold",
                    str(float(trade_decision_cfg.get("threshold", 0.55))),
                    "--min-rows",
                    str(int(trade_decision_cfg.get("min_rows", 200))),
                    "--min-candidate-rows",
                    str(int(trade_decision_cfg.get("min_candidate_rows", 60))),
                    "--min-oof-rows",
                    str(int(trade_decision_cfg.get("min_oof_rows", 40))),
                    "--min-positive-oof-bins",
                    str(int(trade_decision_cfg.get("min_positive_oof_bins", 2))),
                    "--ev-calibration-source",
                    str(trade_decision_cfg.get("ev_calibration_source", "hybrid")),
                    "--ev-min-candidate-rows",
                    str(int(trade_decision_cfg.get("ev_min_candidate_rows", 100))),
                    "--ev-bins",
                    str(int(trade_decision_cfg.get("ev_bins", 8))),
                    "--min-bin-samples",
                    str(int(trade_decision_cfg.get("min_bin_samples", 8))),
                    "--raw-ev-calibration-source",
                    str(trade_decision_cfg.get("raw_ev_calibration_source", "weighted_hybrid")),
                    "--raw-ev-candidate-weight",
                    str(float(trade_decision_cfg.get("raw_ev_candidate_weight", 4.0))),
                    "--feature-meta-path",
                    str(decision_feature_meta_path),
                    "--output",
                    str(trade_decision_model_path),
                ]
                reference_feature_controls_cfg = (
                    trade_decision_cfg.get("reference_feature_controls")
                    if isinstance(trade_decision_cfg.get("reference_feature_controls"), dict)
                    else {}
                )
                if bool(reference_feature_controls_cfg.get("enabled", False)):
                    decision_cmd.extend(
                        [
                            "--reference-feature-mode",
                            str(reference_feature_controls_cfg.get("mode", "allow")),
                        ]
                    )
                    expected_reference_source = reference_feature_controls_cfg.get("expected_source_path")
                    if expected_reference_source is not None:
                        decision_cmd.extend(
                            ["--reference-feature-expected-source", str(expected_reference_source)]
                        )
                    if reference_feature_controls_cfg.get("max_abs_value") is not None:
                        decision_cmd.extend(
                            [
                                "--reference-feature-max-abs-value",
                                str(float(reference_feature_controls_cfg.get("max_abs_value"))),
                            ]
                        )
                midband_focus_cfg_obj = trade_decision_cfg.get("midband_focus", {})
                midband_focus_cfg = midband_focus_cfg_obj if isinstance(midband_focus_cfg_obj, dict) else {}
                if bool(midband_focus_cfg.get("enabled", False)):
                    decision_cmd.extend(
                        [
                            "--midband-focus-enabled",
                            "--midband-focus-pup-low",
                            str(float(midband_focus_cfg.get("p_up_low", 0.55))),
                            "--midband-focus-pup-high",
                            str(float(midband_focus_cfg.get("p_up_high", 0.60))),
                            "--midband-focus-min-abs-ret-pred",
                            str(float(midband_focus_cfg.get("min_abs_ret_pred", 0.0005))),
                            "--midband-focus-negative-weight",
                            str(float(midband_focus_cfg.get("negative_weight", 1.0))),
                            "--midband-focus-positive-weight",
                            str(float(midband_focus_cfg.get("positive_weight", 1.0))),
                        ]
                    )
                    if bool(midband_focus_cfg.get("high_inclusive", False)):
                        decision_cmd.append("--midband-focus-high-inclusive")
                    max_abs_ret_pred = midband_focus_cfg.get("max_abs_ret_pred")
                    if max_abs_ret_pred is not None:
                        decision_cmd.extend(["--midband-focus-max-abs-ret-pred", str(float(max_abs_ret_pred))])
                if bool(trade_decision_cfg.get("candidate_only", False)):
                    decision_cmd.append("--candidate-only")
                results.append(
                    _run_step(
                        "trade_decision_model",
                        decision_cmd,
                        logs_dir / "trade_decision_model.log",
                        args.dry_run,
                    )
                )
                reference_feature_ablation_cfg = (
                    trade_decision_cfg.get("reference_feature_ablation")
                    if isinstance(trade_decision_cfg.get("reference_feature_ablation"), dict)
                    else {}
                )
                if bool(reference_feature_ablation_cfg.get("enabled", False)):
                    ablation_cmd = list(decision_cmd)
                    if "--output" in ablation_cmd:
                        output_index = ablation_cmd.index("--output") + 1
                        if output_index < len(ablation_cmd):
                            ablation_cmd[output_index] = str(trade_decision_ablation_model_path)
                    ablation_cmd.extend(
                        [
                            "--reference-feature-mode",
                            str(reference_feature_ablation_cfg.get("mode", "disable")),
                        ]
                    )
                    expected_reference_source = reference_feature_ablation_cfg.get("expected_source_path")
                    if expected_reference_source is not None:
                        ablation_cmd.extend(
                            ["--reference-feature-expected-source", str(expected_reference_source)]
                        )
                    if reference_feature_ablation_cfg.get("max_abs_value") is not None:
                        ablation_cmd.extend(
                            [
                                "--reference-feature-max-abs-value",
                                str(float(reference_feature_ablation_cfg.get("max_abs_value"))),
                            ]
                        )
                    results.append(
                        _run_step(
                            "trade_decision_model_reference_feature_ablation",
                            ablation_cmd,
                            logs_dir / "trade_decision_model_reference_feature_ablation.log",
                            args.dry_run,
                        )
                    )
                if trade_decision_model_path.exists() and not args.dry_run:
                    decision_payload = _load_json(trade_decision_model_path)
                    trade_decision_deploy_ready = bool(decision_payload.get("deploy_ready", False))
                elif args.dry_run:
                    trade_decision_deploy_ready = bool(trade_decision_cfg.get("enabled", True))
                trade_decision_ablation_deploy_ready = False
                if trade_decision_ablation_model_path.exists() and not args.dry_run:
                    trade_decision_ablation_payload_raw = _load_json(trade_decision_ablation_model_path)
                    trade_decision_ablation_deploy_ready = bool(
                        trade_decision_ablation_payload_raw.get("deploy_ready", False)
                    )
                elif args.dry_run and bool(reference_feature_ablation_cfg.get("enabled", False)):
                    trade_decision_ablation_deploy_ready = bool(trade_decision_cfg.get("enabled", True))

                if trade_decision_deploy_ready and (candidate_quality_input.exists() or args.dry_run):
                    paper_live_cfg_path = Path(str(search_cfg.get("paper_live_config", "")))
                    trade_policy_cfg: Dict[str, Any] = {}
                    if paper_live_cfg_path.exists():
                        try:
                            paper_live_cfg = _load_yaml(paper_live_cfg_path)
                            trade_policy_obj = paper_live_cfg.get("trade_decision_policy", {})
                            if isinstance(trade_policy_obj, dict):
                                trade_policy_cfg = trade_policy_obj
                        except Exception as exc:
                            print(
                                f"Warning: failed to load trade_decision_policy from {paper_live_cfg_path}: {exc}",
                                file=sys.stderr,
                            )

                    weak_veto_cfg_obj = trade_decision_cfg.get("weak_band_candidate_only_veto", {})
                    weak_veto_cfg = weak_veto_cfg_obj if isinstance(weak_veto_cfg_obj, dict) else {}
                    weak_veto_enabled_official = bool(weak_veto_cfg.get("enabled", False))
                    weak_veto_official_mode = str(weak_veto_cfg.get("official_mode", "weak_band")).strip().lower()
                    if weak_veto_official_mode not in {"weak_band", "midband"}:
                        weak_veto_official_mode = "weak_band"
                    official_shadow_variant = str(weak_veto_cfg.get("official_shadow_variant", "none")).strip().lower()
                    official_shadow_selection_cfg_obj = weak_veto_cfg.get("official_shadow_selection", {})
                    official_shadow_selection_cfg = (
                        official_shadow_selection_cfg_obj
                        if isinstance(official_shadow_selection_cfg_obj, dict)
                        else {}
                    )
                    if not _is_supported_official_shadow_variant(official_shadow_variant):
                        official_shadow_variant = "none"
                    weak_veto_low = float(weak_veto_cfg.get("p_up_low", 0.55))
                    weak_veto_high = float(weak_veto_cfg.get("p_up_high", 0.60))
                    weak_veto_high_inclusive = bool(weak_veto_cfg.get("high_inclusive", False))
                    weak_veto_reference_path_cfg = weak_veto_cfg.get("incumbent_reference_path")
                    refined_veto_cfg_obj = weak_veto_cfg.get("refined", {})
                    refined_veto_cfg = refined_veto_cfg_obj if isinstance(refined_veto_cfg_obj, dict) else {}
                    refined_veto_low = float(refined_veto_cfg.get("p_up_low", weak_veto_low))
                    refined_veto_high = float(refined_veto_cfg.get("p_up_high", weak_veto_high))
                    refined_veto_high_inclusive = bool(refined_veto_cfg.get("high_inclusive", weak_veto_high_inclusive))
                    refined_veto_min_abs_ret_pred = float(refined_veto_cfg.get("min_abs_ret_pred", 0.001))
                    refined_veto_reference_path_cfg = refined_veto_cfg.get("incumbent_reference_path")
                    midband_veto_cfg_obj = weak_veto_cfg.get("midband", {})
                    midband_veto_cfg = midband_veto_cfg_obj if isinstance(midband_veto_cfg_obj, dict) else {}
                    midband_veto_low = float(midband_veto_cfg.get("p_up_low", weak_veto_low))
                    midband_veto_high = float(midband_veto_cfg.get("p_up_high", weak_veto_high))
                    midband_veto_high_inclusive = bool(midband_veto_cfg.get("high_inclusive", weak_veto_high_inclusive))
                    midband_veto_min_abs_ret_pred = float(midband_veto_cfg.get("min_abs_ret_pred", 0.0005))
                    midband_veto_max_abs_ret_pred = float(midband_veto_cfg.get("max_abs_ret_pred", 0.001))
                    midband_veto_reference_path_cfg = midband_veto_cfg.get("incumbent_reference_path")
                    direct_midband_policy_cfg_obj = trade_policy_cfg.get("midband_veto", {})
                    direct_midband_policy_cfg = (
                        direct_midband_policy_cfg_obj if isinstance(direct_midband_policy_cfg_obj, dict) else {}
                    )
                    direct_midband_policy_enabled = bool(direct_midband_policy_cfg.get("enabled", False))
                    direct_midband_policy_low = float(direct_midband_policy_cfg.get("p_up_low", 0.55))
                    direct_midband_policy_high = float(direct_midband_policy_cfg.get("p_up_high", 0.60))
                    direct_midband_policy_high_inclusive = bool(direct_midband_policy_cfg.get("high_inclusive", False))
                    direct_midband_policy_min_abs_ret_pred = direct_midband_policy_cfg.get("min_abs_ret_pred")
                    direct_midband_policy_max_abs_ret_pred = direct_midband_policy_cfg.get("max_abs_ret_pred")
                    direct_midband_policy_regime_states = [
                        str(value).strip().lower()
                        for value in (
                            direct_midband_policy_cfg.get("regime_states", [])
                            if isinstance(direct_midband_policy_cfg.get("regime_states", []), list)
                            else []
                        )
                        if str(value).strip()
                    ]
                    raw_ev_veto_cfg_obj = weak_veto_cfg.get("raw_ev_sign", {})
                    raw_ev_veto_cfg = raw_ev_veto_cfg_obj if isinstance(raw_ev_veto_cfg_obj, dict) else {}
                    raw_ev_veto_low = float(raw_ev_veto_cfg.get("p_up_low", weak_veto_low))
                    raw_ev_veto_high = float(raw_ev_veto_cfg.get("p_up_high", weak_veto_high))
                    raw_ev_veto_high_inclusive = bool(raw_ev_veto_cfg.get("high_inclusive", weak_veto_high_inclusive))
                    raw_ev_veto_max = float(raw_ev_veto_cfg.get("raw_ev_sign_max", 0.0))
                    raw_ev_veto_reference_path_cfg = raw_ev_veto_cfg.get("incumbent_reference_path")
                    direction_align_veto_cfg_obj = weak_veto_cfg.get("direction_alignment", {})
                    direction_align_veto_cfg = (
                        direction_align_veto_cfg_obj if isinstance(direction_align_veto_cfg_obj, dict) else {}
                    )
                    direction_align_veto_low = float(direction_align_veto_cfg.get("p_up_low", weak_veto_low))
                    direction_align_veto_high = float(direction_align_veto_cfg.get("p_up_high", weak_veto_high))
                    direction_align_veto_high_inclusive = bool(
                        direction_align_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    direction_align_veto_require_aligned = bool(
                        direction_align_veto_cfg.get("require_aligned", False)
                    )
                    direction_align_veto_use_midband_slice = bool(
                        direction_align_veto_cfg.get("use_midband_slice", True)
                    )
                    direction_align_veto_min_abs_ret_pred = float(
                        direction_align_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    direction_align_veto_max_abs_ret_pred = float(
                        direction_align_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    direction_align_veto_reference_path_cfg = direction_align_veto_cfg.get("incumbent_reference_path")
                    joint_direction_midband_veto_cfg_obj = weak_veto_cfg.get("joint_direction_midband", {})
                    joint_direction_midband_veto_cfg = (
                        joint_direction_midband_veto_cfg_obj if isinstance(joint_direction_midband_veto_cfg_obj, dict) else {}
                    )
                    joint_direction_midband_veto_low = float(joint_direction_midband_veto_cfg.get("p_up_low", weak_veto_low))
                    joint_direction_midband_veto_high = float(joint_direction_midband_veto_cfg.get("p_up_high", weak_veto_high))
                    joint_direction_midband_veto_high_inclusive = bool(
                        joint_direction_midband_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    joint_direction_midband_veto_require_aligned = bool(
                        joint_direction_midband_veto_cfg.get("require_aligned", False)
                    )
                    joint_direction_midband_veto_min_abs_ret_pred = float(
                        joint_direction_midband_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    joint_direction_midband_veto_max_abs_ret_pred = float(
                        joint_direction_midband_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    joint_direction_midband_veto_reference_path_cfg = joint_direction_midband_veto_cfg.get("incumbent_reference_path")
                    regime_state_veto_cfg_obj = weak_veto_cfg.get("regime_state", {})
                    regime_state_veto_cfg = (
                        regime_state_veto_cfg_obj if isinstance(regime_state_veto_cfg_obj, dict) else {}
                    )
                    regime_state_veto_low = float(regime_state_veto_cfg.get("p_up_low", weak_veto_low))
                    regime_state_veto_high = float(regime_state_veto_cfg.get("p_up_high", weak_veto_high))
                    regime_state_veto_high_inclusive = bool(
                        regime_state_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    regime_state_veto_use_midband_slice = bool(
                        regime_state_veto_cfg.get("use_midband_slice", False)
                    )
                    regime_state_veto_min_abs_ret_pred = float(
                        regime_state_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    regime_state_veto_max_abs_ret_pred = float(
                        regime_state_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    regime_state_veto_min_regime_rows = int(regime_state_veto_cfg.get("min_regime_rows", 1))
                    regime_state_veto_reference_path_cfg = regime_state_veto_cfg.get("incumbent_reference_path")
                    regime_state_veto_override_obj = regime_state_veto_cfg.get("regime_states", [])
                    regime_state_veto_override = [
                        str(value).strip().lower()
                        for value in (regime_state_veto_override_obj if isinstance(regime_state_veto_override_obj, list) else [])
                        if str(value).strip()
                    ]
                    chop_high_vol_veto_cfg_obj = weak_veto_cfg.get("chop_high_volatility", {})
                    chop_high_vol_veto_cfg = (
                        chop_high_vol_veto_cfg_obj if isinstance(chop_high_vol_veto_cfg_obj, dict) else {}
                    )
                    chop_high_vol_veto_low = float(chop_high_vol_veto_cfg.get("p_up_low", weak_veto_low))
                    chop_high_vol_veto_high = float(chop_high_vol_veto_cfg.get("p_up_high", weak_veto_high))
                    chop_high_vol_veto_high_inclusive = bool(
                        chop_high_vol_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    chop_high_vol_veto_regime_state = str(
                        chop_high_vol_veto_cfg.get("regime_state", "chop")
                    ).strip().lower()
                    chop_high_vol_veto_volatility_col = str(
                        chop_high_vol_veto_cfg.get("volatility_col", "volatility_realized_24h")
                    )
                    chop_high_vol_veto_use_midband_slice = bool(
                        chop_high_vol_veto_cfg.get("use_midband_slice", False)
                    )
                    chop_high_vol_veto_min_abs_ret_pred = float(
                        chop_high_vol_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    chop_high_vol_veto_max_abs_ret_pred = float(
                        chop_high_vol_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    chop_high_vol_veto_reference_path_cfg = chop_high_vol_veto_cfg.get("incumbent_reference_path")
                    chop_high_vol_veto_min_volatility_cfg = chop_high_vol_veto_cfg.get("min_volatility")
                    volatility_only_veto_cfg_obj = weak_veto_cfg.get("volatility_only", {})
                    volatility_only_veto_cfg = (
                        volatility_only_veto_cfg_obj if isinstance(volatility_only_veto_cfg_obj, dict) else {}
                    )
                    volatility_only_veto_low = float(volatility_only_veto_cfg.get("p_up_low", weak_veto_low))
                    volatility_only_veto_high = float(volatility_only_veto_cfg.get("p_up_high", weak_veto_high))
                    volatility_only_veto_high_inclusive = bool(
                        volatility_only_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    volatility_only_veto_volatility_col = str(
                        volatility_only_veto_cfg.get("volatility_col", chop_high_vol_veto_volatility_col)
                    )
                    volatility_only_veto_use_midband_slice = bool(
                        volatility_only_veto_cfg.get("use_midband_slice", False)
                    )
                    volatility_only_veto_min_abs_ret_pred = float(
                        volatility_only_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    volatility_only_veto_max_abs_ret_pred = float(
                        volatility_only_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    volatility_only_veto_reference_path_cfg = volatility_only_veto_cfg.get("incumbent_reference_path")
                    volatility_only_veto_min_volatility_cfg = volatility_only_veto_cfg.get("min_volatility")

                    resolved_trade_decision_threshold = float(
                        trade_decision_cfg.get("threshold", trade_policy_cfg.get("threshold", 0.55))
                    )
                    aligned_candidate_input = summary_dir / "backtest_signals_meta_ensemble_decision_aligned.csv"
                    common_align_args = [
                        "--threshold",
                        str(resolved_trade_decision_threshold),
                        "--fee-bps",
                        str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                        "--slippage-bps",
                        str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                        "--replace-threshold-rule",
                        "1" if bool(trade_policy_cfg.get("replace_threshold_rule", True)) else "0",
                        "--require-direction-ret-alignment",
                        "1" if bool(trade_policy_cfg.get("require_direction_ret_alignment", True)) else "0",
                        "--use-oof-expected-value",
                        "1" if bool(trade_policy_cfg.get("use_oof_expected_value", True)) else "0",
                        "--oof-expected-value-mode",
                        str(trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")),
                        "--enforce-positive-oof-envelope",
                        "1" if bool(trade_policy_cfg.get("enforce_positive_oof_envelope", False)) else "0",
                        "--positive-oof-envelope-mode",
                        str(trade_policy_cfg.get("positive_oof_envelope_mode", "strict_positive_bin")),
                        "--block-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("block_when_no_positive_oof_bin", True)) else "0",
                        "--positive-oof-min-samples",
                        str(int(trade_policy_cfg.get("positive_oof_min_samples", 4))),
                        "--allow-raw-ev-fallback-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)) else "0",
                        "--raw-ev-fallback-quantile",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9))),
                        "--raw-ev-fallback-min-edge-over-fee",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0))),
                        "--min-expected-net",
                        str(float(trade_policy_cfg.get("min_expected_net", 0.0))),
                        "--min-edge-over-fee",
                        str(float(trade_policy_cfg.get("min_edge_over_fee", 0.0))),
                        "--policy-midband-veto",
                        "1" if direct_midband_policy_enabled else "0",
                        "--policy-midband-pup-low",
                        str(direct_midband_policy_low),
                        "--policy-midband-pup-high",
                        str(direct_midband_policy_high),
                        "--policy-midband-high-inclusive",
                        "1" if direct_midband_policy_high_inclusive else "0",
                    ]
                    if direct_midband_policy_min_abs_ret_pred is not None:
                        common_align_args.extend(
                            ["--policy-midband-min-abs-ret-pred", str(float(direct_midband_policy_min_abs_ret_pred))]
                        )
                    if direct_midband_policy_max_abs_ret_pred is not None:
                        common_align_args.extend(
                            ["--policy-midband-max-abs-ret-pred", str(float(direct_midband_policy_max_abs_ret_pred))]
                        )
                    if direct_midband_policy_regime_states:
                        common_align_args.extend(
                            ["--policy-midband-regime-states", ",".join(direct_midband_policy_regime_states)]
                        )

                    align_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(decision_model_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(aligned_candidate_input),
                        "--meta-output",
                        str(summary_dir / "backtest_signals_meta_ensemble_decision_aligned_meta.json"),
                    ]
                    align_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        align_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        align_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    if weak_veto_official_mode == "midband":
                        align_cmd.extend(
                            [
                                "--midband-candidate-only-veto",
                                "1" if weak_veto_enabled_official else "0",
                                "--midband-pup-low",
                                str(midband_veto_low),
                                "--midband-pup-high",
                                str(midband_veto_high),
                                "--midband-high-inclusive",
                                "1" if midband_veto_high_inclusive else "0",
                                "--midband-min-abs-ret-pred",
                                str(midband_veto_min_abs_ret_pred),
                                "--midband-max-abs-ret-pred",
                                str(midband_veto_max_abs_ret_pred),
                            ]
                        )
                        if midband_veto_reference_path_cfg:
                            align_cmd.extend(["--midband-incumbent-reference", str(midband_veto_reference_path_cfg)])
                    else:
                        align_cmd.extend(
                            [
                                "--weak-band-candidate-only-veto",
                                "1" if weak_veto_enabled_official else "0",
                                "--weak-band-pup-low",
                                str(weak_veto_low),
                                "--weak-band-pup-high",
                                str(weak_veto_high),
                                "--weak-band-high-inclusive",
                                "1" if weak_veto_high_inclusive else "0",
                            ]
                        )
                        if weak_veto_reference_path_cfg:
                            align_cmd.extend(["--weak-band-incumbent-reference", str(weak_veto_reference_path_cfg)])
                    results.append(
                        _run_step(
                            "trade_decision_align_candidate",
                            align_cmd,
                            logs_dir / "trade_decision_align_candidate.log",
                            args.dry_run,
                        )
                    )

                    raw_candidate_diag_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(decision_model_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(summary_dir / "backtest_signals_meta_ensemble_policy_diagnostics_candidate_raw.csv"),
                        "--diagnostics-output",
                        str(summary_dir / "trade_decision_diagnostics_candidate_raw.json"),
                        "--meta-output",
                        str(summary_dir / "trade_decision_diagnostics_candidate_raw_meta.json"),
                        "--diagnostics-only",
                    ]
                    raw_candidate_diag_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        raw_candidate_diag_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        raw_candidate_diag_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    results.append(
                        _run_step(
                            "trade_decision_diagnostics_candidate_raw",
                            raw_candidate_diag_cmd,
                            logs_dir / "trade_decision_diagnostics_candidate_raw.log",
                            args.dry_run,
                        )
                    )

                    if aligned_candidate_input.exists() or args.dry_run:
                        candidate_gate_input = aligned_candidate_input
                        aligned_quality_cmd = [
                            python,
                            "-m",
                            "src.scripts.evaluate_model_quality",
                            "--input",
                            str(candidate_gate_input),
                            "--output",
                            str(summary_dir / "model_quality_candidate.json"),
                        ]
                        results.append(
                            _run_step(
                                "model_quality_eval_decision_aligned",
                                aligned_quality_cmd,
                                logs_dir / "model_quality_eval_decision_aligned.log",
                                args.dry_run,
                            )
                        )

                        overlap_stability_output = summary_dir / "overlap_trust_stability.json"
                        if (
                            quality_input.exists()
                            and labeled_overlap_dataset is not None
                            and labeled_overlap_dataset.exists()
                            and full_selected_path is not None
                            and overlap_selected_path is not None
                            and full_selected_path.exists()
                            and overlap_selected_path.exists()
                        ):
                            overlap_feature_sources = [aligned_candidate_input, quality_input]
                            overlap_diag_cmd = [
                                python,
                                "-m",
                                "src.scripts.analyze_overlap_trust_stability",
                                "--full-walkforward",
                                str(full_selected_path),
                                "--overlap-walkforward",
                                str(overlap_selected_path),
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--labeled-csv",
                                str(quality_input),
                                "--feature-source",
                                str(overlap_feature_sources[0]),
                                "--feature-source",
                                str(overlap_feature_sources[1]),
                                "--ts-col",
                                str(reconcile_cfg.get("ts_col", "ts")),
                                "--return-col",
                                str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                "--signal-col",
                                str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                "--output",
                                str(overlap_stability_output),
                            ]
                            results.append(
                                _run_step(
                                    "overlap_trust_stability_decision_aligned_refresh",
                                    overlap_diag_cmd,
                                    logs_dir / "overlap_trust_stability_decision_aligned_refresh.log",
                                    args.dry_run,
                                )
                            )

            hygiene_cmd = [
                python,
                "-m",
                "src.scripts.analyze_ensemble_hygiene",
                "--input",
                str(quality_input),
                "--output",
                str(summary_dir / "ensemble_hygiene.json"),
            ]
            results.append(
                _run_step(
                    "ensemble_hygiene",
                    hygiene_cmd,
                    logs_dir / "ensemble_hygiene.log",
                    args.dry_run,
                    allowed_returncodes=[0, 2],
                )
            )

            if bool(tuning_cfg.get("enabled", True)):
                optimize_on_overlap = bool(tuning_cfg.get("optimize_on_labeled_overlap", False)) and bool(overlap_as_primary)
                overlap_pruning_cfg_obj = overlap_pre_tuning_cfg.get("model_pruning", {}) if isinstance(overlap_pre_tuning_cfg, dict) else {}
                overlap_pruning_cfg = overlap_pruning_cfg_obj if isinstance(overlap_pruning_cfg_obj, dict) else {}
                fallback_tuning_cfg_obj = overlap_pruning_cfg.get("fallback_tuning", {}) if isinstance(overlap_pruning_cfg, dict) else {}
                fallback_tuning_cfg = fallback_tuning_cfg_obj if isinstance(fallback_tuning_cfg_obj, dict) else {}
                fallback_tuning_enabled = bool(fallback_tuning_cfg.get("enabled", False))

                def _build_tuning_cmd(
                    *,
                    p_up_grid: str,
                    ret_min_grid: str,
                    direction_threshold_grid: str,
                    min_trades: int,
                    max_drawdown: float,
                    min_cum_ret: float,
                    selection_metric: str,
                    economics_turnover_penalty: float,
                    economics_downside_penalty: float,
                    stability_gap_penalty: float,
                    max_stability_gap: float,
                    min_overlap_rows: int,
                    strict_accept: bool,
                ) -> List[str]:
                    tuning_input = tuning_quality_input if tuning_quality_input.exists() else quality_input
                    cmd = [
                        python,
                        "-m",
                        "src.scripts.tune_joint_signal_thresholds",
                        "--input",
                        str(tuning_input),
                        f"--p-up-grid={p_up_grid}",
                        f"--ret-min-grid={ret_min_grid}",
                        f"--direction-threshold-grid={direction_threshold_grid}",
                        "--min-trades",
                        str(int(min_trades)),
                        "--max-dd",
                        str(float(max_drawdown)),
                        "--min-cum-ret",
                        str(float(min_cum_ret)),
                        "--selection-metric",
                        str(selection_metric),
                        "--economics-turnover-penalty",
                        str(float(economics_turnover_penalty)),
                        "--economics-downside-penalty",
                        str(float(economics_downside_penalty)),
                        "--stability-gap-penalty",
                        str(float(stability_gap_penalty)),
                        "--max-stability-gap",
                        str(float(max_stability_gap)),
                        "--min-overlap-rows",
                        str(int(min_overlap_rows)),
                        "--output",
                        str(joint_tuning_output_path),
                    ]
                    if optimize_on_overlap and labeled_overlap_dataset is not None:
                        cmd.extend(
                            [
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--ts-col",
                                str(reconcile_cfg.get("ts_col", "ts")),
                            ]
                        )
                    if bool(strict_accept):
                        cmd.append("--strict-accept")
                    return cmd

                run_tuning = True
                used_fallback_tuning = False
                if optimize_on_overlap and not overlap_pruning_allows_tuning:
                    if fallback_tuning_enabled:
                        used_fallback_tuning = True
                        print(
                            "Overlap model pruning found no viable candidates; running constrained fallback joint-threshold tuning.",
                            file=sys.stderr,
                        )
                        tuning_cmd = _build_tuning_cmd(
                            p_up_grid=str(fallback_tuning_cfg.get("p_up_grid", "0.58,0.60,0.62")),
                            ret_min_grid=str(fallback_tuning_cfg.get("ret_min_grid", "0.0,0.0001,0.0002")),
                            direction_threshold_grid=str(fallback_tuning_cfg.get("direction_threshold_grid", "0.54,0.56,0.58")),
                            min_trades=int(fallback_tuning_cfg.get("min_trades", tuning_cfg.get("min_trades", 10))),
                            max_drawdown=float(fallback_tuning_cfg.get("max_drawdown", tuning_cfg.get("max_drawdown", -0.12))),
                            min_cum_ret=float(fallback_tuning_cfg.get("min_cum_ret", tuning_cfg.get("min_cum_ret", 0.0))),
                            selection_metric=str(fallback_tuning_cfg.get("selection_metric", tuning_cfg.get("selection_metric", "economics_score"))),
                            economics_turnover_penalty=float(
                                fallback_tuning_cfg.get(
                                    "economics_turnover_penalty",
                                    tuning_cfg.get("economics_turnover_penalty", 0.002),
                                )
                            ),
                            economics_downside_penalty=float(
                                fallback_tuning_cfg.get(
                                    "economics_downside_penalty",
                                    tuning_cfg.get("economics_downside_penalty", 2.0),
                                )
                            ),
                            stability_gap_penalty=float(
                                fallback_tuning_cfg.get(
                                    "stability_gap_penalty",
                                    tuning_cfg.get("stability_gap_penalty", 0.0),
                                )
                            ),
                            max_stability_gap=float(
                                fallback_tuning_cfg.get(
                                    "max_stability_gap",
                                    tuning_cfg.get("max_stability_gap", 1e9),
                                )
                            ),
                            min_overlap_rows=int(
                                fallback_tuning_cfg.get(
                                    "min_overlap_rows",
                                    tuning_cfg.get("min_overlap_rows", 0),
                                )
                            ),
                            strict_accept=bool(
                                fallback_tuning_cfg.get(
                                    "enforce_deployable",
                                    tuning_cfg.get("enforce_deployable", True),
                                )
                            ),
                        )
                    else:
                        run_tuning = False
                        joint_tuning_accepted = False
                        blocked_payload = {
                            "rows": 0,
                            "constraints": {
                                "selection_metric": str(tuning_cfg.get("selection_metric", "cum_ret")),
                                "optimize_on_labeled_overlap": True,
                            },
                            "accepted": False,
                            "reason": "blocked_by_overlap_model_pruning",
                            "overlap_model_pruning_path": str(overlap_model_pruning_path),
                            "fallback_tuning_enabled": False,
                        }
                        joint_tuning_output_path.write_text(json.dumps(blocked_payload, indent=2), encoding="utf-8")
                        print(
                            "Skipping joint_threshold_tuning: overlap model pruning found no viable model candidates.",
                            file=sys.stderr,
                        )
                else:
                    tuning_cmd = _build_tuning_cmd(
                        p_up_grid=str(tuning_cfg.get("p_up_grid", "0.50,0.55,0.60,0.65")),
                        ret_min_grid=str(tuning_cfg.get("ret_min_grid", "-0.0002,0.0,0.0002,0.0005")),
                        direction_threshold_grid=str(tuning_cfg.get("direction_threshold_grid", "0.50,0.55,0.60")),
                        min_trades=int(tuning_cfg.get("min_trades", 10)),
                        max_drawdown=float(tuning_cfg.get("max_drawdown", -0.12)),
                        min_cum_ret=float(tuning_cfg.get("min_cum_ret", 0.0)),
                        selection_metric=str(tuning_cfg.get("selection_metric", "cum_ret")),
                        economics_turnover_penalty=float(tuning_cfg.get("economics_turnover_penalty", 0.002)),
                        economics_downside_penalty=float(tuning_cfg.get("economics_downside_penalty", 2.0)),
                        stability_gap_penalty=float(tuning_cfg.get("stability_gap_penalty", 0.0)),
                        max_stability_gap=float(tuning_cfg.get("max_stability_gap", 1e9)),
                        min_overlap_rows=int(tuning_cfg.get("min_overlap_rows", 0)),
                        strict_accept=bool(tuning_cfg.get("enforce_deployable", True)),
                    )

                if run_tuning:
                    results.append(
                        _run_step(
                            "joint_threshold_tuning_fallback_overlap" if used_fallback_tuning else "joint_threshold_tuning",
                            tuning_cmd,
                            logs_dir / ("joint_threshold_tuning_fallback_overlap.log" if used_fallback_tuning else "joint_threshold_tuning.log"),
                            args.dry_run,
                            allowed_returncodes=[0, 2],
                        )
                    )

                    if joint_tuning_output_path.exists() and not args.dry_run:
                        joint_tuning_payload = _load_json(joint_tuning_output_path)
                        if used_fallback_tuning:
                            joint_tuning_payload["fallback_tuning_used"] = True
                            joint_tuning_payload["fallback_tuning_source"] = "overlap_model_pruning"
                            joint_tuning_output_path.write_text(json.dumps(joint_tuning_payload, indent=2), encoding="utf-8")
                        joint_tuning_accepted = bool(joint_tuning_payload.get("accepted", False))

                        sweep_cfg_obj = tuning_cfg.get("overlap_threshold_sweep_fallback", {}) if isinstance(tuning_cfg, dict) else {}
                        sweep_cfg = sweep_cfg_obj if isinstance(sweep_cfg_obj, dict) else {}
                        if (
                            (not joint_tuning_accepted)
                            and optimize_on_overlap
                            and bool(sweep_cfg.get("enabled", True))
                            and labeled_overlap_dataset is not None
                            and labeled_overlap_dataset.exists()
                        ):
                            overlap_compare_path = summary_dir / "walkforward_model_compare_labeled_overlap.json"
                            overlap_selected_model = "xgb"
                            resolved_walkforward: Dict[str, Any] = {}
                            if overlap_compare_path.exists():
                                overlap_compare_payload = _load_json(overlap_compare_path)
                                overlap_selected_model = str(
                                    overlap_compare_payload.get("selected_model_kind", overlap_selected_model)
                                )
                                resolved_obj = overlap_compare_payload.get("resolved_walkforward", {})
                                if isinstance(resolved_obj, dict):
                                    resolved_walkforward = resolved_obj

                            threshold_grid_raw = str(sweep_cfg.get("threshold_grid", "0.48,0.50,0.52,0.55,0.58,0.60"))
                            threshold_grid = [
                                float(v.strip())
                                for v in threshold_grid_raw.split(",")
                                if v is not None and v.strip() != ""
                            ]
                            min_trades_sweep = int(sweep_cfg.get("min_trades", tuning_cfg.get("min_trades", 10)))
                            min_cum_ret_sweep = float(sweep_cfg.get("min_cum_ret", tuning_cfg.get("min_cum_ret", 0.0)))
                            full_min_trades_sweep = int(sweep_cfg.get("full_min_trades", min_trades_sweep))
                            full_min_cum_ret_sweep = float(sweep_cfg.get("full_min_cum_ret", min_cum_ret_sweep))
                            folds_sweep = int(resolved_walkforward.get("folds", reconcile_cfg.get("overlap_compare", {}).get("folds", cv_folds)))
                            train_sweep = int(resolved_walkforward.get("train_size", reconcile_cfg.get("overlap_compare", {}).get("train_size", cv_train_size)))
                            val_sweep = int(resolved_walkforward.get("val_size", reconcile_cfg.get("overlap_compare", {}).get("val_size", cv_val_size)))
                            test_sweep = int(resolved_walkforward.get("test_size", reconcile_cfg.get("overlap_compare", {}).get("test_size", cv_test_size)))
                            gap_sweep = int(resolved_walkforward.get("gap", reconcile_cfg.get("overlap_compare", {}).get("gap", cv_gap)))
                            purge_sweep = int(resolved_walkforward.get("purge_size", reconcile_cfg.get("overlap_compare", {}).get("purge_size", cv_purge_size)))
                            embargo_sweep = int(resolved_walkforward.get("embargo_size", reconcile_cfg.get("overlap_compare", {}).get("embargo_size", cv_embargo_size)))
                            mode_sweep = str(reconcile_cfg.get("overlap_compare", {}).get("mode", cv_mode))

                            sweep_rows: List[Dict[str, Any]] = []
                            for threshold in threshold_grid:
                                sweep_name = str(threshold).replace(".", "p")
                                sweep_output = summary_dir / f"overlap_threshold_sweep_{sweep_name}.json"
                                full_sweep_output = summary_dir / f"full_threshold_sweep_{sweep_name}.json"
                                sweep_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.run_walkforward_validation",
                                    "--dataset-path",
                                    str(labeled_overlap_dataset),
                                    "--y-key",
                                    str(walkforward_target),
                                    "--folds",
                                    str(int(folds_sweep)),
                                    "--train-size",
                                    str(int(train_sweep)),
                                    "--val-size",
                                    str(int(val_sweep)),
                                    "--test-size",
                                    str(int(test_sweep)),
                                    "--gap",
                                    str(int(gap_sweep)),
                                    "--purge-size",
                                    str(int(purge_sweep)),
                                    "--embargo-size",
                                    str(int(embargo_sweep)),
                                    "--mode",
                                    str(mode_sweep),
                                    "--model-kind",
                                    str(overlap_selected_model),
                                    "--signal-threshold",
                                    str(float(threshold)),
                                    "--fee-bps",
                                    str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                                    "--slippage-bps",
                                    str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                                    "--output",
                                    str(sweep_output),
                                ]
                                full_sweep_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.run_walkforward_validation",
                                    "--dataset-path",
                                    str(walkforward_dataset),
                                    "--y-key",
                                    str(walkforward_target),
                                    "--folds",
                                    str(int(compare_cfg.get("folds", cv_folds))),
                                    "--train-size",
                                    str(int(compare_cfg.get("train_size", cv_train_size))),
                                    "--val-size",
                                    str(int(compare_cfg.get("val_size", cv_val_size))),
                                    "--test-size",
                                    str(int(compare_cfg.get("test_size", cv_test_size))),
                                    "--gap",
                                    str(int(compare_cfg.get("gap", cv_gap))),
                                    "--purge-size",
                                    str(int(compare_cfg.get("purge_size", cv_purge_size))),
                                    "--embargo-size",
                                    str(int(compare_cfg.get("embargo_size", cv_embargo_size))),
                                    "--mode",
                                    str(compare_cfg.get("mode", cv_mode)),
                                    "--model-kind",
                                    str(overlap_selected_model),
                                    "--signal-threshold",
                                    str(float(threshold)),
                                    "--fee-bps",
                                    str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                                    "--slippage-bps",
                                    str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                                    "--output",
                                    str(full_sweep_output),
                                ]
                                results.append(
                                    _run_step(
                                        f"overlap_threshold_sweep_{sweep_name}",
                                        sweep_cmd,
                                        logs_dir / f"overlap_threshold_sweep_{sweep_name}.log",
                                        args.dry_run,
                                    )
                                )
                                results.append(
                                    _run_step(
                                        f"full_threshold_sweep_{sweep_name}",
                                        full_sweep_cmd,
                                        logs_dir / f"full_threshold_sweep_{sweep_name}.log",
                                        args.dry_run,
                                    )
                                )
                                if sweep_output.exists():
                                    sweep_payload = _load_json(sweep_output)
                                    full_sweep_payload = _load_json(full_sweep_output) if full_sweep_output.exists() else {}
                                    sweep_rows.append(
                                        {
                                            "signal_threshold": float(threshold),
                                            "cum_ret_net_total": float(sweep_payload.get("cum_ret_net_total", float("nan"))),
                                            "trade_count_total": int(sweep_payload.get("trade_count_total", 0) or 0),
                                            "auc_mean": float(sweep_payload.get("auc_mean", float("nan"))),
                                            "full_cum_ret_net_total": float(full_sweep_payload.get("cum_ret_net_total", float("nan"))),
                                            "full_trade_count_total": int(full_sweep_payload.get("trade_count_total", 0) or 0),
                                            "full_auc_mean": float(full_sweep_payload.get("auc_mean", float("nan"))),
                                            "path": str(sweep_output),
                                            "full_path": str(full_sweep_output),
                                        }
                                    )

                            deployable_rows = [
                                row
                                for row in sweep_rows
                                if int(row.get("trade_count_total", 0) or 0) >= int(min_trades_sweep)
                                and float(row.get("cum_ret_net_total", float("-inf"))) >= float(min_cum_ret_sweep)
                                and int(row.get("full_trade_count_total", 0) or 0) >= int(full_min_trades_sweep)
                                and float(row.get("full_cum_ret_net_total", float("-inf"))) >= float(full_min_cum_ret_sweep)
                            ]
                            if deployable_rows:
                                best_sweep = max(
                                    deployable_rows,
                                    key=lambda r: (
                                        float(r.get("full_cum_ret_net_total", float("-inf"))),
                                        float(r.get("cum_ret_net_total", float("-inf"))),
                                        int(r.get("trade_count_total", 0) or 0),
                                    ),
                                )
                                horizon_key = str(int(walkforward_horizon))
                                ret_min_from_thresholds = 0.0
                                if thresholds_path.exists():
                                    current_thresholds_payload = _load_json(thresholds_path)
                                    horizons_payload = current_thresholds_payload.get("horizons", {})
                                    if isinstance(horizons_payload, dict):
                                        horizon_payload = horizons_payload.get(horizon_key, {})
                                        if isinstance(horizon_payload, dict):
                                            ret_min_from_thresholds = float(horizon_payload.get("ret_min", 0.0) or 0.0)

                                joint_tuning_payload = {
                                    "rows": int(len(sweep_rows)),
                                    "constraints": {
                                        "selection_metric": "cum_ret_net_total",
                                        "optimize_on_labeled_overlap": True,
                                        "min_trades": int(min_trades_sweep),
                                        "min_cum_ret": float(min_cum_ret_sweep),
                                        "full_min_trades": int(full_min_trades_sweep),
                                        "full_min_cum_ret": float(full_min_cum_ret_sweep),
                                        "source": "overlap_threshold_sweep_fallback",
                                    },
                                    "accepted": True,
                                    "best": {
                                        "p_up_min": float(best_sweep.get("signal_threshold", 0.5) or 0.5),
                                        "ret_min": float(ret_min_from_thresholds),
                                        "direction_threshold": 0.5,
                                        "n_trades": int(best_sweep.get("trade_count_total", 0) or 0),
                                        "cum_ret": float(best_sweep.get("cum_ret_net_total", 0.0) or 0.0),
                                        "full_cum_ret": float(best_sweep.get("full_cum_ret_net_total", 0.0) or 0.0),
                                        "stability_gap": abs(
                                            float(best_sweep.get("full_cum_ret_net_total", 0.0) or 0.0)
                                            - float(best_sweep.get("cum_ret_net_total", 0.0) or 0.0)
                                        ),
                                        "max_drawdown": float("nan"),
                                        "economics_score": float(best_sweep.get("full_cum_ret_net_total", 0.0) or 0.0),
                                        "selection_value": float(best_sweep.get("full_cum_ret_net_total", 0.0) or 0.0),
                                    },
                                    "n_candidates": int(len(sweep_rows)),
                                    "n_feasible": int(len(deployable_rows)),
                                    "n_deployable": int(len(deployable_rows)),
                                    "fallback_tuning_used": True,
                                    "fallback_tuning_source": "overlap_threshold_sweep_fallback",
                                    "overlap_selected_model": str(overlap_selected_model),
                                    "full_selected_model": str(overlap_selected_model),
                                    "sweep_rows": sweep_rows,
                                }
                                joint_tuning_output_path.write_text(json.dumps(joint_tuning_payload, indent=2), encoding="utf-8")
                                joint_tuning_accepted = True
                                print(
                                    "Joint threshold tuning fallback produced a candidate that remained deployable on both overlap and full walk-forward slices.",
                                    file=sys.stderr,
                                )

                        if joint_tuning_accepted and thresholds_path.exists():
                            applied = _apply_joint_tuning_to_thresholds(
                                thresholds_path,
                                joint_tuning_payload=joint_tuning_payload,
                                horizon=int(walkforward_horizon),
                            )
                            if not applied:
                                print(
                                    "Warning: joint threshold tuning accepted but thresholds artifact could not be patched.",
                                    file=sys.stderr,
                                )
                            last_deployable_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(thresholds_path, last_deployable_thresholds_path)
                        elif (not joint_tuning_accepted) and use_last_deployable_on_joint_reject:
                            if last_deployable_thresholds_path.exists():
                                thresholds_path = last_deployable_thresholds_path
                                used_last_deployable_thresholds = True
                            else:
                                print(
                                    "Joint threshold tuning was non-deployable and no last deployable thresholds artifact was found; "
                                    f"continuing with current run thresholds at {thresholds_path}.",
                                    file=sys.stderr,
                                )
                    elif args.dry_run:
                        joint_tuning_accepted = None

            if bool(feature_rel_cfg.get("enabled", True)):
                feature_rel_input = Path(feature_rel_cfg.get("input") or str(walkforward_dataset))
                feature_rel_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_feature_reliability",
                    "--input",
                    str(feature_rel_input),
                    "--baseline-window",
                    str(int(feature_rel_cfg.get("baseline_window", 240))),
                    "--recent-window",
                    str(int(feature_rel_cfg.get("recent_window", 120))),
                    "--min-score",
                    str(float(feature_rel_cfg.get("min_score", 0.55))),
                    "--max-features",
                    str(int(feature_rel_cfg.get("max_features", 0))),
                    "--output",
                    str(summary_dir / "feature_reliability.json"),
                ]
                results.append(
                    _run_step(
                        "feature_reliability",
                        feature_rel_cmd,
                        logs_dir / "feature_reliability.log",
                        args.dry_run,
                    )
                )

            if bool(calibration_cfg.get("enabled", True)):
                calibration_horizon_key = str(quality_cfg.get("calibration_horizon", "1h"))
                if bool(calibration_cfg.get("regime_aware", True)):
                    regime_calib_cmd = [
                        python,
                        "-m",
                        "src.scripts.train_platt_calibration",
                        "--horizons",
                        *[str(h) for h in calibration_horizons],
                        "--output-path",
                        str(summary_dir / "platt_calibration.json"),
                        "--coverage-output-path",
                        str(summary_dir / "platt_calibration_coverage.json"),
                        "--method",
                        str(calibration_cfg.get("method", "platt")),
                        "--labeled-input",
                        str(quality_input),
                        "--regime-col",
                        str(calibration_cfg.get("regime_col", "regime_state")),
                        "--min-regime-rows",
                        str(int(calibration_cfg.get("min_regime_rows", 100))),
                    ]
                    if bool(calibration_cfg.get("fit_base_horizons_from_labeled_input", True)):
                        regime_calib_cmd.append("--fit-base-horizons-from-labeled-input")
                    if bool(calibration_cfg.get("skip_model_fit_when_labeled_input", True)):
                        regime_calib_cmd.append("--skip-model-fit")
                    results.append(
                        _run_step(
                            "platt_calibration_regime_aware",
                            regime_calib_cmd,
                            logs_dir / "platt_calibration_regime_aware.log",
                            args.dry_run,
                        )
                    )

                calibration_quality_input = quality_input
                if bool(calibration_cfg.get("use_calibrated_input", True)):
                    calibration_calibrated_input_path = summary_dir / "labeled_backtest.calibrated.csv"
                    calibration_input_status: Dict[str, Any] | None = None
                    if not args.dry_run and quality_input.exists() and (summary_dir / "platt_calibration.json").exists():
                        try:
                            calibration_input_status = _write_calibrated_quality_input(
                                source_path=quality_input,
                                calibration_path=summary_dir / "platt_calibration.json",
                                output_path=calibration_calibrated_input_path,
                                regime_col=str(calibration_cfg.get("regime_col", "regime_state")),
                            )
                            if bool(calibration_input_status.get("written", False)):
                                calibration_quality_input = calibration_calibrated_input_path
                        except Exception as exc:
                            calibration_input_status = {
                                "written": False,
                                "reason": f"exception:{exc}",
                                "source": str(quality_input),
                                "output": str(calibration_calibrated_input_path),
                                "calibration": str(summary_dir / "platt_calibration.json"),
                            }
                            print(f"Warning: failed to write calibrated robustness input: {exc}", file=sys.stderr)
                        if calibration_input_status is not None:
                            (summary_dir / "calibration_calibrated_input_status.json").write_text(
                                json.dumps(calibration_input_status, indent=2),
                                encoding="utf-8",
                            )

                calibration_cmd = _build_calibration_robustness_command(
                    python=python,
                    input_path=calibration_quality_input,
                    output_path=summary_dir / "calibration_robustness.json",
                    calibration_cfg=calibration_cfg,
                    quality_cfg=quality_cfg,
                    trade_decision_cfg=trade_decision_cfg,
                )
                results.append(_run_step("calibration_robustness", calibration_cmd, logs_dir / "calibration_robustness.log", args.dry_run))
                if not args.dry_run:
                    calibration_output_path = summary_dir / "calibration_robustness.json"
                    if calibration_output_path.exists():
                        try:
                            calibration_payload = _load_json(calibration_output_path)
                            recent_diag_payload = _extract_recent_calibration_payload(
                                calibration_payload,
                                horizon_key=calibration_horizon_key,
                            )
                            (summary_dir / "recent_calibration_diagnostics.json").write_text(
                                json.dumps(recent_diag_payload, indent=2),
                                encoding="utf-8",
                            )
                            shutil.copyfile(
                                calibration_output_path,
                                summary_dir / "calibration_robustness_raw.json",
                            )
                            shutil.copyfile(
                                summary_dir / "recent_calibration_diagnostics.json",
                                summary_dir / "recent_calibration_diagnostics_raw.json",
                            )
                        except Exception as exc:
                            print(f"Warning: failed to extract recent calibration diagnostics: {exc}", file=sys.stderr)

            directional_quality_input = calibration_quality_input if 'calibration_quality_input' in locals() else quality_input
            if bool(directional_objectives_cfg.get("use_calibrated_input", True)):
                directional_calibrated_input_path = summary_dir / "labeled_backtest.directional_calibrated.csv"
                directional_calibration_status: Dict[str, Any] | None = None
                if directional_quality_input == quality_input and not args.dry_run and quality_input.exists() and (summary_dir / "platt_calibration.json").exists():
                    try:
                        directional_calibration_status = _write_calibrated_quality_input(
                            source_path=quality_input,
                            calibration_path=summary_dir / "platt_calibration.json",
                            output_path=directional_calibrated_input_path,
                            regime_col=str(calibration_cfg.get("regime_col", "regime_state")),
                        )
                        if bool(directional_calibration_status.get("written", False)):
                            directional_quality_input = directional_calibrated_input_path
                    except Exception as exc:
                        directional_calibration_status = {
                            "written": False,
                            "reason": f"exception:{exc}",
                            "source": str(quality_input),
                            "output": str(directional_calibrated_input_path),
                            "calibration": str(summary_dir / "platt_calibration.json"),
                        }
                        print(f"Warning: failed to write calibrated directional input: {exc}", file=sys.stderr)
                    if directional_calibration_status is not None:
                        (summary_dir / "directional_calibrated_input_status.json").write_text(
                            json.dumps(directional_calibration_status, indent=2),
                            encoding="utf-8",
                        )
                elif directional_quality_input != quality_input and not args.dry_run:
                    directional_calibration_status = {
                        "written": True,
                        "source": str(quality_input),
                        "output": str(directional_quality_input),
                        "calibration": str(summary_dir / "platt_calibration.json"),
                        "reused_from": "calibration_robustness",
                    }
                    (summary_dir / "directional_calibrated_input_status.json").write_text(
                        json.dumps(directional_calibration_status, indent=2),
                        encoding="utf-8",
                    )

            if bool(directional_objectives_cfg.get("enabled", False)):
                directional_cmd = _build_directional_objectives_command(
                    python=python,
                    input_path=directional_quality_input,
                    output_path=summary_dir / "directional_objectives.json",
                    directional_cfg=directional_objectives_cfg,
                )
                results.append(
                    _run_step(
                        "directional_objectives",
                        directional_cmd,
                        logs_dir / "directional_objectives.log",
                        args.dry_run,
                    )
                )

            if bool(regime_weakness_cfg.get("enabled", True)):
                calibration_path = summary_dir / "calibration_robustness.json"
                if (calibration_path.exists() and walkforward_output.exists()) or args.dry_run:
                    regime_weakness_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_regime_weakness",
                        "--calibration",
                        str(calibration_path),
                        "--walkforward",
                        str(walkforward_output),
                        "--horizon",
                        str(calibration_horizon_key),
                        "--max-ece-drift",
                        str(float(regime_weakness_cfg.get("max_ece_drift", calibration_cfg.get("max_ece_drift", 0.02)))),
                        "--min-recent-auc",
                        str(float(quality_cfg.get("min_recent_auc", 0.0))),
                        "--min-net-return",
                        str(float(regime_weakness_cfg.get("min_net_return", 0.0))),
                        "--output",
                        str(summary_dir / "regime_weakness.json"),
                    ]
                    results.append(
                        _run_step(
                            "regime_weakness",
                            regime_weakness_cmd,
                            logs_dir / "regime_weakness.log",
                            args.dry_run,
                        )
                    )

            if bool(rolling_ab_cfg.get("enabled", False)):
                baseline_input = rolling_ab_cfg.get("baseline_input")
                candidate_input = rolling_ab_cfg.get("candidate_input")
                if baseline_input and candidate_input:
                    rolling_cmd = _build_rolling_ab_command(
                        python=python,
                        baseline_path=str(baseline_input),
                        candidate_path=str(candidate_input),
                        rolling_cfg=rolling_ab_cfg,
                        output_path=summary_dir / "rolling_ab_reference_report.json",
                        output_md_path=summary_dir / "rolling_ab_reference_report.md",
                    )
                    results.append(
                        _run_step(
                            "rolling_ab_reference_report",
                            rolling_cmd,
                            logs_dir / "rolling_ab_reference_report.log",
                            args.dry_run,
                        )
                    )
                else:
                    print(
                        "Warning: quality.rolling_ab.enabled=true but baseline_input/candidate_input not set; skipping rolling A/B.",
                        file=sys.stderr,
                    )

        incumbent_quality_path = quality_cfg.get("incumbent_quality_path")
        incumbent_backtest_csv = quality_cfg.get("incumbent_backtest_csv")
        incumbent_labeled = summary_dir / "labeled_backtest_incumbent_1h.csv"
        if incumbent_quality_path and incumbent_backtest_csv:
            incumbent_labeled_cmd = [
                python,
                "-m",
                "src.scripts.build_labeled_backtest_from_history",
                "--backtest-csv",
                str(incumbent_backtest_csv),
                "--prefer-backtest",
                "--output",
                str(incumbent_labeled),
                "--meta-output",
                str(summary_dir / "labeled_backtest_incumbent_meta.json"),
                "--fold-size",
                str(int(quality_cfg.get("fold_size", 12))),
                "--lookback-rows",
                str(int(quality_cfg.get("lookback_rows", 2000))),
                "--lookback-hours",
                str(int(quality_cfg.get("lookback_hours", 0))),
                "--min-rows",
                str(int(quality_cfg.get("min_labeled_rows", 200))),
            ]
            results.append(
                _run_step(
                    "build_incumbent_labeled_dataset",
                    incumbent_labeled_cmd,
                    logs_dir / "build_incumbent_labeled_dataset.log",
                    args.dry_run,
                )
            )

            incumbent_quality_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_model_quality",
                "--input",
                str(incumbent_labeled),
                "--output",
                str(incumbent_quality_path),
            ]
            results.append(
                _run_step(
                    "incumbent_quality_eval",
                    incumbent_quality_cmd,
                    logs_dir / "incumbent_quality_eval.log",
                    args.dry_run,
                )
            )

        candidate_quality_path = summary_dir / "model_quality_candidate.json"
        champion_gate_payload: Dict[str, Any] | None = None
        if bool(champ_cfg.get("enabled", False)):
            baseline_raw = champ_cfg.get("baseline_input")
            candidate_raw = champ_cfg.get("candidate_input")
            resolved_incumbent_labeled = (
                str(incumbent_labeled)
                if incumbent_quality_path and incumbent_backtest_csv and incumbent_labeled.exists()
                else None
            )
            baseline_raw_normalized = str(baseline_raw).strip() if baseline_raw is not None else ""
            baseline_points_to_incumbent_raw = bool(
                baseline_raw_normalized
                and incumbent_backtest_csv
                and Path(baseline_raw_normalized).resolve() == Path(str(incumbent_backtest_csv)).resolve()
            )

            baseline_input = (
                resolved_incumbent_labeled or incumbent_backtest_csv
                if baseline_raw is None
                or baseline_raw_normalized.lower() in {"", "auto"}
                or baseline_points_to_incumbent_raw
                else str(baseline_raw)
            )
            candidate_input = (
                str(candidate_gate_input)
                if candidate_raw is None or str(candidate_raw).strip().lower() in {"", "auto"}
                else str(candidate_raw)
            )

            baseline_resolution = {
                "champion_baseline_config": baseline_raw,
                "incumbent_backtest_csv": incumbent_backtest_csv,
                "incumbent_labeled_csv": resolved_incumbent_labeled,
                "resolved_baseline_input": baseline_input,
                "resolved_candidate_input": candidate_input,
                "baseline_exists": bool(Path(str(baseline_input)).exists()) if baseline_input else False,
                "candidate_exists": bool(Path(str(candidate_input)).exists()) if candidate_input else False,
                "baseline_matches_incumbent_backtest_csv": bool(
                    baseline_input and incumbent_backtest_csv and Path(str(baseline_input)).resolve() == Path(str(incumbent_backtest_csv)).resolve()
                ),
                "baseline_matches_incumbent_labeled_csv": bool(
                    baseline_input and resolved_incumbent_labeled and Path(str(baseline_input)).resolve() == Path(str(resolved_incumbent_labeled)).resolve()
                ),
                "baseline_config_points_to_incumbent_raw": baseline_points_to_incumbent_raw,
            }
            (summary_dir / "champion_baseline_resolution.json").write_text(
                json.dumps(baseline_resolution, indent=2),
                encoding="utf-8",
            )

            if baseline_input and candidate_input:
                if bool(trade_decision_cfg.get("enabled", True)) and bool(trade_decision_deploy_ready):
                    paper_live_cfg_path = Path(str(search_cfg.get("paper_live_config", "")))
                    trade_policy_cfg: Dict[str, Any] = {}
                    if paper_live_cfg_path.exists():
                        try:
                            paper_live_cfg = _load_yaml(paper_live_cfg_path)
                            trade_policy_obj = paper_live_cfg.get("trade_decision_policy", {})
                            if isinstance(trade_policy_obj, dict):
                                trade_policy_cfg = trade_policy_obj
                        except Exception as exc:
                            print(
                                f"Warning: failed to load trade_decision_policy from {paper_live_cfg_path}: {exc}",
                                file=sys.stderr,
                            )

                    baseline_diag_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(baseline_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(summary_dir / "backtest_signals_baseline_policy_diagnostics.csv"),
                        "--diagnostics-output",
                        str(summary_dir / "trade_decision_diagnostics_baseline.json"),
                        "--meta-output",
                        str(summary_dir / "trade_decision_diagnostics_baseline_meta.json"),
                        "--diagnostics-only",
                        "--threshold",
                        str(resolved_trade_decision_threshold),
                        "--fee-bps",
                        str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                        "--slippage-bps",
                        str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                        "--replace-threshold-rule",
                        "1" if bool(trade_policy_cfg.get("replace_threshold_rule", True)) else "0",
                        "--require-direction-ret-alignment",
                        "1" if bool(trade_policy_cfg.get("require_direction_ret_alignment", True)) else "0",
                        "--use-oof-expected-value",
                        "1" if bool(trade_policy_cfg.get("use_oof_expected_value", True)) else "0",
                        "--oof-expected-value-mode",
                        str(trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")),
                        "--enforce-positive-oof-envelope",
                        "1" if bool(trade_policy_cfg.get("enforce_positive_oof_envelope", False)) else "0",
                        "--positive-oof-envelope-mode",
                        str(trade_policy_cfg.get("positive_oof_envelope_mode", "strict_positive_bin")),
                        "--block-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("block_when_no_positive_oof_bin", True)) else "0",
                        "--positive-oof-min-samples",
                        str(int(trade_policy_cfg.get("positive_oof_min_samples", 4))),
                        "--allow-raw-ev-fallback-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)) else "0",
                        "--raw-ev-fallback-quantile",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9))),
                        "--raw-ev-fallback-min-edge-over-fee",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0))),
                        "--min-expected-net",
                        str(float(trade_policy_cfg.get("min_expected_net", 0.0))),
                        "--min-edge-over-fee",
                        str(float(trade_policy_cfg.get("min_edge_over_fee", 0.0))),
                        "--policy-midband-veto",
                        "1" if direct_midband_policy_enabled else "0",
                        "--policy-midband-pup-low",
                        str(direct_midband_policy_low),
                        "--policy-midband-pup-high",
                        str(direct_midband_policy_high),
                        "--policy-midband-high-inclusive",
                        "1" if direct_midband_policy_high_inclusive else "0",
                    ]
                    if direct_midband_policy_min_abs_ret_pred is not None:
                        baseline_diag_cmd.extend(
                            ["--policy-midband-min-abs-ret-pred", str(float(direct_midband_policy_min_abs_ret_pred))]
                        )
                    if direct_midband_policy_max_abs_ret_pred is not None:
                        baseline_diag_cmd.extend(
                            ["--policy-midband-max-abs-ret-pred", str(float(direct_midband_policy_max_abs_ret_pred))]
                        )
                    if direct_midband_policy_regime_states:
                        baseline_diag_cmd.extend(
                            ["--policy-midband-regime-states", ",".join(direct_midband_policy_regime_states)]
                        )
                    if quality_input.exists() or args.dry_run:
                        baseline_diag_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_csv:
                        baseline_diag_cmd.extend(["--feature-source", str(incumbent_backtest_csv)])
                    if candidate_input:
                        baseline_diag_cmd.extend(["--feature-source", str(candidate_input)])
                    results.append(
                        _run_step(
                            "trade_decision_diagnostics_baseline",
                            baseline_diag_cmd,
                            logs_dir / "trade_decision_diagnostics_baseline.log",
                            args.dry_run,
                        )
                    )

                    # Build a policy-aligned incumbent artifact for informational (non-gating) companion analysis.
                    incumbent_aligned_path = summary_dir / "backtest_signals_incumbent_decision_aligned.csv"
                    incumbent_aligned_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(baseline_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(incumbent_aligned_path),
                        "--meta-output",
                        str(summary_dir / "backtest_signals_incumbent_decision_aligned_meta.json"),
                        "--threshold",
                        str(resolved_trade_decision_threshold),
                        "--fee-bps",
                        str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                        "--slippage-bps",
                        str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                        "--replace-threshold-rule",
                        "1" if bool(trade_policy_cfg.get("replace_threshold_rule", True)) else "0",
                        "--require-direction-ret-alignment",
                        "1" if bool(trade_policy_cfg.get("require_direction_ret_alignment", True)) else "0",
                        "--use-oof-expected-value",
                        "1" if bool(trade_policy_cfg.get("use_oof_expected_value", True)) else "0",
                        "--oof-expected-value-mode",
                        str(trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")),
                        "--enforce-positive-oof-envelope",
                        "1" if bool(trade_policy_cfg.get("enforce_positive_oof_envelope", False)) else "0",
                        "--positive-oof-envelope-mode",
                        str(trade_policy_cfg.get("positive_oof_envelope_mode", "strict_positive_bin")),
                        "--block-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("block_when_no_positive_oof_bin", True)) else "0",
                        "--positive-oof-min-samples",
                        str(int(trade_policy_cfg.get("positive_oof_min_samples", 4))),
                        "--allow-raw-ev-fallback-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)) else "0",
                        "--raw-ev-fallback-quantile",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9))),
                        "--raw-ev-fallback-min-edge-over-fee",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0))),
                        "--min-expected-net",
                        str(float(trade_policy_cfg.get("min_expected_net", 0.0))),
                        "--min-edge-over-fee",
                        str(float(trade_policy_cfg.get("min_edge_over_fee", 0.0))),
                        "--policy-midband-veto",
                        "1" if direct_midband_policy_enabled else "0",
                        "--policy-midband-pup-low",
                        str(direct_midband_policy_low),
                        "--policy-midband-pup-high",
                        str(direct_midband_policy_high),
                        "--policy-midband-high-inclusive",
                        "1" if direct_midband_policy_high_inclusive else "0",
                    ]
                    if direct_midband_policy_min_abs_ret_pred is not None:
                        incumbent_aligned_cmd.extend(
                            ["--policy-midband-min-abs-ret-pred", str(float(direct_midband_policy_min_abs_ret_pred))]
                        )
                    if direct_midband_policy_max_abs_ret_pred is not None:
                        incumbent_aligned_cmd.extend(
                            ["--policy-midband-max-abs-ret-pred", str(float(direct_midband_policy_max_abs_ret_pred))]
                        )
                    if direct_midband_policy_regime_states:
                        incumbent_aligned_cmd.extend(
                            ["--policy-midband-regime-states", ",".join(direct_midband_policy_regime_states)]
                        )
                    if quality_input.exists() or args.dry_run:
                        incumbent_aligned_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_csv:
                        incumbent_aligned_cmd.extend(["--feature-source", str(incumbent_backtest_csv)])
                    if candidate_input:
                        incumbent_aligned_cmd.extend(["--feature-source", str(candidate_input)])
                    results.append(
                        _run_step(
                            "trade_decision_align_incumbent_companion",
                            incumbent_aligned_cmd,
                            logs_dir / "trade_decision_align_incumbent_companion.log",
                            args.dry_run,
                        )
                    )

                    companion_candidate_input = str(candidate_input)
                    companion_baseline_input = str(incumbent_aligned_path)
                    official_profile_summary_path = summary_dir / "official_profile_policy_metrics.json"
                    companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(companion_candidate_input),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_companion",
                            companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_companion.log",
                            args.dry_run,
                        )
                    )

                    official_profile_summary_cmd = [
                        python,
                        "-m",
                        "src.scripts.summarize_policy_aligned_profile_metrics",
                        "--candidate",
                        str(companion_candidate_input),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--candidate-meta",
                        str(summary_dir / "backtest_signals_meta_ensemble_decision_aligned_meta.json"),
                        "--profile-id",
                        str(run_profile_id),
                        "--profile-name",
                        str(run_profile_name),
                        "--run-id",
                        str(run_dir.name),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(official_profile_summary_path),
                    ]
                    results.append(
                        _run_step(
                            "official_profile_policy_metrics",
                            official_profile_summary_cmd,
                            logs_dir / "official_profile_policy_metrics.log",
                            args.dry_run,
                        )
                    )

                    if weak_veto_enabled_official and weak_veto_official_mode == "midband":
                        paper_longitudinal_output_path = args.run_root / "midband_shadow_longitudinal.json"
                        paper_longitudinal_cmd = [
                            python,
                            "-m",
                            "src.scripts.update_midband_shadow_longitudinal",
                            "--run-id",
                            str(run_dir.name),
                            "--comparison",
                            str(official_profile_summary_path),
                            "--track",
                            "paper_profile",
                            "--output",
                            str(paper_longitudinal_output_path),
                        ]
                        results.append(
                            _run_step(
                                "midband_paper_profile_longitudinal_update",
                                paper_longitudinal_cmd,
                                logs_dir / "midband_paper_profile_longitudinal_update.log",
                                args.dry_run,
                            )
                        )

                    paired_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(companion_candidate_input),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned",
                            paired_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned.log",
                            args.dry_run,
                        )
                    )

                    overlap_triggered_trade_diag_path = summary_dir / "overlap_triggered_trade_diagnostics.json"
                    companion_candidate_input_path = Path(str(companion_candidate_input))
                    if (
                        labeled_overlap_dataset is not None
                        and labeled_overlap_dataset.exists()
                        and companion_candidate_input_path.exists()
                    ):
                        overlap_triggered_trade_diag_cmd = [
                            python,
                            "-m",
                            "src.scripts.analyze_overlap_triggered_trade_diagnostics",
                            "--candidate",
                            str(companion_candidate_input_path),
                            "--overlap-dataset",
                            str(labeled_overlap_dataset),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--return-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--output",
                            str(overlap_triggered_trade_diag_path),
                        ]
                        results.append(
                            _run_step(
                                "overlap_triggered_trade_diagnostics",
                                overlap_triggered_trade_diag_cmd,
                                logs_dir / "overlap_triggered_trade_diagnostics.log",
                                args.dry_run,
                            )
                        )

                    weak_band_regime_diag_path = summary_dir / "weak_band_candidate_only_regime_diagnostics.json"
                    weak_band_regime_diag_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_weak_band_candidate_regime_diagnostics",
                        "--candidate",
                        str(companion_candidate_input),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--incumbent-signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--return-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--p-up-low",
                        str(regime_state_veto_low),
                        "--p-up-high",
                        str(regime_state_veto_high),
                        "--high-inclusive",
                        "1" if regime_state_veto_high_inclusive else "0",
                        "--use-midband-slice",
                        "1" if regime_state_veto_use_midband_slice else "0",
                        "--min-abs-ret-pred",
                        str(regime_state_veto_min_abs_ret_pred),
                        "--max-abs-ret-pred",
                        str(regime_state_veto_max_abs_ret_pred),
                        "--min-regime-rows",
                        str(regime_state_veto_min_regime_rows),
                        "--output",
                        str(weak_band_regime_diag_path),
                    ]
                    results.append(
                        _run_step(
                            "weak_band_candidate_only_regime_diagnostics",
                            weak_band_regime_diag_cmd,
                            logs_dir / "weak_band_candidate_only_regime_diagnostics.log",
                            args.dry_run,
                        )
                    )

                    selected_regime_states = list(regime_state_veto_override)
                    if (not selected_regime_states) and weak_band_regime_diag_path.exists() and (not args.dry_run):
                        try:
                            regime_diag_payload = _load_json(weak_band_regime_diag_path)
                            selected_obj = regime_diag_payload.get("selected_harmful_regimes", [])
                            if isinstance(selected_obj, list):
                                selected_regime_states = [
                                    str(value).strip().lower()
                                    for value in selected_obj
                                    if str(value).strip()
                                ]
                        except Exception as exc:
                            print(
                                f"Warning: failed to load selected harmful regimes from {weak_band_regime_diag_path}: {exc}",
                                file=sys.stderr,
                            )

                    selected_high_volatility_threshold = None
                    if chop_high_vol_veto_min_volatility_cfg is not None:
                        selected_high_volatility_threshold = float(chop_high_vol_veto_min_volatility_cfg)
                    elif weak_band_regime_diag_path.exists() and (not args.dry_run):
                        try:
                            regime_diag_payload = _load_json(weak_band_regime_diag_path)
                            vol_rule_obj = regime_diag_payload.get("selected_high_volatility_rule", {})
                            if isinstance(vol_rule_obj, dict):
                                threshold_obj = vol_rule_obj.get("threshold")
                                if threshold_obj is not None:
                                    threshold_value = float(threshold_obj)
                                    if np.isfinite(threshold_value):
                                        selected_high_volatility_threshold = threshold_value
                        except Exception as exc:
                            print(
                                f"Warning: failed to load selected high-volatility threshold from {weak_band_regime_diag_path}: {exc}",
                                file=sys.stderr,
                            )
                    volatility_only_min_volatility = selected_high_volatility_threshold
                    if volatility_only_veto_min_volatility_cfg is not None:
                        volatility_only_min_volatility = float(volatility_only_veto_min_volatility_cfg)

                    broad_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto.csv"
                    broad_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto_meta.json"
                    broad_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(broad_shadow_candidate_path),
                        "--meta-output",
                        str(broad_shadow_meta_path),
                    ]
                    broad_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        broad_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        broad_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    broad_shadow_build_cmd.extend(
                        [
                            "--weak-band-candidate-only-veto",
                            "1",
                            "--weak-band-pup-low",
                            str(weak_veto_low),
                            "--weak-band-pup-high",
                            str(weak_veto_high),
                            "--weak-band-high-inclusive",
                            "1" if weak_veto_high_inclusive else "0",
                            "--weak-band-incumbent-reference",
                            str(companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_veto_candidate_band",
                            broad_shadow_build_cmd,
                            logs_dir / "build_shadow_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    broad_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(broad_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_veto_companion",
                            broad_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    broad_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(broad_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_veto",
                            broad_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_veto.log",
                            args.dry_run,
                        )
                    )

                    refined_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto.csv"
                    refined_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto_meta.json"
                    refined_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(refined_shadow_candidate_path),
                        "--meta-output",
                        str(refined_shadow_meta_path),
                    ]
                    refined_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        refined_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        refined_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    refined_shadow_build_cmd.extend(
                        [
                            "--refined-candidate-only-veto",
                            "1",
                            "--refined-pup-low",
                            str(refined_veto_low),
                            "--refined-pup-high",
                            str(refined_veto_high),
                            "--refined-high-inclusive",
                            "1" if refined_veto_high_inclusive else "0",
                            "--refined-min-abs-ret-pred",
                            str(refined_veto_min_abs_ret_pred),
                            "--refined-incumbent-reference",
                            str(refined_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_refined_veto_candidate_band",
                            refined_shadow_build_cmd,
                            logs_dir / "build_shadow_refined_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    refined_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(refined_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_refined_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_refined_veto_companion",
                            refined_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_refined_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    refined_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(refined_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_refined_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_refined_veto",
                            refined_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_refined_veto.log",
                            args.dry_run,
                        )
                    )

                    midband_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto.csv"
                    midband_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto_meta.json"
                    midband_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(midband_shadow_candidate_path),
                        "--meta-output",
                        str(midband_shadow_meta_path),
                    ]
                    midband_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        midband_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        midband_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    midband_shadow_build_cmd.extend(
                        [
                            "--midband-candidate-only-veto",
                            "1",
                            "--midband-pup-low",
                            str(midband_veto_low),
                            "--midband-pup-high",
                            str(midband_veto_high),
                            "--midband-high-inclusive",
                            "1" if midband_veto_high_inclusive else "0",
                            "--midband-min-abs-ret-pred",
                            str(midband_veto_min_abs_ret_pred),
                            "--midband-max-abs-ret-pred",
                            str(midband_veto_max_abs_ret_pred),
                            "--midband-incumbent-reference",
                            str(midband_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_midband_veto_candidate_band",
                            midband_shadow_build_cmd,
                            logs_dir / "build_shadow_midband_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    midband_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(midband_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_midband_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_midband_veto_companion",
                            midband_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_midband_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    midband_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(midband_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_midband_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_midband_veto",
                            midband_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_midband_veto.log",
                            args.dry_run,
                        )
                    )

                    raw_ev_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto.csv"
                    raw_ev_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto_meta.json"
                    raw_ev_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(raw_ev_shadow_candidate_path),
                        "--meta-output",
                        str(raw_ev_shadow_meta_path),
                    ]
                    raw_ev_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        raw_ev_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        raw_ev_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    raw_ev_shadow_build_cmd.extend(
                        [
                            "--raw-ev-sign-candidate-only-veto",
                            "1",
                            "--raw-ev-sign-pup-low",
                            str(raw_ev_veto_low),
                            "--raw-ev-sign-pup-high",
                            str(raw_ev_veto_high),
                            "--raw-ev-sign-high-inclusive",
                            "1" if raw_ev_veto_high_inclusive else "0",
                            "--raw-ev-sign-max",
                            str(raw_ev_veto_max),
                            "--raw-ev-sign-incumbent-reference",
                            str(raw_ev_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_raw_ev_veto_candidate_band",
                            raw_ev_shadow_build_cmd,
                            logs_dir / "build_shadow_raw_ev_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    raw_ev_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(raw_ev_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion",
                            raw_ev_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    raw_ev_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(raw_ev_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_raw_ev_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_raw_ev_veto",
                            raw_ev_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_raw_ev_veto.log",
                            args.dry_run,
                        )
                    )

                    direction_align_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto.csv"
                    )
                    direction_align_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto_meta.json"
                    )
                    direction_align_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(direction_align_shadow_candidate_path),
                        "--meta-output",
                        str(direction_align_shadow_meta_path),
                    ]
                    direction_align_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        direction_align_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        direction_align_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    direction_align_shadow_build_cmd.extend(
                        [
                            "--direction-align-candidate-only-veto",
                            "1",
                            "--direction-align-pup-low",
                            str(direction_align_veto_low),
                            "--direction-align-pup-high",
                            str(direction_align_veto_high),
                            "--direction-align-high-inclusive",
                            "1" if direction_align_veto_high_inclusive else "0",
                            "--direction-align-require-aligned",
                            "1" if direction_align_veto_require_aligned else "0",
                            "--direction-align-use-midband-slice",
                            "1" if direction_align_veto_use_midband_slice else "0",
                            "--direction-align-min-abs-ret-pred",
                            str(direction_align_veto_min_abs_ret_pred),
                            "--direction-align-max-abs-ret-pred",
                            str(direction_align_veto_max_abs_ret_pred),
                            "--direction-align-incumbent-reference",
                            str(direction_align_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_direction_align_veto_candidate_band",
                            direction_align_shadow_build_cmd,
                            logs_dir / "build_shadow_direction_align_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    direction_align_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(direction_align_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_direction_align_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_direction_align_veto_companion",
                            direction_align_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_direction_align_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    direction_align_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(direction_align_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_direction_align_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_direction_align_veto",
                            direction_align_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_direction_align_veto.log",
                            args.dry_run,
                        )
                    )

                    joint_direction_midband_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto.csv"
                    )
                    joint_direction_midband_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto_meta.json"
                    )
                    joint_direction_midband_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(joint_direction_midband_shadow_candidate_path),
                        "--meta-output",
                        str(joint_direction_midband_shadow_meta_path),
                    ]
                    joint_direction_midband_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        joint_direction_midband_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        joint_direction_midband_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    joint_direction_midband_shadow_build_cmd.extend(
                        [
                            "--joint-direction-midband-candidate-only-veto",
                            "1",
                            "--joint-direction-midband-pup-low",
                            str(joint_direction_midband_veto_low),
                            "--joint-direction-midband-pup-high",
                            str(joint_direction_midband_veto_high),
                            "--joint-direction-midband-high-inclusive",
                            "1" if joint_direction_midband_veto_high_inclusive else "0",
                            "--joint-direction-midband-require-aligned",
                            "1" if joint_direction_midband_veto_require_aligned else "0",
                            "--joint-direction-midband-min-abs-ret-pred",
                            str(joint_direction_midband_veto_min_abs_ret_pred),
                            "--joint-direction-midband-max-abs-ret-pred",
                            str(joint_direction_midband_veto_max_abs_ret_pred),
                            "--joint-direction-midband-incumbent-reference",
                            str(joint_direction_midband_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_joint_direction_midband_veto_candidate_band",
                            joint_direction_midband_shadow_build_cmd,
                            logs_dir / "build_shadow_joint_direction_midband_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    joint_direction_midband_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(joint_direction_midband_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion",
                            joint_direction_midband_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    joint_direction_midband_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(joint_direction_midband_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_joint_direction_midband_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_joint_direction_midband_veto",
                            joint_direction_midband_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_joint_direction_midband_veto.log",
                            args.dry_run,
                        )
                    )

                    regime_state_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto.csv"
                    )
                    regime_state_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto_meta.json"
                    )
                    regime_state_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(regime_state_shadow_candidate_path),
                        "--meta-output",
                        str(regime_state_shadow_meta_path),
                    ]
                    regime_state_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        regime_state_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        regime_state_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    regime_state_shadow_build_cmd.extend(
                        [
                            "--regime-state-candidate-only-veto",
                            "1",
                            "--regime-state-pup-low",
                            str(regime_state_veto_low),
                            "--regime-state-pup-high",
                            str(regime_state_veto_high),
                            "--regime-state-high-inclusive",
                            "1" if regime_state_veto_high_inclusive else "0",
                            "--regime-state-regimes",
                            ",".join(selected_regime_states),
                            "--regime-state-use-midband-slice",
                            "1" if regime_state_veto_use_midband_slice else "0",
                            "--regime-state-min-abs-ret-pred",
                            str(regime_state_veto_min_abs_ret_pred),
                            "--regime-state-max-abs-ret-pred",
                            str(regime_state_veto_max_abs_ret_pred),
                            "--regime-state-incumbent-reference",
                            str(regime_state_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_regime_state_veto_candidate_band",
                            regime_state_shadow_build_cmd,
                            logs_dir / "build_shadow_regime_state_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    regime_state_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(regime_state_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_regime_state_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_regime_state_veto_companion",
                            regime_state_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_regime_state_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    regime_state_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(regime_state_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto",
                            regime_state_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto.log",
                            args.dry_run,
                        )
                    )

                    chop_high_vol_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto.csv"
                    )
                    chop_high_vol_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto_meta.json"
                    )
                    chop_high_vol_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(chop_high_vol_shadow_candidate_path),
                        "--meta-output",
                        str(chop_high_vol_shadow_meta_path),
                    ]
                    chop_high_vol_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        chop_high_vol_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        chop_high_vol_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    chop_high_vol_shadow_build_cmd.extend(
                        [
                            "--chop-high-vol-candidate-only-veto",
                            "1",
                            "--chop-high-vol-pup-low",
                            str(chop_high_vol_veto_low),
                            "--chop-high-vol-pup-high",
                            str(chop_high_vol_veto_high),
                            "--chop-high-vol-high-inclusive",
                            "1" if chop_high_vol_veto_high_inclusive else "0",
                            "--chop-high-vol-regime-state",
                            str(chop_high_vol_veto_regime_state),
                            "--chop-high-vol-volatility-col",
                            str(chop_high_vol_veto_volatility_col),
                            "--chop-high-vol-min-volatility",
                            str(selected_high_volatility_threshold if selected_high_volatility_threshold is not None else "nan"),
                            "--chop-high-vol-use-midband-slice",
                            "1" if chop_high_vol_veto_use_midband_slice else "0",
                            "--chop-high-vol-min-abs-ret-pred",
                            str(chop_high_vol_veto_min_abs_ret_pred),
                            "--chop-high-vol-max-abs-ret-pred",
                            str(chop_high_vol_veto_max_abs_ret_pred),
                            "--chop-high-vol-incumbent-reference",
                            str(chop_high_vol_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_chop_high_vol_veto_candidate_band",
                            chop_high_vol_shadow_build_cmd,
                            logs_dir / "build_shadow_chop_high_vol_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    chop_high_vol_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(chop_high_vol_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion",
                            chop_high_vol_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    chop_high_vol_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(chop_high_vol_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_chop_high_vol_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_chop_high_vol_veto",
                            chop_high_vol_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_chop_high_vol_veto.log",
                            args.dry_run,
                        )
                    )

                    volatility_only_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto.csv"
                    )
                    volatility_only_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto_meta.json"
                    )
                    volatility_only_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(volatility_only_shadow_candidate_path),
                        "--meta-output",
                        str(volatility_only_shadow_meta_path),
                    ]
                    volatility_only_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        volatility_only_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        volatility_only_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    volatility_only_shadow_build_cmd.extend(
                        [
                            "--volatility-only-candidate-only-veto",
                            "1",
                            "--volatility-only-pup-low",
                            str(volatility_only_veto_low),
                            "--volatility-only-pup-high",
                            str(volatility_only_veto_high),
                            "--volatility-only-high-inclusive",
                            "1" if volatility_only_veto_high_inclusive else "0",
                            "--volatility-only-volatility-col",
                            str(volatility_only_veto_volatility_col),
                            "--volatility-only-min-volatility",
                            str(volatility_only_min_volatility if volatility_only_min_volatility is not None else "nan"),
                            "--volatility-only-use-midband-slice",
                            "1" if volatility_only_veto_use_midband_slice else "0",
                            "--volatility-only-min-abs-ret-pred",
                            str(volatility_only_veto_min_abs_ret_pred),
                            "--volatility-only-max-abs-ret-pred",
                            str(volatility_only_veto_max_abs_ret_pred),
                            "--volatility-only-incumbent-reference",
                            str(volatility_only_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_volatility_only_veto_candidate_band",
                            volatility_only_shadow_build_cmd,
                            logs_dir / "build_shadow_volatility_only_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    volatility_only_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(volatility_only_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion",
                            volatility_only_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    volatility_only_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(volatility_only_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_volatility_only_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_volatility_only_veto",
                            volatility_only_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_volatility_only_veto.log",
                            args.dry_run,
                        )
                    )

                    weak_veto_retro_cfg_obj = weak_veto_cfg.get("retrospective", {})
                    weak_veto_retro_cfg = weak_veto_retro_cfg_obj if isinstance(weak_veto_retro_cfg_obj, dict) else {}
                    if bool(weak_veto_retro_cfg.get("enabled", True)):
                        midband_shadow_comparison_path = summary_dir / "midband_shadow_comparison.json"
                        weak_veto_comparison_cmd = [
                            python,
                            "-m",
                            "src.scripts.evaluate_midband_shadow_retrospective",
                            "--default-candidate",
                            str(companion_candidate_input),
                            "--midband-shadow-candidate",
                            str(midband_shadow_candidate_path),
                            "--incumbent",
                            str(companion_baseline_input),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--incumbent-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--window-size",
                            str(int(weak_veto_retro_cfg.get("window_size", 120))),
                            "--step-size",
                            str(int(weak_veto_retro_cfg.get("step_size", 24))),
                            "--min-rows",
                            str(int(weak_veto_retro_cfg.get("min_rows", 80))),
                            "--n-boot",
                            str(int(weak_veto_retro_cfg.get("n_boot", 1000))),
                            "--seed",
                            str(int(weak_veto_retro_cfg.get("seed", champ_cfg.get("seed", 42)))),
                            "--midband-shadow-meta",
                            str(midband_shadow_meta_path),
                            "--output",
                            str(midband_shadow_comparison_path),
                        ]
                        results.append(
                            _run_step(
                                "midband_shadow_retrospective_comparison",
                                weak_veto_comparison_cmd,
                                logs_dir / "midband_shadow_retrospective_comparison.log",
                                args.dry_run,
                            )
                        )

                        longitudinal_output_cfg = weak_veto_retro_cfg.get("longitudinal_output")
                        if isinstance(longitudinal_output_cfg, str) and longitudinal_output_cfg.strip():
                            longitudinal_output_path = Path(longitudinal_output_cfg)
                        else:
                            longitudinal_output_path = args.run_root / "midband_shadow_longitudinal.json"
                        if not longitudinal_output_path.is_absolute():
                            longitudinal_output_path = Path.cwd() / longitudinal_output_path

                        midband_longitudinal_cmd = [
                            python,
                            "-m",
                            "src.scripts.update_midband_shadow_longitudinal",
                            "--run-id",
                            str(run_dir.name),
                            "--comparison",
                            str(midband_shadow_comparison_path),
                            "--track",
                            "shadow_retrospective",
                            "--output",
                            str(longitudinal_output_path),
                        ]
                        results.append(
                            _run_step(
                                "midband_shadow_longitudinal_update",
                                midband_longitudinal_cmd,
                                logs_dir / "midband_shadow_longitudinal_update.log",
                                args.dry_run,
                            )
                        )

                official_shadow_selection_cfg_obj = weak_veto_cfg.get("official_shadow_selection", {})
                official_shadow_selection_cfg = (
                    official_shadow_selection_cfg_obj if isinstance(official_shadow_selection_cfg_obj, dict) else {}
                )
                recent_window_rows_for_shadow = int(
                    official_shadow_selection_cfg.get(
                        "recent_window_rows",
                        calibration_cfg.get("recent_window", 120),
                    )
                )
                recent_triggered_cfg_obj = official_shadow_selection_cfg.get("recent_triggered_regime_volatility", {})
                recent_triggered_cfg = recent_triggered_cfg_obj if isinstance(recent_triggered_cfg_obj, dict) else {}
                recent_triggered_enabled = bool(recent_triggered_cfg.get("enabled", True))
                recent_triggered_rule = _derive_recent_triggered_regime_volatility_rule(
                    candidate_path=aligned_candidate_input,
                    recent_window_rows=recent_window_rows_for_shadow,
                    signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                    return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                    regime_col=str(recent_triggered_cfg.get("regime_col", calibration_cfg.get("regime_col", "regime_state"))),
                    volatility_col=str(recent_triggered_cfg.get("volatility_col", "volatility_realized_24h")),
                    min_regime_rows=int(recent_triggered_cfg.get("min_regime_rows", 2)),
                    require_overall_regime_negative=bool(
                        recent_triggered_cfg.get("require_overall_regime_negative", True)
                    ),
                )
                recent_triggered_rule["enabled_by_config"] = recent_triggered_enabled
                (summary_dir / "recent_triggered_regime_volatility_rule.json").write_text(
                    json.dumps(recent_triggered_rule, indent=2),
                    encoding="utf-8",
                )
                if recent_triggered_enabled and bool(recent_triggered_rule.get("enabled", False)):
                    triggered_regime_volatility_shadow_candidate_path = (
                        summary_dir
                        / "backtest_signals_meta_ensemble_decision_aligned_shadow_triggered_regime_volatility_veto.csv"
                    )
                    triggered_regime_volatility_shadow_meta_path = (
                        summary_dir
                        / "backtest_signals_meta_ensemble_decision_aligned_shadow_triggered_regime_volatility_veto_meta.json"
                    )
                    triggered_regime_volatility_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(triggered_regime_volatility_shadow_candidate_path),
                        "--meta-output",
                        str(triggered_regime_volatility_shadow_meta_path),
                    ]
                    triggered_regime_volatility_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        triggered_regime_volatility_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        triggered_regime_volatility_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    triggered_regime_volatility_shadow_build_cmd.extend(
                        [
                            "--triggered-regime-volatility-veto",
                            "1",
                            "--triggered-regime-volatility-regimes",
                            ",".join(str(value) for value in recent_triggered_rule.get("selected_regimes", [])),
                            "--triggered-regime-volatility-regime-col",
                            str(recent_triggered_rule.get("regime_col", "regime_state")),
                            "--triggered-regime-volatility-volatility-col",
                            str(recent_triggered_rule.get("volatility_col", "volatility_realized_24h")),
                            "--triggered-regime-volatility-min-volatility",
                            str(recent_triggered_rule.get("min_volatility")),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_triggered_regime_volatility_veto",
                            triggered_regime_volatility_shadow_build_cmd,
                            logs_dir / "build_shadow_triggered_regime_volatility_veto.log",
                            args.dry_run,
                        )
                    )

                    triggered_regime_volatility_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(triggered_regime_volatility_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_triggered_regime_volatility_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_triggered_regime_volatility_veto_companion",
                            triggered_regime_volatility_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_triggered_regime_volatility_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    triggered_regime_volatility_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(triggered_regime_volatility_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_triggered_regime_volatility_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_triggered_regime_volatility_veto",
                            triggered_regime_volatility_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_triggered_regime_volatility_veto.log",
                            args.dry_run,
                        )
                    )

                selection_guard_cfg_obj = official_shadow_selection_cfg.get("selection_calibration_guard", {})
                selection_guard_cfg = selection_guard_cfg_obj if isinstance(selection_guard_cfg_obj, dict) else {}
                selection_guard_enabled = bool(selection_guard_cfg.get("enabled", False))
                selection_guard_regime_col = str(
                    selection_guard_cfg.get("regime_col", calibration_cfg.get("regime_col", "regime_state"))
                )
                selection_guard_p_col = str(selection_guard_cfg.get("p_col", "p_up"))
                selection_guard_manual_rules = _normalize_selection_calibration_guard_rules(selection_guard_cfg.get("rules", []))
                selection_guard_auto_cfg_obj = selection_guard_cfg.get("auto_derive", {})
                selection_guard_auto_cfg = (
                    selection_guard_auto_cfg_obj if isinstance(selection_guard_auto_cfg_obj, dict) else {}
                )
                selection_guard_auto_enabled = bool(selection_guard_auto_cfg.get("enabled", False))
                selection_guard_auto_floors_obj = selection_guard_auto_cfg.get("candidate_p_up_floors", [0.48, 0.49, 0.54, 0.55])
                selection_guard_auto_floors: List[float] = []
                if isinstance(selection_guard_auto_floors_obj, list):
                    for raw_value in selection_guard_auto_floors_obj:
                        try:
                            selection_guard_auto_floors.append(float(raw_value))
                        except (TypeError, ValueError):
                            continue
                elif isinstance(selection_guard_auto_floors_obj, str):
                    for raw_value in selection_guard_auto_floors_obj.split(","):
                        try:
                            selection_guard_auto_floors.append(float(raw_value.strip()))
                        except (TypeError, ValueError):
                            continue
                deploy_cfg_obj = quality_cfg.get("promotion_deploy", {})
                deploy_cfg = deploy_cfg_obj if isinstance(deploy_cfg_obj, dict) else {}
                selection_guard_reference_payload = _load_reusable_selection_calibration_guard_rules(
                    deployed_rule_path=Path(
                        str(
                            selection_guard_auto_cfg.get(
                                "deployed_rule_path",
                                deploy_cfg.get(
                                    "selection_guard_rule_target",
                                    "artifacts/monitoring/selection_calibration_guard_rule_1h.json",
                                ),
                            )
                        )
                    ),
                    deploy_manifest_path=Path(
                        str(
                            selection_guard_auto_cfg.get(
                                "deploy_manifest_path",
                                deploy_cfg.get(
                                    "manifest_target",
                                    "artifacts/monitoring/reliability_promotion_deploy_manifest.json",
                                ),
                            )
                        )
                    ),
                    expected_regime_col=selection_guard_regime_col,
                    expected_p_col=selection_guard_p_col,
                )
                if bool(selection_guard_auto_cfg.get("augment_with_deployed_rule_neighborhood", True)):
                    selection_guard_auto_floors = _augment_selection_guard_candidate_floors(
                        base_floors=selection_guard_auto_floors,
                        reference_rules=_normalize_selection_calibration_guard_rules(
                            selection_guard_reference_payload.get("rules", [])
                        ),
                        step=float(selection_guard_auto_cfg.get("deployed_rule_floor_step", 0.01)),
                        lower_steps=int(selection_guard_auto_cfg.get("deployed_rule_lower_steps", 2)),
                        upper_steps=int(selection_guard_auto_cfg.get("deployed_rule_upper_steps", 0)),
                    )
                selection_guard_auto_payload: Dict[str, Any] = {
                    "enabled": False,
                    "reason": "auto_derive_disabled",
                    "rules": [],
                    "steps": [],
                }
                selection_min_candidate_trades = int(official_shadow_selection_cfg.get("min_candidate_trades", 5))
                if selection_guard_enabled and selection_guard_auto_enabled and not args.dry_run:
                    selection_guard_auto_payload = _derive_selection_calibration_guard_rules(
                        candidate_path=aligned_candidate_input,
                        recent_window_rows=int(
                            official_shadow_selection_cfg.get(
                                "recent_window_rows",
                                calibration_cfg.get("recent_window", 120),
                            )
                        ),
                        baseline_window_rows=int(calibration_cfg.get("baseline_window", 240)),
                        signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        regime_col=selection_guard_regime_col,
                        p_col=selection_guard_p_col,
                        y_col=str(selection_guard_cfg.get("y_col", "y_true")),
                        min_selection_rows=int(calibration_cfg.get("min_selection_rows", 0)),
                        adaptive_selection_cfg=(
                            calibration_cfg.get("adaptive_selection_rows")
                            if isinstance(calibration_cfg.get("adaptive_selection_rows"), dict)
                            else {}
                        ),
                        floors=selection_guard_auto_floors,
                        min_blocked_recent_rows=int(selection_guard_auto_cfg.get("min_blocked_recent_rows", 3)),
                        max_rules=int(selection_guard_auto_cfg.get("max_rules", 2)),
                        min_recent_ece_improvement=float(selection_guard_auto_cfg.get("min_recent_ece_improvement", 0.0)),
                        min_ece_drift_improvement=float(selection_guard_auto_cfg.get("min_ece_drift_improvement", 0.0)),
                        max_recent_ece=(
                            float(quality_cfg.get("max_recent_ece"))
                            if quality_cfg.get("max_recent_ece") is not None
                            else None
                        ),
                        max_ece_drift=(
                            float(calibration_cfg.get("max_ece_drift"))
                            if calibration_cfg.get("max_ece_drift") is not None
                            else None
                        ),
                        min_recent_auc=(
                            float(quality_cfg.get("min_recent_auc"))
                            if quality_cfg.get("min_recent_auc") is not None
                            else None
                        ),
                        require_blocked_recent_net_nonpositive=bool(
                            selection_guard_auto_cfg.get("require_blocked_recent_net_nonpositive", True)
                        ),
                        max_blocked_recent_net_return_total=(
                            float(selection_guard_auto_cfg.get("max_blocked_recent_net_return_total"))
                            if selection_guard_auto_cfg.get("max_blocked_recent_net_return_total") is not None
                            else None
                        ),
                        require_recent_net_nonnegative=bool(
                            selection_guard_auto_cfg.get("require_recent_net_nonnegative", True)
                        ),
                        sparse_active_trade_cap=int(selection_guard_auto_cfg.get("sparse_active_trade_cap", 5)),
                        sparse_min_blocked_recent_rows=int(selection_guard_auto_cfg.get("sparse_min_blocked_recent_rows", 1)),
                        sparse_min_retained_recent_rows=int(selection_guard_auto_cfg.get("sparse_min_retained_recent_rows", 1)),
                        sparse_allow_row_policy_override=bool(
                            selection_guard_auto_cfg.get("sparse_allow_row_policy_override", True)
                        ),
                        sparse_allow_missing_baseline=bool(
                            selection_guard_auto_cfg.get("sparse_allow_missing_baseline", True)
                        ),
                        sparse_use_observed_p_up_values=bool(
                            selection_guard_auto_cfg.get("sparse_use_observed_p_up_values", True)
                        ),
                    )
                selection_guard_reuse_payload: Dict[str, Any] = {
                    "enabled": False,
                    "reason": "not_used",
                    "rules": [],
                }
                selection_guard_reuse_viability: Dict[str, Any] = {
                    "enabled": False,
                    "reason": "not_evaluated",
                    "rules": [],
                }
                if (
                    selection_guard_enabled
                    and not args.dry_run
                    and not selection_guard_manual_rules
                    and not selection_guard_auto_payload.get("rules")
                    and bool(selection_guard_auto_cfg.get("reuse_last_deployed_on_empty", True))
                ):
                    selection_guard_reuse_payload = selection_guard_reference_payload
                    if bool(selection_guard_reuse_payload.get("enabled", False)):
                        selection_guard_reuse_viability = _evaluate_selection_calibration_guard_rule_viability(
                            candidate_path=aligned_candidate_input,
                            rules=_normalize_selection_calibration_guard_rules(
                                selection_guard_reuse_payload.get("rules", [])
                            ),
                            recent_window_rows=int(
                                official_shadow_selection_cfg.get(
                                    "recent_window_rows",
                                    calibration_cfg.get("recent_window", 120),
                                )
                            ),
                            baseline_window_rows=int(calibration_cfg.get("baseline_window", 240)),
                            signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            regime_col=selection_guard_regime_col,
                            p_col=selection_guard_p_col,
                            y_col=str(selection_guard_cfg.get("y_col", "y_true")),
                            min_selection_rows=int(calibration_cfg.get("min_selection_rows", 0)),
                            adaptive_selection_cfg=(
                                calibration_cfg.get("adaptive_selection_rows")
                                if isinstance(calibration_cfg.get("adaptive_selection_rows"), dict)
                                else {}
                            ),
                            min_candidate_trades=selection_min_candidate_trades,
                        )
                        if not bool(selection_guard_reuse_viability.get("enabled", False)):
                            selection_guard_reuse_payload = {
                                **selection_guard_reuse_payload,
                                "enabled": False,
                                "reason": str(selection_guard_reuse_viability.get("reason", "guard_reuse_not_viable")),
                                "rules": [],
                            }
                    else:
                        selection_guard_reuse_viability = {
                            "enabled": False,
                            "reason": str(selection_guard_reuse_payload.get("reason", "not_available")),
                            "rules": _normalize_selection_calibration_guard_rules(
                                selection_guard_reuse_payload.get("rules", [])
                            ),
                        }
                selection_guard_rules = _dedupe_selection_calibration_guard_rules(
                    [
                        *selection_guard_manual_rules,
                        *(
                            selection_guard_auto_payload.get("rules", [])
                            if isinstance(selection_guard_auto_payload.get("rules", []), list)
                            else []
                        ),
                        *(
                            selection_guard_reuse_payload.get("rules", [])
                            if isinstance(selection_guard_reuse_payload.get("rules", []), list)
                            else []
                        ),
                    ]
                )
                selection_guard_candidate_path = (
                    summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard.csv"
                )
                selection_guard_meta_path = (
                    summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard_meta.json"
                )
                selection_guard_source_candidate_path = (
                    Path(str(selection_guard_reference_payload.get("source_candidate_path")))
                    if selection_guard_reference_payload.get("source_candidate_path")
                    else None
                )
                selection_guard_source_summary_dir = (
                    selection_guard_source_candidate_path.parent
                    if selection_guard_source_candidate_path is not None
                    else None
                )
                trade_decision_distribution_policy_cfg = {
                    "threshold": resolved_trade_decision_threshold,
                    "replace_threshold_rule": bool(trade_policy_cfg.get("replace_threshold_rule", True)),
                    "require_direction_ret_alignment": bool(
                        trade_policy_cfg.get("require_direction_ret_alignment", True)
                    ),
                    "use_oof_expected_value": bool(trade_policy_cfg.get("use_oof_expected_value", True)),
                    "oof_expected_value_mode": str(
                        trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")
                    ),
                    "enforce_positive_oof_envelope": bool(
                        trade_policy_cfg.get("enforce_positive_oof_envelope", False)
                    ),
                    "positive_oof_envelope_mode": str(
                        trade_policy_cfg.get("positive_oof_envelope_mode", "strict_positive_bin")
                    ),
                    "block_when_no_positive_oof_bin": bool(
                        trade_policy_cfg.get("block_when_no_positive_oof_bin", True)
                    ),
                    "positive_oof_min_samples": int(trade_policy_cfg.get("positive_oof_min_samples", 4)),
                    "allow_raw_ev_fallback_when_no_positive_oof_bin": bool(
                        trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)
                    ),
                    "raw_ev_fallback_quantile": float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9)),
                    "raw_ev_fallback_min_edge_over_fee": float(
                        trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0)
                    ),
                    "min_expected_net": float(trade_policy_cfg.get("min_expected_net", 0.0)),
                    "min_edge_over_fee": float(trade_policy_cfg.get("min_edge_over_fee", 0.0)),
                    "midband_veto": {
                        "enabled": bool(direct_midband_policy_enabled),
                        "p_up_low": float(direct_midband_policy_low),
                        "p_up_high": float(direct_midband_policy_high),
                        "high_inclusive": bool(direct_midband_policy_high_inclusive),
                        "min_abs_ret_pred": (
                            None
                            if direct_midband_policy_min_abs_ret_pred is None
                            else float(direct_midband_policy_min_abs_ret_pred)
                        ),
                        "max_abs_ret_pred": (
                            None
                            if direct_midband_policy_max_abs_ret_pred is None
                            else float(direct_midband_policy_max_abs_ret_pred)
                        ),
                        "regime_states": list(direct_midband_policy_regime_states),
                    },
                }
                trade_decision_distribution_shift_payload = _build_trade_decision_distribution_shift(
                    current_candidate_path=decision_model_input,
                    current_model_path=trade_decision_model_path,
                    source_candidate_path=(
                        selection_guard_source_summary_dir / "backtest_signals_meta_ensemble_decision_features.csv"
                        if selection_guard_source_summary_dir is not None
                        else None
                    ),
                    source_model_path=(
                        selection_guard_source_summary_dir / "trade_decision_model.json"
                        if selection_guard_source_summary_dir is not None
                        else None
                    ),
                    trade_policy_cfg=trade_decision_distribution_policy_cfg,
                    fee_bps=float(quality_cfg.get("walkforward_fee_bps", 2.0)),
                    slippage_bps=float(quality_cfg.get("walkforward_slippage_bps", 1.0)),
                    current_diagnostics_path=summary_dir / "trade_decision_diagnostics_candidate_raw.json",
                    source_diagnostics_path=(
                        selection_guard_source_summary_dir / "trade_decision_diagnostics_candidate_raw.json"
                        if selection_guard_source_summary_dir is not None
                        else None
                    ),
                    source_run_id=(
                        str(selection_guard_reference_payload.get("source_run_id"))
                        if selection_guard_reference_payload.get("source_run_id") is not None
                        else None
                    ),
                )
                trade_decision_model_shift_payload = _build_trade_decision_model_shift(
                    current_candidate_path=decision_model_input,
                    current_model_path=trade_decision_model_path,
                    current_feature_meta_path=decision_feature_meta_path,
                    source_candidate_path=(
                        selection_guard_source_summary_dir / "backtest_signals_meta_ensemble_decision_features.csv"
                        if selection_guard_source_summary_dir is not None
                        else None
                    ),
                    source_model_path=(
                        selection_guard_source_summary_dir / "trade_decision_model.json"
                        if selection_guard_source_summary_dir is not None
                        else None
                    ),
                    source_feature_meta_path=(
                        selection_guard_source_summary_dir / "backtest_signals_meta_ensemble_decision_features_meta.json"
                        if selection_guard_source_summary_dir is not None
                        else None
                    ),
                    source_run_id=(
                        str(selection_guard_reference_payload.get("source_run_id"))
                        if selection_guard_reference_payload.get("source_run_id") is not None
                        else None
                    ),
                )
                trade_decision_ablation_payload = None
                if trade_decision_ablation_model_path.exists() or args.dry_run:
                    trade_decision_ablation_payload = _build_trade_decision_ablation_comparison(
                        candidate_path=decision_model_input,
                        base_model_path=trade_decision_model_path,
                        ablation_model_path=trade_decision_ablation_model_path,
                        trade_policy_cfg=trade_decision_distribution_policy_cfg,
                        fee_bps=float(quality_cfg.get("walkforward_fee_bps", 2.0)),
                        slippage_bps=float(quality_cfg.get("walkforward_slippage_bps", 1.0)),
                    )
                (summary_dir / "trade_decision_distribution_shift.json").write_text(
                    json.dumps(trade_decision_distribution_shift_payload, indent=2),
                    encoding="utf-8",
                )
                (summary_dir / "trade_decision_model_shift.json").write_text(
                    json.dumps(trade_decision_model_shift_payload, indent=2),
                    encoding="utf-8",
                )
                if trade_decision_ablation_payload is not None:
                    (summary_dir / "trade_decision_reference_feature_ablation.json").write_text(
                        json.dumps(trade_decision_ablation_payload, indent=2),
                        encoding="utf-8",
                    )
                selection_guard_rule_payload = {
                    "enabled": bool(selection_guard_enabled and bool(selection_guard_rules)),
                    "regime_col": selection_guard_regime_col,
                    "p_col": selection_guard_p_col,
                    "manual_rules": selection_guard_manual_rules,
                    "auto_derive": selection_guard_auto_payload,
                    "reused_last_deployed": selection_guard_reuse_payload,
                    "reuse_viability": selection_guard_reuse_viability,
                    "trade_decision_distribution_shift": trade_decision_distribution_shift_payload,
                    "trade_decision_model_shift": trade_decision_model_shift_payload,
                    "trade_decision_reference_feature_ablation": trade_decision_ablation_payload,
                    "distribution_shift": _build_selection_calibration_guard_distribution_shift(
                        current_candidate_path=aligned_candidate_input,
                        source_candidate_path=selection_guard_source_candidate_path,
                        recent_window_rows=int(
                            official_shadow_selection_cfg.get(
                                "recent_window_rows",
                                calibration_cfg.get("recent_window", 120),
                            )
                        ),
                        signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        regime_col=selection_guard_regime_col,
                        p_col=selection_guard_p_col,
                        reference_rules=_normalize_selection_calibration_guard_rules(
                            selection_guard_reference_payload.get("rules", [])
                        ),
                        source_run_id=(
                            str(selection_guard_reference_payload.get("source_run_id"))
                            if selection_guard_reference_payload.get("source_run_id") is not None
                            else None
                        ),
                    ),
                    "rules": selection_guard_rules,
                }
                (summary_dir / "selection_calibration_guard_rule.json").write_text(
                    json.dumps(selection_guard_rule_payload, indent=2),
                    encoding="utf-8",
                )
                if selection_guard_enabled and selection_guard_rules and not args.dry_run:
                    _build_selection_calibration_guard_shadow(
                        input_path=aligned_candidate_input,
                        output_path=selection_guard_candidate_path,
                        meta_path=selection_guard_meta_path,
                        signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        regime_col=selection_guard_regime_col,
                        p_col=selection_guard_p_col,
                        rules=selection_guard_rules,
                    )
                if selection_guard_enabled and selection_guard_rules and (selection_guard_candidate_path.exists() or args.dry_run):
                    selection_guard_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(selection_guard_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_selection_calibration_guard_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_selection_calibration_guard_companion",
                            selection_guard_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_selection_calibration_guard_companion.log",
                            args.dry_run,
                        )
                    )
                    selection_guard_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(selection_guard_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_selection_calibration_guard.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_selection_calibration_guard",
                            selection_guard_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_selection_calibration_guard.log",
                            args.dry_run,
                        )
                    )

                threshold_variants_obj = official_shadow_selection_cfg.get("threshold_variants", [0.56, 0.57, 0.58, 0.60])
                threshold_variants: List[float] = []
                if isinstance(threshold_variants_obj, list):
                    for raw_value in threshold_variants_obj:
                        try:
                            threshold_variants.append(float(raw_value))
                        except (TypeError, ValueError):
                            continue
                elif isinstance(threshold_variants_obj, str):
                    for raw_value in threshold_variants_obj.split(","):
                        try:
                            threshold_variants.append(float(raw_value.strip()))
                        except (TypeError, ValueError):
                            continue
                threshold_variant_artifacts: Dict[str, Dict[str, Path]] = {}
                seen_threshold_variants: set[str] = set()
                for threshold_variant in threshold_variants:
                    if abs(float(threshold_variant) - float(resolved_trade_decision_threshold)) < 1e-9:
                        continue
                    variant_name = _format_threshold_variant_name(float(threshold_variant))
                    if variant_name in seen_threshold_variants:
                        continue
                    seen_threshold_variants.add(variant_name)
                    threshold_shadow_candidate_path = (
                        summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{variant_name}.csv"
                    )
                    threshold_shadow_meta_path = (
                        summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{variant_name}_meta.json"
                    )
                    threshold_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(threshold_shadow_candidate_path),
                        "--meta-output",
                        str(threshold_shadow_meta_path),
                    ]
                    threshold_shadow_build_cmd.extend(
                        _override_cli_arg(common_align_args, "--threshold", str(float(threshold_variant)))
                    )
                    if quality_input.exists() or args.dry_run:
                        threshold_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        threshold_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    results.append(
                        _run_step(
                            f"build_shadow_{variant_name}",
                            threshold_shadow_build_cmd,
                            logs_dir / f"build_shadow_{variant_name}.log",
                            args.dry_run,
                        )
                    )

                    threshold_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(threshold_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / f"champion_challenger_policy_aligned_shadow_{variant_name}_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            f"champion_challenger_policy_aligned_shadow_{variant_name}_companion",
                            threshold_shadow_companion_cmd,
                            logs_dir / f"champion_challenger_policy_aligned_shadow_{variant_name}_companion.log",
                            args.dry_run,
                        )
                    )

                    threshold_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(threshold_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}.json"),
                    ]
                    results.append(
                        _run_step(
                            f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}",
                            threshold_shadow_overlap_cmd,
                            logs_dir / f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}.log",
                            args.dry_run,
                        )
                    )
                    threshold_variant_artifacts[variant_name] = {
                        "candidate_path": threshold_shadow_candidate_path,
                        "meta_path": threshold_shadow_meta_path,
                        "companion_path": summary_dir / f"champion_challenger_policy_aligned_shadow_{variant_name}_companion.json",
                    }

                reference_feature_ablation_shadow_candidate_path = (
                    summary_dir
                    / "backtest_signals_meta_ensemble_decision_aligned_shadow_reference_feature_ablation.csv"
                )
                reference_feature_ablation_shadow_meta_path = (
                    summary_dir
                    / "backtest_signals_meta_ensemble_decision_aligned_shadow_reference_feature_ablation_meta.json"
                )
                reference_feature_ablation_shadow_companion_path = (
                    summary_dir
                    / "champion_challenger_policy_aligned_shadow_reference_feature_ablation_companion.json"
                )
                if trade_decision_ablation_model_path.exists() or args.dry_run:
                    reference_feature_ablation_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_ablation_model_path),
                        "--output",
                        str(reference_feature_ablation_shadow_candidate_path),
                        "--meta-output",
                        str(reference_feature_ablation_shadow_meta_path),
                    ]
                    reference_feature_ablation_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        reference_feature_ablation_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        reference_feature_ablation_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    results.append(
                        _run_step(
                            "build_shadow_reference_feature_ablation",
                            reference_feature_ablation_shadow_build_cmd,
                            logs_dir / "build_shadow_reference_feature_ablation.log",
                            args.dry_run,
                        )
                    )

                    reference_feature_ablation_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(reference_feature_ablation_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(reference_feature_ablation_shadow_companion_path),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_reference_feature_ablation_companion",
                            reference_feature_ablation_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_reference_feature_ablation_companion.log",
                            args.dry_run,
                        )
                    )

                    reference_feature_ablation_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(reference_feature_ablation_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_reference_feature_ablation.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_reference_feature_ablation",
                            reference_feature_ablation_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_reference_feature_ablation.log",
                            args.dry_run,
                        )
                    )

                reference_feature_ablation_cfg = (
                    trade_decision_cfg.get("reference_feature_ablation")
                    if isinstance(trade_decision_cfg.get("reference_feature_ablation"), dict)
                    else {}
                )
                ablation_threshold_variants_obj = reference_feature_ablation_cfg.get(
                    "threshold_variants",
                    official_shadow_selection_cfg.get("threshold_variants", [0.56, 0.57, 0.58, 0.60]),
                )
                ablation_threshold_variants: List[float] = []
                if isinstance(ablation_threshold_variants_obj, list):
                    for raw_value in ablation_threshold_variants_obj:
                        try:
                            ablation_threshold_variants.append(float(raw_value))
                        except (TypeError, ValueError):
                            continue
                elif isinstance(ablation_threshold_variants_obj, str):
                    for raw_value in ablation_threshold_variants_obj.split(","):
                        try:
                            ablation_threshold_variants.append(float(raw_value.strip()))
                        except (TypeError, ValueError):
                            continue
                reference_feature_ablation_threshold_artifacts: Dict[str, Dict[str, Path]] = {}
                seen_ablation_threshold_variants: set[str] = set()
                for threshold_variant in ablation_threshold_variants:
                    if abs(float(threshold_variant) - float(resolved_trade_decision_threshold)) < 1e-9:
                        continue
                    variant_name = _format_reference_feature_ablation_threshold_variant_name(float(threshold_variant))
                    if variant_name in seen_ablation_threshold_variants:
                        continue
                    seen_ablation_threshold_variants.add(variant_name)
                    threshold_shadow_candidate_path = (
                        summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{variant_name}.csv"
                    )
                    threshold_shadow_meta_path = (
                        summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{variant_name}_meta.json"
                    )
                    if trade_decision_ablation_model_path.exists() or args.dry_run:
                        threshold_shadow_build_cmd = [
                            python,
                            "-m",
                            "src.scripts.apply_trade_decision_policy_to_backtest",
                            "--input",
                            str(companion_candidate_input),
                            "--model",
                            str(trade_decision_ablation_model_path),
                            "--output",
                            str(threshold_shadow_candidate_path),
                            "--meta-output",
                            str(threshold_shadow_meta_path),
                        ]
                        threshold_shadow_build_cmd.extend(
                            _override_cli_arg(common_align_args, "--threshold", str(float(threshold_variant)))
                        )
                        if quality_input.exists() or args.dry_run:
                            threshold_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                        if incumbent_backtest_for_features:
                            threshold_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                        results.append(
                            _run_step(
                                f"build_shadow_{variant_name}",
                                threshold_shadow_build_cmd,
                                logs_dir / f"build_shadow_{variant_name}.log",
                                args.dry_run,
                            )
                        )

                        threshold_shadow_companion_path = (
                            summary_dir / f"champion_challenger_policy_aligned_shadow_{variant_name}_companion.json"
                        )
                        threshold_shadow_companion_cmd = [
                            python,
                            "-m",
                            "src.scripts.evaluate_champion_challenger",
                            "--baseline",
                            str(companion_baseline_input),
                            "--candidate",
                            str(threshold_shadow_candidate_path),
                            "--baseline-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--n-boot",
                            str(int(champ_cfg.get("n_boot", 2000))),
                            "--alpha",
                            str(float(champ_cfg.get("alpha", 0.05))),
                            "--seed",
                            str(int(champ_cfg.get("seed", 42))),
                            "--output",
                            str(threshold_shadow_companion_path),
                        ]
                        results.append(
                            _run_step(
                                f"champion_challenger_policy_aligned_shadow_{variant_name}_companion",
                                threshold_shadow_companion_cmd,
                                logs_dir / f"champion_challenger_policy_aligned_shadow_{variant_name}_companion.log",
                                args.dry_run,
                            )
                        )

                        threshold_shadow_overlap_cmd = [
                            python,
                            "-m",
                            "src.scripts.analyze_paired_trigger_overlap",
                            "--candidate",
                            str(threshold_shadow_candidate_path),
                            "--incumbent",
                            str(companion_baseline_input),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--incumbent-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--output",
                            str(summary_dir / f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}.json"),
                        ]
                        results.append(
                            _run_step(
                                f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}",
                                threshold_shadow_overlap_cmd,
                                logs_dir / f"paired_trigger_overlap_policy_aligned_shadow_{variant_name}.log",
                                args.dry_run,
                            )
                        )
                        reference_feature_ablation_threshold_artifacts[variant_name] = {
                            "candidate_path": threshold_shadow_candidate_path,
                            "meta_path": threshold_shadow_meta_path,
                            "companion_path": threshold_shadow_companion_path,
                        }

                reference_feature_ablation_selection_guard_cfg_obj = reference_feature_ablation_cfg.get(
                    "selection_calibration_guard",
                    {},
                )
                reference_feature_ablation_selection_guard_cfg = (
                    reference_feature_ablation_selection_guard_cfg_obj
                    if isinstance(reference_feature_ablation_selection_guard_cfg_obj, dict)
                    else {}
                )
                reference_feature_ablation_selection_guard_artifacts: Dict[str, Dict[str, Path]] = {}
                if bool(reference_feature_ablation_selection_guard_cfg.get("enabled", False)):
                    source_threshold = _safe_float(
                        reference_feature_ablation_selection_guard_cfg.get("source_threshold", 0.555),
                        default=float("nan"),
                    )
                    if np.isfinite(source_threshold):
                        source_variant_name = _format_reference_feature_ablation_threshold_variant_name(source_threshold)
                        source_artifacts = reference_feature_ablation_threshold_artifacts.get(source_variant_name)
                        if source_artifacts is not None:
                            selection_guard_variant_name = _format_reference_feature_ablation_selection_guard_variant_name(
                                source_threshold
                            )
                            selection_guard_candidate_path = (
                                summary_dir
                                / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{selection_guard_variant_name}.csv"
                            )
                            selection_guard_meta_path = (
                                summary_dir
                                / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{selection_guard_variant_name}_meta.json"
                            )
                            selection_guard_companion_path = (
                                summary_dir
                                / f"champion_challenger_policy_aligned_shadow_{selection_guard_variant_name}_companion.json"
                            )
                            selection_guard_rules = _normalize_selection_calibration_guard_rules(
                                reference_feature_ablation_selection_guard_cfg.get("rules", [])
                            )
                            selection_guard_rule_path = (
                                summary_dir / f"{selection_guard_variant_name}_rule.json"
                            )
                            selection_guard_rule_payload = {
                                "enabled": bool(selection_guard_rules),
                                "source_variant": source_variant_name,
                                "source_threshold": source_threshold,
                                "rules": selection_guard_rules,
                                "reason": "configured_rules" if selection_guard_rules else "no_rules_configured",
                            }
                            selection_guard_rule_path.write_text(
                                json.dumps(selection_guard_rule_payload, indent=2),
                                encoding="utf-8",
                            )
                            if selection_guard_rules and (source_artifacts["candidate_path"].exists() or args.dry_run):
                                if not args.dry_run:
                                    _build_selection_calibration_guard_shadow(
                                        input_path=source_artifacts["candidate_path"],
                                        output_path=selection_guard_candidate_path,
                                        meta_path=selection_guard_meta_path,
                                        signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                        return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                        regime_col=str(
                                            reference_feature_ablation_selection_guard_cfg.get(
                                                "regime_col",
                                                selection_guard_regime_col,
                                            )
                                        ),
                                        p_col=str(
                                            reference_feature_ablation_selection_guard_cfg.get(
                                                "p_col",
                                                selection_guard_p_col,
                                            )
                                        ),
                                        rules=selection_guard_rules,
                                    )
                                reference_feature_ablation_selection_guard_companion_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.evaluate_champion_challenger",
                                    "--baseline",
                                    str(companion_baseline_input),
                                    "--candidate",
                                    str(selection_guard_candidate_path),
                                    "--baseline-col",
                                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                                    "--candidate-col",
                                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                    "--n-boot",
                                    str(int(champ_cfg.get("n_boot", 2000))),
                                    "--alpha",
                                    str(float(champ_cfg.get("alpha", 0.05))),
                                    "--seed",
                                    str(int(champ_cfg.get("seed", 42))),
                                    "--output",
                                    str(selection_guard_companion_path),
                                ]
                                results.append(
                                    _run_step(
                                        f"champion_challenger_policy_aligned_shadow_{selection_guard_variant_name}_companion",
                                        reference_feature_ablation_selection_guard_companion_cmd,
                                        logs_dir
                                        / f"champion_challenger_policy_aligned_shadow_{selection_guard_variant_name}_companion.log",
                                        args.dry_run,
                                    )
                                )

                                reference_feature_ablation_selection_guard_overlap_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.analyze_paired_trigger_overlap",
                                    "--candidate",
                                    str(selection_guard_candidate_path),
                                    "--incumbent",
                                    str(companion_baseline_input),
                                    "--candidate-col",
                                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                    "--incumbent-col",
                                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                                    "--signal-col",
                                    str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                    "--output",
                                    str(summary_dir / f"paired_trigger_overlap_policy_aligned_shadow_{selection_guard_variant_name}.json"),
                                ]
                                results.append(
                                    _run_step(
                                        f"paired_trigger_overlap_policy_aligned_shadow_{selection_guard_variant_name}",
                                        reference_feature_ablation_selection_guard_overlap_cmd,
                                        logs_dir
                                        / f"paired_trigger_overlap_policy_aligned_shadow_{selection_guard_variant_name}.log",
                                        args.dry_run,
                                    )
                                )
                                reference_feature_ablation_selection_guard_artifacts[selection_guard_variant_name] = {
                                    "candidate_path": selection_guard_candidate_path,
                                    "meta_path": selection_guard_meta_path,
                                    "companion_path": selection_guard_companion_path,
                                }

                reference_feature_ablation_ranking_cfg_obj = reference_feature_ablation_cfg.get(
                    "neutral_abs_ret_pred_floor",
                    {},
                )
                reference_feature_ablation_ranking_cfg = (
                    reference_feature_ablation_ranking_cfg_obj
                    if isinstance(reference_feature_ablation_ranking_cfg_obj, dict)
                    else {}
                )
                reference_feature_ablation_ranking_artifacts: Dict[str, Dict[str, Path]] = {}
                if bool(reference_feature_ablation_ranking_cfg.get("enabled", False)):
                    source_threshold = _safe_float(
                        reference_feature_ablation_ranking_cfg.get("source_threshold", 0.555),
                        default=float("nan"),
                    )
                    min_abs_ret_pred = _safe_float(
                        reference_feature_ablation_ranking_cfg.get("min_abs_ret_pred", 0.00212),
                        default=float("nan"),
                    )
                    if np.isfinite(source_threshold) and np.isfinite(min_abs_ret_pred):
                        source_variant_name = _format_reference_feature_ablation_threshold_variant_name(source_threshold)
                        source_artifacts = reference_feature_ablation_threshold_artifacts.get(source_variant_name)
                        if source_artifacts is not None:
                            ranking_variant_name = _format_reference_feature_ablation_abs_ret_pred_variant_name(
                                source_threshold,
                                min_abs_ret_pred,
                            )
                            ranking_candidate_path = (
                                summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{ranking_variant_name}.csv"
                            )
                            ranking_meta_path = (
                                summary_dir / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{ranking_variant_name}_meta.json"
                            )
                            ranking_companion_path = (
                                summary_dir / f"champion_challenger_policy_aligned_shadow_{ranking_variant_name}_companion.json"
                            )
                            ranking_rule_path = summary_dir / f"{ranking_variant_name}_rule.json"
                            ranking_rule_payload = {
                                "enabled": True,
                                "source_variant": source_variant_name,
                                "source_threshold": source_threshold,
                                "regime_state": str(reference_feature_ablation_ranking_cfg.get("regime_state", "neutral")),
                                "ret_pred_col": str(reference_feature_ablation_ranking_cfg.get("ret_pred_col", "ret_pred")),
                                "min_abs_ret_pred": float(min_abs_ret_pred),
                            }
                            ranking_rule_path.write_text(json.dumps(ranking_rule_payload, indent=2), encoding="utf-8")
                            if source_artifacts["candidate_path"].exists() or args.dry_run:
                                if not args.dry_run:
                                    _build_regime_abs_ret_pred_floor_shadow(
                                        input_path=source_artifacts["candidate_path"],
                                        output_path=ranking_candidate_path,
                                        meta_path=ranking_meta_path,
                                        signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                        return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                        regime_col=str(reference_feature_ablation_ranking_cfg.get("regime_col", "regime_state")),
                                        ret_pred_col=str(reference_feature_ablation_ranking_cfg.get("ret_pred_col", "ret_pred")),
                                        regime_state=str(reference_feature_ablation_ranking_cfg.get("regime_state", "neutral")),
                                        min_abs_ret_pred=float(min_abs_ret_pred),
                                    )
                                ranking_companion_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.evaluate_champion_challenger",
                                    "--baseline",
                                    str(companion_baseline_input),
                                    "--candidate",
                                    str(ranking_candidate_path),
                                    "--baseline-col",
                                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                                    "--candidate-col",
                                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                    "--n-boot",
                                    str(int(champ_cfg.get("n_boot", 2000))),
                                    "--alpha",
                                    str(float(champ_cfg.get("alpha", 0.05))),
                                    "--seed",
                                    str(int(champ_cfg.get("seed", 42))),
                                    "--output",
                                    str(ranking_companion_path),
                                ]
                                results.append(
                                    _run_step(
                                        f"champion_challenger_policy_aligned_shadow_{ranking_variant_name}_companion",
                                        ranking_companion_cmd,
                                        logs_dir / f"champion_challenger_policy_aligned_shadow_{ranking_variant_name}_companion.log",
                                        args.dry_run,
                                    )
                                )
                                ranking_overlap_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.analyze_paired_trigger_overlap",
                                    "--candidate",
                                    str(ranking_candidate_path),
                                    "--incumbent",
                                    str(companion_baseline_input),
                                    "--candidate-col",
                                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                    "--incumbent-col",
                                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                                    "--signal-col",
                                    str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                    "--output",
                                    str(summary_dir / f"paired_trigger_overlap_policy_aligned_shadow_{ranking_variant_name}.json"),
                                ]
                                results.append(
                                    _run_step(
                                        f"paired_trigger_overlap_policy_aligned_shadow_{ranking_variant_name}",
                                        ranking_overlap_cmd,
                                        logs_dir / f"paired_trigger_overlap_policy_aligned_shadow_{ranking_variant_name}.log",
                                        args.dry_run,
                                    )
                                )
                                reference_feature_ablation_ranking_artifacts[ranking_variant_name] = {
                                    "candidate_path": ranking_candidate_path,
                                    "meta_path": ranking_meta_path,
                                    "companion_path": ranking_companion_path,
                                }

                reference_feature_ablation_neutral_p_up_cap_cfg_obj = reference_feature_ablation_cfg.get(
                    "neutral_p_up_cap",
                    {},
                )
                reference_feature_ablation_neutral_p_up_cap_cfg = (
                    reference_feature_ablation_neutral_p_up_cap_cfg_obj
                    if isinstance(reference_feature_ablation_neutral_p_up_cap_cfg_obj, dict)
                    else {}
                )
                reference_feature_ablation_neutral_p_up_cap_artifacts: Dict[str, Dict[str, Path]] = {}
                if bool(reference_feature_ablation_neutral_p_up_cap_cfg.get("enabled", False)):
                    source_threshold = _safe_float(
                        reference_feature_ablation_neutral_p_up_cap_cfg.get("source_threshold", 0.555),
                        default=float("nan"),
                    )
                    max_p_up_exclusive = _safe_float(
                        reference_feature_ablation_neutral_p_up_cap_cfg.get("max_p_up_exclusive", 0.499),
                        default=float("nan"),
                    )
                    if np.isfinite(source_threshold) and np.isfinite(max_p_up_exclusive):
                        source_variant_name = _format_reference_feature_ablation_threshold_variant_name(source_threshold)
                        source_artifacts = reference_feature_ablation_threshold_artifacts.get(source_variant_name)
                        if source_artifacts is not None:
                            neutral_p_up_cap_variant_name = _format_reference_feature_ablation_neutral_p_up_cap_variant_name(
                                source_threshold,
                                max_p_up_exclusive,
                            )
                            neutral_p_up_cap_candidate_path = (
                                summary_dir
                                / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{neutral_p_up_cap_variant_name}.csv"
                            )
                            neutral_p_up_cap_meta_path = (
                                summary_dir
                                / f"backtest_signals_meta_ensemble_decision_aligned_shadow_{neutral_p_up_cap_variant_name}_meta.json"
                            )
                            neutral_p_up_cap_companion_path = (
                                summary_dir
                                / f"champion_challenger_policy_aligned_shadow_{neutral_p_up_cap_variant_name}_companion.json"
                            )
                            neutral_p_up_cap_rule_path = summary_dir / f"{neutral_p_up_cap_variant_name}_rule.json"
                            neutral_p_up_cap_rule_payload = {
                                "enabled": True,
                                "source_variant": source_variant_name,
                                "source_threshold": source_threshold,
                                "regime_state": str(reference_feature_ablation_neutral_p_up_cap_cfg.get("regime_state", "neutral")),
                                "p_col": str(reference_feature_ablation_neutral_p_up_cap_cfg.get("p_col", "p_up")),
                                "max_p_up_exclusive": float(max_p_up_exclusive),
                            }
                            neutral_p_up_cap_rule_path.write_text(
                                json.dumps(neutral_p_up_cap_rule_payload, indent=2),
                                encoding="utf-8",
                            )
                            if source_artifacts["candidate_path"].exists() or args.dry_run:
                                if not args.dry_run:
                                    _build_regime_max_p_up_shadow(
                                        input_path=source_artifacts["candidate_path"],
                                        output_path=neutral_p_up_cap_candidate_path,
                                        meta_path=neutral_p_up_cap_meta_path,
                                        signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                        return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                        regime_col=str(reference_feature_ablation_neutral_p_up_cap_cfg.get("regime_col", "regime_state")),
                                        p_col=str(reference_feature_ablation_neutral_p_up_cap_cfg.get("p_col", "p_up")),
                                        regime_state=str(reference_feature_ablation_neutral_p_up_cap_cfg.get("regime_state", "neutral")),
                                        max_p_up_exclusive=float(max_p_up_exclusive),
                                    )
                                neutral_p_up_cap_companion_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.evaluate_champion_challenger",
                                    "--baseline",
                                    str(companion_baseline_input),
                                    "--candidate",
                                    str(neutral_p_up_cap_candidate_path),
                                    "--baseline-col",
                                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                                    "--candidate-col",
                                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                    "--n-boot",
                                    str(int(champ_cfg.get("n_boot", 2000))),
                                    "--alpha",
                                    str(float(champ_cfg.get("alpha", 0.05))),
                                    "--seed",
                                    str(int(champ_cfg.get("seed", 42))),
                                    "--output",
                                    str(neutral_p_up_cap_companion_path),
                                ]
                                results.append(
                                    _run_step(
                                        f"champion_challenger_policy_aligned_shadow_{neutral_p_up_cap_variant_name}_companion",
                                        neutral_p_up_cap_companion_cmd,
                                        logs_dir / f"champion_challenger_policy_aligned_shadow_{neutral_p_up_cap_variant_name}_companion.log",
                                        args.dry_run,
                                    )
                                )
                                neutral_p_up_cap_overlap_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.analyze_paired_trigger_overlap",
                                    "--candidate",
                                    str(neutral_p_up_cap_candidate_path),
                                    "--incumbent",
                                    str(companion_baseline_input),
                                    "--candidate-col",
                                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                    "--incumbent-col",
                                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                                    "--signal-col",
                                    str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                    "--output",
                                    str(summary_dir / f"paired_trigger_overlap_policy_aligned_shadow_{neutral_p_up_cap_variant_name}.json"),
                                ]
                                results.append(
                                    _run_step(
                                        f"paired_trigger_overlap_policy_aligned_shadow_{neutral_p_up_cap_variant_name}",
                                        neutral_p_up_cap_overlap_cmd,
                                        logs_dir / f"paired_trigger_overlap_policy_aligned_shadow_{neutral_p_up_cap_variant_name}.log",
                                        args.dry_run,
                                    )
                                )
                                reference_feature_ablation_neutral_p_up_cap_artifacts[neutral_p_up_cap_variant_name] = {
                                    "candidate_path": neutral_p_up_cap_candidate_path,
                                    "meta_path": neutral_p_up_cap_meta_path,
                                    "companion_path": neutral_p_up_cap_companion_path,
                                }

                official_shadow_candidates = {
                    REFERENCE_FEATURE_ABLATION_VARIANT: (
                        reference_feature_ablation_shadow_candidate_path,
                        reference_feature_ablation_shadow_meta_path,
                    ),
                    "weak_band": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto_meta.json",
                    ),
                    "refined": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto_meta.json",
                    ),
                    "midband": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto_meta.json",
                    ),
                    "raw_ev_sign": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto_meta.json",
                    ),
                    "direction_alignment": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto_meta.json",
                    ),
                    "joint_direction_midband": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto_meta.json",
                    ),
                    "regime_state": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto_meta.json",
                    ),
                    "chop_high_volatility": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto_meta.json",
                    ),
                    "volatility_only": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto_meta.json",
                    ),
                    "triggered_regime_volatility": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_triggered_regime_volatility_veto.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_triggered_regime_volatility_veto_meta.json",
                    ),
                    "selection_calibration_guard": (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard.csv",
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard_meta.json",
                    ),
                }
                official_shadow_candidates.update(
                    {
                        variant_name: (paths["candidate_path"], paths["meta_path"])
                        for variant_name, paths in threshold_variant_artifacts.items()
                    }
                )
                official_shadow_artifacts: Dict[str, Dict[str, Path]] = {
                    "none": {
                        "candidate_path": aligned_candidate_input,
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_companion.json",
                    },
                    REFERENCE_FEATURE_ABLATION_VARIANT: {
                        "candidate_path": reference_feature_ablation_shadow_candidate_path,
                        "meta_path": reference_feature_ablation_shadow_meta_path,
                        "companion_path": reference_feature_ablation_shadow_companion_path,
                    },
                    "selection_calibration_guard": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_selection_calibration_guard_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_selection_calibration_guard_companion.json",
                    },
                    "weak_band": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_veto_companion.json",
                    },
                    "refined": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_refined_veto_companion.json",
                    },
                    "midband": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_midband_veto_companion.json",
                    },
                    "raw_ev_sign": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion.json",
                    },
                    "direction_alignment": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_direction_align_veto_companion.json",
                    },
                    "joint_direction_midband": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion.json",
                    },
                    "regime_state": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_regime_state_veto_companion.json",
                    },
                    "chop_high_volatility": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion.json",
                    },
                    "volatility_only": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion.json",
                    },
                    "triggered_regime_volatility": {
                        "candidate_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_triggered_regime_volatility_veto.csv",
                        "meta_path": summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_triggered_regime_volatility_veto_meta.json",
                        "companion_path": summary_dir / "champion_challenger_policy_aligned_shadow_triggered_regime_volatility_veto_companion.json",
                    },
                }
                official_shadow_artifacts.update(threshold_variant_artifacts)
                official_shadow_artifacts.update(reference_feature_ablation_threshold_artifacts)
                official_shadow_artifacts.update(reference_feature_ablation_selection_guard_artifacts)
                official_shadow_artifacts.update(reference_feature_ablation_ranking_artifacts)
                official_shadow_artifacts.update(reference_feature_ablation_neutral_p_up_cap_artifacts)
                if official_shadow_variant == "auto" and not args.dry_run:
                    calibration_horizon_key = str(quality_cfg.get("calibration_horizon", "1h"))
                    recent_window_rows = int(
                        official_shadow_selection_cfg.get(
                            "recent_window_rows",
                            calibration_cfg.get("recent_window", 120),
                        )
                    )
                    min_candidate_trades = int(official_shadow_selection_cfg.get("min_candidate_trades", 5))
                    require_companion_promote = bool(
                        official_shadow_selection_cfg.get("require_companion_promote", True)
                    )
                    require_nonnegative_recent_delta = bool(
                        official_shadow_selection_cfg.get("require_nonnegative_recent_delta", True)
                    )
                    require_nonnegative_rolling_delta = bool(
                        official_shadow_selection_cfg.get("require_nonnegative_rolling_delta", True)
                    )
                    require_candidate_not_dominated = bool(
                        official_shadow_selection_cfg.get("require_candidate_not_dominated", True)
                    )
                    require_recent_calibration_ok = bool(
                        official_shadow_selection_cfg.get("require_recent_calibration_ok", True)
                    )
                    prefer_calibration_stability_within_eligible = bool(
                        official_shadow_selection_cfg.get("prefer_calibration_stability_within_eligible", True)
                    )
                    fallback_variant_when_no_eligible = str(
                        official_shadow_selection_cfg.get("fallback_variant_when_no_eligible", "none")
                    ).strip().lower()
                    if fallback_variant_when_no_eligible not in official_shadow_artifacts:
                        fallback_variant_when_no_eligible = "none"
                    selection_payload: Dict[str, Any] = {
                        "selected_variant": "none",
                        "criteria": {
                            "recent_window_rows": recent_window_rows,
                            "min_candidate_trades": min_candidate_trades,
                            "require_companion_promote": require_companion_promote,
                            "require_nonnegative_recent_delta": require_nonnegative_recent_delta,
                            "require_nonnegative_rolling_delta": require_nonnegative_rolling_delta,
                            "require_candidate_not_dominated": require_candidate_not_dominated,
                            "require_recent_calibration_ok": require_recent_calibration_ok,
                            "prefer_calibration_stability_within_eligible": prefer_calibration_stability_within_eligible,
                            "fallback_variant_when_no_eligible": fallback_variant_when_no_eligible,
                            "calibration_horizon": calibration_horizon_key,
                            "calibration_selection_scope": str(calibration_cfg.get("selection_scope", "all")),
                        },
                        "candidates": [],
                    }
                    best_variant = fallback_variant_when_no_eligible
                    best_score: tuple[Any, ...] | None = None
                    best_ineligible_variant = fallback_variant_when_no_eligible
                    best_ineligible_score: tuple[Any, ...] | None = None
                    eligible_variant_found = False
                    for variant_name, artifact_paths in official_shadow_artifacts.items():
                        variant_candidate_path = artifact_paths["candidate_path"]
                        variant_meta_path = artifact_paths["meta_path"]
                        variant_companion_path = artifact_paths["companion_path"]
                        if not variant_candidate_path.exists():
                            continue
                        rolling_variant_json = summary_dir / f"rolling_ab_policy_aligned_{variant_name}.json"
                        rolling_variant_md = summary_dir / f"rolling_ab_policy_aligned_{variant_name}.md"
                        rolling_variant_cmd = _build_rolling_ab_command(
                            python=python,
                            baseline_path=companion_baseline_input,
                            candidate_path=variant_candidate_path,
                            rolling_cfg=rolling_ab_cfg,
                            output_path=rolling_variant_json,
                            output_md_path=rolling_variant_md,
                        )
                        results.append(
                            _run_step(
                                f"rolling_ab_policy_aligned_{variant_name}",
                                rolling_variant_cmd,
                                logs_dir / f"rolling_ab_policy_aligned_{variant_name}.log",
                                args.dry_run,
                            )
                        )
                        calibration_variant_json = summary_dir / f"calibration_robustness_policy_aligned_{variant_name}.json"
                        calibration_variant_cmd = _build_calibration_robustness_command(
                            python=python,
                            input_path=variant_candidate_path,
                            output_path=calibration_variant_json,
                            calibration_cfg=calibration_cfg,
                            quality_cfg=quality_cfg,
                            trade_decision_cfg=trade_decision_cfg,
                        )
                        results.append(
                            _run_step(
                                f"calibration_robustness_policy_aligned_{variant_name}",
                                calibration_variant_cmd,
                                logs_dir / f"calibration_robustness_policy_aligned_{variant_name}.log",
                                args.dry_run,
                            )
                        )
                        rolling_payload = _load_json(rolling_variant_json)
                        calibration_variant_payload = _load_json(calibration_variant_json)
                        companion_payload = _load_json(variant_companion_path) if variant_companion_path.exists() else {}
                        meta_payload = _load_json(variant_meta_path) if variant_meta_path.exists() else {}
                        calibration_variant_recent = _extract_recent_calibration_payload(
                            calibration_variant_payload,
                            horizon_key=calibration_horizon_key,
                        )
                        calibration_hardening = (
                            calibration_variant_recent.get("promotion_hardening", {})
                            if isinstance(calibration_variant_recent.get("promotion_hardening", {}), dict)
                            else {}
                        )
                        recent_payload = _compute_recent_policy_slice_metrics(
                            baseline_path=Path(str(companion_baseline_input)),
                            candidate_path=variant_candidate_path,
                            recent_window_rows=recent_window_rows,
                            signal_col=str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            return_col=str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        )
                        rolling_summary = rolling_payload.get("rolling_summary", {}) if isinstance(rolling_payload, dict) else {}
                        rolling_overall = rolling_payload.get("overall", {}) if isinstance(rolling_payload, dict) else {}
                        companion_stats = companion_payload.get("stats", {}) if isinstance(companion_payload, dict) else {}
                        candidate_wins = int(rolling_summary.get("candidate_wins", 0) or 0)
                        baseline_wins = int(rolling_summary.get("baseline_wins", 0) or 0)
                        rolling_delta = _safe_float(
                            rolling_overall.get("delta_cum_ret") if isinstance(rolling_overall, dict) else 0.0,
                            default=float("-inf"),
                        )
                        recent_delta = _safe_float(recent_payload.get("delta_net_return"), default=float("-inf"))
                        companion_promote = bool(companion_payload.get("promote", False))
                        companion_mean_diff = _safe_float(companion_stats.get("mean_diff"), default=float("-inf"))
                        companion_pvalue = _safe_float(companion_stats.get("pvalue_one_sided"), default=1e9)
                        candidate_trade_count = int(
                            meta_payload.get(
                                "trade_count",
                                ((recent_payload.get("candidate") or {}).get("trade_count") if isinstance(recent_payload, dict) else 0),
                            )
                            or 0
                        )
                        eligible = True
                        rejection_reasons: List[str] = []
                        if candidate_trade_count < min_candidate_trades:
                            eligible = False
                            rejection_reasons.append("candidate_trade_count_below_min")
                        if require_companion_promote and not companion_promote:
                            eligible = False
                            rejection_reasons.append("companion_not_significant")
                        if require_nonnegative_recent_delta and recent_delta < 0.0:
                            eligible = False
                            rejection_reasons.append("recent_delta_negative")
                        if require_nonnegative_rolling_delta and rolling_delta < 0.0:
                            eligible = False
                            rejection_reasons.append("rolling_delta_negative")
                        if require_candidate_not_dominated and candidate_wins < baseline_wins:
                            eligible = False
                            rejection_reasons.append("rolling_candidate_dominated")
                        if require_recent_calibration_ok and not bool(calibration_hardening.get("selection_ok", True)):
                            eligible = False
                            rejection_reasons.append("recent_calibration_failed")
                        calibration_rank = _extract_selection_scope_ranking_metrics(calibration_variant_recent)
                        if prefer_calibration_stability_within_eligible:
                            score = (
                                1 if eligible else 0,
                                1 if bool(calibration_hardening.get("selection_ok", True)) else 0,
                                -int(calibration_rank.get("failed_check_count", 0)),
                                -_safe_float(calibration_rank.get("recent_ece"), default=1e9),
                                -_safe_float(calibration_rank.get("ece_drift"), default=1e9),
                                _safe_float(calibration_rank.get("recent_auc"), default=float("-inf")),
                                int(calibration_rank.get("recent_rows", 0)),
                                recent_delta,
                                candidate_wins - baseline_wins,
                                rolling_delta,
                                1 if companion_promote else 0,
                                companion_mean_diff,
                                -companion_pvalue,
                                _safe_float(meta_payload.get("net_return_total"), default=float("-inf")),
                            )
                        else:
                            score = (
                                1 if eligible else 0,
                                1 if bool(calibration_hardening.get("selection_ok", True)) else 0,
                                recent_delta,
                                candidate_wins - baseline_wins,
                                rolling_delta,
                                1 if companion_promote else 0,
                                companion_mean_diff,
                                -companion_pvalue,
                                _safe_float(meta_payload.get("net_return_total"), default=float("-inf")),
                            )
                        ineligible_score = (
                            1 if bool(calibration_hardening.get("selection_ok", True)) else 0,
                            -int(calibration_rank.get("failed_check_count", 0)),
                            -_safe_float(calibration_rank.get("recent_ece"), default=1e9),
                            -_safe_float(calibration_rank.get("ece_drift"), default=1e9),
                            _safe_float(calibration_rank.get("recent_auc"), default=float("-inf")),
                            int(calibration_rank.get("recent_rows", 0)),
                            -len(rejection_reasons),
                            recent_delta,
                            candidate_wins - baseline_wins,
                            rolling_delta,
                            1 if companion_promote else 0,
                            companion_mean_diff,
                            -companion_pvalue,
                            _safe_float(meta_payload.get("net_return_total"), default=float("-inf")),
                        )
                        selection_payload["candidates"].append(
                            {
                                "variant": variant_name,
                                "eligible": eligible,
                                "rejection_reasons": rejection_reasons,
                                "companion": {
                                    "promote": companion_promote,
                                    "mean_diff": companion_mean_diff,
                                    "pvalue_one_sided": companion_pvalue,
                                },
                                "calibration_hardening": calibration_hardening,
                                "calibration_rank": calibration_rank,
                                "calibration_recent": calibration_variant_recent,
                                "rolling": {
                                    "candidate_wins": candidate_wins,
                                    "baseline_wins": baseline_wins,
                                    "delta_cum_ret": rolling_delta,
                                },
                                "recent": recent_payload,
                                "trade_count": candidate_trade_count,
                                "net_return_total": _safe_float(meta_payload.get("net_return_total")),
                                "score": list(score),
                                "ineligible_score": list(ineligible_score),
                            }
                        )
                        if eligible and (best_score is None or score > best_score):
                            best_score = score
                            best_variant = variant_name
                            eligible_variant_found = True
                        if best_ineligible_score is None or ineligible_score > best_ineligible_score:
                            best_ineligible_score = ineligible_score
                            best_ineligible_variant = variant_name
                    if not eligible_variant_found:
                        best_variant = fallback_variant_when_no_eligible
                    official_shadow_variant = best_variant
                    selection_payload["selected_variant"] = best_variant
                    selection_payload["eligible_variant_found"] = bool(eligible_variant_found)
                    selection_payload["best_eligible_variant"] = best_variant if eligible_variant_found else None
                    selection_payload["best_ineligible_variant"] = best_ineligible_variant
                    selection_payload["fallback_variant_used"] = bool(not eligible_variant_found)
                    (summary_dir / "official_shadow_selection.json").write_text(
                        json.dumps(selection_payload, indent=2),
                        encoding="utf-8",
                    )
                    reference_feature_ablation_candidate = next(
                        (
                            candidate
                            for candidate in selection_payload.get("candidates", [])
                            if isinstance(candidate, dict)
                            and str(candidate.get("variant", "")).strip().lower() == REFERENCE_FEATURE_ABLATION_VARIANT
                        ),
                        None,
                    )
                    if reference_feature_ablation_candidate is not None:
                        (summary_dir / "trade_decision_reference_feature_ablation_shadow_evaluation.json").write_text(
                            json.dumps(
                                {
                                    "variant": REFERENCE_FEATURE_ABLATION_VARIANT,
                                    "selected": bool(best_variant == REFERENCE_FEATURE_ABLATION_VARIANT),
                                    "trade_decision_model_path": str(trade_decision_ablation_model_path),
                                    "trade_decision_model_deploy_ready": bool(trade_decision_ablation_deploy_ready),
                                    **reference_feature_ablation_candidate,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                selected_shadow_artifact = official_shadow_artifacts.get(official_shadow_variant, {})
                selected_shadow_path = (
                    selected_shadow_artifact.get("candidate_path")
                    if isinstance(selected_shadow_artifact, dict)
                    else None
                )
                selected_shadow_meta_path = (
                    selected_shadow_artifact.get("meta_path")
                    if isinstance(selected_shadow_artifact, dict)
                    else None
                )
                selected_shadow_companion_path = (
                    selected_shadow_artifact.get("companion_path")
                    if isinstance(selected_shadow_artifact, dict)
                    else None
                )
                selected_shadow_overlap_triggered_trade_diag_path = _official_shadow_overlap_triggered_trade_diag_path(
                    summary_dir,
                    official_shadow_variant,
                )
                if not args.dry_run:
                    current_trade_decision_model_path = _resolve_trade_decision_model_path_for_variant(
                        summary_dir,
                        official_shadow_variant,
                    )
                    source_trade_decision_model_path = (
                        _resolve_trade_decision_model_path_for_variant(
                            selection_guard_source_summary_dir,
                            str(selection_guard_reference_payload.get("source_official_shadow_variant", "none")),
                        )
                        if selection_guard_source_summary_dir is not None
                        else None
                    )
                    source_trade_decision_candidate_path = (
                        selection_guard_source_summary_dir / "backtest_signals_meta_ensemble_decision_features.csv"
                        if selection_guard_source_summary_dir is not None
                        else None
                    )
                    source_trade_decision_feature_meta_path = (
                        selection_guard_source_summary_dir / "backtest_signals_meta_ensemble_decision_features_meta.json"
                        if selection_guard_source_summary_dir is not None
                        else None
                    )
                    current_reference_source = _extract_trade_decision_reference_source(decision_feature_meta_path)
                    source_reference_source = _extract_trade_decision_reference_source(source_trade_decision_feature_meta_path)
                    if (
                        current_reference_source
                        and selection_guard_source_summary_dir is not None
                        and source_reference_source != current_reference_source
                    ):
                        source_labeled_snapshot_path = selection_guard_source_summary_dir / "labeled_backtest.snapshot.csv"
                        reconstructed_source_candidate_path = (
                            summary_dir / "source_trade_decision_model_shift_decision_features.csv"
                        )
                        reconstructed_source_feature_meta_path = (
                            summary_dir / "source_trade_decision_model_shift_decision_features_meta.json"
                        )
                        if source_labeled_snapshot_path.exists():
                            rebuild_source_feature_cmd = [
                                python,
                                "-m",
                                "src.scripts.enrich_backtest_with_decision_features",
                                "--input",
                                str(source_labeled_snapshot_path),
                                "--output",
                                str(reconstructed_source_candidate_path),
                                "--meta-output",
                                str(reconstructed_source_feature_meta_path),
                                "--auto-discover-sources",
                                "--feature-source",
                                str(source_labeled_snapshot_path),
                                "--feature-source",
                                str(current_reference_source),
                                "--incumbent-reference-source",
                                str(current_reference_source),
                            ]
                            results.append(
                                _run_step(
                                    "rebuild_source_trade_decision_features_model_shift",
                                    rebuild_source_feature_cmd,
                                    logs_dir / "rebuild_source_trade_decision_features_model_shift.log",
                                    args.dry_run,
                                )
                            )
                            if reconstructed_source_candidate_path.exists() and reconstructed_source_feature_meta_path.exists():
                                source_trade_decision_candidate_path = reconstructed_source_candidate_path
                                source_trade_decision_feature_meta_path = reconstructed_source_feature_meta_path
                    if current_trade_decision_model_path.exists() and source_trade_decision_model_path is not None:
                        trade_decision_model_shift_payload = _build_trade_decision_model_shift(
                            current_candidate_path=decision_model_input,
                            current_model_path=current_trade_decision_model_path,
                            current_feature_meta_path=decision_feature_meta_path,
                            source_candidate_path=source_trade_decision_candidate_path,
                            source_model_path=source_trade_decision_model_path,
                            source_feature_meta_path=source_trade_decision_feature_meta_path,
                            source_run_id=(
                                str(selection_guard_reference_payload.get("source_run_id"))
                                if selection_guard_reference_payload.get("source_run_id") is not None
                                else None
                            ),
                        )
                        (summary_dir / "trade_decision_model_shift.json").write_text(
                            json.dumps(trade_decision_model_shift_payload, indent=2),
                            encoding="utf-8",
                        )
                if selected_shadow_path is not None and (selected_shadow_path.exists() or args.dry_run):
                    candidate_input = str(selected_shadow_path)
                    candidate_gate_input = selected_shadow_path

                    official_quality_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_model_quality",
                        "--input",
                        str(selected_shadow_path),
                        "--output",
                        str(candidate_quality_path),
                    ]
                    results.append(
                        _run_step(
                            "model_quality_eval_official_shadow",
                            official_quality_cmd,
                            logs_dir / "model_quality_eval_official_shadow.log",
                            args.dry_run,
                        )
                    )

                if bool(rolling_ab_cfg.get("enabled", False)):
                    current_candidate_for_rolling = (
                        selected_shadow_path if selected_shadow_path is not None else Path(str(candidate_input))
                    )
                    if current_candidate_for_rolling.exists() or args.dry_run:
                        rolling_cmd = _build_rolling_ab_command(
                            python=python,
                            baseline_path=companion_baseline_input,
                            candidate_path=current_candidate_for_rolling,
                            rolling_cfg=rolling_ab_cfg,
                            output_path=summary_dir / "rolling_ab_report.json",
                            output_md_path=summary_dir / "rolling_ab_report.md",
                        )
                        results.append(
                            _run_step(
                                "rolling_ab_report",
                                rolling_cmd,
                                logs_dir / "rolling_ab_report.log",
                                args.dry_run,
                            )
                        )

                    if selected_shadow_path is not None and (selected_shadow_path.exists() or args.dry_run):
                        selected_policy_aligned_companion_path = (
                            selected_shadow_companion_path
                            if selected_shadow_companion_path is not None
                            else summary_dir / "champion_challenger_policy_aligned_companion.json"
                        )
                        companion_cmd = [
                            python,
                            "-m",
                            "src.scripts.evaluate_champion_challenger",
                            "--baseline",
                            str(companion_baseline_input),
                            "--candidate",
                            str(selected_shadow_path),
                            "--baseline-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--n-boot",
                            str(int(champ_cfg.get("n_boot", 2000))),
                            "--alpha",
                            str(float(champ_cfg.get("alpha", 0.05))),
                            "--seed",
                            str(int(champ_cfg.get("seed", 42))),
                            "--output",
                            str(selected_policy_aligned_companion_path),
                        ]
                        results.append(
                            _run_step(
                                "champion_challenger_policy_aligned_companion_official_shadow",
                                companion_cmd,
                                logs_dir / "champion_challenger_policy_aligned_companion_official_shadow.log",
                                args.dry_run,
                            )
                        )
                        if (
                            not args.dry_run
                            and selected_policy_aligned_companion_path.exists()
                            and selected_policy_aligned_companion_path
                            != summary_dir / "champion_challenger_policy_aligned_companion.json"
                        ):
                            shutil.copyfile(
                                selected_policy_aligned_companion_path,
                                summary_dir / "champion_challenger_policy_aligned_companion.json",
                            )

                        official_profile_summary_cmd = [
                            python,
                            "-m",
                            "src.scripts.summarize_policy_aligned_profile_metrics",
                            "--candidate",
                            str(selected_shadow_path),
                            "--incumbent",
                            str(companion_baseline_input),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--incumbent-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--candidate-meta",
                            str(selected_shadow_meta_path),
                            "--profile-id",
                            str(run_profile_id),
                            "--profile-name",
                            str(run_profile_name),
                            "--run-id",
                            str(run_dir.name),
                            "--n-boot",
                            str(int(champ_cfg.get("n_boot", 2000))),
                            "--seed",
                            str(int(champ_cfg.get("seed", 42))),
                            "--output",
                            str(official_profile_summary_path),
                        ]
                        results.append(
                            _run_step(
                                "official_profile_policy_metrics_official_shadow",
                                official_profile_summary_cmd,
                                logs_dir / "official_profile_policy_metrics_official_shadow.log",
                                args.dry_run,
                            )
                        )

                        paired_overlap_cmd = [
                            python,
                            "-m",
                            "src.scripts.analyze_paired_trigger_overlap",
                            "--candidate",
                            str(selected_shadow_path),
                            "--incumbent",
                            str(companion_baseline_input),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--incumbent-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--output",
                            str(summary_dir / "paired_trigger_overlap_policy_aligned.json"),
                        ]
                        results.append(
                            _run_step(
                                "paired_trigger_overlap_policy_aligned_official_shadow",
                                paired_overlap_cmd,
                                logs_dir / "paired_trigger_overlap_policy_aligned_official_shadow.log",
                                args.dry_run,
                            )
                        )

                        if (
                            labeled_overlap_dataset is not None
                            and (labeled_overlap_dataset.exists() or args.dry_run)
                            and (selected_shadow_path.exists() or args.dry_run)
                        ):
                            overlap_triggered_trade_diag_cmd = [
                                python,
                                "-m",
                                "src.scripts.analyze_overlap_triggered_trade_diagnostics",
                                "--candidate",
                                str(selected_shadow_path),
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--signal-col",
                                str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                "--return-col",
                                str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                "--output",
                                str(selected_shadow_overlap_triggered_trade_diag_path),
                            ]
                            results.append(
                                _run_step(
                                    "overlap_triggered_trade_diagnostics_official_shadow",
                                    overlap_triggered_trade_diag_cmd,
                                    logs_dir / "overlap_triggered_trade_diagnostics_official_shadow.log",
                                    args.dry_run,
                                )
                            )
                            if (
                                not args.dry_run
                                and selected_shadow_overlap_triggered_trade_diag_path.exists()
                                and selected_shadow_overlap_triggered_trade_diag_path
                                != summary_dir / "overlap_triggered_trade_diagnostics.json"
                            ):
                                shutil.copyfile(
                                    selected_shadow_overlap_triggered_trade_diag_path,
                                    summary_dir / "overlap_triggered_trade_diagnostics.json",
                                )

                if bool(calibration_cfg.get("enabled", True)):
                    calibration_horizon_key = str(quality_cfg.get("calibration_horizon", "1h"))
                    current_candidate_for_calibration = (
                        selected_shadow_path if selected_shadow_path is not None else Path(str(candidate_input))
                    )
                    if current_candidate_for_calibration.exists() or args.dry_run:
                        policy_calibration_cmd = _build_calibration_robustness_command(
                            python=python,
                            input_path=current_candidate_for_calibration,
                            output_path=summary_dir / "calibration_robustness.json",
                            calibration_cfg=calibration_cfg,
                            quality_cfg=quality_cfg,
                            trade_decision_cfg=trade_decision_cfg,
                        )
                        results.append(
                            _run_step(
                                "calibration_robustness_policy_aligned_official",
                                policy_calibration_cmd,
                                logs_dir / "calibration_robustness_policy_aligned_official.log",
                                args.dry_run,
                            )
                        )
                        if not args.dry_run and (summary_dir / "calibration_robustness.json").exists():
                            try:
                                policy_calibration_payload = _load_json(summary_dir / "calibration_robustness.json")
                                recent_diag_payload = _extract_recent_calibration_payload(
                                    policy_calibration_payload,
                                    horizon_key=calibration_horizon_key,
                                )
                                (summary_dir / "recent_calibration_diagnostics.json").write_text(
                                    json.dumps(recent_diag_payload, indent=2),
                                    encoding="utf-8",
                                )
                            except Exception as exc:
                                print(f"Warning: failed to refresh policy-aligned calibration diagnostics: {exc}", file=sys.stderr)

                        if bool(calibration_cfg.get("regime_aware", True)):
                            policy_regime_labeled_path = summary_dir / "platt_calibration_policy_aligned_labeled.csv"
                            policy_regime_labeled_meta_path = summary_dir / "platt_calibration_policy_aligned_labeled_meta.json"
                            policy_regime_labeled_input = current_candidate_for_calibration
                            policy_regime_horizons = [
                                _format_horizon_label(horizon)
                                for horizon in calibration_horizons
                                if float(horizon) >= 1.0
                            ]
                            if policy_regime_horizons:
                                policy_regime_labeled_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.build_labeled_backtest_from_history",
                                    "--no-prefer-backtest",
                                    "--include-reliability-snapshots",
                                    "--output",
                                    str(policy_regime_labeled_path),
                                    "--meta-output",
                                    str(policy_regime_labeled_meta_path),
                                    "--fold-size",
                                    str(int(quality_cfg.get("fold_size", 12))),
                                    "--lookback-rows",
                                    str(int(quality_cfg.get("lookback_rows", 2000))),
                                    "--lookback-hours",
                                    str(int(quality_cfg.get("lookback_hours", 0))),
                                    "--min-rows",
                                    str(int(quality_cfg.get("min_labeled_rows", 200))),
                                    "--horizons",
                                    *policy_regime_horizons,
                                ]
                                try:
                                    results.append(
                                        _run_step(
                                            "build_policy_aligned_regime_calibration_dataset",
                                            policy_regime_labeled_cmd,
                                            logs_dir / "build_policy_aligned_regime_calibration_dataset.log",
                                            args.dry_run,
                                        )
                                    )
                                except RuntimeError as exc:
                                    print(
                                        "Warning: failed to build policy-aligned regime calibration dataset; "
                                        f"falling back to policy-aligned candidate input: {exc}",
                                        file=sys.stderr,
                                    )
                                if policy_regime_labeled_path.exists() or args.dry_run:
                                    policy_regime_labeled_input = policy_regime_labeled_path

                            policy_regime_calib_cmd = [
                                python,
                                "-m",
                                "src.scripts.train_platt_calibration",
                                "--horizons",
                                *[str(h) for h in calibration_horizons],
                                "--output-path",
                                str(summary_dir / "platt_calibration.json"),
                                "--coverage-output-path",
                                str(summary_dir / "platt_calibration_coverage.json"),
                                "--method",
                                str(calibration_cfg.get("method", "platt")),
                                "--labeled-input",
                                str(policy_regime_labeled_input),
                                "--regime-col",
                                str(calibration_cfg.get("regime_col", "regime_state")),
                                "--min-regime-rows",
                                str(int(calibration_cfg.get("min_regime_rows", 100))),
                            ]
                            if bool(calibration_cfg.get("fit_base_horizons_from_labeled_input", True)):
                                policy_regime_calib_cmd.append("--fit-base-horizons-from-labeled-input")
                            if bool(calibration_cfg.get("skip_model_fit_when_labeled_input", True)):
                                policy_regime_calib_cmd.append("--skip-model-fit")
                            results.append(
                                _run_step(
                                    "platt_calibration_regime_aware_policy_aligned_official",
                                    policy_regime_calib_cmd,
                                    logs_dir / "platt_calibration_regime_aware_policy_aligned_official.log",
                                    args.dry_run,
                                )
                            )

                    if bool(regime_weakness_cfg.get("enabled", True)):
                        calibration_path = summary_dir / "calibration_robustness.json"
                        if (calibration_path.exists() and walkforward_output.exists()) or args.dry_run:
                            regime_weakness_cmd = [
                                python,
                                "-m",
                                "src.scripts.evaluate_regime_weakness",
                                "--calibration",
                                str(calibration_path),
                                "--walkforward",
                                str(walkforward_output),
                                "--horizon",
                                str(calibration_horizon_key),
                                "--max-ece-drift",
                                str(float(regime_weakness_cfg.get("max_ece_drift", calibration_cfg.get("max_ece_drift", 0.02)))),
                                "--min-recent-auc",
                                str(float(quality_cfg.get("min_recent_auc", 0.0))),
                                "--min-net-return",
                                str(float(regime_weakness_cfg.get("min_net_return", 0.0))),
                                "--output",
                                str(summary_dir / "regime_weakness.json"),
                            ]
                            results.append(
                                _run_step(
                                    "regime_weakness_policy_aligned_official",
                                    regime_weakness_cmd,
                                    logs_dir / "regime_weakness_policy_aligned_official.log",
                                    args.dry_run,
                                )
                            )

                champ_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_champion_challenger",
                    "--baseline",
                    str(baseline_input),
                    "--candidate",
                    str(candidate_input),
                    "--baseline-col",
                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                    "--candidate-col",
                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                    "--n-boot",
                    str(int(champ_cfg.get("n_boot", 2000))),
                    "--alpha",
                    str(float(champ_cfg.get("alpha", 0.05))),
                    "--seed",
                    str(int(champ_cfg.get("seed", 42))),
                    "--output",
                    str(summary_dir / "champion_challenger_gate.json"),
                ]
                results.append(
                    _run_step(
                        "champion_challenger_gate",
                        champ_cmd,
                        logs_dir / "champion_challenger_gate.log",
                        args.dry_run,
                    )
                )
                gate_path = summary_dir / "champion_challenger_gate.json"
                if gate_path.exists() and not args.dry_run:
                    champion_gate_payload = _load_json(gate_path)

        champion_gate_source = str(quality_cfg.get("champion_gate_source", "auto")).strip().lower()
        effective_champion_gate_path, effective_champion_gate_payload, champion_gate_resolution = _resolve_effective_champion_gate(
            summary_dir=summary_dir,
            champion_gate_payload=champion_gate_payload,
            official_shadow_variant=official_shadow_variant,
            champion_gate_source=champion_gate_source,
            policy_aligned_gate_path=selected_shadow_companion_path,
        )
        (summary_dir / "champion_gate_resolution.json").write_text(
            json.dumps(champion_gate_resolution, indent=2),
            encoding="utf-8",
        )
        if not args.dry_run:
            champion_gate_alignment = _build_champion_gate_alignment_check(
                summary_dir=summary_dir,
                official_shadow_variant=official_shadow_variant,
                champion_gate_source=champion_gate_source,
                selection_payload=selection_payload,
                effective_champion_gate_path=effective_champion_gate_path,
                effective_champion_gate_payload=effective_champion_gate_payload,
                champion_gate_resolution=champion_gate_resolution,
                policy_aligned_gate_path=selected_shadow_companion_path,
            )
            (summary_dir / "champion_gate_alignment_check.json").write_text(
                json.dumps(champion_gate_alignment, indent=2),
                encoding="utf-8",
            )
            if not bool(champion_gate_alignment.get("passed", False)):
                raise RuntimeError(
                    "Champion gate alignment regression detected: "
                    + "; ".join(str(item) for item in champion_gate_alignment.get("errors", []))
                )

        champion_blocked = bool(
            effective_champion_gate_payload
            and bool(champ_cfg.get("enforce", False))
            and not bool(effective_champion_gate_payload.get("promote", False))
        )

        if incumbent_quality_path and (candidate_quality_path.exists() or args.dry_run):
            if champion_blocked and not args.dry_run:
                synthetic_gate = {
                    "promote": False,
                    "reason": "champion_challenger_blocked",
                    "champion_gate_resolution": champion_gate_resolution,
                    "champion_challenger": effective_champion_gate_payload,
                    "champion_challenger_labeled": champion_gate_payload,
                }
                gate_out = summary_dir / "promotion_gate.json"
                gate_out.write_text(json.dumps(synthetic_gate, indent=2), encoding="utf-8")
                print("Promotion gate blocked by champion-challenger significance gate.", file=sys.stderr)
            else:
                promote_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_shadow_promotion",
                    "--incumbent",
                    str(incumbent_quality_path),
                    "--candidate",
                    str(candidate_quality_path),
                    "--min-auc-delta",
                    str(float(quality_cfg.get("min_auc_delta", 0.002))),
                    "--max-brier-increase",
                    str(float(quality_cfg.get("max_brier_increase", 0.0))),
                    "--max-ece-increase",
                    str(float(quality_cfg.get("max_ece_increase", 0.01))),
                    "--min-trade-count",
                    str(int(quality_cfg.get("min_trade_count", 10))),
                    "--trade-count-key",
                    str(quality_cfg.get("trade_count_key", "trade_count")),
                    "--min-net-return",
                    str(float(quality_cfg.get("min_net_return", 0.0))),
                    "--net-return-key",
                    str(quality_cfg.get("net_return_key", "net_return_total")),
                    "--output",
                    str(summary_dir / "promotion_gate.json"),
                ]
                calibration_robustness_path = summary_dir / "calibration_robustness.json"
                if calibration_robustness_path.exists() or args.dry_run:
                    promote_cmd.extend(
                        [
                            "--calibration-robustness",
                            str(calibration_robustness_path),
                            "--calibration-horizon",
                            str(quality_cfg.get("calibration_horizon", "1h")),
                        ]
                    )
                    if quality_cfg.get("max_ece_drift") is not None:
                        promote_cmd.extend(["--max-ece-drift", str(float(quality_cfg.get("max_ece_drift")))])
                    if quality_cfg.get("max_recent_ece") is not None:
                        promote_cmd.extend(["--max-recent-ece", str(float(quality_cfg.get("max_recent_ece")))])
                    if quality_cfg.get("min_recent_auc") is not None:
                        promote_cmd.extend(["--min-recent-auc", str(float(quality_cfg.get("min_recent_auc")))])

                rolling_ab_report_path = summary_dir / "rolling_ab_report.json"
                if rolling_ab_report_path.exists() or args.dry_run:
                    promote_cmd.extend(["--rolling-ab-report", str(rolling_ab_report_path)])
                    if quality_cfg.get("max_negative_rolling_windows") is not None:
                        promote_cmd.extend(
                            [
                                "--max-negative-rolling-windows",
                                str(int(quality_cfg.get("max_negative_rolling_windows"))),
                            ]
                        )
                    if bool(quality_cfg.get("require_nonnegative_rolling_delta", False)):
                        promote_cmd.append("--require-nonnegative-rolling-delta")

                overlap_triggered_path = summary_dir / "overlap_triggered_trade_diagnostics.json"
                if overlap_triggered_path.exists() or args.dry_run:
                    promote_cmd.extend(["--overlap-triggered-diagnostics", str(overlap_triggered_path)])
                    if quality_cfg.get("min_overlap_triggered_trades") is not None:
                        promote_cmd.extend(
                            [
                                "--min-overlap-triggered-trades",
                                str(int(quality_cfg.get("min_overlap_triggered_trades"))),
                            ]
                        )
                    if quality_cfg.get("min_overlap_triggered_net_return") is not None:
                        promote_cmd.extend(
                            [
                                "--min-overlap-triggered-net-return",
                                str(float(quality_cfg.get("min_overlap_triggered_net_return"))),
                            ]
                        )
                    if quality_cfg.get("min_overlap_triggered_hit_rate") is not None:
                        promote_cmd.extend(
                            [
                                "--min-overlap-triggered-hit-rate",
                                str(float(quality_cfg.get("min_overlap_triggered_hit_rate"))),
                            ]
                        )
                borderline_selection_cfg = (
                    quality_cfg.get("borderline_selection_exception")
                    if isinstance(quality_cfg.get("borderline_selection_exception"), dict)
                    else {}
                )
                if bool(borderline_selection_cfg.get("enabled", False)):
                    promote_cmd.append("--allow-borderline-selection-rows-exception")
                    if bool(borderline_selection_cfg.get("require_champion_significance", True)):
                        promote_cmd.append("--require-champion-for-borderline")
                    if bool(borderline_selection_cfg.get("require_rolling_stability", True)):
                        promote_cmd.append("--require-rolling-for-borderline")
                    if bool(borderline_selection_cfg.get("require_overlap_triggered_strength", True)):
                        promote_cmd.append("--require-overlap-for-borderline")
                if bool(quality_cfg.get("require_champion_significance", False)) and bool(champ_cfg.get("enabled", False)):
                    promote_cmd.extend(
                        [
                            "--champion-gate",
                            str(effective_champion_gate_path),
                        ]
                    )
                promotion_allowed = [0, 3] if args.continue_on_promotion_fail else [0]
                results.append(
                    _run_step(
                        "promotion_gate",
                        promote_cmd,
                        logs_dir / "promotion_gate.log",
                        args.dry_run,
                        allowed_returncodes=promotion_allowed,
                    )
                )
                if not args.dry_run:
                    promotion_gate_path = summary_dir / "promotion_gate.json"
                    promotion_gate_payload = _load_json(promotion_gate_path) if promotion_gate_path.exists() else {}
                    deploy_cfg_obj = quality_cfg.get("promotion_deploy", {})
                    deploy_cfg = deploy_cfg_obj if isinstance(deploy_cfg_obj, dict) else {}
                    promoted_profile_path = selected_shadow_path if selected_shadow_path is not None else Path(str(candidate_gate_input))
                    promoted_profile_meta_path = selected_shadow_meta_path
                    selected_trade_decision_model_path = (
                        trade_decision_ablation_model_path
                        if _shadow_variant_uses_reference_feature_ablation_model(official_shadow_variant)
                        else trade_decision_model_path
                    )
                    selected_trade_decision_deploy_ready = (
                        trade_decision_ablation_deploy_ready
                        if _shadow_variant_uses_reference_feature_ablation_model(official_shadow_variant)
                        else trade_decision_deploy_ready
                    )
                    if promoted_profile_meta_path is None:
                        fallback_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_meta.json"
                        if fallback_meta_path.exists():
                            promoted_profile_meta_path = fallback_meta_path
                    promotion_gate_payload = _apply_trade_decision_model_shift_guard(
                        summary_dir=summary_dir,
                        promotion_gate_payload=promotion_gate_payload,
                        trade_decision_cfg=trade_decision_cfg,
                        model_shift_payload=trade_decision_model_shift_payload,
                    )
                    if promotion_gate_path.exists() or promotion_gate_payload:
                        promotion_gate_path.write_text(json.dumps(promotion_gate_payload, indent=2), encoding="utf-8")
                    if bool(deploy_cfg.get("enabled", False)) and bool(promotion_gate_payload.get("promote", False)):
                        deploy_manifest = _deploy_promoted_reliability_artifacts(
                            run_dir=run_dir,
                            deploy_cfg=deploy_cfg,
                            thresholds_path=thresholds_path,
                            platt_calibration_path=summary_dir / "platt_calibration.json",
                            trade_decision_model_path=selected_trade_decision_model_path,
                            trade_decision_deploy_ready=selected_trade_decision_deploy_ready,
                            promoted_profile_path=promoted_profile_path,
                            promoted_profile_meta_path=promoted_profile_meta_path,
                            candidate_quality_path=candidate_quality_path,
                            promotion_gate_path=promotion_gate_path,
                            calibration_robustness_path=summary_dir / "calibration_robustness.json",
                            rolling_ab_report_path=summary_dir / "rolling_ab_report.json",
                            rolling_ab_md_path=summary_dir / "rolling_ab_report.md",
                            selection_guard_rule_path=summary_dir / "selection_calibration_guard_rule.json",
                            official_shadow_variant=official_shadow_variant,
                            champion_gate_resolution=champion_gate_resolution,
                        )
                        (summary_dir / "promotion_deploy_manifest.json").write_text(
                            json.dumps(deploy_manifest, indent=2),
                            encoding="utf-8",
                        )
        elif not args.dry_run:
            # Emit an explicit non-evaluated promotion artifact so downstream
            # matched-cycle summaries can compare promotion evidence on both profiles.
            if not incumbent_quality_path:
                skip_reason = "not_evaluated_incumbent_quality_missing"
            elif not candidate_quality_path.exists():
                skip_reason = "not_evaluated_candidate_quality_missing"
            else:
                skip_reason = "not_evaluated"

            skipped_gate = {
                "promote": False,
                "evaluated": False,
                "reason": skip_reason,
                "incumbent_quality_path": str(incumbent_quality_path) if incumbent_quality_path else None,
                "candidate_quality_path": str(candidate_quality_path),
                "candidate_quality_exists": bool(candidate_quality_path.exists()),
            }
            (summary_dir / "promotion_gate.json").write_text(
                json.dumps(skipped_gate, indent=2),
                encoding="utf-8",
            )

        # If quality indicates a no-trade or low-trade regime, trigger a relaxed threshold search for paper-live continuity.
        no_trade_regime = {
            "detected": False,
            "trigger_trade_count": None,
            "min_trade_count": int(quality_cfg.get("min_trade_count", 10)),
            "used_fallback_thresholds": False,
            "fallback_thresholds_path": None,
        }
        quality_candidate_path = summary_dir / "model_quality_candidate.json"
        if bool(no_trade_cfg.get("enabled", True)) and (quality_candidate_path.exists() or args.dry_run):
            trigger_below = int(no_trade_cfg.get("trigger_trade_count_below", quality_cfg.get("min_trade_count", 10)))
            trade_count = 0
            if quality_candidate_path.exists():
                quality_payload = _load_json(quality_candidate_path)
                trade_count = int(quality_payload.get("trade_count", 0) or 0)
            no_trade_regime["trigger_trade_count"] = int(trade_count)

            if (args.dry_run or trade_count < trigger_below) and not used_last_deployable_thresholds:
                no_trade_regime["detected"] = True
                fallback_thresholds_path = summary_dir / "calibrated_thresholds_fallback.json"
                fallback_cmd = [
                    python,
                    "-m",
                    "src.scripts.search_ensemble_thresholds",
                    "--targets",
                    _join_horizons(calibration_horizons),
                    "--objective",
                    str(no_trade_cfg.get("objective", "cumret_with_dd_constraint")),
                    "--max-dd",
                    str(float(no_trade_cfg.get("max_drawdown", -0.15))),
                    "--min-trades",
                    str(int(no_trade_cfg.get("min_trades", 1))),
                    f"--p-up-grid={str(no_trade_cfg.get('p_up_grid', '0.45,0.50,0.55,0.60'))}",
                    f"--ret-min-grid={str(no_trade_cfg.get('ret_min_grid', '-0.0005,-0.0002,0.0,0.0002'))}",
                    "--output-dir",
                    str(summary_dir / "threshold_search_fallback"),
                    "--output",
                    str(fallback_thresholds_path),
                ]
                results.append(
                    _run_step(
                        "fallback_threshold_search",
                        fallback_cmd,
                        logs_dir / "fallback_threshold_search.log",
                        args.dry_run,
                    )
                )
                thresholds_path = fallback_thresholds_path
                no_trade_regime["used_fallback_thresholds"] = True
                no_trade_regime["fallback_thresholds_path"] = str(fallback_thresholds_path)

            no_trade_regime["joint_threshold_accepted"] = joint_tuning_accepted
            no_trade_regime["used_last_deployable_thresholds"] = bool(used_last_deployable_thresholds)
            no_trade_regime["last_deployable_thresholds_path"] = str(last_deployable_thresholds_path)
            no_trade_regime["paper_live_thresholds_path"] = str(thresholds_path)

        regime_path = summary_dir / "regime_diagnostics.json"
        regime_path.write_text(json.dumps(no_trade_regime, indent=2), encoding="utf-8")

    if not args.skip_monitoring:
        monitoring_horizons = monitoring_cfg.get("horizons", ["1h", "4h", "8h", "12h"])
        monitoring_cmd = [
            python,
            "-m",
            "src.scripts.check_reliability_triggers",
            "--config-path",
            str(args.config),
            "--horizons",
            ",".join(str(h) for h in monitoring_horizons),
            "--output-path",
            str(summary_dir / "reliability_triggers.json"),
        ]
        results.append(_run_step("reliability_trigger_check", monitoring_cmd, logs_dir / "reliability_trigger_check.log", args.dry_run))

    if not args.skip_paper_live:
        paper_live_config = str(search_cfg.get("paper_live_config", "configs/run_refresh_and_predict.default.yaml"))
        direction_output_shadow_enabled = bool(direction_output_shadow_cfg.get("enabled", False))
        direction_output_shadow_apply_to_paper_live = bool(direction_output_shadow_cfg.get("apply_to_paper_live", False))
        upstream_direction_candidate_enabled = bool(upstream_direction_candidate_cfg.get("enabled", False))
        upstream_direction_candidate_apply_to_paper_live = bool(
            upstream_direction_candidate_cfg.get("apply_to_paper_live", False)
        )
        trade_decision_chop_suppression_candidate_enabled = bool(
            trade_decision_chop_suppression_candidate_cfg.get("enabled", False)
        )
        trade_decision_chop_suppression_candidate_apply_to_paper_live = bool(
            trade_decision_chop_suppression_candidate_cfg.get("apply_to_paper_live", False)
        )
        direction_output_shadow_horizons = _resolve_direction_output_shadow_horizons(direction_output_shadow_cfg)
        direction_output_shadow_horizon_labels = [_format_horizon_label(horizon) for horizon in direction_output_shadow_horizons]
        direction_output_shadow_label = (
            direction_output_shadow_horizon_labels[0]
            if len(direction_output_shadow_horizon_labels) == 1
            else "multi_horizon"
        )
        direction_output_shadow_method = str(direction_output_shadow_cfg.get("calibration_method", "isotonic"))
        direction_output_shadow_neutral_band = float(direction_output_shadow_cfg.get("neutral_band", 0.02))
        direction_output_shadow_include_snapshots = bool(
            direction_output_shadow_cfg.get("include_reliability_snapshots", True)
        )
        direction_output_shadow_apply_marginal_weights = bool(
            direction_output_shadow_cfg.get("apply_marginal_audit_weights", True)
        )
        marginal_audit_enabled = bool(direction_output_shadow_enabled or upstream_direction_candidate_enabled)
        marginal_horizon = str(
            upstream_direction_candidate_cfg.get(
                "marginal_horizon",
                direction_output_shadow_cfg.get(
                    "marginal_horizon",
                    direction_output_shadow_horizon_labels[0] if direction_output_shadow_horizon_labels else "1h",
                ),
            )
        )
        direction_output_shadow_labeled_path = summary_dir / f"direction_output_labeled_{direction_output_shadow_label}.csv"
        direction_output_shadow_labeled_meta_path = summary_dir / f"direction_output_labeled_{direction_output_shadow_label}_meta.json"
        direction_output_shadow_calibration_path = summary_dir / f"direction_output_{direction_output_shadow_method}_{direction_output_shadow_label}.json"
        direction_output_shadow_coverage_path = summary_dir / f"direction_output_{direction_output_shadow_method}_{direction_output_shadow_label}_coverage.json"
        direction_output_shadow_config_path = summary_dir / "paper_live_direction_output_shadow_config.yaml"
        direction_output_shadow_config_meta_path = summary_dir / "paper_live_direction_output_shadow_config_meta.json"
        upstream_direction_candidate_path = summary_dir / "paper_live_upstream_direction_candidate.yaml"
        upstream_direction_candidate_meta_path = summary_dir / "paper_live_upstream_direction_candidate_meta.json"
        trade_decision_chop_suppression_candidate_path = (
            summary_dir / "paper_live_trade_decision_chop_suppression_candidate.yaml"
        )
        trade_decision_chop_suppression_candidate_meta_path = (
            summary_dir / "paper_live_trade_decision_chop_suppression_candidate_meta.json"
        )
        direction_output_shadow_marginal_audit_path = summary_dir / f"direction_marginal_{marginal_horizon}.json"
        direction_output_shadow_marginal_rows_path = summary_dir / f"direction_marginal_{marginal_horizon}_rows.csv"
        direction_audit_input = summary_dir / "backtest_signals_meta_ensemble_decision_aligned.csv"
        if not direction_audit_input.exists() and not args.dry_run:
            direction_audit_input = summary_dir / "backtest_signals_meta_ensemble.csv"
        pre_paper_audit_path = summary_dir / "direction_model_audit_pre_paper_live.json"
        audit_weighted_config_path = summary_dir / "paper_live_direction_audit_config.yaml"
        if direction_audit_input.exists() or args.dry_run:
            pre_direction_audit_cmd = [
                python,
                "-m",
                "src.scripts.audit_direction_models",
                "--backtest-csv",
                str(direction_audit_input),
                "--latest-predictions",
                "artifacts/predictions/latest.json",
                "--output",
                str(pre_paper_audit_path),
            ]
            results.append(
                _run_step(
                    "direction_model_audit_pre_paper_live",
                    pre_direction_audit_cmd,
                    logs_dir / "direction_model_audit_pre_paper_live.log",
                    args.dry_run,
                )
            )
            if not args.dry_run and bool(search_cfg.get("apply_direction_audit_to_paper_live", True)):
                try:
                    if _build_audit_weighted_runtime_config(
                        base_config_path=Path(paper_live_config),
                        audit_payload=_load_json(pre_paper_audit_path),
                        output_path=audit_weighted_config_path,
                    ):
                        paper_live_config = str(audit_weighted_config_path)
                except Exception as exc:
                    print(f"Warning: failed to derive audit-weighted paper-live config: {exc}", file=sys.stderr)
        if marginal_audit_enabled:
            direction_output_shadow_min_rows = int(
                direction_output_shadow_cfg.get("min_rows", 200)
            )
            direction_output_shadow_min_regime_rows = int(
                direction_output_shadow_cfg.get("min_regime_rows", calibration_cfg.get("min_regime_rows", 20))
            )
            direction_output_shadow_fold_size = int(direction_output_shadow_cfg.get("fold_size", quality_cfg.get("fold_size", 12)))
            direction_output_shadow_lookback_rows = int(
                direction_output_shadow_cfg.get("lookback_rows", quality_cfg.get("lookback_rows", 2000))
            )
            direction_output_shadow_lookback_hours = int(
                direction_output_shadow_cfg.get("lookback_hours", quality_cfg.get("lookback_hours", 0))
            )

            marginal_audit_cmd = [
                python,
                "-m",
                "src.scripts.analyze_direction_marginal_calibration",
                "--history-path",
                "artifacts/predictions/history.json",
                "--spot-ohlcv-path",
                "data/spot_klines",
                "--horizon",
                marginal_horizon,
                "--benchmark-config",
                str(paper_live_config),
                "--fold-size",
                str(direction_output_shadow_fold_size),
                "--lookback-rows",
                str(direction_output_shadow_lookback_rows),
                "--lookback-hours",
                str(direction_output_shadow_lookback_hours),
                "--output",
                str(direction_output_shadow_marginal_audit_path),
                "--rows-output",
                str(direction_output_shadow_marginal_rows_path),
            ]
            if direction_output_shadow_include_snapshots:
                marginal_audit_cmd.append("--include-reliability-snapshots")
            results.append(
                _run_step(
                    "direction_marginal_audit",
                    marginal_audit_cmd,
                    logs_dir / "direction_marginal_audit.log",
                    args.dry_run,
                )
            )

            if not args.dry_run and upstream_direction_candidate_enabled and direction_output_shadow_marginal_audit_path.exists():
                try:
                    candidate_payload = _write_upstream_direction_candidate_config(
                        base_config_path=Path(paper_live_config),
                        marginal_audit_path=direction_output_shadow_marginal_audit_path,
                        output_path=upstream_direction_candidate_path,
                        meta_output_path=upstream_direction_candidate_meta_path,
                        apply_to_paper_live=upstream_direction_candidate_apply_to_paper_live,
                    )
                    if (
                        candidate_payload.get("internal_direction_weight_update_applied")
                        and upstream_direction_candidate_apply_to_paper_live
                    ):
                        paper_live_config = str(upstream_direction_candidate_path)
                except Exception as exc:
                    print(f"Warning: failed to derive upstream direction candidate config: {exc}", file=sys.stderr)

        if not args.dry_run and trade_decision_chop_suppression_candidate_enabled and direction_audit_input.exists():
            try:
                candidate_payload = _write_trade_decision_midband_candidate_config(
                    base_config_path=Path(paper_live_config),
                    candidate_path=direction_audit_input,
                    output_path=trade_decision_chop_suppression_candidate_path,
                    meta_output_path=trade_decision_chop_suppression_candidate_meta_path,
                    recent_window_rows=int(
                        trade_decision_chop_suppression_candidate_cfg.get("recent_window_rows", 288)
                    ),
                    signal_col=str(trade_decision_chop_suppression_candidate_cfg.get("signal_col", "signal_ensemble")),
                    p_col=str(trade_decision_chop_suppression_candidate_cfg.get("p_col", "p_up")),
                    ret_pred_col=str(
                        trade_decision_chop_suppression_candidate_cfg.get("ret_pred_col", "ret_pred")
                    ),
                    return_col=str(
                        trade_decision_chop_suppression_candidate_cfg.get("return_col", "ret_ensemble_net")
                    ),
                    regime_col=str(
                        trade_decision_chop_suppression_candidate_cfg.get("regime_col", "regime_state")
                    ),
                    min_regime_rows=int(
                        trade_decision_chop_suppression_candidate_cfg.get("min_regime_rows", 2)
                    ),
                    require_overall_regime_negative=bool(
                        trade_decision_chop_suppression_candidate_cfg.get("require_overall_regime_negative", True)
                    ),
                    apply_to_paper_live=trade_decision_chop_suppression_candidate_apply_to_paper_live,
                )
                if (
                    candidate_payload.get("trade_decision_midband_update_applied")
                    and trade_decision_chop_suppression_candidate_apply_to_paper_live
                ):
                    paper_live_config = str(trade_decision_chop_suppression_candidate_path)
            except Exception as exc:
                print(
                    f"Warning: failed to derive trade decision chop suppression candidate config: {exc}",
                    file=sys.stderr,
                )

        if direction_output_shadow_enabled:
            direction_output_labeled_cmd = [
                python,
                "-m",
                "src.scripts.build_labeled_backtest_from_history",
                "--no-prefer-backtest",
                "--output",
                str(direction_output_shadow_labeled_path),
                "--meta-output",
                str(direction_output_shadow_labeled_meta_path),
                "--fold-size",
                str(direction_output_shadow_fold_size),
                "--lookback-rows",
                str(direction_output_shadow_lookback_rows),
                "--lookback-hours",
                str(direction_output_shadow_lookback_hours),
                "--min-rows",
                str(direction_output_shadow_min_rows),
                "--horizons",
                *direction_output_shadow_horizon_labels,
            ]
            if direction_output_shadow_include_snapshots:
                direction_output_labeled_cmd.append("--include-reliability-snapshots")
            try:
                results.append(
                    _run_step(
                        "build_direction_output_shadow_dataset",
                        direction_output_labeled_cmd,
                        logs_dir / "build_direction_output_shadow_dataset.log",
                        args.dry_run,
                    )
                )
            except RuntimeError as exc:
                print(
                    "Warning: failed to build direction output shadow dataset; skipping direction output shadow calibration: "
                    f"{exc}",
                    file=sys.stderr,
                )

            if direction_output_shadow_labeled_path.exists() or args.dry_run:
                direction_output_calibration_cmd = [
                    python,
                    "-m",
                    "src.scripts.train_platt_calibration",
                    "--horizons",
                    *[str(horizon) for horizon in direction_output_shadow_horizons],
                    "--output-path",
                    str(direction_output_shadow_calibration_path),
                    "--coverage-output-path",
                    str(direction_output_shadow_coverage_path),
                    "--method",
                    direction_output_shadow_method,
                    "--labeled-input",
                    str(direction_output_shadow_labeled_path),
                    "--fit-base-horizons-from-labeled-input",
                    "--skip-model-fit",
                    "--regime-col",
                    str(calibration_cfg.get("regime_col", "regime_state")),
                    "--min-regime-rows",
                    str(direction_output_shadow_min_regime_rows),
                ]
                results.append(
                    _run_step(
                        "direction_output_shadow_calibration",
                        direction_output_calibration_cmd,
                        logs_dir / "direction_output_shadow_calibration.log",
                        args.dry_run,
                    )
                )

            if args.dry_run:
                if direction_output_shadow_apply_to_paper_live:
                    paper_live_config = str(direction_output_shadow_config_path)
            elif direction_output_shadow_calibration_path.exists():
                try:
                    _write_direction_output_shadow_config(
                        base_config_path=Path(paper_live_config),
                        direction_output_calibration_path=direction_output_shadow_calibration_path,
                        output_path=direction_output_shadow_config_path,
                        meta_output_path=direction_output_shadow_config_meta_path,
                        marginal_audit_path=(
                            direction_output_shadow_marginal_audit_path
                            if direction_output_shadow_apply_marginal_weights
                            and direction_output_shadow_marginal_audit_path.exists()
                            else None
                        ),
                        neutral_band=direction_output_shadow_neutral_band,
                        horizons=direction_output_shadow_horizons,
                    )
                    if direction_output_shadow_apply_to_paper_live:
                        paper_live_config = str(direction_output_shadow_config_path)
                except Exception as exc:
                    print(f"Warning: failed to derive direction-output shadow config: {exc}", file=sys.stderr)

                if direction_output_shadow_marginal_audit_path.exists():
                    latest_direction_marginal_path = Path(f"artifacts/analysis/direction_marginal_{marginal_horizon}_latest.json")
                    latest_direction_marginal_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(direction_output_shadow_marginal_audit_path, latest_direction_marginal_path)
                if direction_output_shadow_marginal_rows_path.exists():
                    latest_direction_marginal_rows_path = Path(f"artifacts/analysis/direction_marginal_{marginal_horizon}_rows.csv")
                    latest_direction_marginal_rows_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(direction_output_shadow_marginal_rows_path, latest_direction_marginal_rows_path)
        paper_cmd = [
            python,
            "-m",
            "src.scripts.run_refresh_and_predict",
            "--config",
            paper_live_config,
            "--targets",
            _join_horizons(calibration_horizons),
            "--thresholds-json",
            str(thresholds_path),
            "--platt-calibration",
            str(summary_dir / "platt_calibration.json"),
            "--write-artifacts",
        ]
        if not edge_trustworthy_for_paper_live and not args.dry_run:
            # Force conservative hold behavior when overlap evidence disagrees with canonical walk-forward results.
            paper_cmd.extend(["--p-up-min", "0.99", "--ret-min", "0.005"])
            print(
                "Edge trustworthiness check failed; paper-live run forced to conservative hold thresholds.",
                file=sys.stderr,
            )
        paper_trade_decision_model_path = (
            trade_decision_ablation_model_path
            if _shadow_variant_uses_reference_feature_ablation_model(official_shadow_variant)
            else trade_decision_model_path
        )
        paper_trade_decision_deploy_ready = (
            trade_decision_ablation_deploy_ready
            if _shadow_variant_uses_reference_feature_ablation_model(official_shadow_variant)
            else trade_decision_deploy_ready
        )
        if bool(trade_decision_cfg.get("enabled", True)) and bool(paper_trade_decision_deploy_ready):
            paper_cmd.extend(["--trade-decision-enabled", "--trade-decision-model", str(paper_trade_decision_model_path)])
            if trade_decision_cfg.get("threshold") is not None:
                paper_cmd.extend(["--trade-decision-threshold", str(float(trade_decision_cfg.get("threshold")))])
        elif bool(trade_decision_cfg.get("enabled", True)) and not args.dry_run:
            paper_cmd.append("--trade-decision-disabled")
            print(
                "Skipping trade-decision gate in paper-live because trained sample size is below configured minimum.",
                file=sys.stderr,
            )
        if bool(search_cfg.get("paper_live_dry_run", False)):
            paper_cmd.append("--dry-run")
        results.append(_run_step("paper_live_refresh", paper_cmd, logs_dir / "paper_live_refresh.log", args.dry_run))

        if direction_audit_input.exists() or args.dry_run:
            direction_audit_cmd = [
                python,
                "-m",
                "src.scripts.audit_direction_models",
                "--backtest-csv",
                str(direction_audit_input),
                "--latest-predictions",
                "artifacts/predictions/latest.json",
                "--output",
                str(summary_dir / "direction_model_audit.json"),
            ]
            results.append(
                _run_step(
                    "direction_model_audit",
                    direction_audit_cmd,
                    logs_dir / "direction_model_audit.log",
                    args.dry_run,
                )
            )
            if not args.dry_run:
                latest_audit_path = Path("artifacts/analysis/direction_model_audit_latest.json")
                latest_audit_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(summary_dir / "direction_model_audit.json", latest_audit_path)

        prediction_coherence_cmd = [
            python,
            "-m",
            "src.scripts.analyze_prediction_coherence",
            "--history-path",
            "artifacts/predictions/history.json",
            "--output",
            str(summary_dir / "prediction_coherence.json"),
        ]
        results.append(
            _run_step(
                "prediction_coherence",
                prediction_coherence_cmd,
                logs_dir / "prediction_coherence.log",
                args.dry_run,
            )
        )
        if not args.dry_run:
            latest_prediction_coherence_path = Path("artifacts/analysis/prediction_coherence_latest.json")
            latest_prediction_coherence_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(summary_dir / "prediction_coherence.json", latest_prediction_coherence_path)

        # Mark monitoring artifact with no-trade/fallback regime diagnostics.
        regime_path = summary_dir / "regime_diagnostics.json"
        if regime_path.exists() and not args.dry_run:
            try:
                regime_payload = _load_json(regime_path)
                _annotate_monitoring_with_regime(Path("artifacts/monitoring/latest.json"), regime_payload)
            except Exception as exc:  # pragma: no cover - best effort annotation
                print(f"Warning: failed to annotate monitoring latest with regime diagnostics: {exc}", file=sys.stderr)

        live_snapshot_path = summary_dir / "live_predictions_snapshot.json"
        live_snapshot_cmd = [
            python,
            "-m",
            "src.scripts.snapshot_live_predictions",
            "--run-id",
            str(run_dir.name),
            "--profile-id",
            str(run_profile_id),
            "--profile-name",
            str(run_profile_name),
            "--predictions-latest",
            "artifacts/predictions/latest.json",
            "--monitoring-latest",
            "artifacts/monitoring/latest.json",
            "--output",
            str(live_snapshot_path),
        ]
        results.append(
            _run_step(
                "live_prediction_snapshot",
                live_snapshot_cmd,
                logs_dir / "live_prediction_snapshot.log",
                args.dry_run,
            )
        )

        if run_profile_id == "midband_paper_evaluation":
            default_snapshot = _find_latest_profile_snapshot_run(
                run_root=args.run_root,
                current_run_id=str(run_dir.name),
                profile_id="default_runtime",
            )
            if default_snapshot is None:
                print(
                    "Warning: no prior default-runtime live snapshot found; skipping default-vs-midband live comparison.",
                    file=sys.stderr,
                )
            else:
                _, default_snapshot_path = default_snapshot
                live_comparison_cmd = [
                    python,
                    "-m",
                    "src.scripts.compare_default_vs_midband_paper_live_snapshots",
                    "--default-snapshot",
                    str(default_snapshot_path),
                    "--midband-snapshot",
                    str(live_snapshot_path),
                    "--output",
                    str(summary_dir / "default_vs_midband_paper_live_comparison.json"),
                ]
                results.append(
                    _run_step(
                        "default_vs_midband_paper_live_comparison",
                        live_comparison_cmd,
                        logs_dir / "default_vs_midband_paper_live_comparison.log",
                        args.dry_run,
                    )
                )

            longitudinal_live_cmd = [
                python,
                "-m",
                "src.scripts.build_default_vs_midband_paper_live_longitudinal",
                "--run-root",
                str(args.run_root),
                "--include-run-id",
                str(run_dir.name),
                "--include-profile-id",
                str(run_profile_id),
                "--include-profile-name",
                str(run_profile_name),
                "--include-snapshot",
                str(live_snapshot_path),
                "--output",
                str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
            ]
            results.append(
                _run_step(
                    "default_vs_midband_paper_live_longitudinal",
                    longitudinal_live_cmd,
                    logs_dir / "default_vs_midband_paper_live_longitudinal.log",
                    args.dry_run,
                )
            )

            watchlist_output = summary_dir / "default_vs_midband_paper_live_watchlist.json"
            watchlist_cmd = [
                python,
                "-m",
                "src.scripts.build_default_vs_midband_paper_live_watchlist",
                "--longitudinal-input",
                str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
                "--output",
                str(watchlist_output),
                "--target-matched-pairs",
                "8",
                "--early-operational-streak",
                "2",
                "--early-actionable-asymmetry-streak",
                "2",
            ]
            results.append(
                _run_step(
                    "default_vs_midband_paper_live_watchlist",
                    watchlist_cmd,
                    logs_dir / "default_vs_midband_paper_live_watchlist.log",
                    args.dry_run,
                )
            )

            canonical_watchlist_cmd = [
                python,
                "-m",
                "src.scripts.build_default_vs_midband_paper_live_watchlist",
                "--longitudinal-input",
                str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
                "--output",
                str(args.run_root / "default_vs_midband_paper_live_watchlist.json"),
                "--target-matched-pairs",
                "8",
                "--early-operational-streak",
                "2",
                "--early-actionable-asymmetry-streak",
                "2",
            ]
            results.append(
                _run_step(
                    "default_vs_midband_paper_live_watchlist_canonical",
                    canonical_watchlist_cmd,
                    logs_dir / "default_vs_midband_paper_live_watchlist_canonical.log",
                    args.dry_run,
                )
            )

    cadence_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "monthly_retrain_day": int(cadence_cfg.get("monthly_retrain_day", 1)),
        "weekly_recalibration_weekday": str(cadence_cfg.get("weekly_recalibration_weekday", "mon")),
        "trigger_file": str(summary_dir / "reliability_triggers.json"),
        "notes": "Trigger immediate retrain when reliability_triggers.global_trigger = true.",
    }
    cadence_path = summary_dir / "cadence_plan.json"
    cadence_path.write_text(json.dumps(cadence_payload, indent=2), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": str(args.config),
        "profile": {
            "id": run_profile_id,
            "name": run_profile_name,
        },
        "run_dir": str(run_dir),
        "steps": [
            {
                "name": step.name,
                "returncode": step.returncode,
                "log": str(step.log_path),
                "command": step.command,
            }
            for step in results
        ],
        "cadence_plan": str(cadence_path),
    }
    manifest_path = summary_dir / "workflow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nReliability workflow finished. Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return execute_reliability_workflow(args)


if __name__ == "__main__":
    main()
