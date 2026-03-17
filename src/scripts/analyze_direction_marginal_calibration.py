from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.scripts.build_labeled_backtest_from_history import _build_multi_horizon_from_history


DEFAULT_COMPONENTS = [
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
]


def _parse_weight_spec(spec: str | None) -> Dict[str, float]:
    if not spec:
        return {}
    parsed: Dict[str, float] = {}
    for raw_chunk in str(spec).split(","):
        chunk = raw_chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        raw_name, raw_value = chunk.split(":", 1)
        try:
            weight = float(raw_value.strip())
        except ValueError:
            continue
        if weight > 0.0:
            parsed[raw_name.strip()] = weight
    return parsed


def _format_weight_spec(weights: Dict[str, float]) -> str:
    merged = {name: 0.0 for name in DEFAULT_COMPONENTS}
    for name, value in weights.items():
        if name in merged:
            merged[name] = float(value)
    return ",".join(f"{name}:{merged[name]:.1f}" for name in DEFAULT_COMPONENTS)


def _parse_horizon_value(value: str | float | int) -> float | None:
    try:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered.endswith("h"):
                return float(lowered[:-1])
            if lowered.endswith("m"):
                return float(lowered[:-1]) / 60.0
            return float(lowered)
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_benchmark_regime_specs(config_path: Path | None, horizon: str) -> Dict[str, str]:
    if config_path is None or not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    regime_weights = payload.get("regime_model_weights")
    if not isinstance(regime_weights, dict):
        return {}
    target_horizon = _parse_horizon_value(horizon)
    if target_horizon is None:
        return {}

    resolved: Dict[str, str] = {}
    for regime in ("trend_ignition", "neutral", "chop"):
        block = regime_weights.get(regime)
        if not isinstance(block, dict):
            continue
        spec = None
        for raw_key, raw_value in block.items():
            if _parse_horizon_value(raw_key) == target_horizon and raw_value is not None:
                spec = str(raw_value)
                break
        if spec:
            resolved[regime] = spec
    return resolved


def _mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.mean())
    if not np.isfinite(value):
        return None
    return value


def _expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if y_true.size == 0:
        return float("nan")
    bins_arr = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(bins_arr[:-1], bins_arr[1:]):
        if right >= 1.0:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not np.any(mask):
            continue
        total += abs(float(y_true[mask].mean()) - float(p[mask].mean())) * (np.sum(mask) / y_true.size)
    return float(total)


def _direction_accuracy(frame: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    valid = frame.loc[frame["p_up"].notna() & frame["y_true"].notna()].copy()
    if valid.empty:
        return {"rows": 0, "accuracy": float("nan"), "ece_10": float("nan")}
    predicted = (pd.to_numeric(valid["p_up"], errors="coerce") >= threshold).astype(int)
    labels = pd.to_numeric(valid["y_true"], errors="coerce").astype(int).to_numpy(dtype=int)
    probs = pd.to_numeric(valid["p_up"], errors="coerce").clip(0.0, 1.0).to_numpy(dtype=float)
    return {
        "rows": int(len(valid)),
        "accuracy": float((predicted.to_numpy(dtype=int) == labels).mean()),
        "ece_10": _expected_calibration_error(labels, probs, bins=10),
    }


def _top_feature_shifts(frame: pd.DataFrame, marginal_mask: pd.Series, top_n: int) -> List[Dict[str, Any]]:
    numeric_columns = [
        column
        for column in frame.columns
        if column.startswith("p_up_")
        or column in {
            "ret_pred",
            "expected_value",
            "volatility_realized_24h",
            "volatility_ewm_24h",
            "volatility_garch_like",
        }
    ]

    shifts: List[Dict[str, Any]] = []
    marginal = frame.loc[marginal_mask].copy()
    baseline = frame.loc[~marginal_mask].copy()
    for column in numeric_columns:
        marginal_values = pd.to_numeric(marginal[column], errors="coerce").dropna()
        baseline_values = pd.to_numeric(baseline[column], errors="coerce").dropna()
        if len(marginal_values) < 5 or len(baseline_values) < 5:
            continue
        baseline_std = float(baseline_values.std(ddof=0))
        if not np.isfinite(baseline_std) or baseline_std <= 1e-9:
            continue
        marginal_mean = float(marginal_values.mean())
        baseline_mean = float(baseline_values.mean())
        z_score = (marginal_mean - baseline_mean) / baseline_std
        shifts.append(
            {
                "feature": column,
                "marginal_mean": marginal_mean,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "mean_shift_z": float(z_score),
                "abs_mean_shift_z": float(abs(z_score)),
            }
        )
    shifts.sort(key=lambda item: item["abs_mean_shift_z"], reverse=True)
    return shifts[:top_n]


def _component_metrics(frame: pd.DataFrame, component_col: str) -> Dict[str, Any]:
    series = pd.to_numeric(frame[component_col], errors="coerce")
    labels = pd.to_numeric(frame["y_true"], errors="coerce")
    realized = pd.to_numeric(frame.get("ret_realized"), errors="coerce")
    valid = pd.DataFrame({"prob": series, "y_true": labels, "ret_realized": realized}).dropna(subset=["prob", "y_true"])
    if valid.empty:
        return {
            "rows": 0,
            "accuracy": float("nan"),
            "ece_10": float("nan"),
            "mean_realized_return": float("nan"),
        }

    y_true = valid["y_true"].astype(int).to_numpy(dtype=int)
    y_prob = valid["prob"].clip(0.0, 1.0).to_numpy(dtype=float)
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "rows": int(valid.shape[0]),
        "accuracy": float(np.mean(y_hat == y_true)),
        "ece_10": _expected_calibration_error(y_true, y_prob, bins=10),
        "mean_realized_return": _mean_or_none(valid["ret_realized"].dropna()),
    }


def _build_weight_recommendations(
    frame: pd.DataFrame,
    *,
    min_rows: int = 20,
    min_accuracy: float = 0.5,
    max_ece_10: float = 0.12,
    top_band_accuracy_tolerance: float = 0.01,
    method: str = "marginal_1h_component_filter",
) -> Dict[str, Any]:
    component_cols = [col for col in frame.columns if col.startswith("p_up_") and col not in {"p_up_meta", "p_up_gate"}]
    metrics = {col.removeprefix("p_up_"): _component_metrics(frame, col) for col in sorted(component_cols)}
    ranked = sorted(
        (
            {
                "component": component,
                **values,
            }
            for component, values in metrics.items()
        ),
        key=lambda item: (
            -999.0 if not np.isfinite(item.get("accuracy", float("nan"))) else item["accuracy"],
            999.0 if not np.isfinite(item.get("ece_10", float("nan"))) else -item["ece_10"],
        ),
        reverse=True,
    )

    recommended_weights = {name: 0.0 for name in DEFAULT_COMPONENTS}
    viable: List[Dict[str, Any]] = []
    demoted: List[str] = []
    for row in ranked:
        component = str(row.get("component"))
        accuracy = row.get("accuracy", float("nan"))
        ece = row.get("ece_10", float("nan"))
        rows = int(row.get("rows", 0) or 0)
        if (
            rows < int(min_rows)
            or not np.isfinite(accuracy)
            or float(accuracy) < float(min_accuracy)
            or (np.isfinite(ece) and float(ece) > float(max_ece_10))
        ):
            demoted.append(component)
            continue
        viable.append(row)

    if viable:
        best_accuracy = max(float(row["accuracy"]) for row in viable)
        for row in viable:
            component = str(row["component"])
            accuracy = float(row["accuracy"])
            recommended_weights[component] = 1.5 if accuracy >= best_accuracy - float(top_band_accuracy_tolerance) else 1.0

    weight_spec = _format_weight_spec(recommended_weights)
    return {
        "method": method,
        "criteria": {
            "min_rows": int(min_rows),
            "min_accuracy": float(min_accuracy),
            "max_ece_10": float(max_ece_10),
            "top_band_accuracy_tolerance": float(top_band_accuracy_tolerance),
        },
        "components": metrics,
        "ranked_components": ranked,
        "promoted_components": [name for name, value in recommended_weights.items() if value > 0.0],
        "demoted_components": sorted(set(demoted)),
        "recommended_weights": recommended_weights,
        "recommended_weight_spec_1h": weight_spec,
    }


def _ensemble_weight_metrics(frame: pd.DataFrame, weight_spec: str | None) -> Dict[str, Any] | None:
    weights = _parse_weight_spec(weight_spec)
    if not weights:
        return None
    component_cols = [f"p_up_{name}" for name, weight in weights.items() if weight > 0.0]
    if not component_cols:
        return None
    valid = frame.copy()
    valid["y_true"] = pd.to_numeric(valid["y_true"], errors="coerce")
    valid["ret_realized"] = pd.to_numeric(valid.get("ret_realized"), errors="coerce")
    for column in component_cols:
        valid[column] = pd.to_numeric(valid.get(column), errors="coerce")
    valid = valid.dropna(subset=["y_true", *component_cols])
    if valid.empty:
        return None

    weight_values = np.asarray([weights[column.removeprefix("p_up_")] for column in component_cols], dtype=float)
    component_matrix = valid[component_cols].to_numpy(dtype=float)
    probability = (component_matrix * weight_values).sum(axis=1) / weight_values.sum()
    labels = valid["y_true"].astype(int).to_numpy(dtype=int)
    predicted = (probability >= 0.5).astype(int)
    return {
        "rows": int(labels.size),
        "accuracy": float(np.mean(predicted == labels)),
        "ece_10": _expected_calibration_error(labels, probability, bins=10),
        "mean_realized_return": _mean_or_none(valid["ret_realized"].dropna()),
        "mean_p_up": float(np.mean(probability)),
    }


def _build_regime_specific_override_analysis(
    frame: pd.DataFrame,
    *,
    fallback_spec: str | None,
    benchmark_specs: Dict[str, str] | None,
    regime_min_rows: int,
    min_ece_improvement: float,
) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {
        "method": "regime_full_slice_component_filter",
        "criteria": {
            "regime_min_rows": int(regime_min_rows),
            "min_ece_improvement": float(min_ece_improvement),
            "requires_non_decreasing_accuracy": True,
        },
        "selected_regime_overrides": {},
        "apply_fallback_for_missing_regimes": True,
        "benchmark_source": "config" if benchmark_specs else "fallback_spec",
        "per_regime": {},
    }
    if "regime_state" not in frame.columns or not fallback_spec:
        return analysis

    selected: Dict[str, str] = {}
    for regime, group in frame.groupby("regime_state"):
        regime_name = str(regime)
        regime_rows = int(len(group))
        baseline_spec = (benchmark_specs or {}).get(regime_name, fallback_spec)
        baseline_metrics = _ensemble_weight_metrics(group, baseline_spec)
        candidate_recommendations = _build_weight_recommendations(
            group,
            min_rows=int(regime_min_rows),
            method="regime_full_slice_component_filter",
        )
        candidate_spec = candidate_recommendations.get("recommended_weight_spec_1h")
        candidate_metrics = _ensemble_weight_metrics(group, str(candidate_spec) if candidate_spec else None)

        selected_override = False
        reason = "no_material_improvement"
        if regime_rows < int(regime_min_rows):
            reason = "insufficient_rows"
        elif not candidate_recommendations.get("promoted_components"):
            reason = "no_viable_components"
        elif candidate_spec == fallback_spec:
            reason = "matches_fallback_spec"
        elif baseline_metrics is None or candidate_metrics is None:
            reason = "metrics_unavailable"
        else:
            baseline_accuracy = float(baseline_metrics.get("accuracy", float("nan")))
            candidate_accuracy = float(candidate_metrics.get("accuracy", float("nan")))
            baseline_ece = float(baseline_metrics.get("ece_10", float("nan")))
            candidate_ece = float(candidate_metrics.get("ece_10", float("nan")))
            if (
                np.isfinite(candidate_accuracy)
                and np.isfinite(baseline_accuracy)
                and np.isfinite(candidate_ece)
                and np.isfinite(baseline_ece)
                and candidate_accuracy + 1e-12 >= baseline_accuracy
                and baseline_ece - candidate_ece >= float(min_ece_improvement)
            ):
                selected_override = True
                reason = "ece_improved_without_accuracy_regression"
                selected[regime_name] = str(candidate_spec)

        analysis["per_regime"][regime_name] = {
            "rows": regime_rows,
            "fallback_spec": fallback_spec,
            "benchmark_spec": baseline_spec,
            "candidate_spec": candidate_spec,
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "selected": bool(selected_override),
            "selection_reason": reason,
            "candidate_promoted_components": candidate_recommendations.get("promoted_components", []),
        }

    analysis["selected_regime_overrides"] = selected
    analysis["apply_fallback_for_missing_regimes"] = not bool(selected)
    return analysis


def _regime_summary(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if "regime_state" not in frame.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for regime, group in frame.groupby("regime_state"):
        p_up = pd.to_numeric(group["p_up"], errors="coerce")
        y_true = pd.to_numeric(group["y_true"], errors="coerce")
        ret_realized = pd.to_numeric(group["ret_realized"], errors="coerce")
        valid = p_up.notna() & y_true.notna() & ret_realized.notna()
        if not valid.any():
            continue
        pred_05 = (p_up[valid] >= 0.5).astype(int)
        pred_06 = (p_up[valid] >= 0.6).astype(int)
        labels = y_true[valid].astype(int)
        rows.append(
            {
                "regime_state": str(regime),
                "rows": int(valid.sum()),
                "mean_p_up": _mean_or_none(p_up[valid]),
                "mean_ret_pred": _mean_or_none(pd.to_numeric(group.loc[valid, "ret_pred"], errors="coerce")),
                "mean_ret_realized": _mean_or_none(ret_realized[valid]),
                "accuracy_threshold_0p50": float((pred_05.to_numpy(dtype=int) == labels.to_numpy(dtype=int)).mean()),
                "accuracy_threshold_0p60": float((pred_06.to_numpy(dtype=int) == labels.to_numpy(dtype=int)).mean()),
            }
        )
    rows.sort(key=lambda item: item["rows"], reverse=True)
    return rows


def _bin_summary(frame: pd.DataFrame, lower: float, upper: float, step: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    left = lower
    while left < upper:
        right = min(left + step, upper)
        if right >= upper:
            mask = (frame["p_up"] >= left) & (frame["p_up"] <= right)
        else:
            mask = (frame["p_up"] >= left) & (frame["p_up"] < right)
        group = frame.loc[mask].copy()
        if not group.empty:
            y_true = pd.to_numeric(group["y_true"], errors="coerce").astype(int)
            pred = (pd.to_numeric(group["p_up"], errors="coerce") >= 0.5).astype(int)
            rows.append(
                {
                    "band": f"[{left:.3f}, {right:.3f}{']' if right >= upper else ')'}",
                    "rows": int(len(group)),
                    "mean_p_up": _mean_or_none(pd.to_numeric(group["p_up"], errors="coerce")),
                    "mean_ret_pred": _mean_or_none(pd.to_numeric(group["ret_pred"], errors="coerce")),
                    "mean_ret_realized": _mean_or_none(pd.to_numeric(group["ret_realized"], errors="coerce")),
                    "accuracy_threshold_0p50": float((pred.to_numpy(dtype=int) == y_true.to_numpy(dtype=int)).mean()),
                }
            )
        left = right
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the marginal 1h direction slice from labeled prediction history and summarize ranking drift."
    )
    parser.add_argument("--history-path", type=Path, default=Path("artifacts/predictions/history.json"))
    parser.add_argument("--spot-ohlcv-path", type=Path, default=Path("data/spot_klines"))
    parser.add_argument("--horizon", type=str, default="1h")
    parser.add_argument("--lookback-rows", type=int, default=2000)
    parser.add_argument("--lookback-hours", type=int, default=0)
    parser.add_argument("--fold-size", type=int, default=6)
    parser.add_argument("--include-reliability-snapshots", action="store_true")
    parser.add_argument("--lower", type=float, default=0.50)
    parser.add_argument("--upper", type=float, default=0.60)
    parser.add_argument("--bin-step", type=float, default=0.025)
    parser.add_argument("--top-feature-shifts", type=int, default=10)
    parser.add_argument("--regime-min-rows", type=int, default=15)
    parser.add_argument("--regime-min-ece-improvement", type=float, default=0.01)
    parser.add_argument("--benchmark-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/direction_marginal_1h_latest.json"))
    parser.add_argument("--rows-output", type=Path, default=Path("artifacts/analysis/direction_marginal_1h_rows.csv"))
    args = parser.parse_args()

    lookback_rows = int(args.lookback_rows) if int(args.lookback_rows) > 0 else None
    lookback_hours = int(args.lookback_hours) if int(args.lookback_hours) > 0 else None
    frame, meta = _build_multi_horizon_from_history(
        history_path=args.history_path,
        horizons=[args.horizon],
        spot_ohlcv_path=args.spot_ohlcv_path,
        fold_size=int(args.fold_size),
        lookback_rows=lookback_rows,
        lookback_hours=lookback_hours,
        include_reliability_snapshots=bool(args.include_reliability_snapshots),
    )
    frame = frame.copy()
    frame["p_up"] = pd.to_numeric(frame["p_up"], errors="coerce")
    frame["ret_pred"] = pd.to_numeric(frame.get("ret_pred"), errors="coerce")
    frame["ret_realized"] = pd.to_numeric(frame.get("ret_realized"), errors="coerce")
    frame["signal_dir_threshold_0p50"] = (frame["p_up"] >= 0.5).astype(int)
    frame["signal_dir_threshold_0p60"] = (frame["p_up"] >= 0.6).astype(int)
    frame["direction_ret_agree"] = (
        ((frame["signal_dir_threshold_0p50"] == 1) & (frame["ret_pred"] > 0.0))
        | ((frame["signal_dir_threshold_0p50"] == 0) & (frame["ret_pred"] < 0.0))
    )
    marginal_mask = frame["p_up"].between(float(args.lower), float(args.upper), inclusive="both")
    marginal = frame.loc[marginal_mask].copy()
    weight_recommendations = _build_weight_recommendations(marginal)
    benchmark_specs = _load_benchmark_regime_specs(args.benchmark_config, str(args.horizon))
    regime_override_analysis = _build_regime_specific_override_analysis(
        frame,
        fallback_spec=weight_recommendations.get("recommended_weight_spec_1h"),
        benchmark_specs=benchmark_specs,
        regime_min_rows=int(args.regime_min_rows),
        min_ece_improvement=float(args.regime_min_ece_improvement),
    )
    weight_recommendations["recommended_regime_weights_1h"] = regime_override_analysis.get(
        "selected_regime_overrides",
        {},
    )
    weight_recommendations["apply_fallback_for_missing_regimes"] = bool(
        regime_override_analysis.get("apply_fallback_for_missing_regimes", True)
    )
    weight_recommendations["regime_override_analysis"] = regime_override_analysis

    payload: Dict[str, Any] = {
        "meta": meta,
        "horizon": str(args.horizon),
        "benchmark_config": None if args.benchmark_config is None else str(args.benchmark_config),
        "marginal_band": {"lower": float(args.lower), "upper": float(args.upper)},
        "overall": {
            "rows": int(len(frame)),
            "threshold_0p50": _direction_accuracy(frame, 0.5),
            "threshold_0p60": _direction_accuracy(frame, 0.6),
        },
        "marginal": {
            "rows": int(len(marginal)),
            "share_of_total": float(len(marginal) / len(frame)) if len(frame) else 0.0,
            "threshold_0p50": _direction_accuracy(marginal, 0.5),
            "threshold_0p60": _direction_accuracy(marginal, 0.6),
            "direction_ret_agreement_rate": float(marginal["direction_ret_agree"].mean()) if len(marginal) else float("nan"),
            "mean_ret_pred": _mean_or_none(marginal["ret_pred"]) if len(marginal) else None,
            "mean_ret_realized": _mean_or_none(marginal["ret_realized"]) if len(marginal) else None,
        },
        "full_regime_summary": _regime_summary(frame),
        "regime_summary": _regime_summary(marginal),
        "probability_band_summary": _bin_summary(frame, float(args.lower), float(args.upper), float(args.bin_step)),
        "top_feature_shifts": _top_feature_shifts(frame, marginal_mask, int(args.top_feature_shifts)),
        "weight_recommendations": weight_recommendations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    marginal.to_csv(args.rows_output, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()