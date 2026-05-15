from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.runtime.component_diversity_support import resolve_component_group_map


def _clip_probability(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=1e-6, upper=1.0 - 1e-6)


def _infer_model_columns(frame: pd.DataFrame) -> list[str]:
    known_models = set(resolve_component_group_map().keys())
    columns: list[str] = []
    for column in frame.columns:
        if not column.startswith("p_up_"):
            continue
        suffix = column[len("p_up_") :].strip().lower()
        if not suffix or suffix not in known_models:
            continue
        columns.append(column)
    return sorted(columns)


def _infer_label_series(frame: pd.DataFrame, *, horizon_hours: float | None = None) -> pd.Series:
    if "y_true" in frame.columns:
        values = pd.to_numeric(frame["y_true"], errors="coerce")
        return values.astype("Int64")

    candidate_columns: list[str] = []
    if horizon_hours is not None:
        if float(horizon_hours).is_integer():
            candidate_columns.append(f"ret_{int(horizon_hours)}h")
        else:
            candidate_columns.append(f"ret_{horizon_hours:g}h")
    candidate_columns.extend(["ret_1h", "ret_4h", "ret_8h", "ret_12h"])
    for column in candidate_columns:
        if column in frame.columns:
            returns = pd.to_numeric(frame[column], errors="coerce")
            return (returns > 0.0).astype("Int64")
    raise ValueError("Could not infer a label column. Expected y_true or a return column such as ret_1h.")


def _decision_proxy(probability: pd.Series, *, trade_band: float) -> pd.Series:
    probability = probability.astype(float)
    decision = pd.Series(np.zeros(len(probability), dtype=int), index=probability.index)
    decision.loc[probability >= (0.5 + trade_band)] = 1
    decision.loc[probability <= (0.5 - trade_band)] = -1
    return decision


def _decision_proxy_return(decision: pd.Series, returns: pd.Series, *, fee_bps: float) -> pd.Series:
    fee = float(fee_bps) / 10_000.0
    gross = decision.astype(float) * returns.astype(float)
    active = decision.astype(int).ne(0).astype(float)
    return gross - active * fee


def _classification_metrics(
    probability: pd.Series,
    labels: pd.Series,
    returns: pd.Series,
    *,
    trade_band: float,
    fee_bps: float,
) -> Dict[str, float]:
    probability = _clip_probability(probability)
    labels = labels.astype(int)
    returns = returns.astype(float)
    predicted = (probability >= 0.5).astype(int)
    decision = _decision_proxy(probability, trade_band=trade_band)
    proxy_return = _decision_proxy_return(decision, returns, fee_bps=fee_bps)
    traded = decision.ne(0)
    accuracy = float((predicted == labels).mean()) if len(labels) else math.nan
    brier = float(np.mean(np.square(probability.to_numpy(dtype=float) - labels.to_numpy(dtype=float)))) if len(labels) else math.nan
    log_loss = float(
        -np.mean(
            labels.to_numpy(dtype=float) * np.log(probability.to_numpy(dtype=float))
            + (1.0 - labels.to_numpy(dtype=float)) * np.log(1.0 - probability.to_numpy(dtype=float))
        )
    ) if len(labels) else math.nan
    alignment = float((np.sign(decision[traded].astype(float)) == np.sign(returns[traded].astype(float))).mean()) if traded.any() else math.nan
    return {
        "rows": float(len(labels)),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "trade_rate": float(traded.mean()) if len(traded) else math.nan,
        "decision_alignment_rate": alignment,
        "decision_proxy_return_mean": float(proxy_return.mean()) if len(proxy_return) else math.nan,
    }


def _summarize_regime_stability(
    frame: pd.DataFrame,
    probability: pd.Series,
    labels: pd.Series,
    returns: pd.Series,
    *,
    trade_band: float,
    fee_bps: float,
) -> Dict[str, Any]:
    if "regime_state" not in frame.columns:
        return {"available": False, "by_regime": {}, "brier_range": None, "proxy_return_range": None}
    valid = frame.loc[frame["regime_state"].notna()].copy()
    if valid.empty:
        return {"available": False, "by_regime": {}, "brier_range": None, "proxy_return_range": None}

    by_regime: Dict[str, Dict[str, float]] = {}
    for regime, regime_frame in valid.groupby("regime_state"):
        regime_index = regime_frame.index
        by_regime[str(regime)] = _classification_metrics(
            probability.loc[regime_index],
            labels.loc[regime_index],
            returns.loc[regime_index],
            trade_band=trade_band,
            fee_bps=fee_bps,
        )

    brier_values = [metrics["brier"] for metrics in by_regime.values() if metrics.get("brier") is not None and not math.isnan(metrics["brier"])]
    proxy_values = [metrics["decision_proxy_return_mean"] for metrics in by_regime.values() if metrics.get("decision_proxy_return_mean") is not None and not math.isnan(metrics["decision_proxy_return_mean"])]
    return {
        "available": True,
        "by_regime": by_regime,
        "brier_range": float(max(brier_values) - min(brier_values)) if brier_values else None,
        "proxy_return_range": float(max(proxy_values) - min(proxy_values)) if proxy_values else None,
    }


def _mean_correlation(frame: pd.DataFrame, columns: Sequence[str]) -> float | None:
    if len(columns) < 2:
        return None
    corr = frame.loc[:, list(columns)].corr().to_numpy(dtype=float)
    if corr.shape[0] < 2:
        return None
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr[mask]
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def _family_probability(frame: pd.DataFrame, model_columns: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(model_columns)].mean(axis=1)


def _leave_one_out_probability(frame: pd.DataFrame, keep_columns: Sequence[str]) -> pd.Series:
    if not keep_columns:
        return pd.Series(np.full(len(frame), 0.5, dtype=float), index=frame.index)
    return frame.loc[:, list(keep_columns)].mean(axis=1)


def _fold_summary(
    frame: pd.DataFrame,
    probability_column: pd.Series,
    labels: pd.Series,
    returns: pd.Series,
    *,
    trade_band: float,
    fee_bps: float,
) -> Dict[str, Any]:
    if "fold" not in frame.columns:
        return {"available": False, "by_fold": {}, "metric_mean": {}, "metric_std": {}}

    by_fold: Dict[str, Dict[str, float]] = {}
    for fold, fold_frame in frame.groupby("fold"):
        fold_index = fold_frame.index
        by_fold[str(fold)] = _classification_metrics(
            probability_column.loc[fold_index],
            labels.loc[fold_index],
            returns.loc[fold_index],
            trade_band=trade_band,
            fee_bps=fee_bps,
        )

    metric_names = ["accuracy", "brier", "log_loss", "trade_rate", "decision_proxy_return_mean"]
    metric_mean: Dict[str, float] = {}
    metric_std: Dict[str, float] = {}
    for metric in metric_names:
        values = [float(metrics[metric]) for metrics in by_fold.values() if metric in metrics and not math.isnan(float(metrics[metric]))]
        if not values:
            continue
        metric_mean[metric] = float(np.mean(values))
        metric_std[metric] = float(np.std(values))
    return {"available": True, "by_fold": by_fold, "metric_mean": metric_mean, "metric_std": metric_std}


def _pairwise_model_correlation(frame: pd.DataFrame, model_columns: Sequence[str]) -> list[Dict[str, Any]]:
    if len(model_columns) < 2:
        return []
    corr = frame.loc[:, list(model_columns)].corr()
    rows: list[Dict[str, Any]] = []
    for left_idx, left in enumerate(model_columns):
        for right in model_columns[left_idx + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "corr": float(corr.loc[left, right]),
                }
            )
    rows.sort(key=lambda row: abs(float(row["corr"])), reverse=True)
    return rows


def analyze_direction_family_value(
    frame: pd.DataFrame,
    *,
    horizon_hours: float | None = None,
    trade_band: float = 0.05,
    fee_bps: float = 3.0,
) -> Dict[str, Any]:
    analysis = frame.copy()
    if horizon_hours is not None and "horizon_hours" in analysis.columns:
        analysis = analysis.loc[pd.to_numeric(analysis["horizon_hours"], errors="coerce") == float(horizon_hours)].copy()
    model_columns = _infer_model_columns(analysis)
    if not model_columns:
        raise ValueError("No component probability columns matching p_up_<model> were found.")

    labels = _infer_label_series(analysis, horizon_hours=horizon_hours)
    returns_column = "ret_1h" if "ret_1h" in analysis.columns else None
    if returns_column is None:
        raise ValueError("Expected ret_1h in the labeled backtest input.")
    returns = pd.to_numeric(analysis[returns_column], errors="coerce")

    valid = analysis.copy()
    valid["label"] = pd.to_numeric(labels, errors="coerce")
    valid["return_value"] = returns
    valid = valid.dropna(subset=model_columns + ["label", "return_value"]).copy()
    if valid.empty:
        raise ValueError("No valid rows remained after filtering for component probabilities and labels.")

    labels_valid = valid["label"].astype(int)
    returns_valid = valid["return_value"].astype(float)
    system_probability = pd.to_numeric(valid["p_up"], errors="coerce") if "p_up" in valid.columns else _family_probability(valid, model_columns)
    component_baseline = _family_probability(valid, model_columns)

    group_map = resolve_component_group_map()
    family_members: Dict[str, list[str]] = {}
    for column in model_columns:
        model_name = column[len("p_up_") :].strip().lower()
        family = group_map.get(model_name, "default")
        family_members.setdefault(family, []).append(column)
    family_members = {family: sorted(columns) for family, columns in sorted(family_members.items())}

    baseline_component_metrics = _classification_metrics(
        component_baseline,
        labels_valid,
        returns_valid,
        trade_band=trade_band,
        fee_bps=fee_bps,
    )
    baseline_system_metrics = _classification_metrics(
        system_probability.fillna(component_baseline),
        labels_valid,
        returns_valid,
        trade_band=trade_band,
        fee_bps=fee_bps,
    )
    baseline_decision = _decision_proxy(component_baseline, trade_band=trade_band)

    families_payload: Dict[str, Any] = {}
    for family, columns in family_members.items():
        probability = _family_probability(valid, columns)
        without_columns = [column for column in model_columns if column not in columns]
        without_probability = _leave_one_out_probability(valid, without_columns)
        without_metrics = _classification_metrics(
            without_probability,
            labels_valid,
            returns_valid,
            trade_band=trade_band,
            fee_bps=fee_bps,
        )
        without_decision = _decision_proxy(without_probability, trade_band=trade_band)
        families_payload[family] = {
            "members": [column[len("p_up_") :] for column in columns],
            "blend_metrics": _classification_metrics(
                probability,
                labels_valid,
                returns_valid,
                trade_band=trade_band,
                fee_bps=fee_bps,
            ),
            "mean_intra_family_correlation": _mean_correlation(valid, columns),
            "mean_correlation_to_component_baseline": float(valid.loc[:, columns].corrwith(component_baseline).mean()),
            "regime_stability": _summarize_regime_stability(
                valid,
                probability,
                labels_valid,
                returns_valid,
                trade_band=trade_band,
                fee_bps=fee_bps,
            ),
            "leave_one_out": {
                "metrics": without_metrics,
                "delta_vs_component_baseline": {
                    "accuracy": float(without_metrics["accuracy"] - baseline_component_metrics["accuracy"]),
                    "brier": float(without_metrics["brier"] - baseline_component_metrics["brier"]),
                    "log_loss": float(without_metrics["log_loss"] - baseline_component_metrics["log_loss"]),
                    "decision_proxy_return_mean": float(
                        without_metrics["decision_proxy_return_mean"] - baseline_component_metrics["decision_proxy_return_mean"]
                    ),
                },
                "decision_proxy_flip_rate": float((without_decision != baseline_decision).mean()),
                "fold_summary": _fold_summary(
                    valid,
                    without_probability,
                    labels_valid,
                    returns_valid,
                    trade_band=trade_band,
                    fee_bps=fee_bps,
                ),
            },
        }

    models_payload: Dict[str, Any] = {}
    for column in model_columns:
        model_name = column[len("p_up_") :].strip().lower()
        model_probability = pd.to_numeric(valid[column], errors="coerce")
        without_columns = [candidate for candidate in model_columns if candidate != column]
        without_probability = _leave_one_out_probability(valid, without_columns)
        without_metrics = _classification_metrics(
            without_probability,
            labels_valid,
            returns_valid,
            trade_band=trade_band,
            fee_bps=fee_bps,
        )
        models_payload[model_name] = {
            "family": group_map.get(model_name, "default"),
            "metrics": _classification_metrics(
                model_probability,
                labels_valid,
                returns_valid,
                trade_band=trade_band,
                fee_bps=fee_bps,
            ),
            "mean_correlation_to_others": _mean_correlation(valid, [column, *[candidate for candidate in model_columns if candidate != column]]),
            "leave_one_out_delta_vs_component_baseline": {
                "accuracy": float(without_metrics["accuracy"] - baseline_component_metrics["accuracy"]),
                "brier": float(without_metrics["brier"] - baseline_component_metrics["brier"]),
                "log_loss": float(without_metrics["log_loss"] - baseline_component_metrics["log_loss"]),
                "decision_proxy_return_mean": float(
                    without_metrics["decision_proxy_return_mean"] - baseline_component_metrics["decision_proxy_return_mean"]
                ),
            },
        }

    strongest_family = None
    strongest_family_delta = None
    for family, payload in families_payload.items():
        delta = payload["leave_one_out"]["delta_vs_component_baseline"]["brier"]
        if strongest_family is None or float(delta) > float(strongest_family_delta):
            strongest_family = family
            strongest_family_delta = delta

    pairwise = _pairwise_model_correlation(valid, model_columns)
    prune_candidates = [
        row for row in pairwise
        if abs(float(row["corr"])) >= 0.95
        and models_payload[row["right"][len("p_up_") :] if row["right"].startswith("p_up_") else row["right"]]["leave_one_out_delta_vs_component_baseline"]["brier"] <= 0.0
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(valid)),
        "trade_band": float(trade_band),
        "fee_bps": float(fee_bps),
        "model_columns": model_columns,
        "family_members": family_members,
        "baseline": {
            "system_probability": baseline_system_metrics,
            "component_mean": baseline_component_metrics,
        },
        "families": families_payload,
        "models": models_payload,
        "pairwise_model_correlation": pairwise,
        "recommendations": {
            "most_incremental_family_by_brier": {
                "family": strongest_family,
                "delta_vs_component_baseline": strongest_family_delta,
            },
            "high_correlation_prune_candidates": prune_candidates[:10],
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit direction-model families on a labeled backtest using calibration, decision-impact proxy, regime stability, and correlation metrics."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/monitoring/labeled_backtest_1h.csv"),
        help="Labeled backtest CSV containing per-model probabilities.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/direction_family_value_latest.json"),
        help="JSON file to write the audit summary to.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=None,
        help="Optional horizon filter applied when the input contains multiple horizons.",
    )
    parser.add_argument(
        "--trade-band",
        type=float,
        default=0.05,
        help="Neutral band around 0.5 used for the decision-impact proxy.",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=3.0,
        help="Per-trade friction used in the decision-impact return proxy.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    frame = pd.read_csv(args.input)
    summary = analyze_direction_family_value(
        frame,
        horizon_hours=args.horizon_hours,
        trade_band=args.trade_band,
        fee_bps=args.fee_bps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "rows": summary["rows"],
        "families": sorted(summary["families"].keys()),
    }, indent=2))


if __name__ == "__main__":
    main()