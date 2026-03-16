from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_COMPONENT_THRESHOLD = 0.5

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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ece_10(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, 11)
    total = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right >= 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        bucket_true = y_true[mask]
        bucket_prob = y_prob[mask]
        total += abs(float(bucket_true.mean()) - float(bucket_prob.mean())) * (bucket_true.size / y_true.size)
    return float(total)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _component_metrics(frame: pd.DataFrame, component_col: str, threshold: float) -> Dict[str, Any]:
    series = pd.to_numeric(frame[component_col], errors="coerce")
    label = pd.to_numeric(frame["y_true"], errors="coerce")
    ret = pd.to_numeric(frame.get("ret_1h"), errors="coerce")
    valid = pd.DataFrame({"prob": series, "y_true": label, "ret_1h": ret}).dropna(subset=["prob", "y_true"])
    if valid.empty:
        return {
            "rows": 0,
            "auc": float("nan"),
            "brier": float("nan"),
            "ece_10": float("nan"),
            "accuracy": float("nan"),
            "trade_count": 0,
            "hit_rate": float("nan"),
            "mean_realized_return": float("nan"),
            "cum_realized_return": float("nan"),
        }

    y_true = valid["y_true"].to_numpy(dtype=float)
    y_prob = valid["prob"].clip(0.0, 1.0).to_numpy(dtype=float)
    y_hat = (y_prob >= threshold).astype(int)
    realized = valid["ret_1h"].fillna(0.0).to_numpy(dtype=float)
    trade_mask = y_hat == 1
    trade_returns = realized[trade_mask]
    trade_hits = (realized[trade_mask] > 0.0).astype(float) if trade_returns.size else np.array([], dtype=float)

    return {
        "rows": int(valid.shape[0]),
        "auc": _safe_auc(y_true, y_prob),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "ece_10": _ece_10(y_true, y_prob),
        "accuracy": float(np.mean(y_hat == y_true.astype(int))),
        "trade_count": int(trade_returns.size),
        "hit_rate": float(trade_hits.mean()) if trade_hits.size else float("nan"),
        "mean_realized_return": float(trade_returns.mean()) if trade_returns.size else 0.0,
        "cum_realized_return": float(trade_returns.sum()) if trade_returns.size else 0.0,
    }


def _build_historical_audit(backtest_path: Path, threshold: float) -> Dict[str, Any]:
    frame = pd.read_csv(backtest_path)
    component_cols = [col for col in frame.columns if col.startswith("p_up_") and col not in {"p_up", "p_up_meta", "p_up_gate"}]
    if "y_true" not in frame.columns:
        raise KeyError(f"Expected 'y_true' in {backtest_path}")
    if not component_cols:
        raise KeyError(f"No component probability columns found in {backtest_path}")

    metrics = {col.removeprefix("p_up_"): _component_metrics(frame, col, threshold) for col in sorted(component_cols)}
    ranked = sorted(
        (
            {
                "component": component,
                **values,
            }
            for component, values in metrics.items()
        ),
        key=lambda item: (
            -999.0 if not np.isfinite(item["auc"]) else item["auc"],
            item["cum_realized_return"],
        ),
        reverse=True,
    )
    return {
        "source": str(backtest_path),
        "row_count": int(frame.shape[0]),
        "components": metrics,
        "ranked_components": ranked,
        "best_component_by_auc": ranked[0]["component"] if ranked else None,
        "worst_component_by_auc": ranked[-1]["component"] if ranked else None,
    }


def _direction_vote(probability: float) -> str:
    return "up" if probability >= 0.5 else "down"


def _build_live_audit(latest_payload: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(latest_payload, dict):
        return {
            "generated_at": None,
            "horizons": {},
            "disagreement_horizon_count": 0,
        }
    predictions = latest_payload.get("predictions", {}) if isinstance(latest_payload, dict) else {}
    horizon_rows: Dict[str, Any] = {}
    component_disagreement_count = 0
    for label, entry in predictions.items():
        components = entry.get("p_up_components", {}) if isinstance(entry.get("p_up_components"), dict) else {}
        if not components:
            continue
        probabilities = {name: float(value) for name, value in components.items()}
        votes = {name: _direction_vote(value) for name, value in probabilities.items()}
        up_count = sum(1 for vote in votes.values() if vote == "up")
        down_count = sum(1 for vote in votes.values() if vote == "down")
        dominant_direction = "up" if up_count > down_count else "down" if down_count > up_count else "split"
        dominant_ratio = max(up_count, down_count) / max(len(votes), 1)
        if dominant_direction == "split" or dominant_ratio < 0.67:
            component_disagreement_count += 1
        horizon_rows[label] = {
            "direction_next": entry.get("direction_next"),
            "p_up": float(entry.get("p_up", 0.0)),
            "ret_pred": float(entry.get("ret_pred", 0.0)),
            "trade_action": entry.get("trade_action"),
            "regime_state": entry.get("regime_state"),
            "component_probabilities": probabilities,
            "component_votes": votes,
            "dominant_component_direction": dominant_direction,
            "dominant_component_ratio": float(dominant_ratio),
            "component_prob_span": float(max(probabilities.values()) - min(probabilities.values())),
        }
    return {
        "generated_at": latest_payload.get("generated_at"),
        "horizons": horizon_rows,
        "disagreement_horizon_count": int(component_disagreement_count),
    }


def _build_recommendations(historical: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    recommendations: List[str] = []
    ranked = historical.get("ranked_components", [])
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        if np.isfinite(best.get("auc", float("nan"))):
            recommendations.append(
                f"1h historical leader is {best['component']} (auc={best['auc']:.3f}, cum_ret={best['cum_realized_return']:.4f})."
            )
        if np.isfinite(worst.get("auc", float("nan"))):
            recommendations.append(
                f"1h historical laggard is {worst['component']} (auc={worst['auc']:.3f}, cum_ret={worst['cum_realized_return']:.4f})."
            )
    disagreement_count = int(live.get("disagreement_horizon_count", 0))
    if disagreement_count > 0:
        recommendations.append(
            f"Live component disagreement is elevated across {disagreement_count} horizon(s); avoid increasing ensemble confidence when component votes are split."
        )
    for label, row in (live.get("horizons") or {}).items():
        if row.get("dominant_component_ratio", 0.0) < 0.67:
            recommendations.append(f"{label} is internally split across model families; treat its direction as low conviction.")
    return recommendations


def _build_weight_recommendations(historical: Dict[str, Any]) -> Dict[str, Any]:
    ranked = historical.get("ranked_components", []) or []
    component_metrics = historical.get("components", {}) or {}
    recommended_weights = {name: 0.0 for name in DEFAULT_COMPONENTS}
    promoted_components: List[str] = []
    demoted_components: List[str] = []

    viable: List[Dict[str, Any]] = []
    for row in ranked:
        component = str(row.get("component"))
        auc = row.get("auc", float("nan"))
        ece = row.get("ece_10", float("nan"))
        cum_ret = row.get("cum_realized_return", float("nan"))
        hit_rate = row.get("hit_rate", float("nan"))
        trade_count = row.get("trade_count", 0)
        if not np.isfinite(auc) or not np.isfinite(cum_ret):
            demoted_components.append(component)
            continue
        if float(auc) < 0.5 or float(cum_ret) <= 0.0 or (np.isfinite(ece) and float(ece) > 0.1) or int(trade_count) < 20:
            demoted_components.append(component)
            continue
        viable.append(row)

    if viable:
        best_auc = max(float(row["auc"]) for row in viable)
        for row in viable:
            component = str(row["component"])
            auc = float(row["auc"])
            if auc >= best_auc - 0.0025:
                recommended_weights[component] = 1.5
            else:
                recommended_weights[component] = 1.0
            promoted_components.append(component)

    weight_spec_parts = [f"{name}:{recommended_weights[name]:.1f}" for name in DEFAULT_COMPONENTS]
    return {
        "method": "historical_1h_component_filter",
        "criteria": {
            "min_auc": 0.5,
            "min_cum_realized_return": 0.0,
            "max_ece_10": 0.1,
            "min_trade_count": 20,
            "top_band_auc_tolerance": 0.0025,
        },
        "promoted_components": promoted_components,
        "demoted_components": sorted(set(demoted_components)),
        "recommended_weights": recommended_weights,
        "recommended_weight_spec_1h": ",".join(weight_spec_parts),
        "recommended_regime_weights_1h": {
            "trend_ignition": ",".join(weight_spec_parts),
            "neutral": ",".join(weight_spec_parts),
            "chop": ",".join(weight_spec_parts),
        },
        "input_components": sorted(component_metrics.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit direction model families using historical 1h components and live multi-horizon predictions.")
    parser.add_argument(
        "--backtest-csv",
        type=Path,
        default=Path("artifacts/reliability/20260313T231343Z/summary/backtest_signals_meta_ensemble_decision_aligned.csv"),
        help="Historical component-level backtest CSV. Defaults to the latest audited 1h meta-ensemble summary checked into the repo.",
    )
    parser.add_argument(
        "--latest-predictions",
        type=Path,
        default=Path("artifacts/predictions/latest.json"),
        help="Live prediction summary JSON produced by run_refresh_and_predict.",
    )
    parser.add_argument(
        "--component-threshold",
        type=float,
        default=DEFAULT_COMPONENT_THRESHOLD,
        help="Probability threshold used to convert component probabilities into directional trades for audit purposes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/direction_model_audit_latest.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    historical = _build_historical_audit(args.backtest_csv, threshold=float(args.component_threshold))
    latest_payload = _load_json(args.latest_predictions) if args.latest_predictions.exists() else None
    live = _build_live_audit(latest_payload)
    payload = {
        "historical_1h": historical,
        "live_multi_horizon": live,
        "weight_recommendations": _build_weight_recommendations(historical),
        "recommendations": _build_recommendations(historical, live),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()