from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_MODEL_PATH = "artifacts/models/trade_decision_model.json"
DEFAULT_REFERENCE_CSV = "artifacts/reliability/20260515T023232Z/summary/backtest_signals_meta_ensemble_decision_aligned_shadow_reference_feature_ablation.csv"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/trade_decision_sign_sanity_audit_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/trade_decision_sign_sanity_audit_latest.md"
DEFAULT_FOCUS_FEATURES = [
    "component_agreement_ratio",
    "component_entropy",
    "p_up_ret_mismatch",
    "confluence_direction_matches_dominant",
    "regime_is_neutral",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether trade-decision coefficient signs match empirical profitable-vs-losing feature direction."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-csv", default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--signal-col", default="signal_ensemble")
    parser.add_argument("--target-col", default="ret_ensemble_net")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--focus-feature",
        dest="focus_features",
        action="append",
        default=None,
        help="Feature to audit. Repeat to provide multiple features. Defaults to recurring 4h offenders.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(result):
        return float(default)
    return result


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines = ["# Trade Decision Sign Sanity Audit", ""]
    lines.append("## Summary")
    lines.append(f"- reference_csv: {payload.get('inputs', {}).get('reference_csv')}")
    lines.append(f"- candidate_rows: {payload.get('candidate_rows')}")
    lines.append(f"- profitable_candidate_rows: {payload.get('profitable_candidate_rows')}")
    lines.append("")
    lines.append("## Feature Audit")
    for item in payload.get("features", []):
        lines.append(
            f"- {item.get('feature')}: coefficient={item.get('coefficient')}, positive_mean={item.get('positive_mean')}, "
            f"negative_mean={item.get('negative_mean')}, mean_delta={item.get('mean_delta_positive_minus_negative')}, "
            f"sign_alignment={item.get('sign_alignment')}"
        )
    lines.append("")
    lines.append("## Recommendation")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    model_payload = json.loads(Path(args.model_path).read_text(encoding="utf-8"))
    reference_df = pd.read_csv(args.reference_csv)

    feature_columns = [str(value) for value in model_payload.get("feature_columns", [])]
    coefficients = [float(value) for value in model_payload.get("coefficients", [])]
    coef_map = dict(zip(feature_columns, coefficients))
    focus_features = list(args.focus_features or DEFAULT_FOCUS_FEATURES)

    signal = pd.to_numeric(reference_df.get(str(args.signal_col), 0.0), errors="coerce").fillna(0.0)
    target = pd.to_numeric(reference_df.get(str(args.target_col), 0.0), errors="coerce").fillna(0.0)
    candidate_df = reference_df.loc[signal > 0.0].copy()
    if candidate_df.empty:
        raise RuntimeError("No candidate rows found in reference CSV.")
    candidate_target = pd.to_numeric(candidate_df.get(str(args.target_col), 0.0), errors="coerce").fillna(0.0)
    positive_mask = candidate_target > 0.0
    negative_mask = ~positive_mask

    feature_results: List[Dict[str, Any]] = []
    for feature_name in focus_features:
        if feature_name not in coef_map:
            continue
        if feature_name not in candidate_df.columns:
            feature_results.append(
                {
                    "feature": feature_name,
                    "coefficient": float(coef_map[feature_name]),
                    "samples": 0,
                    "positive_mean": None,
                    "negative_mean": None,
                    "mean_delta_positive_minus_negative": None,
                    "sign_alignment": "missing_feature_column",
                }
            )
            continue

        values = pd.to_numeric(candidate_df[feature_name], errors="coerce").dropna()
        candidate_values = pd.to_numeric(candidate_df[feature_name], errors="coerce")
        valid_mask = candidate_values.notna()
        valid_positive = valid_mask & positive_mask
        valid_negative = valid_mask & negative_mask
        sample_count = int(valid_mask.sum())
        if sample_count < int(args.min_samples) or int(valid_positive.sum()) == 0 or int(valid_negative.sum()) == 0:
            feature_results.append(
                {
                    "feature": feature_name,
                    "coefficient": float(coef_map[feature_name]),
                    "samples": sample_count,
                    "positive_mean": None,
                    "negative_mean": None,
                    "mean_delta_positive_minus_negative": None,
                    "sign_alignment": "insufficient_samples",
                }
            )
            continue

        positive_mean = float(candidate_values.loc[valid_positive].mean())
        negative_mean = float(candidate_values.loc[valid_negative].mean())
        delta = float(positive_mean - negative_mean)
        coefficient = float(coef_map[feature_name])
        if delta == 0.0 or coefficient == 0.0:
            sign_alignment = "flat_or_zero"
        elif delta > 0.0 and coefficient > 0.0:
            sign_alignment = "aligned"
        elif delta < 0.0 and coefficient < 0.0:
            sign_alignment = "aligned"
        else:
            sign_alignment = "sign_mismatch"

        feature_results.append(
            {
                "feature": feature_name,
                "coefficient": coefficient,
                "samples": sample_count,
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "mean_delta_positive_minus_negative": delta,
                "sign_alignment": sign_alignment,
            }
        )

    mismatch_features = [item["feature"] for item in feature_results if item["sign_alignment"] == "sign_mismatch"]
    recommendations: List[str] = []
    if mismatch_features:
        recommendations.append(
            "The audited trade-decision features include empirical sign mismatches, which supports reviewing model weighting before relying on threshold changes."
        )
    else:
        recommendations.append(
            "The audited trade-decision feature signs are directionally consistent with the reference candidate rows, so the remaining 4h issue is more about live feature state than obvious coefficient inversion."
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model_path": str(args.model_path),
            "reference_csv": str(args.reference_csv),
            "signal_col": str(args.signal_col),
            "target_col": str(args.target_col),
            "focus_features": focus_features,
        },
        "candidate_rows": int(len(candidate_df)),
        "profitable_candidate_rows": int(positive_mask.sum()),
        "features": feature_results,
        "recommendations": recommendations,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote diagnostic JSON: {output_json}")
    print(f"Wrote diagnostic memo: {output_md}")


if __name__ == "__main__":
    main()