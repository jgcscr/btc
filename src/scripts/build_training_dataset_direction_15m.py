from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_15M
from src.data.labeling import binary_direction_labels, triple_barrier_direction_labels
from src.data.labeling import binary_direction_labels_with_no_trade
from src.data.bq_loader import load_btc_features_15m
from src.data.dataset_preparation import make_features_and_target, time_series_train_val_test_split
from src.scripts import build_training_dataset as hourly_builder
from src.scripts.build_training_dataset_15m import (
    CORE_MODEL_FEATURES_15M,
    EXPECTED_FREQ,
    PERIODS_PER_HOUR,
    _augment_price_features_15m,
    _merge_processed_features_15m,
    _recompute_return_targets_15m,
)
from src.trading.volatility import add_volatility_columns, split_volatility_arrays
from src.data.dataset_preparation import enforce_unique_hourly_index, repair_hourly_continuity

META_PATH = Path("artifacts/datasets/btc_features_15m_direction_meta.json")
OUTPUT_FILENAME = "btc_features_15m_direction_splits.npz"


def _build_direction_targets(
    df: pd.DataFrame,
    *,
    scheme: str,
    threshold: float,
    tb_horizon_steps: int,
    tb_vol_window: int,
    tb_upper_mult: float,
    tb_lower_mult: float,
    no_trade_abs_ret: float,
    no_trade_vol_mult: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    if "ret_15m" not in result.columns:
        result = _recompute_return_targets_15m(result)
    if scheme == "triple_barrier":
        labels, stats = triple_barrier_direction_labels(
            result["close"],
            horizon_steps=tb_horizon_steps,
            vol_window=tb_vol_window,
            upper_mult=tb_upper_mult,
            lower_mult=tb_lower_mult,
        )
        result["dir_15m"] = labels
        return result, stats
    if no_trade_abs_ret > 0.0 or no_trade_vol_mult > 0.0:
        labels, stats = binary_direction_labels_with_no_trade(
            result["ret_15m"],
            threshold=threshold,
            no_trade_abs_ret=no_trade_abs_ret,
            no_trade_vol_mult=no_trade_vol_mult,
            vol_window=tb_vol_window,
        )
        result["dir_15m"] = labels.astype(float)
        return result, stats
    result["dir_15m"] = np.where(
        result["ret_15m"].notna(),
        binary_direction_labels(result["ret_15m"], threshold).astype(float),
        np.nan,
    )
    return result, {}


def build_direction_dataset(
    output_dir: str,
    train_frac: float,
    val_frac: float,
    *,
    labeling_scheme: str,
    threshold: float,
    tb_horizon_steps: int,
    tb_vol_window: int,
    tb_upper_mult: float,
    tb_lower_mult: float,
    no_trade_abs_ret: float,
    no_trade_vol_mult: float,
    feature_reliability_json: str | None,
    feature_reliability_min_score: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    df = load_btc_features_15m(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_15M,
    )

    if df.empty:
        raise RuntimeError("Loaded empty DataFrame from BigQuery; check 15m curated table content.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df, dup_count, gap_count = enforce_unique_hourly_index(
        df,
        label="curated_features_15m",
        raise_on_gap=False,
        normalize_to_hour=False,
        expected_freq=EXPECTED_FREQ,
    )
    if dup_count == 0 and gap_count == 0:
        print("[curated_features_15m] 15m spacing verified; no duplicates detected.")
    elif gap_count:
        print(f"[curated_features_15m] Logged {gap_count} non-15m intervals; upstream gaps remain.")

    df, backfilled = repair_hourly_continuity(
        df,
        label="curated_features_15m",
        expected_freq=EXPECTED_FREQ,
    )
    if backfilled:
        print(f"[curated_features_15m] Reindexed with {backfilled} synthetic 15m rows via forward/back fill.")

    df = _merge_processed_features_15m(df, hourly_builder.PROCESSED_PATHS)
    df = hourly_builder._drop_external_source_columns(df)
    df = _augment_price_features_15m(df)
    df, volatility_columns = add_volatility_columns(
        df,
        realized_windows=hourly_builder.DEFAULT_REALIZED_WINDOWS,
        periods_per_hour=PERIODS_PER_HOUR,
    )
    df, _ = hourly_builder._drop_constant_features(df, hourly_builder.ZERO_VARIANCE_CANDIDATES)
    df = hourly_builder._drop_excluded_features(df)
    df, labeling_stats = _build_direction_targets(
        df,
        scheme=labeling_scheme,
        threshold=threshold,
        tb_horizon_steps=tb_horizon_steps,
        tb_vol_window=tb_vol_window,
        tb_upper_mult=tb_upper_mult,
        tb_lower_mult=tb_lower_mult,
        no_trade_abs_ret=no_trade_abs_ret,
        no_trade_vol_mult=no_trade_vol_mult,
    )

    df, dup_after_merge, gap_after_merge = enforce_unique_hourly_index(
        df,
        label="curated_features_15m_merged",
        raise_on_gap=False,
        normalize_to_hour=False,
        expected_freq=EXPECTED_FREQ,
    )
    if dup_after_merge:
        print(f"[curated_features_15m_merged] Removed {dup_after_merge} duplicates introduced during merge.")
    if gap_after_merge:
        print(
            f"[curated_features_15m_merged] Logged {gap_after_merge} non-15m intervals after merge; "
            "downstream consumers should handle upstream gaps.",
        )

    allowed_features = [feature for feature in CORE_MODEL_FEATURES_15M if feature in df.columns]
    for column in volatility_columns:
        if column in df.columns and column not in allowed_features:
            allowed_features.append(column)
    allowed_features = hourly_builder._append_technical_feature_columns(df, allowed_features)
    if feature_reliability_json:
        reliability_path = Path(feature_reliability_json)
        if reliability_path.exists():
            payload = json.loads(reliability_path.read_text(encoding="utf-8"))
            accepted = payload.get("accepted_features")
            score_map = payload.get("feature_scores", {}) if isinstance(payload, dict) else {}
            if isinstance(accepted, list):
                accepted_set = {str(v) for v in accepted}
                filtered: list[str] = []
                for feature in allowed_features:
                    if feature in accepted_set:
                        filtered.append(feature)
                        continue
                    score_obj = score_map.get(feature) if isinstance(score_map, dict) else None
                    score = None
                    if isinstance(score_obj, dict) and "score" in score_obj:
                        try:
                            score = float(score_obj["score"])
                        except Exception:
                            score = None
                    if score is not None and score >= float(feature_reliability_min_score):
                        filtered.append(feature)
                if filtered:
                    print(f"Feature reliability filter kept {len(filtered)} / {len(allowed_features)} features.")
                    allowed_features = filtered
    df = hourly_builder._enforce_feature_coverage(df, allowed_features)

    X, y = make_features_and_target(
        df,
        target_column="dir_15m",
        dropna=True,
        allowed_features=allowed_features,
    )

    splits = time_series_train_val_test_split(X, y, train_frac=train_frac, val_frac=val_frac)

    output_path = os.path.join(output_dir, OUTPUT_FILENAME)

    volatility_arrays = split_volatility_arrays(
        df.loc[X.index],
        volatility_columns,
        n_train=splits.X_train.shape[0],
        n_val=splits.X_val.shape[0],
    )

    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        feature_names=np.array(splits.feature_names),
        threshold=np.array([0.0], dtype=np.float32),
        **volatility_arrays,
    )

    meta_payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "row_count": int(len(X)),
        "feature_count": int(len(splits.feature_names)),
        "labeling_scheme": labeling_scheme,
        "threshold": float(threshold),
        "triple_barrier": {
            "horizon_steps": int(tb_horizon_steps),
            "vol_window": int(tb_vol_window),
            "upper_mult": float(tb_upper_mult),
            "lower_mult": float(tb_lower_mult),
            "stats": labeling_stats,
        },
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta_payload, indent=2))
    print(f"Saved 15m direction dataset to {output_path}")
    print(f"Wrote dataset meta summary to {META_PATH}")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a 15m direction dataset for sequence and tree models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets",
        help="Directory to save the prepared dataset splits.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of samples allocated to the training split (default: 0.7).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of samples allocated to the validation split (default: 0.15).",
    )
    parser.add_argument("--labeling-scheme", choices=("binary", "triple_barrier"), default="binary")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--tb-horizon-steps", type=int, default=1)
    parser.add_argument("--tb-vol-window", type=int, default=96)
    parser.add_argument("--tb-upper-mult", type=float, default=1.0)
    parser.add_argument("--tb-lower-mult", type=float, default=1.0)
    parser.add_argument("--no-trade-abs-ret", type=float, default=0.0)
    parser.add_argument("--no-trade-vol-mult", type=float, default=0.0)
    parser.add_argument("--feature-reliability-json", type=str, default=None)
    parser.add_argument("--feature-reliability-min-score", type=float, default=0.55)
    args = parser.parse_args(argv)

    build_direction_dataset(
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        labeling_scheme=args.labeling_scheme,
        threshold=args.threshold,
        tb_horizon_steps=args.tb_horizon_steps,
        tb_vol_window=args.tb_vol_window,
        tb_upper_mult=args.tb_upper_mult,
        tb_lower_mult=args.tb_lower_mult,
        no_trade_abs_ret=args.no_trade_abs_ret,
        no_trade_vol_mult=args.no_trade_vol_mult,
        feature_reliability_json=args.feature_reliability_json,
        feature_reliability_min_score=args.feature_reliability_min_score,
    )


if __name__ == "__main__":
    main()
