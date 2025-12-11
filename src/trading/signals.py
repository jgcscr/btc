from dataclasses import dataclass
from datetime import timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h


@dataclass
class PreparedData:
    df_all: pd.DataFrame
    X_all_ordered: pd.DataFrame
    scaler: StandardScaler
    feature_names: List[str]


def _load_full_features_df() -> pd.DataFrame:
    df = load_btc_features_1h(
        project_id=PROJECT_ID,
        dataset_id=BQ_DATASET_CURATED,
        table_id=BQ_TABLE_FEATURES_1H,
    )
    if df.empty:
        raise RuntimeError(
            "Loaded empty DataFrame from BigQuery; check that the curated table has data.",
        )
    return df


def _load_feature_names_from_npz(path: str) -> Optional[List[str]]:
    try:
        data = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        return None

    if "feature_names" not in data.files:
        return None

    return data["feature_names"].tolist()


def _build_scaler_from_training(X_all_ordered: pd.DataFrame) -> StandardScaler:
    n = len(X_all_ordered)
    if n == 0:
        raise ValueError("Empty feature matrix; cannot build scaler.")

    n_train = int(n * 0.7)
    if n_train <= 0:
        raise ValueError("Not enough samples to define a training split.")

    X_train = X_all_ordered.iloc[:n_train]

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def prepare_data_for_signals(dataset_npz_path: str, target_column: str = "ret_1h") -> PreparedData:
    """Load full features from BigQuery and prepare ordered features + scaler.

    This mirrors the logic used in training and in the live signal script:
    - sort by ts
    - build X using make_features_and_target
    - enforce feature order from the NPZ dataset (if available)
    - fit a StandardScaler on the train split only
    """
    df_all_raw = _load_full_features_df()
    if "ts" not in df_all_raw.columns:
        raise ValueError("Expected a 'ts' column in the curated features table.")

    # Sort by ts and drop rows with NaN in the target column, mirroring
    # make_features_and_target and the dataset construction used for training.
    df_all_sorted = df_all_raw.sort_values("ts").reset_index(drop=True)
    df_all = df_all_sorted.dropna(subset=[target_column]).reset_index(drop=True)

    non_feature_cols = {"ts", target_column, "ret_fwd_3h"}
    feature_cols = [c for c in df_all.columns if c not in non_feature_cols]
    X_all = df_all[feature_cols].copy()

    feature_names = _load_feature_names_from_npz(dataset_npz_path)
    if feature_names is None:
        feature_names = list(X_all.columns)

    missing_in_all = set(feature_names) - set(X_all.columns)
    if missing_in_all:
        raise RuntimeError(f"Full dataset is missing expected feature columns: {sorted(missing_in_all)}")

    X_all_ordered = X_all[feature_names]

    scaler = _build_scaler_from_training(X_all_ordered)

    return PreparedData(
        df_all=df_all,
        X_all_ordered=X_all_ordered,
        scaler=scaler,
        feature_names=feature_names,
    )


def prepare_data_for_signals_from_ohlcv(
    df_features: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    train_frac: float = 0.7,
) -> PreparedData:
    """Build a ``PreparedData`` bundle directly from an OHLCV-derived dataframe.

    This is used for fallback realtime predictions when BigQuery-curated rows are
    unavailable; callers must supply a dataframe containing the same feature
    columns expected by the 1h models. Scaling is refit on the earliest portion
    of the data (``train_frac``) so the ensemble logic can reuse
    ``compute_signal_for_index`` unchanged.
    """

    if "ts" not in df_features.columns:
        raise ValueError("Expected dataframe to include a 'ts' column.")

    if feature_names is None:
        non_feature_cols = {"ts"}
        feature_names = [c for c in df_features.columns if c not in non_feature_cols]

    missing = set(feature_names) - set(df_features.columns)
    if missing:
        raise ValueError(f"Dataframe missing required feature columns: {sorted(missing)}")

    df_all = df_features.sort_values("ts").reset_index(drop=True)
    X_all_ordered = df_all[feature_names].copy()

    n_rows = len(X_all_ordered)
    if n_rows == 0:
        raise ValueError("Empty dataframe; cannot build PreparedData.")

    n_train = max(int(n_rows * train_frac), 1)
    scaler = StandardScaler()
    scaler.fit(X_all_ordered.iloc[:n_train])

    return PreparedData(
        df_all=df_all,
        X_all_ordered=X_all_ordered,
        scaler=scaler,
        feature_names=feature_names,
    )


def format_ts_iso(ts_value: Any) -> str:
    """Format a timestamp-like value as an RFC3339-like string with Z suffix.

    The curated table stores ``ts`` as an integer nanosecond timestamp. This
    helper accepts either a pandas ``Timestamp`` or an integer-like value and
    normalizes to UTC.
    """
    if isinstance(ts_value, pd.Timestamp):
        dt = ts_value.to_pydatetime().astimezone(timezone.utc)
    else:
        ts = pd.to_datetime(ts_value, unit="ns", utc=True)
        dt = ts.to_pydatetime().astimezone(timezone.utc)

    iso = dt.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def find_row_index_for_ts(df_all: pd.DataFrame, ts_str: str) -> int:
    """Find the row index for a given timestamp string.

    The ts column is stored as integer nanoseconds; parse the input and
    compare on that basis.
    """
    ts_parsed = pd.to_datetime(ts_str, utc=True)
    target_ns = int(ts_parsed.value)

    matches = np.where(df_all["ts"].to_numpy() == target_ns)[0]
    if matches.size == 0:
        raise ValueError(f"No row found with ts = {ts_str!r}")
    return int(matches[-1])


def load_models(reg_model_path: str, dir_model_path: str) -> Dict[str, Any]:
    reg = XGBRegressor()
    reg.load_model(reg_model_path)

    dir_model = XGBClassifier()
    dir_model.load_model(dir_model_path)

    return {"reg": reg, "dir": dir_model}


def compute_signal_for_index(
    prepared: PreparedData,
    index: int,
    models: Dict[str, Any],
    p_up_min: float,
    ret_min: float,
) -> Dict[str, Any]:
    if not (0 <= index < len(prepared.df_all)):
        raise IndexError("Index out of range for prepared data.")

    ts_value = prepared.df_all["ts"].iloc[index]
    X_row = prepared.X_all_ordered.iloc[[index]]
    X_scaled = prepared.scaler.transform(X_row)

    reg = models["reg"]
    dir_model = models["dir"]

    ret_pred_arr = reg.predict(X_scaled)
    p_up_arr = dir_model.predict_proba(X_scaled)[:, 1]

    ret_pred = float(ret_pred_arr[0])
    p_up = float(p_up_arr[0])

    signal_ensemble = int((p_up >= p_up_min) and (ret_pred >= ret_min))
    signal_dir_only = int(p_up >= 0.5)

    return {
        "ts": format_ts_iso(ts_value),
        "p_up": p_up,
        "ret_pred": ret_pred,
        "signal_ensemble": signal_ensemble,
        "signal_dir_only": signal_dir_only,
    }
