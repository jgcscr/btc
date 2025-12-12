import json
import os
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target
from src.data.onchain_loader import load_onchain_cached
from src.data.targets_multi_horizon import add_multi_horizon_targets
from src.training.lstm_model import LSTMDirectionClassifier


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


def _build_features_from_csv(
    features_path: str,
    target_column: str,
    horizons: List[int],
    onchain_path: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(features_path, parse_dates=["ts"])
    if "ts" not in df.columns:
        raise ValueError("Features CSV must include a 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    if onchain_path:
        df_onchain = load_onchain_cached(onchain_path)
        df_onchain = df_onchain.set_index("ts").reindex(df["ts"]).ffill().bfill().reset_index()
        df = df.merge(df_onchain, on="ts", how="left")

    df_targets = add_multi_horizon_targets(df, horizons=horizons, price_col="close")
    ret_cols = [f"ret_{h}h" for h in horizons]
    df_targets = df_targets.dropna(subset=ret_cols)

    X, _ = make_features_and_target(df_targets, target_column=target_column, dropna=False)

    drop_cols = [f"ret_{h}h" for h in horizons if f"ret_{h}h" in X.columns and f"ret_{h}h" != target_column]
    drop_cols.extend([f"dir_{h}h" for h in horizons if f"dir_{h}h" in X.columns])
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    return df_targets.reset_index(drop=True), X.reset_index(drop=True)


def prepare_data_for_signals(
    dataset_npz_path: str,
    target_column: str = "ret_1h",
    features_path: Optional[str] = None,
    onchain_path: Optional[str] = None,
) -> PreparedData:
    """Load full features from BigQuery and prepare ordered features + scaler.

    This mirrors the logic used in training and in the live signal script:
    - sort by ts
    - build X using make_features_and_target
    - enforce feature order from the NPZ dataset (if available)
    - fit a StandardScaler on the train split only
    """
    horizons: List[int] = [1, 4]
    feature_names = _load_feature_names_from_npz(dataset_npz_path)
    with np.load(dataset_npz_path, allow_pickle=True) as dataset_npz:
        if "horizons" in dataset_npz.files:
            horizons_arr = dataset_npz["horizons"].tolist()
            if isinstance(horizons_arr, list):
                horizons = [int(h) for h in horizons_arr]

    if features_path:
        df_all, X_all = _build_features_from_csv(
            features_path=features_path,
            target_column=target_column,
            horizons=horizons,
            onchain_path=onchain_path,
        )
        if feature_names is None:
            feature_names = list(X_all.columns)
    else:
        df_all_raw = _load_full_features_df()
        if "ts" not in df_all_raw.columns:
            raise ValueError("Expected a 'ts' column in the curated features table.")

        df_all_sorted = df_all_raw.sort_values("ts").reset_index(drop=True)
        df_all = df_all_sorted.dropna(subset=[target_column]).reset_index(drop=True)

        non_feature_cols = {"ts", target_column, "ret_fwd_3h"}
        feature_cols = [c for c in df_all.columns if c not in non_feature_cols]
        X_all = df_all[feature_cols].copy()

        if feature_names is None:
            feature_names = list(X_all.columns)

    missing_in_all = set(feature_names) - set(X_all.columns)
    if missing_in_all:
        raise RuntimeError(f"Full dataset is missing expected feature columns: {sorted(missing_in_all)}")

    X_all_ordered = X_all[feature_names].copy()

    n_total = len(X_all_ordered)
    if n_total == 0:
        raise RuntimeError("Feature matrix is empty after ordering columns.")

    n_train = int(n_total * 0.7)
    if n_train <= 0:
        raise RuntimeError("Not enough samples to compute training statistics for scaling.")

    col_means = X_all_ordered.iloc[:n_train].mean(axis=0, skipna=True)
    X_all_ordered = X_all_ordered.fillna(col_means)
    X_all_ordered = X_all_ordered.fillna(0.0)

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


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_lstm_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    resolved_dir = os.path.abspath(model_dir)
    summary_path = os.path.join(resolved_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"LSTM summary not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seq_len = int(summary.get("seq_len"))
    feature_names = summary.get("feature_names", [])
    if not feature_names:
        raise ValueError("LSTM summary missing feature_names")
    hyperparams = summary.get("hyperparams", {})
    hidden_size = int(hyperparams.get("hidden_size"))
    num_layers = int(hyperparams.get("num_layers"))
    dropout = float(hyperparams.get("dropout", 0.0))
    norm_type = str(hyperparams.get("norm_type", "none"))

    model_path = os.path.join(resolved_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM weights not found at {model_path}")

    torch_device = _resolve_device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    input_size = int(checkpoint.get("input_size", len(feature_names)))

    lstm_model = LSTMDirectionClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        norm_type=norm_type,
    )
    lstm_model.load_state_dict(state_dict)
    lstm_model.to(torch_device)
    lstm_model.eval()

    scaler_mean = None
    scaler_std = None
    scaler_path = summary.get("scaler_path")
    if scaler_path:
        resolved_scaler = scaler_path
        if not os.path.isabs(resolved_scaler):
            resolved_scaler = os.path.join(resolved_dir, os.path.basename(resolved_scaler))
        if os.path.exists(resolved_scaler):
            if resolved_scaler.endswith(".joblib"):
                scaler_payload = joblib_load(resolved_scaler)
                scaler_mean = scaler_payload.get("mean")
                scaler_std = scaler_payload.get("std")
            else:
                with np.load(resolved_scaler) as scaler_npz:
                    scaler_mean = scaler_npz.get("mean")
                    scaler_std = scaler_npz.get("std")

    return {
        "model": lstm_model,
        "device": torch_device,
        "seq_len": seq_len,
        "feature_names": feature_names,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


def load_models(
    reg_model_path: str,
    dir_model_path: Optional[str] = None,
    lstm_model_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    reg = XGBRegressor()
    reg.load_model(reg_model_path)
    models["reg"] = reg

    if dir_model_path:
        dir_model = XGBClassifier()
        dir_model.load_model(dir_model_path)
        models["dir"] = dir_model

    if lstm_model_dir:
        models["dir_lstm"] = _load_lstm_direction_model(lstm_model_dir, device)

    if "dir" not in models and "dir_lstm" not in models:
        raise ValueError("At least one direction model must be provided.")

    return models


def populate_lstm_cache_from_prepared(prepared: PreparedData, models: Dict[str, Any]) -> None:
    """Populate the cached scaled feature matrix required for LSTM inference.

    The realtime scripts reuse :func:`compute_signal_for_index`, which expects
    ``models["dir_lstm"]["scaled_features"]`` to contain the full scaled
    feature matrix aligned to the prepared dataframe. This helper centralizes
    that preparation so backtests and realtime pipelines stay consistent.
    """

    lstm_info = models.get("dir_lstm")
    if lstm_info is None:
        return

    lstm_feature_names = list(lstm_info.get("feature_names", []))
    if lstm_feature_names and lstm_feature_names != list(prepared.feature_names):
        raise RuntimeError("Feature names mismatch between LSTM model and prepared data.")

    feature_frame = prepared.X_all_ordered
    if lstm_feature_names:
        missing = set(lstm_feature_names) - set(feature_frame.columns)
        if missing:
            raise RuntimeError(f"Prepared data missing required LSTM feature columns: {sorted(missing)}")
        feature_frame = feature_frame[lstm_feature_names]

    scaler_mean = lstm_info.get("scaler_mean")
    scaler_std = lstm_info.get("scaler_std")

    if scaler_mean is not None and scaler_std is not None:
        mean_arr = np.asarray(scaler_mean, dtype=np.float32)
        std_arr = np.asarray(scaler_std, dtype=np.float32)
        std_arr[std_arr == 0.0] = 1.0
        matrix = feature_frame.to_numpy(dtype=np.float32, copy=False)
        scaled_matrix = (matrix - mean_arr) / std_arr
    else:
        scaled_matrix = prepared.scaler.transform(feature_frame).astype(np.float32)

    lstm_info["scaled_features"] = scaled_matrix


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
    dir_model = models.get("dir")
    lstm_info = models.get("dir_lstm")

    ret_pred_arr = reg.predict(X_scaled)
    ret_pred = float(ret_pred_arr[0])

    p_up: Optional[float] = None
    if lstm_info is not None:
        seq_len = int(lstm_info["seq_len"])
        scaled_features = lstm_info.get("scaled_features")
        if scaled_features is None:
            raise RuntimeError("LSTM model missing precomputed feature matrix.")
        if index + 1 >= seq_len:
            start = index + 1 - seq_len
            window = scaled_features[start : index + 1].astype(np.float32, copy=False)
            lstm_model: LSTMDirectionClassifier = lstm_info["model"]
            torch_device: torch.device = lstm_info["device"]
            with torch.no_grad():
                tensor = torch.from_numpy(window).unsqueeze(0).to(torch_device)
                logits = lstm_model(tensor)
                prob = torch.sigmoid(logits).item()
            p_up = float(prob)
        elif dir_model is None:
            # Insufficient history for the LSTM and no backup classifier; stay neutral.
            p_up = 0.5

    if p_up is None:
        if dir_model is None:
            raise RuntimeError("No direction model available to compute probabilities.")
        p_up_arr = dir_model.predict_proba(X_scaled)[:, 1]
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
