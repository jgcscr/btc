import json
import os
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
def _augment_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    def _safe_diff(series: pd.Series) -> pd.Series:
        return series.diff().fillna(0.0)

    def _safe_pct(series: pd.Series) -> pd.Series:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for base in ("close", "volume", "fut_close", "fut_volume"):
        if base not in result.columns:
            continue
        result[f"{base}_delta_1h"] = _safe_diff(result[base])
        result[f"{base}_pct_change_1h"] = _safe_pct(result[base])

    if "close" in result.columns:
        std_7 = result["close"].rolling(window=7, min_periods=3).std(ddof=0)
        std_24 = result["close"].rolling(window=24, min_periods=6).std(ddof=0)
        if "ma_close_7h" in result.columns:
            denom = std_7.replace(0.0, np.nan)
            result["close_zscore_7h"] = ((result["close"] - result["ma_close_7h"]) / denom).fillna(0.0)
        if "ma_close_24h" in result.columns:
            denom = std_24.replace(0.0, np.nan)
            result["close_zscore_24h"] = ((result["close"] - result["ma_close_24h"]) / denom).fillna(0.0)

    if "fut_close" in result.columns:
        rolling_mean = result["fut_close"].rolling(window=7, min_periods=3).mean()
        rolling_std = result["fut_close"].rolling(window=7, min_periods=3).std(ddof=0).replace(0.0, np.nan)
        result["fut_close_zscore_7h"] = ((result["fut_close"] - rolling_mean) / rolling_std).fillna(0.0)

    return result

from xgboost import XGBClassifier, XGBRegressor

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H
from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import enforce_unique_hourly_index, make_features_and_target
from src.data.onchain_loader import load_onchain_cached
from src.data.targets_multi_horizon import add_multi_horizon_targets
from src.scripts.build_training_dataset import PROCESSED_PATHS as REG_PROCESSED_PATHS
from src.scripts.build_training_dataset import _merge_processed_features as merge_curated_features
from src.training.lstm_model import LSTMDirectionClassifier
from src.models.transformer_classifier import TransformerDirectionClassifier
from src.trading.ensembles import simple_average, weighted_average


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
    df, _, gap_csv = enforce_unique_hourly_index(
        df,
        label="offline_features",
        raise_on_gap=False,
    )
    if gap_csv:
        print(f"[offline_features] Logged {gap_csv} non-hourly intervals; proceeding with gaps.")

    if onchain_path:
        df_onchain = load_onchain_cached(onchain_path)
        df_onchain = df_onchain.set_index("ts").reindex(df["ts"]).ffill().bfill().reset_index()
        df = df.merge(df_onchain, on="ts", how="left")
        df, _, gap_csv_merged = enforce_unique_hourly_index(
            df,
            label="offline_features_merged",
            raise_on_gap=False,
        )
        if gap_csv_merged:
            print(
                f"[offline_features_merged] Logged {gap_csv_merged} non-hourly intervals after merge; proceeding with gaps."
            )

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
        df_all_raw, _, gap_live = enforce_unique_hourly_index(
            df_all_raw,
            label="curated_features_live",
            raise_on_gap=False,
        )
        if gap_live:
            print(f"[curated_features_live] Logged {gap_live} non-hourly intervals; upstream feed has gaps.")
        if "ts" not in df_all_raw.columns:
            raise ValueError("Expected a 'ts' column in the curated features table.")

        df_all_sorted = df_all_raw.sort_values("ts").reset_index(drop=True)
        df_all_augmented = merge_curated_features(df_all_sorted, REG_PROCESSED_PATHS)
        df_all_augmented = _augment_price_features(df_all_augmented)
        df_all_augmented, _, gap_live_merged = enforce_unique_hourly_index(
            df_all_augmented,
            label="curated_features_live_merged",
            raise_on_gap=False,
        )
        if gap_live_merged:
            print(
                f"[curated_features_live_merged] Logged {gap_live_merged} non-hourly intervals after merge; upstream feed has gaps."
            )
        df_all = df_all_augmented.dropna(subset=[target_column]).reset_index(drop=True)

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
    df_all, _, _ = enforce_unique_hourly_index(df_all, label="realtime_features")
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


def _load_transformer_direction_model(model_dir: str, device: Optional[str]) -> Dict[str, Any]:
    resolved_dir = os.path.abspath(model_dir)
    summary_path = os.path.join(resolved_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Transformer summary not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seq_len = int(summary.get("seq_len"))
    feature_names = summary.get("feature_names", [])
    if not feature_names:
        raise ValueError("Transformer summary missing feature_names")
    hyperparams = summary.get("hyperparams", {})

    model_path = os.path.join(resolved_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer weights not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    input_size = int(checkpoint.get("input_size"))
    hidden_dim = int(checkpoint.get("hidden_dim", hyperparams.get("hidden_dim", 128)))
    num_heads = int(checkpoint.get("num_heads", hyperparams.get("num_heads", 4)))
    ffn_dim = int(checkpoint.get("ffn_dim", hyperparams.get("ffn_dim", hidden_dim * 2)))
    num_layers = int(checkpoint.get("num_layers", hyperparams.get("num_layers", 2)))
    dropout = float(checkpoint.get("dropout", hyperparams.get("dropout", 0.1)))
    use_layer_norm = bool(checkpoint.get("use_layer_norm", hyperparams.get("use_layer_norm", True)))

    torch_device = _resolve_device(device)
    transformer_model = TransformerDirectionClassifier(
        input_size=input_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=seq_len,
        use_layer_norm=use_layer_norm,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    transformer_model.load_state_dict(state_dict)
    transformer_model.to(torch_device)
    transformer_model.eval()

    scaler_mean = None
    scaler_std = None
    scaler_path = os.path.join(resolved_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler_payload = joblib_load(scaler_path)
        scaler_mean = scaler_payload.get("mean")
        scaler_std = scaler_payload.get("std")

    return {
        "model": transformer_model,
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
    transformer_model_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    reg = XGBRegressor()
    reg.load_model(reg_model_path)
    models["reg"] = reg
    reg_meta_path = Path(reg_model_path).with_name("model_metadata.json")
    if reg_meta_path.exists():
        try:
            metadata = json.loads(reg_meta_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        feature_names = metadata.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            models["reg_feature_names"] = [str(name) for name in feature_names]

    if dir_model_path:
        dir_model = XGBClassifier()
        dir_model.load_model(dir_model_path)
        models["dir"] = dir_model
        dir_meta_path = Path(dir_model_path).with_name("model_metadata_direction.json")
        if dir_meta_path.exists():
            try:
                dir_metadata = json.loads(dir_meta_path.read_text())
            except json.JSONDecodeError:
                dir_metadata = {}
            dir_feature_names = dir_metadata.get("feature_names")
            if isinstance(dir_feature_names, list) and dir_feature_names:
                models["dir_feature_names"] = [str(name) for name in dir_feature_names]

    if lstm_model_dir:
        models["dir_lstm"] = _load_lstm_direction_model(lstm_model_dir, device)

    if transformer_model_dir:
        models["dir_transformer"] = _load_transformer_direction_model(transformer_model_dir, device)

    if "dir" not in models and "dir_lstm" not in models and "dir_transformer" not in models:
        raise ValueError("At least one direction model must be provided.")

    return models


def populate_sequence_cache_from_prepared(prepared: PreparedData, models: Dict[str, Any]) -> None:
    """Populate cached scaled feature matrices required for sequence models."""

    sequence_keys = ["dir_lstm", "dir_transformer"]

    for key in sequence_keys:
        model_info = models.get(key)
        if model_info is None:
            continue

        model_feature_names = list(model_info.get("feature_names", []))
        if model_feature_names and model_feature_names != list(prepared.feature_names):
            raise RuntimeError("Feature names mismatch between sequence model and prepared data.")

        feature_frame = prepared.X_all_ordered
        if model_feature_names:
            missing = set(model_feature_names) - set(feature_frame.columns)
            if missing:
                raise RuntimeError(f"Prepared data missing required sequence feature columns: {sorted(missing)}")
            feature_frame = feature_frame[model_feature_names]

        scaler_mean = model_info.get("scaler_mean")
        scaler_std = model_info.get("scaler_std")

        if scaler_mean is not None and scaler_std is not None:
            mean_arr = np.asarray(scaler_mean, dtype=np.float32)
            std_arr = np.asarray(scaler_std, dtype=np.float32)
            std_arr[std_arr == 0.0] = 1.0
            matrix = feature_frame.to_numpy(dtype=np.float32, copy=False)
            scaled_matrix = (matrix - mean_arr) / std_arr
        else:
            scaled_matrix = prepared.scaler.transform(feature_frame).astype(np.float32)

        model_info["scaled_features"] = scaled_matrix


def _sequence_model_probability(model_info: Dict[str, Any], index: int) -> Optional[float]:
    seq_len = int(model_info.get("seq_len", 0))
    if seq_len <= 0:
        return None

    scaled_features = model_info.get("scaled_features")
    if scaled_features is None:
        raise RuntimeError("Sequence model missing precomputed feature matrix.")

    if index + 1 < seq_len:
        return None

    start = index + 1 - seq_len
    window = scaled_features[start : index + 1].astype(np.float32, copy=False)
    tensor = torch.from_numpy(window).unsqueeze(0).to(model_info["device"])
    model: torch.nn.Module = model_info["model"]
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def compute_signal_for_index(
    prepared: PreparedData,
    index: int,
    models: Dict[str, Any],
    p_up_min: float,
    ret_min: float,
    dir_model_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if not (0 <= index < len(prepared.df_all)):
        raise IndexError("Index out of range for prepared data.")

    ts_value = prepared.df_all["ts"].iloc[index]
    X_row = prepared.X_all_ordered.iloc[[index]]
    X_scaled = prepared.scaler.transform(X_row)
    X_scaled_df = pd.DataFrame(X_scaled, columns=prepared.feature_names)

    reg = models["reg"]
    dir_model = models.get("dir")
    lstm_info = models.get("dir_lstm")
    transformer_info = models.get("dir_transformer")

    reg_feature_names = models.get("reg_feature_names")
    if reg_feature_names:
        missing = [name for name in reg_feature_names if name not in X_scaled_df.columns]
        if missing:
            raise RuntimeError(f"Prepared data missing regression feature columns: {missing}")
        reg_input = X_scaled_df[reg_feature_names].to_numpy()
    else:
        reg_input = X_scaled

    ret_pred_arr = reg.predict(reg_input)
    ret_pred = float(ret_pred_arr[0])

    probabilities: Dict[str, float] = {}
    display_labels = {"xgb": "xgboost", "lstm": "lstm", "transformer": "transformer"}

    if dir_model is not None:
        dir_feature_names = models.get("dir_feature_names")
        if dir_feature_names:
            missing_dir = [name for name in dir_feature_names if name not in X_scaled_df.columns]
            if missing_dir:
                raise RuntimeError(f"Prepared data missing direction feature columns: {missing_dir}")
            dir_input = X_scaled_df[dir_feature_names].to_numpy()
        else:
            dir_input = X_scaled
        p_up_arr = dir_model.predict_proba(dir_input)[:, 1]
        probabilities["xgb"] = float(p_up_arr[0])

    if lstm_info is not None:
        seq_prob = _sequence_model_probability(lstm_info, index)
        if seq_prob is not None:
            probabilities["lstm"] = seq_prob

    if transformer_info is not None:
        seq_prob = _sequence_model_probability(transformer_info, index)
        if seq_prob is not None:
            probabilities["transformer"] = seq_prob

    p_up: Optional[float]
    direction_model_kind: Optional[str]

    if probabilities:
        if dir_model_weights:
            applicable_weights = {k: v for k, v in dir_model_weights.items() if k in probabilities}
        else:
            applicable_weights = {}

        if applicable_weights:
            try:
                p_up = weighted_average(probabilities, applicable_weights)
            except ValueError:
                p_up = simple_average(probabilities.values())
        else:
            p_up = simple_average(probabilities.values())

        direction_model_kind = (
            display_labels[next(iter(probabilities))]
            if len(probabilities) == 1
            else "ensemble"
        )
    else:
        if dir_model is None and (lstm_info is not None or transformer_info is not None):
            p_up = 0.5
            direction_model_kind = "fallback"
        else:
            raise RuntimeError("No direction model available to compute probabilities.")

    signal_ensemble = int((p_up >= p_up_min) and (ret_pred >= ret_min))
    signal_dir_only = int(p_up >= 0.5)

    result = {
        "ts": format_ts_iso(ts_value),
        "p_up": p_up,
        "ret_pred": ret_pred,
        "signal_ensemble": signal_ensemble,
        "signal_dir_only": signal_dir_only,
    }

    if probabilities:
        result["p_up_components"] = probabilities.copy()
        for name, value in probabilities.items():
            result[f"p_up_{name}"] = value

    if direction_model_kind is not None:
        result["direction_model_kind"] = direction_model_kind

    return result
