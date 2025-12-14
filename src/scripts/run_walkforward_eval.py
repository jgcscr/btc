from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.bq_loader import load_btc_features_1h
from src.data.dataset_preparation import make_features_and_target
from src.data.targets_multi_horizon import add_multi_horizon_targets
from src.scripts.build_training_dataset import PROCESSED_PATHS as REG_PROCESSED_PATHS
from src.scripts.build_training_dataset import _merge_processed_features as merge_curated_features
from src.scripts.build_training_dataset_direction import make_direction_labels
from src.training.lstm_data import save_sequence_dataset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduleConfig:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    train_offset: pd.DateOffset
    val_offset: pd.DateOffset
    test_offset: pd.DateOffset
    step_offset: pd.DateOffset
    max_windows: int
    test_label: str


@dataclass(frozen=True)
class DataConfig:
    features_path: Optional[Path]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    strategy: str
    source_dir: Optional[Path]
    script: Optional[str]
    n_trials: Optional[int]
    seq_len: Optional[int]
    timeout_seconds: Optional[int]


@dataclass(frozen=True)
class BacktestConfig:
    name: str
    script: str
    thresholds: Dict[str, float]


@dataclass(frozen=True)
class HarnessConfig:
    schedule: ScheduleConfig
    data: DataConfig
    seq_len: int
    models: List[ModelConfig]
    backtests: List[BacktestConfig]


@dataclass(frozen=True)
class Window:
    label: str
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class DatasetArtifacts:
    regression_path: Path
    direction_path: Path
    multi_horizon_path: Path
    sequence_path: Path
    stats: Dict[str, Dict[str, int]]


def _to_utc(value: str, timezone: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone)
    return ts.tz_convert("UTC")


REQUIRED_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "fut_open",
    "fut_high",
    "fut_low",
    "fut_close",
    "fut_volume",
    "open_interest",
    "funding_rate",
    "ma_close_7h",
    "ma_close_24h",
    "ma_ratio_7_24",
    "vol_24h",
]


def _load_feature_source(where_clause: str, data_config: DataConfig) -> pd.DataFrame:
    if data_config.features_path is None:
        return load_btc_features_1h(where_clause=where_clause)

    path = data_config.features_path
    if not path.exists():
        raise FileNotFoundError(f"Configured features_path not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, parse_dates=["ts"])
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported features file extension for {path}")

    if "ts" not in df.columns:
        raise ValueError(f"Expected 'ts' column in features file {path}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _augment_required_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    rename_map: Dict[str, List[str]] = {
        "fut_open": ["coinapi_futures_open"],
        "fut_high": ["coinapi_futures_high"],
        "fut_low": ["coinapi_futures_low"],
        "fut_close": ["coinapi_futures_close"],
        "fut_volume": ["coinapi_futures_volume"],
        "funding_rate": ["coinapi_funding_funding_rate"],
    }

    for target, sources in rename_map.items():
        if target in result.columns:
            continue
        for source in sources:
            if source in result.columns:
                result[target] = result[source]
                break

    if "quote_volume" not in result.columns:
        if {"close", "volume"}.issubset(result.columns):
            result["quote_volume"] = result["close"].astype(float) * result["volume"].astype(float)
        else:
            result["quote_volume"] = 0.0

    if "num_trades" not in result.columns:
        volume_series = result.get("volume")
        if volume_series is not None:
            result["num_trades"] = volume_series.rolling(window=6, min_periods=1).mean()
        else:
            result["num_trades"] = 0.0

    if "open_interest" not in result.columns:
        fallback = result.get("coinapi_futures_open_interest")
        if fallback is not None:
            result["open_interest"] = fallback
        else:
            volume_series = result.get("volume")
            if volume_series is not None:
                result["open_interest"] = volume_series.rolling(window=12, min_periods=1).sum()
            else:
                result["open_interest"] = 0.0

    close_series = result.get("close")
    if close_series is not None:
        close_values = close_series.astype(float)
        result["ma_close_7h"] = close_values.rolling(window=7, min_periods=1).mean()
        result["ma_close_24h"] = close_values.rolling(window=24, min_periods=1).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = result["ma_close_7h"] / result["ma_close_24h"]
        result["ma_ratio_7_24"] = ratio.replace([np.inf, -np.inf], np.nan)
    else:
        result["ma_close_7h"] = 0.0
        result["ma_close_24h"] = 0.0
        result["ma_ratio_7_24"] = 0.0

    volume_series = result.get("volume")
    if volume_series is not None:
        result["vol_24h"] = volume_series.astype(float).rolling(window=24, min_periods=1).sum()
    else:
        result["vol_24h"] = 0.0

    for feature in REQUIRED_FEATURES:
        if feature not in result.columns:
            result[feature] = 0.0

    numeric_cols = [col for col in result.columns if col != "ts"]
    result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric, errors="coerce")
    result = result.sort_values("ts").reset_index(drop=True)
    result = result.ffill().bfill().fillna(0.0)
    return result


def _parse_offset(cfg: Dict[str, Any], prefix: str, default: Optional[Tuple[pd.DateOffset, str]] = None) -> Tuple[pd.DateOffset, str]:
    units = [
        ("months", "m"),
        ("days", "d"),
        ("hours", "h"),
    ]
    for unit, suffix in units:
        key = f"{prefix}_{unit}"
        if key in cfg:
            value = int(cfg[key])
            if value <= 0:
                raise ValueError(f"{key} must be > 0")
            offset_kwargs = {unit: value}
            return pd.DateOffset(**offset_kwargs), f"{value}{suffix}"
    if default is not None:
        return default
    raise ValueError(
        "Schedule must specify one of {prefix}_months, {prefix}_days, or {prefix}_hours".format(prefix=prefix)
    )


def _load_config(path: Path) -> HarnessConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    schedule_raw = raw.get("schedule") or {}
    preprocessing_raw = raw.get("preprocessing") or {}
    models_raw = raw.get("models") or {}
    backtests_raw = raw.get("backtests") or {}
    data_raw = raw.get("data") or {}

    timezone = schedule_raw.get("timezone", "UTC")

    train_offset, _ = _parse_offset(schedule_raw, "train")
    val_offset, _ = _parse_offset(schedule_raw, "val")
    test_offset, test_label = _parse_offset(schedule_raw, "test")
    step_offset, _ = _parse_offset(schedule_raw, "step", default=(test_offset, test_label))

    schedule = ScheduleConfig(
        name=str(schedule_raw.get("name", "walkforward")),
        start=_to_utc(str(schedule_raw["start"]), timezone),
        end=_to_utc(str(schedule_raw["end"]), timezone),
        train_offset=train_offset,
        val_offset=val_offset,
        test_offset=test_offset,
        step_offset=step_offset,
        max_windows=int(schedule_raw.get("max_windows", 6)),
        test_label=test_label,
    )

    seq_len = int(preprocessing_raw.get("seq_len", 24))

    models: List[ModelConfig] = []
    for name, cfg in models_raw.items():
        models.append(
            ModelConfig(
                name=name,
                strategy=str(cfg.get("strategy", "reuse")),
                source_dir=Path(cfg["source_dir"]) if cfg.get("source_dir") else None,
                script=cfg.get("script"),
                n_trials=int(cfg["n_trials"]) if cfg.get("n_trials") is not None else None,
                seq_len=int(cfg["seq_len"]) if cfg.get("seq_len") is not None else None,
                timeout_seconds=int(cfg["timeout_seconds"]) if cfg.get("timeout_seconds") is not None else None,
            )
        )

    backtests: List[BacktestConfig] = []
    for name, cfg in backtests_raw.items():
        thresholds = cfg.get("thresholds", {})
        backtests.append(
            BacktestConfig(
                name=name,
                script=str(cfg["script"]),
                thresholds={k: float(v) for k, v in thresholds.items()},
            )
        )

    data_cfg = DataConfig(features_path=Path(data_raw["features_path"]).expanduser() if data_raw.get("features_path") else None)

    return HarnessConfig(schedule=schedule, data=data_cfg, seq_len=seq_len, models=models, backtests=backtests)


def _iter_windows(schedule: ScheduleConfig) -> List[Window]:
    windows: List[Window] = []
    test_start = schedule.start
    while True:
        val_start = test_start - schedule.val_offset
        train_start = val_start - schedule.train_offset
        test_end = test_start + schedule.test_offset

        if test_end > schedule.end:
            break

        label = f"test_{test_start.strftime('%Y%m%d')}_{schedule.test_label}"
        windows.append(
            Window(
                label=label,
                train_start=train_start,
                val_start=val_start,
                val_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )

        test_start = test_start + schedule.step_offset
        if test_start >= schedule.end:
            break

    return windows


def _ensure_min_samples(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, context: str) -> None:
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(f"Split for {context} has empty sections (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")


def _fill_with_means(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    train_means = train_df.mean(axis=0, skipna=True)
    train_filled = train_df.fillna(train_means)
    filled = [train_filled]
    for df in other_dfs:
        filled.append(df.fillna(train_means))
    return filled


def _scale_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df)
    X_val = scaler.transform(val_df)
    X_test = scaler.transform(test_df)
    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)


def _save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    logger.info("Saved %s", path)


def _build_datasets(window: Window, seq_len: int, datasets_dir: Path, data_config: DataConfig) -> DatasetArtifacts:
    datasets_dir.mkdir(parents=True, exist_ok=True)

    def _bq_timestamp(ts: pd.Timestamp) -> str:
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        formatted = ts_utc.strftime("%Y-%m-%d %H:%M:%S")
        return f"TIMESTAMP('{formatted} UTC')"

    def _bq_ts_expression(column: str = "ts") -> str:
        """Builds a BigQuery expression that normalizes a timestamp column.

        The curated table stores ts either as TIMESTAMP or as various epoch-based
        integers depending on the ingestion batch. This expression converts the
        column to TIMESTAMP on-the-fly so the WHERE clause stays type-safe.
        """

        safe_int = f"SAFE_CAST({column} AS INT64)"
        safe_str = f"SAFE_CAST({column} AS STRING)"
        return "(CASE " \
            f"WHEN SAFE_CAST({column} AS TIMESTAMP) IS NOT NULL THEN SAFE_CAST({column} AS TIMESTAMP) " \
            f"WHEN {safe_int} IS NOT NULL THEN (CASE " \
            f"WHEN ABS({safe_int}) >= 1000000000000000 THEN TIMESTAMP_MICROS({safe_int}) " \
            f"WHEN ABS({safe_int}) >= 1000000000000 THEN TIMESTAMP_MILLIS({safe_int}) " \
            f"ELSE TIMESTAMP_SECONDS({safe_int}) END) " \
            f"WHEN {safe_str} IS NOT NULL THEN SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', {safe_str}) " \
            "ELSE NULL END)"

    ts_expression = _bq_ts_expression()
    where_clause = " AND ".join(
        [
            f"{ts_expression} >= {_bq_timestamp(window.train_start)}",
            f"{ts_expression} < {_bq_timestamp(window.test_end)}",
        ]
    )

    df_raw = _load_feature_source(where_clause, data_config)
    if df_raw.empty:
        raise RuntimeError(f"No curated features returned for window {window.label}.")

    df_merged = merge_curated_features(df_raw, REG_PROCESSED_PATHS)
    df_merged["ts"] = pd.to_datetime(df_merged["ts"], utc=True)
    df_merged = _augment_required_features(df_merged)

    df_merged = df_merged[(df_merged["ts"] >= window.train_start) & (df_merged["ts"] < window.test_end)].reset_index(drop=True)
    if df_merged.empty:
        raise RuntimeError(f"Merged features empty after filtering for window {window.label}.")

    if "ret_1h" not in df_merged.columns:
        df_merged["ret_1h"] = np.log(df_merged["close"].astype(float)).diff()

    df_filtered = df_merged.dropna(subset=["ret_1h"]).reset_index(drop=True)

    X_df, y_ret = make_features_and_target(df_filtered, target_column="ret_1h", dropna=False)
    ts_series = df_filtered["ts"]

    train_mask = (ts_series >= window.train_start) & (ts_series < window.val_start)
    val_mask = (ts_series >= window.val_start) & (ts_series < window.test_start)
    test_mask = (ts_series >= window.test_start) & (ts_series < window.test_end)

    train_df = X_df.loc[train_mask]
    val_df = X_df.loc[val_mask]
    test_df = X_df.loc[test_mask]

    _ensure_min_samples(train_df, val_df, test_df, f"regression {window.label}")

    filled_train, filled_val, filled_test = _fill_with_means(train_df, [val_df, test_df])
    X_train, X_val, X_test = _scale_splits(filled_train, filled_val, filled_test)

    y_train = y_ret.loc[train_mask].to_numpy(dtype=np.float32)
    y_val = y_ret.loc[val_mask].to_numpy(dtype=np.float32)
    y_test = y_ret.loc[test_mask].to_numpy(dtype=np.float32)

    regression_path = datasets_dir / "btc_features_1h_splits.npz"
    _save_npz(
        regression_path,
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": np.array(X_df.columns.to_list()),
        },
    )

    # Direction dataset
    direction_threshold = 0.0
    y_dir = make_direction_labels(y_ret, threshold=direction_threshold)
    y_dir_train = y_dir.loc[train_mask].to_numpy(dtype=np.float32)
    y_dir_val = y_dir.loc[val_mask].to_numpy(dtype=np.float32)
    y_dir_test = y_dir.loc[test_mask].to_numpy(dtype=np.float32)

    direction_path = datasets_dir / "btc_features_1h_direction_splits.npz"
    _save_npz(
        direction_path,
        {
            "X_train": X_train,
            "y_train": y_dir_train,
            "X_val": X_val,
            "y_val": y_dir_val,
            "X_test": X_test,
            "y_test": y_dir_test,
            "feature_names": np.array(X_df.columns.to_list()),
            "threshold": np.array([direction_threshold], dtype=np.float32),
        },
    )

    seq_path = datasets_dir / f"btc_features_1h_direction_seq_len{seq_len}.npz"
    save_sequence_dataset(str(direction_path), str(seq_path), seq_len)

    # Multi-horizon dataset
    df_multi = add_multi_horizon_targets(df_merged, horizons=[1, 4])
    df_multi = df_multi.dropna(subset=["ret_1h", "ret_4h"]).reset_index(drop=True)

    X_multi, y_multi = make_features_and_target(df_multi, target_column="ret_1h", dropna=False)
    ts_multi = pd.to_datetime(df_multi["ts"], utc=True)
    drop_cols = ["ret_4h", "dir_1h", "dir_4h", "ret_fwd_3h"]
    X_multi = X_multi.drop(columns=[col for col in drop_cols if col in X_multi.columns], errors="ignore")

    m_train_mask = (ts_multi >= window.train_start) & (ts_multi < window.val_start)
    m_val_mask = (ts_multi >= window.val_start) & (ts_multi < window.test_start)
    m_test_mask = (ts_multi >= window.test_start) & (ts_multi < window.test_end)

    m_train_df = X_multi.loc[m_train_mask]
    m_val_df = X_multi.loc[m_val_mask]
    m_test_df = X_multi.loc[m_test_mask]

    _ensure_min_samples(m_train_df, m_val_df, m_test_df, f"multi-horizon {window.label}")

    m_filled_train, m_filled_val, m_filled_test = _fill_with_means(m_train_df, [m_val_df, m_test_df])
    m_X_train, m_X_val, m_X_test = _scale_splits(m_filled_train, m_filled_val, m_filled_test)

    m_y_train = y_multi.loc[m_train_mask].to_numpy(dtype=np.float32)
    m_y_val = y_multi.loc[m_val_mask].to_numpy(dtype=np.float32)
    m_y_test = y_multi.loc[m_test_mask].to_numpy(dtype=np.float32)

    ret4h = df_multi["ret_4h"].to_numpy(dtype=np.float32)
    dir1h = df_multi["dir_1h"].to_numpy(dtype=np.float32)
    dir4h = df_multi["dir_4h"].to_numpy(dtype=np.float32)

    def _split_arr(values: np.ndarray, mask_train: pd.Series, mask_val: pd.Series, mask_test: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            values[mask_train.to_numpy()],
            values[mask_val.to_numpy()],
            values[mask_test.to_numpy()],
        )

    ret4h_train, ret4h_val, ret4h_test = _split_arr(ret4h, m_train_mask, m_val_mask, m_test_mask)
    dir1h_train, dir1h_val, dir1h_test = _split_arr(dir1h, m_train_mask, m_val_mask, m_test_mask)
    dir4h_train, dir4h_val, dir4h_test = _split_arr(dir4h, m_train_mask, m_val_mask, m_test_mask)

    multi_path = datasets_dir / "btc_features_multi_horizon_splits.npz"
    _save_npz(
        multi_path,
        {
            "X_train": m_X_train,
            "y_train": m_y_train,
            "X_val": m_X_val,
            "y_val": m_y_val,
            "X_test": m_X_test,
            "y_test": m_y_test,
            "feature_names": np.array(X_multi.columns.to_list()),
            "horizons": np.array([1, 4], dtype=np.int32),
            "direction_threshold": np.array([0.0], dtype=np.float32),
            "y_ret4h_train": ret4h_train,
            "y_ret4h_val": ret4h_val,
            "y_ret4h_test": ret4h_test,
            "y_dir1h_train": dir1h_train,
            "y_dir1h_val": dir1h_val,
            "y_dir1h_test": dir1h_test,
            "y_dir4h_train": dir4h_train,
            "y_dir4h_val": dir4h_val,
            "y_dir4h_test": dir4h_test,
        },
    )

    stats = {
        "regression": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "multi_horizon": {
            "train": int(m_train_mask.sum()),
            "val": int(m_val_mask.sum()),
            "test": int(m_test_mask.sum()),
        },
    }

    return DatasetArtifacts(
        regression_path=regression_path,
        direction_path=direction_path,
        multi_horizon_path=multi_path,
        sequence_path=seq_path,
        stats=stats,
    )


def _copy_model(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Model source not found: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    logger.info("Reused model artifacts from %s", source)


def _run_subprocess(cmd: List[str], cwd: Optional[Path], env: Optional[Dict[str, str]], capture: bool = False) -> subprocess.CompletedProcess:
    logger.info("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def _handle_models(models: List[ModelConfig], datasets: DatasetArtifacts, models_dir: Path, seq_len: int) -> Dict[str, str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, str] = {}

    for cfg in models:
        target_dir = models_dir / cfg.name
        strategy = cfg.strategy.lower()
        if strategy == "reuse":
            if cfg.source_dir is None:
                raise ValueError(f"Model {cfg.name} configured for reuse without source_dir")
            _copy_model(cfg.source_dir, target_dir)
            summaries[cfg.name] = f"reused from {cfg.source_dir}"
        elif strategy == "retrain_optuna":
            script = cfg.script or "src/scripts/search_transformer_optuna.py"
            trials = cfg.n_trials or 3
            seq = cfg.seq_len or seq_len
            target_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                script.replace(".py", "").replace("/", "."),
                "--dataset-path",
                str(datasets.direction_path),
                "--seq-len",
                str(seq),
                "--n-trials",
                str(trials),
                "--output-dir",
                str(target_dir),
                "--device",
                "cpu",
            ]
            if cfg.timeout_seconds:
                cmd.extend(["--timeout", str(cfg.timeout_seconds)])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""

            result = _run_subprocess(cmd, cwd=None, env=env, capture=True)
            summaries[cfg.name] = "retrained via Optuna"

            log_path = target_dir / "training_stdout.txt"
            log_path.write_text(result.stdout or "", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported strategy '{cfg.strategy}' for model {cfg.name}")

    return summaries


def _run_backtests(backtests: List[BacktestConfig], datasets: DatasetArtifacts, models_dir: Path, backtests_dir: Path) -> Dict[str, str]:
    backtests_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    for cfg in backtests:
        script_module = cfg.script.replace(".py", "").replace("/", ".")
        cmd = [
            sys.executable,
            "-m",
            script_module,
            "--dataset-path-reg",
            str(datasets.regression_path),
            "--reg-model-dir",
            str(models_dir / "xgb_ret1h"),
            "--dir-model-dir",
            str(models_dir / "xgb_dir1h"),
        ]
        if "p_up_min" in cfg.thresholds:
            cmd.extend(["--p-up-min", str(cfg.thresholds["p_up_min"])])
        if "ret_min" in cfg.thresholds:
            cmd.extend(["--ret-min", str(cfg.thresholds["ret_min"])])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        result = _run_subprocess(cmd, cwd=None, env=env, capture=True)
        output_text = result.stdout or ""
        log_path = backtests_dir / f"{cfg.name}.txt"
        log_path.write_text(output_text, encoding="utf-8")
        outputs[cfg.name] = output_text

    return outputs


def _enforce_retention(root_dir: Path, max_windows: int) -> None:
    windows = sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if len(windows) <= max_windows:
        return
    excess = windows[: len(windows) - max_windows]
    for path in excess:
        shutil.rmtree(path)
        logger.info("Pruned old window artifacts at %s", path)


def _parse_backtest_metrics(output_text: str) -> Dict[str, Dict[str, float]]:
    sections: Dict[str, Dict[str, float]] = {}
    current_key: Optional[str] = None
    keys = {"n_trades", "hit_rate", "avg_ret_per_trade", "cum_ret", "max_drawdown"}

    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Ensemble strategy"):
            current_key = "ensemble_strategy"
            sections[current_key] = {}
            continue
        if line.startswith("Direction-only baseline"):
            current_key = "direction_only_baseline"
            sections[current_key] = {}
            continue
        if ":" not in line or current_key is None:
            continue
        name, value = line.split(":", 1)
        name = name.strip()
        if name not in keys:
            continue
        value_str = value.strip().split()[0]
        try:
            sections[current_key][name] = float(value_str)
        except ValueError:
            continue

    return sections


def _write_window_summary(window_dir: Path, window: Window, datasets: DatasetArtifacts, model_notes: Dict[str, str], backtest_outputs: Dict[str, str]) -> Dict[str, object]:
    metrics_payload = {
        "datasets": datasets.stats,
        "models": model_notes,
        "backtests": {name: _parse_backtest_metrics(text) for name, text in backtest_outputs.items()},
    }

    summary = {
        "window": window.label,
        "train_start": window.train_start.isoformat(),
        "val_start": window.val_start.isoformat(),
        "test_start": window.test_start.isoformat(),
        "test_end": window.test_end.isoformat(),
        "datasets": {
            "regression": str(datasets.regression_path),
            "direction": str(datasets.direction_path),
            "multi_horizon": str(datasets.multi_horizon_path),
            "sequence": str(datasets.sequence_path),
            "stats": datasets.stats,
        },
        "models": model_notes,
        "backtests": backtest_outputs,
        "metrics": metrics_payload,
    }

    path = window_dir / "window_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_path = window_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return summary


def _update_schedule_summary(schedule_dir: Path, summaries: List[Dict[str, object]]) -> None:
    summary_path = schedule_dir / "summary_latest.json"
    existing: List[Dict[str, object]] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    existing_labels = {entry.get("window") for entry in existing}
    filtered = [entry for entry in existing if entry.get("window") not in {s["window"] for s in summaries}]
    filtered.extend(summaries)
    summary_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")


def run_harness(config: HarnessConfig, force: bool, dry_run: bool, only_windows: Optional[set[str]] = None) -> None:
    schedule_dir = Path("artifacts") / "walkforward" / config.schedule.name

    if dry_run:
        logger.info("[DRY RUN] Schedule directory would be %s", schedule_dir)
    else:
        schedule_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []

    all_windows = _iter_windows(config.schedule)
    if only_windows is not None:
        missing = [label for label in only_windows if label not in {w.label for w in all_windows}]
        if missing:
            raise ValueError(f"Requested windows not in schedule: {missing}")

    for window in all_windows:
        if only_windows is not None and window.label not in only_windows:
            continue
        if dry_run:
            logger.info("[DRY RUN] Window %s", window.label)
            logger.info(
                "[DRY RUN]  - Build datasets for %s to %s",
                window.train_start.isoformat(),
                window.test_end.isoformat(),
            )
            logger.info(
                "[DRY RUN]  - Model actions: %s",
                ", ".join(cfg.name for cfg in config.models) or "none",
            )
            logger.info(
                "[DRY RUN]  - Backtests: %s",
                ", ".join(cfg.name for cfg in config.backtests) or "none",
            )
            continue

        window_dir = schedule_dir / window.label
        if window_dir.exists() and not force:
            logger.info("Skipping existing window %s", window.label)
            continue
        if window_dir.exists() and force:
            shutil.rmtree(window_dir)
        window_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing window %s", window.label)

        datasets = _build_datasets(window, config.seq_len, window_dir / "datasets", config.data)
        model_notes = _handle_models(config.models, datasets, window_dir / "models", config.seq_len)
        backtest_outputs = _run_backtests(config.backtests, datasets, window_dir / "models", window_dir / "backtests")

        summary = _write_window_summary(window_dir, window, datasets, model_notes, backtest_outputs)
        summaries.append(summary)

        _enforce_retention(schedule_dir, config.schedule.max_windows)

    if dry_run:
        logger.info("[DRY RUN] Completed without writing artifacts.")
        return

    if summaries:
        _update_schedule_summary(schedule_dir, summaries)
    else:
        logger.info("No new windows processed; nothing to update.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only walk-forward evaluation harness.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML schedule configuration.")
    parser.add_argument("--force", action="store_true", help="Rebuild windows even if artifacts already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Log planned actions without writing artifacts.")
    parser.add_argument(
        "--only-window",
        action="append",
        dest="only_windows",
        default=None,
        help="Restrict execution to specific window label (can be passed multiple times).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    config = _load_config(Path(args.config))
    only_windows = set(args.only_windows) if args.only_windows else None
    run_harness(config, force=args.force, dry_run=args.dry_run, only_windows=only_windows)


if __name__ == "__main__":
    main()

