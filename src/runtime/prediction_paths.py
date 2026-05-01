from __future__ import annotations

from pathlib import Path


DATASET_DIR = Path("artifacts/datasets")
DATASET_1H_PATH = DATASET_DIR / "btc_features_1h_splits.npz"
DATASET_MULTI_PATH = DATASET_DIR / "btc_features_multi_horizon_splits.npz"
DATASET_15M_PATH = DATASET_DIR / "btc_features_15m_splits.npz"
MODEL_ROOT = Path("artifacts/models")
LATEST_PREDICTION_PATH = Path("artifacts/predictions/latest.json")
HISTORY_PREDICTION_PATH = Path("artifacts/predictions/history.json")
TRADE_READY_MONITOR_PATH = Path("artifacts/monitoring/trade_ready_summary.json")
MONITORING_LATEST_PATH = Path("artifacts/monitoring/latest.json")
META_BASELINE_JSON_PATH = Path("artifacts/monitoring/meta_baseline.json")
META_BASELINE_PARQUET_PATH = Path("artifacts/monitoring/meta_baseline.parquet")
META_BASELINE_SOURCE_CSV = Path("artifacts/backtests/backtest_signals_meta_ensemble.csv")
DATA_QUALITY_MONITOR_PATH = Path("artifacts/monitoring/data_quality_latest.json")
TREND_IGNITION_STATE_PATH = Path("artifacts/monitoring/trend_ignition_state.json")
DIRECTION_FALLBACK_STATE_PATH = Path("artifacts/monitoring/direction_fallback_state.json")
TARGET_RANGE_MODEL_DIR = Path("artifacts/models/target_ranges")