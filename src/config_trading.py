"""Central configuration for trading-related defaults.

These values capture the current "v1" default configuration for
thresholds and simple transaction-cost assumptions. Scripts should
import from here for their default CLI values but still allow
overrides via command-line flags.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_P_UP_MIN: float = 0.45
DEFAULT_RET_MIN: float = 0.0

# Baseline 1h model artifact locations.
_USE_MLFLOW_REGISTRY = os.getenv("USE_MLFLOW_REGISTRY", "").lower() in {"1", "true", "yes"}

DEFAULT_REG_MODEL_DIR_1H: str = (
	"models:/xgb_ret1h/latest" if _USE_MLFLOW_REGISTRY else "artifacts/models/xgb_ret1h_v1"
)
DEFAULT_DIR_MODEL_DIR_1H: str = (
	"models:/xgb_dir1h/latest" if _USE_MLFLOW_REGISTRY else "artifacts/models/xgb_dir1h_v5"
)
DEFAULT_LSTM_MODEL_DIR_1H: str | None = (
	"models:/lstm_dir1h/latest" if _USE_MLFLOW_REGISTRY else "artifacts/models/lstm_dir1h_v1"
)
DEFAULT_BILSTM_MODEL_DIR_1H: str | None = None
DEFAULT_GRU_MODEL_DIR_1H: str | None = None
DEFAULT_CNN_LSTM_MODEL_DIR_1H: str | None = None
DEFAULT_TRANSFORMER_MODEL_DIR_1H: str | None = (
	"models:/transformer_dir1h/latest" if _USE_MLFLOW_REGISTRY else "artifacts/models/transformer_dir1h_v1"
)
DEFAULT_TRANSFORMER_LARGE_MODEL_DIR_1H: str | None = None

# Per-horizon transformer defaults (prefer registry when enabled).
DEFAULT_TRANSFORMER_MODEL_DIR_BY_SUFFIX: dict[str, str | None] = {
	"15m": "models:/transformer_dir15m/latest"
	if _USE_MLFLOW_REGISTRY
	else "artifacts/models/transformer_dir15m_v1",
	"1h": DEFAULT_TRANSFORMER_MODEL_DIR_1H,
	"4h": "models:/transformer_dir4h/latest"
	if _USE_MLFLOW_REGISTRY
	else "artifacts/models/transformer_dir4h_v1",
	"8h": "models:/transformer_dir8h/latest"
	if _USE_MLFLOW_REGISTRY
	else "artifacts/models/transformer_dir8h_v1",
	"12h": "models:/transformer_dir12h/latest"
	if _USE_MLFLOW_REGISTRY
	else "artifacts/models/transformer_dir12h_v1",
}

DEFAULT_DIR_MODEL_PATH_1H: str = DEFAULT_DIR_MODEL_DIR_1H

# Structured direction-model registry describing each ensemble member.
# Fields: ``type`` (loader key), ``path`` (directory or model file), and
# ``weight`` (relative vote). ``name`` defaults to ``type`` when omitted.
DEFAULT_DIR_MODELS_1H: list[dict[str, object]] = [
	{
		"name": "transformer",
		"type": "transformer",
		"path": DEFAULT_TRANSFORMER_MODEL_DIR_1H,
		"weight": 2.0,
	},
	{
		"name": "lstm",
		"type": "lstm",
		"path": DEFAULT_LSTM_MODEL_DIR_1H,
		"weight": 1.0,
	},
	{
		"name": "bilstm",
		"type": "bilstm",
		"path": "artifacts/models/bilstm_dir1h_v1",
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "gru",
		"type": "gru",
		"path": "artifacts/models/gru_dir1h_v1",
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "cnn_lstm",
		"type": "cnn_lstm",
		"path": "artifacts/models/cnn_lstm_dir1h_v1",
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "cnn_bilstm",
		"type": "cnn_bilstm",
		"path": "artifacts/models/cnn_bilstm_dir1h_v1",
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "garch_lstm",
		"type": "garch_lstm",
		"path": "artifacts/models/garch_lstm_dir1h_v1",
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "xgb",
		"type": "xgb",
		"path": DEFAULT_DIR_MODEL_PATH_1H,
		"weight": 1.0,
	},
]

DEFAULT_DIR_MODEL_WEIGHTS_1H: str | None = ",".join(
	f"{entry['name']}:{entry.get('weight', 1.0)}"
	for entry in DEFAULT_DIR_MODELS_1H
	if entry.get("weight") is not None
)

# Optuna-tuned 1h profile overrides (Dec-2025 vintage).
OPTUNA_REG_MODEL_DIR_1H: str = "artifacts/models/xgb_ret1h_20251218T183003Z"
OPTUNA_DIR_MODEL_DIR_1H: str = "artifacts/models/xgb_dir1h_20251218T181909Z"
OPTUNA_LSTM_MODEL_DIR_1H: str = "artifacts/models/lstm_dir1h_20251218T181937Z"
OPTUNA_TRANSFORMER_MODEL_DIR_1H: str = "artifacts/models/transformer_dir1h_20251218T175733Z"
OPTUNA_P_UP_MIN_1H: float = 0.60
OPTUNA_RET_MIN_1H: float = 0.0
OPTUNA_DIR_MODEL_WEIGHTS_1H: str = "transformer:2,lstm:1,xgb:1"

# Per-trade fee and slippage assumptions in basis points. These are
# intended as conservative but realistic defaults for a liquid BTCUSDT
# market on a major exchange.
DEFAULT_FEE_BPS: float = 2.0
DEFAULT_SLIPPAGE_BPS: float = 1.0
