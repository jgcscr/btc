"""Central configuration for trading-related defaults.

These values capture the current "v1" default configuration for
thresholds and simple transaction-cost assumptions. Scripts should
import from here for their default CLI values but still allow
overrides via command-line flags.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_P_UP_MIN: float = 0.45
DEFAULT_RET_MIN: float = 0.0

# Baseline 1h model artifact locations.
DEFAULT_REG_MODEL_DIR_1H: str = "artifacts/models/xgb_ret1h_v1"
DEFAULT_DIR_MODEL_DIR_1H: str = "artifacts/models/xgb_dir1h_v2"
DEFAULT_LSTM_MODEL_DIR_1H: str | None = "artifacts/models/lstm_dir1h_v1"
DEFAULT_BILSTM_MODEL_DIR_1H: str | None = "artifacts/models/bilstm_dir1h_v1"
DEFAULT_GRU_MODEL_DIR_1H: str | None = "artifacts/models/gru_dir1h_v1"
DEFAULT_CNN_LSTM_MODEL_DIR_1H: str | None = "artifacts/models/cnn_lstm_dir1h_v1"
DEFAULT_TRANSFORMER_MODEL_DIR_1H: str | None = "artifacts/models/transformer_dir1h_v1"
DEFAULT_TRANSFORMER_LARGE_MODEL_DIR_1H: str | None = "artifacts/models/transformer_dir1h_large"

DEFAULT_DIR_MODEL_PATH_1H: str = str(Path(DEFAULT_DIR_MODEL_DIR_1H) / "xgb_dir1h_model.json")

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
		"name": "transformer_large",
		"type": "transformer_large",
		"path": DEFAULT_TRANSFORMER_LARGE_MODEL_DIR_1H,
		"weight": 0.0,
		"optional": True,
		"label": "transformer-large",
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
		"path": DEFAULT_BILSTM_MODEL_DIR_1H,
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "gru",
		"type": "gru",
		"path": DEFAULT_GRU_MODEL_DIR_1H,
		"weight": 1.0,
		"optional": True,
	},
	{
		"name": "cnn_lstm",
		"type": "cnn_lstm",
		"path": DEFAULT_CNN_LSTM_MODEL_DIR_1H,
		"weight": 0.0,
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
