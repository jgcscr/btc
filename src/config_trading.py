"""Central configuration for trading-related defaults.

These values capture the current "v1" default configuration for
thresholds and simple transaction-cost assumptions. Scripts should
import from here for their default CLI values but still allow
overrides via command-line flags.
"""

DEFAULT_P_UP_MIN: float = 0.45
DEFAULT_RET_MIN: float = 0.0

# Baseline 1h model artifact locations.
DEFAULT_REG_MODEL_DIR_1H: str = "artifacts/models/xgb_ret1h_v1"
DEFAULT_DIR_MODEL_DIR_1H: str = "artifacts/models/xgb_dir1h_v1"

# Optuna-tuned 1h profile overrides (Dec-2025 vintage).
OPTUNA_REG_MODEL_DIR_1H: str = "artifacts/models/xgb_ret1h_optuna"
OPTUNA_DIR_MODEL_DIR_1H: str = "artifacts/models/xgb_dir1h_optuna"
OPTUNA_P_UP_MIN_1H: float = 0.50
OPTUNA_RET_MIN_1H: float = -0.001

# Per-trade fee and slippage assumptions in basis points. These are
# intended as conservative but realistic defaults for a liquid BTCUSDT
# market on a major exchange.
DEFAULT_FEE_BPS: float = 2.0
DEFAULT_SLIPPAGE_BPS: float = 1.0
