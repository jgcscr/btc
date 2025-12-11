import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without xgboost
    XGBOOST_AVAILABLE = False

from src.scripts.eval_equity_curves import _trading_metrics, _apply_costs


def _train_tiny_models(tmp_dir: str) -> Dict[str, str]:
    # Tiny synthetic dataset
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y_ret = np.array([0.0, 0.001, -0.001, 0.002], dtype=float)
    y_dir = (y_ret > 0.0).astype(int)

    reg = XGBRegressor(n_estimators=5, max_depth=2, learning_rate=0.3, n_jobs=1)
    reg.fit(X, y_ret)

    clf = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        n_jobs=1,
        eval_metric="logloss",
    )
    clf.fit(X, y_dir)

    reg_path = os.path.join(tmp_dir, "xgb_ret1h_model.json")
    dir_path = os.path.join(tmp_dir, "xgb_dir1h_model.json")
    reg.save_model(reg_path)
    clf.save_model(dir_path)

    return {"reg": reg_path, "dir": dir_path}


def test_trading_metrics_and_costs_smoke() -> None:
    # This test does not require xgboost; it only exercises metrics helpers.
    ret = np.array([0.001, -0.0005, 0.002], dtype=float)
    signal = np.array([1, 0, 1], dtype=int)

    metrics = _trading_metrics(ret, signal)
    assert set(metrics.keys()) == {
        "n_trades",
        "hit_rate",
        "avg_ret_trade",
        "cum_ret",
        "max_drawdown",
        "sharpe_like",
    }

    # Apply simple costs and ensure metrics still compute
    ret_net = _apply_costs(ret, signal, fee_bps=2.0, slippage_bps=1.0)
    metrics_net = _trading_metrics(ret_net, signal)
    assert metrics_net["n_trades"] == metrics["n_trades"]


def test_tiny_models_can_be_trained_and_saved() -> None:
    if not XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed in this environment")
    # This is a very lightweight smoke test ensuring that saving tiny
    # XGBoost models for later use in eval_equity_curves works end-to-end.
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = _train_tiny_models(tmp_dir)
        assert os.path.exists(paths["reg"])  # regression model json
        assert os.path.exists(paths["dir"])  # direction model json
