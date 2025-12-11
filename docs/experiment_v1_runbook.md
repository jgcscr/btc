# Experiment v1 Runbook (BTC 1h)

This runbook describes how to reproduce the v1 backtest, historical paper trading, and realtime signal logging for the BTCUSDT 1h forecasting system.

## 1. Environment Setup

- Ensure Python 3.10+ and `pip` are available.
- From the repo root, install dependencies (adjust as needed):

```bash
pip install -r requirements.txt
```

- Configure access to the curated BigQuery table used for live-style feature loading:
  - Project: `jc-financial-466902`
  - Dataset: `btc_forecast_curated`
  - Table: `btc_features_1h`
- Verify GCP credentials (e.g., `GOOGLE_APPLICATION_CREDENTIALS`) are set so that `load_btc_features_1h` works.

## 2. Reproduce v1 Backtest (Signals on NPZ Splits)

Script: `src/scripts/backtest_signals.py`

- Command (v1, test split, with fees/slippage):

```bash
python -m src.scripts.backtest_signals \
  --dataset-path artifacts/datasets/btc_features_1h_splits.npz \
  --reg-model-dir artifacts/models/xgb_ret1h_v1 \
  --dir-model-dir artifacts/models/xgb_dir1h_v1 \
  --p-up-min 0.45 \
  --ret-min 0.00000 \
  --fee-bps 2.0 \
  --slippage-bps 1.0 \
  --use-test-split \
  --output-dir artifacts/analysis/backtest_signals_v1
```

- Expected key metrics (test split, ensemble vs direction-only baseline):

  - Ensemble (gross): `n_trades ≈ 721`, `hit_rate ≈ 0.717`, `cum_ret ≈ 1.71`.
  - Ensemble (net, 2 bps fee + 1 bp slippage): `cum_ret ≈ 1.50`, `hit_rate ≈ 0.675`.
  - Baseline (gross): `n_trades ≈ 764`, `hit_rate ≈ 0.644`, `cum_ret ≈ 1.35`.
  - Baseline (net): `cum_ret ≈ 1.12`, `hit_rate ≈ 0.607`.

- Output:
  - Per-bar CSV: `artifacts/analysis/backtest_signals_v1/backtest_signals.csv`.

## 3. Historical Paper Trading (Position-Aware)

Script: `src/scripts/paper_trade_loop.py`

- Command (v1, test split, ensemble-based entries/exits):

```bash
python -m src.scripts.paper_trade_loop \
  --dataset-path artifacts/datasets/btc_features_1h_splits.npz \
  --reg-model-dir artifacts/models/xgb_ret1h_v1 \
  --dir-model-dir artifacts/models/xgb_dir1h_v1 \
  --p-up-min 0.45 \
  --ret-min 0.00000 \
  --fee-bps 2.0 \
  --slippage-bps 1.0 \
  --use-test-split \
  --output-dir artifacts/analysis/paper_trade_v1
```

- Expected summary (test split, approximate):

  - `n_trades (entries) ≈ 339`
  - `hit_rate ≈ 0.80`
  - `cum_ret (log-sum) ≈ 1.51`
  - `max_drawdown (log) ≈ -0.085`

- Output:
  - Per-bar CSV with `position` and `equity`: `artifacts/analysis/paper_trade_v1/paper_trade.csv`.

## 4. Realtime Signal Logger (Live-Style)

Script: `src/scripts/run_signal_realtime.py`

- Command (single run, designed for hourly scheduling):

```bash
python -m src.scripts.run_signal_realtime \
  --reg-model-dir artifacts/models/xgb_ret1h_v1 \
  --dir-model-dir artifacts/models/xgb_dir1h_v1 \
  --log-path artifacts/live/paper_trade_realtime.csv
```

- Behavior:

  - Loads full `btc_features_1h` table from BigQuery and reconstructs features via `src/trading/signals.py`.
  - Uses the v1 regression and direction models to compute:
    - `p_up`, `ret_pred`, `signal_ensemble`, `signal_dir_only` for the latest completed bar.
  - Prints a JSON-style summary of the latest signal to stdout.
  - Appends one row per new bar to `artifacts/live/paper_trade_realtime.csv` with columns:
    - `ts`, `p_up`, `ret_pred`, `signal_ensemble`, `signal_dir_only`, `created_at`, `notes`.
  - If the latest bar has the same `ts` as the last logged row, it skips appending (idempotent per bar).

### 4.1 Suggested Scheduling

Example cron entry (run at 5 minutes past every hour, adjust path):

```cron
5 * * * * cd /workspaces/btc && /usr/bin/python -m src.scripts.run_signal_realtime \
  --reg-model-dir artifacts/models/xgb_ret1h_v1 \
  --dir-model-dir artifacts/models/xgb_dir1h_v1 \
  --log-path artifacts/live/paper_trade_realtime.csv >> logs/run_signal_realtime.log 2>&1
```

Ensure the environment (virtualenv, `PYTHONPATH`, GCP credentials) is correctly initialized in the cron context.

## 5. Where to Look

- Datasets: `artifacts/datasets/btc_features_1h_splits.npz`.
- Models:
  - Regression: `artifacts/models/xgb_ret1h_v1`.
  - Direction: `artifacts/models/xgb_dir1h_v1`.
- Backtest logs: `artifacts/analysis/backtest_signals_v1/backtest_signals.csv`.
- Historical paper-trading logs: `artifacts/analysis/paper_trade_v1/paper_trade.csv`.
- Realtime signal logs: `artifacts/live/paper_trade_realtime.csv`.

This runbook corresponds to the v1 experiment documented in `experiment_2024-10_to-2025-12_v1.md`.

## 6. Reading and Plotting the Logs

### 6.1 Backtest signals (`backtest_signals.csv`)

Plot ensemble vs direction-only equity curves:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "artifacts/analysis/backtest_signals_v1/backtest_signals.csv",
    parse_dates=["ts"],
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["ts"], df["equity_ens"], label="Ensemble (net)")
ax.plot(df["ts"], df["equity_dir"], label="Direction-only (net)")
ax.set_ylabel("Equity")
ax.legend()
ax.grid(True)
plt.show()
```

### 6.2 Historical paper trading (`paper_trade.csv`)

Plot the single position-aware equity curve:

```python
import pandas as pd

df = pd.read_csv(
    "artifacts/analysis/paper_trade_v1/paper_trade.csv",
    parse_dates=["ts"],
)
df["equity"].plot(figsize=(10, 5), title="Paper-trade equity (v1)")
```

Inspect trade-level behavior by filtering for entry/exit bars:

```python
import pandas as pd

df = pd.read_csv(
    "artifacts/analysis/paper_trade_v1/paper_trade.csv",
    parse_dates=["ts"],
)

entries = df[(df["position"].shift(1) == 0) & (df["position"] == 1)]
exits = df[(df["position"].shift(1) == 1) & (df["position"] == 0)]
```

### 6.3 Realtime log (`paper_trade_realtime.csv`)

Monitor thresholded signals over time:

```python
import pandas as pd
import matplotlib.pyplot as plt

df_live = pd.read_csv(
    "artifacts/live/paper_trade_realtime.csv",
    parse_dates=["ts"],
)

df_live.set_index("ts")[["p_up", "ret_pred"]].plot(
    subplots=True,
    figsize=(10, 6),
    title="Realtime signals",
)
plt.show()
```

Confirm there is at most one row per `ts`:

```python
import pandas as pd

df_live = pd.read_csv("artifacts/live/paper_trade_realtime.csv")
dup_ts = df_live["ts"].duplicated().any()
print("Any duplicate ts?", dup_ts)
```

### 6.4 Sanity-Check Checklist

- **Backtest (`backtest_signals.csv`)**:
  - `equity_ens` and `equity_dir` are smooth and monotonic in line with the cumulative sum of `ret_net_ens` / `ret_net_dir` (no unexplained huge jumps).

- **Paper trade (`paper_trade.csv`)**:
  - `position` only flips when `signal_ensemble` flips.
  - The sum of `ret_net` over the file is close to `ln(equity[-1])` (see detailed checks in section 9.4 of the main experiment doc).

- **Realtime (`paper_trade_realtime.csv`)**:
  - `ts` is strictly increasing with no duplicates.
  - The pattern of `p_up` / `ret_pred` and thresholded signals is broadly consistent with the historical backtest/paper-trade behavior around overlapping dates.
