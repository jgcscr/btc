
# BTCUSDT Forecasting – Ingestion, Features, and BigQuery Setup

This repository provides a robust, multi-vendor pipeline for BTCUSDT forecasting, with premium macro, on-chain, and market data sources, resilient fallbacks, and transparent monitoring. It supports:

- Ingestion of macro, spot, perp, and funding data from premium providers (Alpha Vantage, CoinAPI, CryptoQuant, FRED)
- Feature engineering and monitoring with provenance and fallback logic
- Partitioned Parquet output for all processed features
- BigQuery integration for raw and processed tables
- End-to-end model training and deployment


## 1. Python environment

```bash
cd /workspaces/btc
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Required environment variables

Set the following environment variables for premium data access:

- `FRED_API_KEY` (required for FRED macro ingestion)
- `ALPHA_VANTAGE_API_KEY` (required for Alpha Vantage macro)
- `COINAPI_KEY` (required for CoinAPI spot, perp, and funding)
- `CQ_TOKEN` (required for CryptoQuant daily metrics)

These providers are now in active use. Free endpoints are no longer sufficient for full feature coverage.


## Quick Manual Prediction

Run the all-in-one refresh script to rebuild features, regenerate datasets, and emit 1h/4h/8h/12h predictions. The command writes `artifacts/predictions/latest.json` and appends to `artifacts/predictions/history.json`.

```bash
# Full refresh with live ingestion
python -m src.scripts.run_refresh_and_predict --targets 1,4,8,12

# CI-friendly smoke test (skips network calls and emits stub predictions)
python -m src.scripts.run_refresh_and_predict --dry-run --targets 1,4,8,12
```

Example `latest.json` payload (values truncated):

```json
{
	"generated_at": "2025-12-16T16:41:13.731354+00:00",
	"predictions": {
		"1h": {
			"timestamp": "2025-12-16T00:00:00Z",
			"horizon_hours": 1,
			"close": 86486.12,
			"p_up": 0.4756,
			"ret_pred": 0.0010,
			"projected_price": 86573.57,
			"signal_ensemble": 1,
			"signal_dir_only": 0
		},
		"4h": { "...": "..." },
		"8h": { "...": "..." },
		"12h": { "...": "..." }
	}
}
```

Pipeline regression tests covering the CLI live in `tests/pipeline/` and can be executed via `pytest tests/pipeline -q`.


## 2. Ingestion and Feature Engineering Overview

### Active Data Loaders

- **Alpha Vantage macro**: Ingests macroeconomic series (e.g., SPX, DXY) using premium Alpha Vantage endpoints. Requires `ALPHA_VANTAGE_API_KEY`. The expanded catalog can be refreshed in one shot:

	```bash
	python -m data.ingestors.alpha_vantage_macro --run-catalog
	```

	Default coverage (configurable via `ALPHA_VANTAGE_CATALOG_PATH`, see below):

	| Symbol | Description | Functions |
	| --- | --- | --- |
	| SPY | S&P 500 ETF | 60min intraday, daily |
	| QQQ | Nasdaq 100 ETF | 60min intraday, daily |
	| DXY | US Dollar Index | daily |
	| ^TNX | US 10Y Treasury Yield | daily |
	| VIX | CBOE Volatility Index | daily |
	| GLD | Gold ETF | 60min intraday, daily |
	| USO | Oil ETF | 60min intraday, daily |
	| HYG | High-Yield Corporate Bond ETF | 60min intraday, daily |

	Customize the list by setting `ALPHA_VANTAGE_CATALOG_PATH` to a JSON file matching the on-disk schema, and control throttling with `ALPHA_VANTAGE_SLEEP_SECONDS`. A minimal catalog override looks like:

	```json
	[
		{
			"symbol": "BTCUSD",
			"name": "Spot Bitcoin",
			"functions": [
				{"function": "TIME_SERIES_INTRADAY", "interval": "60min"},
				{"function": "TIME_SERIES_DAILY"}
			]
		}
	]
	```

	Pass `--audit` to summarize the most recent ingestions.
- **CoinAPI spot/perp/funding**: Loads spot and perpetual BTCUSDT market data, plus funding rates. Funding endpoint is premium and currently under vendor investigation (see status below). Requires `COINAPI_KEY`.
- **CryptoQuant daily fallback**: Ingests daily on-chain metrics (exchange flows, reserves, whale counts) using `CQ_TOKEN`. Hourly access is pending (see status below). Synthetic data is used for fallback if API is unavailable.
- **FRED macro**: Loads macroeconomic indicators (e.g., trade-weighted USD) using `FRED_API_KEY`.

### Feature Processors and Monitoring

After raw ingestion, run the feature processors to generate hourly/daily Parquet and monitoring summaries:

```bash
# Macro features
python -m data.processed.compute_macro_features
# CoinAPI features (spot, perp, funding, realized vol, basis, deltas)
python -m data.processed.compute_coinapi_features
# CryptoQuant features (daily fallback, resampled to hourly)
python -m data.processed.compute_cryptoquant_resampled
# On-chain features (if needed)
python -m data.processed.compute_onchain_features
```

Each processor emits:
- `data/processed/*/hourly_features.parquet` (or daily)
- `artifacts/monitoring/*_summary.json` (coverage, nulls, diagnostics)
- All dataset builders now pull from these processed Parquet files

### Monitoring

#### Alpha Vantage quota monitor

Run the lightweight quota monitor after any ingestion burst (and during nightly automation) to confirm remaining call headroom:

```bash
python -m src.scripts.monitor_alpha_vantage_quota
```

Optional environment variables:

- `ALPHA_VANTAGE_ALERT_THRESHOLD` (default `180`): per-key call ceiling for the current UTC day.

Example summary when all keys remain under the threshold:

```json
{
	"date": "2025-12-17",
	"threshold": 180.0,
	"keys": {
		"HVBSTQAQ43M17SQ1": {
			"calls": 147.0,
			"rate_limit_hits": 0,
			"remaining": 33.0,
			"last_updated": "2025-12-17T14:41:07.431991+00:00"
		}
	},
	"message": "All keys remain under threshold."
}
```


## 3. Vendor Status & Escalations

**CryptoQuant**: Hourly API access is pending (ticket CQ-2025-1213). Daily fallback and synthetic data are in use for now.

**CoinAPI**: Funding endpoint returns 404 for BTCUSDT perpetual (vendor escalation ongoing). Symbol resolution and diagnostics are logged for support.

**Instructions when access is restored:**
- Rerun the relevant ingestors (e.g., `data.ingestors.cryptoquant_daily`, `data.ingestors.coinapi_exchange`)
- Rerun the processors (`compute_cryptoquant_resampled.py`, `compute_coinapi_features.py`)
- Rebuild all dataset splits using the scripts in `src/scripts/` (e.g., `build_training_dataset.py`)

## 4. Create BigQuery dataset (once)

```bash
gcloud config set project jc-financial-466902

bq --location=us-central1 mk -d \
	--description "Raw BTC/crypto data for forecasting" \
	jc-financial-466902:btc_forecast_raw
```

If the dataset already exists, BigQuery will return an "Already exists" error, which is safe to ignore.


## 5. Load Parquet into BigQuery raw tables

### 4.1 Spot klines

Load all 1h spot klines that have been uploaded under the partitioned layout:

```bash
gcloud config set project jc-financial-466902

bq load \
	--source_format=PARQUET \
	jc-financial-466902:btc_forecast_raw.spot_klines \
	gs://jc-financial-466902-btc-forecast-data/raw/spot_klines/interval=1h/yyyy=*/mm=*/dd=*/*.parquet
```

### 4.2 Futures metrics

```bash
bq load \
	--source_format=PARQUET \
	jc-financial-466902:btc_forecast_raw.futures_metrics \
	gs://jc-financial-466902-btc-forecast-data/raw/futures_metrics/interval=1h/yyyy=*/mm=*/dd=*/*.parquet
```

If these commands succeed, the raw BigQuery tables are populated and you can
move on to curated feature tables and model training.


## 6. Model Training, API Serving, and Cloud Run Deployment

### 5.1 Train and save the model

```bash
cd /workspaces/btc
source .venv/bin/activate
pip install -r requirements.txt


# Build dataset splits from processed Parquet features
python -m src.scripts.build_training_dataset --output-dir artifacts/datasets

# Train baseline XGBoost model
python -m src.scripts.train_baseline_model \
	--dataset-path artifacts/datasets/btc_features_1h_splits.npz \
	--output-dir artifacts/models/xgb_ret1h_v1
```

### 5.2 Prepare model artifacts for API serving

```bash
cd /workspaces/btc
chmod +x scripts/prepare_model_for_api.sh
./scripts/prepare_model_for_api.sh
# This copies model files into src/api/model for Docker builds
```

### 5.3 Build and push Docker image to Artifact Registry

```bash
cd /workspaces/btc
gcloud config set project jc-financial-466902
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create btc-forecast-repo \
	--repository-format=docker \
	--location=us-central1 \
	--description="Docker images for BTC forecasting service"

gcloud builds submit \
	--tag us-central1-docker.pkg.dev/jc-financial-466902/btc-forecast-repo/btc-forecast-api:v1 \
	.
```

### 5.4 Deploy to Cloud Run

```bash
gcloud run deploy btc-forecast-api \
	--image=us-central1-docker.pkg.dev/jc-financial-466902/btc-forecast-repo/btc-forecast-api:v1 \
	--platform=managed \
	--region=us-central1 \
	--allow-unauthenticated \
	--memory=2Gi \
	--cpu=2
```

### 5.5 Test the deployed API

```bash
SERVICE_URL="https://btc-forecast-api-<your-id>.us-central1.run.app"

# Health check
curl "$SERVICE_URL/health"

# Prediction
curl -X POST "$SERVICE_URL/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"instances": [
			{
				"close": 43000.0,
				"volume": 123.45
			}
		]
	}'
```


---

## 7. Walk-forward evaluation (CPU-only)

The walk-forward harness clears `CUDA_VISIBLE_DEVICES`, so every refit (including the transformer Optuna sweep) runs on CPU inside the default Codespaces container. Artifacts are stored under `artifacts/walkforward/<schedule>/<window_id>/` and summarized per schedule.

### 7.1 Smoke check (7-day window)

Use the smoke config for a quick validation of the end-to-end wiring:

```bash
python -m src.scripts.run_walkforward_eval \
	--config configs/walkforward/monthly_cpu_smoke.yaml
```

Outputs land in `artifacts/walkforward/monthly_cpu_smoke/`; the latest run is mirrored to `summary_latest.json` for fast inspection.

### 7.2 Monthly schedule (full 1-month windows)

Run the full monthly schedule to refresh all production windows and append a consolidated CSV/JSON summary:

```bash
python -m src.scripts.run_walkforward_eval \
	--config configs/walkforward/monthly_cpu.yaml
```

Setting `--force` rebuilds an existing window directory. Aggregate metrics live in `artifacts/walkforward/monthly_cpu/summary.csv` and `summary.json`; each window folder (for example `test_20241001_1m`) keeps the datasets, retrained transformer checkpoints, and backtests for deeper analysis.

### 7.3 Regression metric checks

Compare fresh metrics against the stored baselines after any walk-forward or backtest run:

```bash
python -m src.scripts.metrics_diff \
	--baseline artifacts/baselines/walkforward_monthly_cpu_summary.json \
	--new artifacts/walkforward/monthly_cpu/summary.json
```

The diff tool inspects `hit_rate`, `cum_ret`/`cum_ret_net`, `max_drawdown`, `n_trades`, and `sharpe_like` with ±2%/±0.05 tolerances (drawdowns may worsen by at most 0.01). Use `--update` to promote a reviewed run as the new baseline.

**Experiment history and detailed status:** See `docs/experiment_2024-10_to-2025-12_v1.md` for a full log of ingestion, feature, and vendor status.

This completes the full pipeline: multi-vendor data ingestion → feature engineering → BigQuery → model training → API serving on Cloud Run.