
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


## 2. Ingestion and Feature Engineering Overview

### Active Data Loaders

- **Alpha Vantage macro**: Ingests macroeconomic series (e.g., SPX, DXY) using premium Alpha Vantage endpoints. Requires `ALPHA_VANTAGE_API_KEY`.
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

**Experiment history and detailed status:** See `docs/experiment_2024-10_to-2025-12_v1.md` for a full log of ingestion, feature, and vendor status.

This completes the full pipeline: multi-vendor data ingestion → feature engineering → BigQuery → model training → API serving on Cloud Run.