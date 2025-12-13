# BTCUSDT Forecasting – Data & BigQuery Setup

This repo sets up a simple pipeline to:

- Generate BTCUSDT spot and futures kline data as Parquet.
- Upload the Parquet files to Google Cloud Storage (GCS) in a partitioned layout.
- Load the data into BigQuery raw tables in project `jc-financial-466902`.

## 1. Python environment

```bash
cd /workspaces/btc
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # installs google-cloud-bigquery-storage for fast table reads
```

## 2. Generate and upload spot & futures data

### 2.1 Spot klines via binance.us

We use the binance.us endpoint `https://api.binance.us/api/v3/klines` to fetch
spot BTCUSDT OHLCV and write it to GCS in a partitioned layout.

```bash
cd /workspaces/btc
source .venv/bin/activate

export DATA_BUCKET=jc-financial-466902-btc-forecast-data

# Ingest one UTC day of 1h BTCUSDT spot klines
python -m data.ingestors.binance_spot_klines \
	--symbol BTCUSDT \
	--interval 1h \
	--date 2025-01-01 \
	--bucket "$DATA_BUCKET"
```

This will:
- fetch 1h candles for that UTC day, using close time as `ts` in UTC.
- write a local parquet file under `data/spot_klines/`.
- upload it to:
	`gs://$DATA_BUCKET/raw/spot_klines/interval=1h/yyyy=2025/mm=01/dd=01/...parquet`.

### 2.2 Futures metrics via Binance futures

For BTCUSDT perpetual futures we use `https://fapi.binance.com/fapi/v1/continuousKlines`.
The first version writes OHLCV and leaves `open_interest` and `funding_rate` as NaN
so the schema matches the planned `btc_forecast_raw.futures_metrics` table.

```bash
cd /workspaces/btc
source .venv/bin/activate

export DATA_BUCKET=jc-financial-466902-btc-forecast-data

# Ingest one UTC day of 1h BTCUSDT futures klines
python -m data.ingestors.binance_futures_metrics \
	--pair BTCUSDT \
	--interval 1h \
	--date 2025-01-01 \
	--bucket "$DATA_BUCKET"
```

This will upload parquet to:
`gs://$DATA_BUCKET/raw/futures_metrics/interval=1h/yyyy=2025/mm=01/dd=01/...parquet`.

> If the Binance endpoints are geo-blocked, you can still use `src/ingest_spot_klines.py --dummy`
> to generate synthetic spot data locally for testing the rest of the pipeline.

### 2.3 Macro, on-chain, and funding scaffolding

Phase 1 macros use free endpoints. Export API keys before calling the loaders:

```bash
export FRED_API_KEY="..."
export ALPHA_VANTAGE_API_KEY="..."
```

Fetch macroeconomic series into `data/raw/macro/`:

```bash
# FRED broad trade-weighted USD index
python -m data.ingestors.fred_macro DTWEXBGS --start 2019-01-01

# Alpha Vantage daily SPX close (uses TIME_SERIES_DAILY JSON)
python -m data.ingestors.alpha_vantage_macro TIME_SERIES_DAILY SPX
```

Fetch on-chain metrics into `data/raw/onchain/`:

```bash
# Active Bitcoin addresses, trailing year
python -m data.ingestors.blockchain_onchain activeaddresses --timespan 1year
```

Convert raw tidy rows into hourly feature Parquet and monitoring summaries:

```bash
# Writes data/processed/*/hourly_features.parquet and artifacts/monitoring/*_summary.json
python -m data.processed.compute_macro_features
python -m data.processed.compute_onchain_features
python -m data.processed.compute_funding_features --pair BTCUSDT --fetch --limit 500
```

Funding features depend on Binance endpoints. The `--fetch` flag hydrates `data/raw/funding/binance/`
before the hourly aggregation runs. If live API access is restricted, copy historical Parquet into that
directory and run the processor without `--fetch`.

## 3. Create BigQuery dataset (once)

```bash
gcloud config set project jc-financial-466902

bq --location=us-central1 mk -d \
	--description "Raw BTC/crypto data for forecasting" \
	jc-financial-466902:btc_forecast_raw
```

If the dataset already exists, BigQuery will return an "Already exists" error, which is safe to ignore.

## 4. Load Parquet into BigQuery raw tables

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

## 5. Model Training, API Serving, and Cloud Run Deployment

### 5.1 Train and save the model

```bash
cd /workspaces/btc
source .venv/bin/activate
pip install -r requirements.txt

# Build dataset splits from BigQuery curated features
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

This completes the full pipeline: data ingestion → BigQuery → model training → API serving on Cloud Run.