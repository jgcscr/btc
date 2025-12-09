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
pip install -r requirements.txt
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