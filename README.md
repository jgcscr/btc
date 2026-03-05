
# BTCUSDT Forecasting – Ingestion, Features, and BigQuery Setup

This repository provides a robust, multi-vendor pipeline for BTCUSDT forecasting, with premium macro, on-chain, and market data sources, resilient fallbacks, and transparent monitoring. It supports:

- Ingestion of macro, spot, perp, and funding data from premium providers (Alpha Vantage, Binance, CryptoQuant, FRED)
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
pip install -r requirements-tests.txt  # pytest + requests-mock helpers
```

### Required environment variables

Set the following environment variables for premium data access:

- `FRED_API_KEY` (required for FRED macro ingestion)
- `ALPHA_VANTAGE_API_KEY` (required for Alpha Vantage macro)
- `TWELVE_DATA_API_KEY` (required when `MACRO_PROVIDER=twelve`)
- `CQ_TOKEN` (required for CryptoQuant daily metrics)
- `LIVE_DATA_OK` (optional; default `0`) – set to `1` when you explicitly want Kaiko/Twelve ingestors to hit live vendor APIs. Leaving it unset keeps the scripts on bundled sample payloads, which is the default for tests and CI.

These providers are now in active use. Free endpoints are no longer sufficient for full feature coverage.

Use the helper script to load the Alpha Vantage key from Secret Manager before local runs:

```bash
source env/load_alpha_vantage_secret.sh  # exports ALPHA_VANTAGE_API_KEY from GCP Secret Manager
```

The script defaults to project `jc-financial-466902` and secret `alpha-vantage-api-key`. Override `PROJECT_ID`, `ALPHA_VANTAGE_SECRET_NAME`, or `ALPHA_VANTAGE_SECRET_VERSION` if you keep the secret elsewhere.

Macro ingestion defaults to Alpha Vantage; set `MACRO_PROVIDER=twelve` (or pass `--provider twelve`) to route through Twelve Data when the Twelve key is present. Clearing the flag falls back to Alpha Vantage automatically.


## Quick Manual Prediction

Run the all-in-one refresh script to rebuild features, regenerate datasets, and emit 0.25h/1h/4h/8h/12h predictions. The command writes `artifacts/predictions/latest.json` and appends to `artifacts/predictions/history.json`.

```bash
# Full refresh with live ingestion
python -m src.scripts.run_refresh_and_predict --targets 0.25,1,4,8,12

# Load the same defaults from a config file (see [configs/run_refresh_and_predict.default.yaml](configs/run_refresh_and_predict.default.yaml))
python -m src.scripts.run_refresh_and_predict --config configs/run_refresh_and_predict.default.yaml

# Optional: include sequence ensembles for richer probability voting
python -m src.scripts.run_refresh_and_predict \
	--targets 0.25,1,4,8,12 \
	--dir-lstm-path artifacts/models/lstm_dir_v1 \
	--dir-bilstm-path artifacts/models/bilstm_dir_v1 \
	--dir-gru-path artifacts/models/gru_dir_v1 \
	--dir-transformer-path artifacts/models/transformer_dir_v1

# CI-friendly smoke test (skips network calls and emits stub predictions)
python -m src.scripts.run_refresh_and_predict --dry-run --targets 0.25,1,4,8,12
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
			"p_up_components": {"xgb": 0.4756, "lstm": 0.5123},
			"expected_value": -0.0003,
			"stop_loss": 85608.34,
			"take_profit": 87561.29,
			"signal_ensemble": 1,
			"signal_dir_only": 0
		},
		"4h": { "...": "..." },
		"8h": { "...": "..." },
		"12h": { "...": "..." }
	}
}
```

Each horizon now surfaces:
- `p_up_components`: raw probabilities from the direction models (XGBoost plus any enabled LSTM, BiLSTM, GRU, transformer checkpoints) used in the ensemble vote.
- `expected_value`: log-return EV computed from the ensemble probability and per-horizon residual bands.
- `stop_loss` / `take_profit`: price targets derived from \(\pm 1\sigma\) residual bands, ensuring the CLI and monitoring payloads expose consistent trade risk parameters.

Pipeline regression tests covering the CLI live in `tests/pipeline/` and can be executed via `pytest tests/pipeline -q`.

You can override every CLI default (hours, targets, threshold paths, providers, monitoring flags, model overrides) via `--config`. The flag accepts either YAML or JSON and only affects values that are not explicitly passed on the command line. Start from [configs/run_refresh_and_predict.default.yaml](configs/run_refresh_and_predict.default.yaml) and tweak the keys you care about:

```yaml
hours: 720
targets: [0.25, 1, 4]
p_up_min: 0.55
thresholds_json: artifacts/models/calibrated_thresholds_custom.json
spot_provider: tiingo
write_artifacts: true
```

Running `python -m src.scripts.run_refresh_and_predict --config my_config.yaml` now reproduces the same CLI as if you had passed each flag manually.

#### Local feature overrides

When vendor feeds lag behind but you have hand-curated hourly features, run the refresh CLI in "local" mode. Supply your merged parquet via `--features-path` and point any auxiliary tables (macro, on-chain, CryptoQuant, funding) to their corresponding parquet snapshots. The CLI skips network ingestion and dataset rebuilds while reusing calibrated models and thresholds:

```bash
python -m src.scripts.run_refresh_and_predict \
	--use-local-features \
	--features-path tmp/live_features/features.parquet \
	--macro-path tmp/live_features/macro.parquet \
	--onchain-path tmp/live_features/onchain.parquet \
	--cryptoquant-path tmp/live_features/cryptoquant.parquet \
	--funding-path tmp/live_features/funding.parquet \
	--targets 0.25,1,4,8,12
```

The overrides must include a `ts` column (UTC, hourly) plus the model feature set (the script aligns to the `feature_names` stored in the latest dataset NPZ and fills any missing columns with zeros). Monitoring payloads in `artifacts/monitoring/latest.json` and `trade_ready_summary.json` record the provenance of each override so it is clear which parquet inputs drove a given signal snapshot.

### Direction-model configuration (JSON or legacy weights)

Direction classifiers are now described through a structured registry (`DEFAULT_DIR_MODELS_1H` in `src/config_trading.py`). Every CLI that loads ensemble members (`run_refresh_and_predict`, `run_signal_realtime*`, `paper_trade_loop`, `backtest_signals`, `run_signal_once`) exposes:


Example JSON override:

```json
[
	{
		"name": "json_xgb",
		"type": "xgb",
		"path": "artifacts/models/xgb_dir1h_experimental/xgb_dir1h_model.json",
		"weight": 2.0
	},
	{
		"name": "seq_lstm",
		"type": "lstm",
		"path": "artifacts/models/lstm_dir1h_experimental",
		"weight": 1.0,
		"label": "lstm v3"
	}
]

To opt into the CNN-LSTM ensemble member, add an entry such as:

```json
{
	"name": "cnn_lstm_v1",
	"type": "cnn_lstm",
	"path": "artifacts/models/cnn_lstm_dir1h_v1",
	"weight": 1.0,
	"optional": true
}
```

Transformer experiments can also include a wider preset recorded in the registry. Enable it via JSON once the checkpoint exists:

```json
{
	"name": "transformer_large",
	"type": "transformer_large",
	"path": "artifacts/models/transformer_dir1h_large",
	"weight": 2.0,
	"optional": true,
	"label": "transformer-large v1"
}
```
```

When a CLI runs it prints the resolved config (after applying JSON + overrides + inferred artifact paths) so you can confirm the active ensemble. Supplying neither flag falls back to the default registry baked into `config_trading`.

Sequence members can be retrained with the dedicated CLIs under `src/scripts/`:

```bash
python -m src.scripts.train_lstm_dir1h       --output-dir artifacts/models/lstm_dir1h_v2
python -m src.scripts.train_bilstm_dir1h    --output-dir artifacts/models/bilstm_dir1h_v1
python -m src.scripts.train_gru_dir1h       --output-dir artifacts/models/gru_dir1h_v1
python -m src.scripts.train_cnn_lstm_dir1h  --output-dir artifacts/models/cnn_lstm_dir1h_v1
python -m src.scripts.train_transformer_dir1h --output-dir artifacts/models/transformer_dir1h_v1
python -m src.scripts.train_transformer_dir1h --preset large \
	--output-dir artifacts/models/transformer_dir1h_large
# Convenience wrapper for the preset above (defaults to --preset large)
python -m src.scripts.train_transformer_dir1h_large --output-dir artifacts/models/transformer_dir1h_large
```

Use `--max-steps 1` for any of the transformer CLIs when you only need a smoke-test pass in CI; it limits each epoch to a single optimizer step without skipping validation/testing.

Each script shares the same hyperparameter flags (and optional `--params-json` overrides) so Optuna sweeps or manual experiments stay consistent across architectures.

### Volatility-aware gating

Both dataset builders now emit realized/EMA/GARCH-lite volatility tensors inside the NPZ files
(`volatility_realized_24h_train`, `volatility_ewm_24h_val`, etc.) and document the available
columns in their companion `*_meta.json`. During inference the same metrics are computed live via
`src/trading/volatility.py`, attached to `PreparedData`, and surfaced in prediction payloads as the
`volatility` block plus a `volatility_flag` boolean. Calibrated thresholds can raise the ensemble
floor when markets dislocate by supplying optional keys per horizon:

```json
{
	"horizons": {
		"1": {
			"p_up_min": 0.55,
			"ret_min": 0.0004,
			"volatility_ceiling": 0.035,
			"volatility_mult": 1.4,
			"volatility_metric": "volatility_realized_24h"
		}
	}
}
```

If the selected metric exceeds `volatility_ceiling`, `run_refresh_and_predict.py` automatically bumps
`p_up_min` by the multiplier (documented in the monitoring summary) or leaves the signal untouched
when volatility is inside the guardrails.

### Showcase refresh + report

For a single-command walkthrough that rebuilds data, emits predictions, snapshots the monitoring payloads, and produces a shareable summary, run the showcase CLI:

```bash
# Full showcase run (writes report + artifact copies under artifacts/monitoring/showcases)
python -m src.scripts.showcase_refresh_and_report \
	--targets 0.25,1,4,8,12 \
	--thresholds-json artifacts/models/calibrated_thresholds.json

# Smoke test version that reuses cached datasets
python -m src.scripts.showcase_refresh_and_report --refresh-dry-run
```

Every run copies `artifacts/predictions/latest.json`, `artifacts/monitoring/trade_ready_summary.json`, and a CSV snapshot of the per-horizon EV/thresolds into a timestamped folder inside `artifacts/monitoring/showcases/`. A Markdown (or JSON via `--report-format json`) digest highlights which horizons fired, their expected value, and any detected blockers (fallback feeds, dry-run mode, or missing horizon payloads).


## 2. Ingestion and Feature Engineering Overview

### Active Data Loaders

- **Macro provider (Alpha Vantage / Twelve Data)**: Ingests macroeconomic series (e.g., SPX, DXY, VIX) from the selected vendor. Defaults to Alpha Vantage (requires `ALPHA_VANTAGE_API_KEY`). Set `MACRO_PROVIDER=twelve` (or `--provider twelve`) to pull from Twelve Data instead (requires `TWELVE_DATA_API_KEY`). The expanded catalog can be refreshed in one shot:

	```bash
	python -m data.ingestors.alpha_vantage_macro --run-catalog
	```

	Default coverage (configurable via `ALPHA_VANTAGE_CATALOG_PATH`, see below):

	| Symbol | Description | Functions | Twelve Data proxy |
	| --- | --- | --- | --- |
	| SPY | S&P 500 ETF | 60min intraday, daily | (same) |
	| QQQ | Nasdaq 100 ETF | 60min intraday, daily | (same) |
	| DXY | US Dollar Index | daily | Uses `UUP` for ETF proxy |
	| ^TNX | US 10Y Treasury Yield | daily | Falls back to Alpha Vantage |
	| VIX | CBOE Volatility Index | daily | Uses `VIXY` for ETF proxy |
	| GLD | Gold ETF | 60min intraday, daily | Twelve Data supports daily only |
	| USO | Oil ETF | 60min intraday, daily | Twelve Data supports daily only |
	| HYG | High-Yield Corporate Bond ETF | 60min intraday, daily | Twelve Data supports daily only |

	Twelve Data currently exposes standard intra-day and daily time series; Alpha Vantage remains the fallback for intraday extended slices and treasury yield endpoints. When a Twelve Data call fails, the ingestor automatically tries the next alias (for example `DXY` → `UUP`, `VIX` → `VIXY`) and reports any unresolved functions in the monitoring summary.

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
- **CryptoQuant daily fallback**: Ingests daily on-chain metrics (exchange flows, reserves, whale counts) using `CQ_TOKEN`. Hourly access is pending (see status below). Synthetic data is used for fallback if API is unavailable.
- **FRED macro**: Loads macroeconomic indicators (e.g., trade-weighted USD) using `FRED_API_KEY`.

#### On-chain API configuration

Set these environment variables before running any on-chain loaders or `compute_onchain_features`:

- `ONCHAIN_API_BASE_URL` (required): REST endpoint for the on-chain provider.
- `ONCHAIN_API_KEY` (required): bearer/API key passed to the upstream service.
- `ONCHAIN_DEFAULT_INTERVAL` (optional): interval string accepted by the provider (defaults to `1h`).

Example shell exports for local development:

```bash
export ONCHAIN_API_BASE_URL="https://my-onchain-provider.example/v1"
export ONCHAIN_API_KEY="sk-your-key"
export ONCHAIN_DEFAULT_INTERVAL="1h"  # omit to inherit the default

# now raw + processed on-chain stages will work
python -m data.ingestors.onchain_loader --start "2025-12-23T00:00:00Z" --end "2025-12-30T00:00:00Z" --interval 1h
python -m data.processed.compute_onchain_features
```

### Feature Processors and Monitoring

After raw ingestion, run the feature processors to generate hourly/daily Parquet and monitoring summaries:

```bash
# Macro features
python -m data.processed.compute_macro_features
# Binance-derived funding + futures features
python -m data.processed.compute_funding_features
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

#### Alpha Vantage/Twelve macro monitor

Run the lightweight quota monitor after any ingestion burst (and during nightly automation) to confirm remaining call headroom:

```bash
python -m src.scripts.monitor_alpha_vantage_quota
```

Optional environment variables:

- `ALPHA_VANTAGE_ALERT_THRESHOLD` (default `180`): per-key call ceiling for the current UTC day.

Set `MACRO_PROVIDER` before broader refresh jobs to ensure the desired vendor remains active; leaving it unset falls back to Alpha Vantage.

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

**Instructions when access is restored:**
- Rerun the relevant ingestors (e.g., `data.ingestors.cryptoquant_daily`)
- Rerun the processors (`compute_cryptoquant_resampled.py`)
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

# Optional: build 15m splits for high-frequency experiments
python -m src.scripts.build_training_dataset_15m --output-dir artifacts/datasets

# Build direction (classification) splits with volatility tensors
python -m src.scripts.build_training_dataset_direction \
	--output-dir artifacts/datasets \
	--threshold 0.0

# Direction NPZ files now include volatility_* arrays that track the realized/EWM
# volatility inputs for each split. Sequence builders automatically copy these
# tensors so downstream trainers can consume the richer schema without any
# additional wiring.

# Train refreshed sequence direction models (Jan-2026 vintage)
python -m src.scripts.train_bilstm_dir1h \
	--device cpu \
	--output-dir artifacts/models/bilstm_dir1h_20260101 \
	--epochs 25 > logs/train_bilstm_20260101.log

python -m src.scripts.train_gru_dir1h \
	--device cpu \
	--output-dir artifacts/models/gru_dir1h_20260101 \
	--epochs 25 > logs/train_gru_20260101.log

python -m src.scripts.train_cnn_lstm_dir1h \
	--device cpu \
	--output-dir artifacts/models/cnn_lstm_dir1h_20260101 \
	--epochs 25 > logs/train_cnn_lstm_20260101.log

python -m src.scripts.train_transformer_dir1h \
	--preset large \
	--device cpu \
	--epochs 30 \
	--patience 6 \
	--output-dir artifacts/models/transformer_dir1h_large_20260101 \
	> logs/train_transformer_large_20260101.log

# Update the structured ensemble config to point at the refreshed models
cp artifacts/models/direction_models_latest.json \
	artifacts/models/direction_models_active.json
# (optionally tweak weights per deployment)

# Pass the JSON into refresh/run loops so the ensemble loads the new members
python -m src.scripts.run_refresh_and_predict \
	--dry-run \
	--dir-model-config-json artifacts/models/direction_models_latest.json

# Train baseline XGBoost model
python -m src.scripts.train_baseline_model \
	--dataset-path artifacts/datasets/btc_features_1h_splits.npz \
	--output-dir artifacts/models/xgb_ret1h_v1

#### Cloud Build training for 15m models (no Vertex)

The Cloud Build assets [cloudbuild/train_15m.Dockerfile](cloudbuild/train_15m.Dockerfile) and
[cloudbuild/train_15m.yaml](cloudbuild/train_15m.yaml) build a self-contained trainer image and run the
15m regression (`train_ret15m`) and direction (`train_dir15m`) jobs directly on Cloud Build using only
Artifact Registry and Cloud Storage. The container entrypoints download the NPZ, launch the underlying
Python trainers, and sync the resulting model directories back to the requested GCS prefix.

Manual trigger example (update the substitution values to match the dataset you want to train on):

```bash
gcloud config set project jc-financial-466902
gcloud builds submit \
	--config cloudbuild/train_15m.yaml \
	--substitutions "_RET15M_DATASET_NPZ=gs://jc-financial-models-prod/datasets/btc_features_15m_splits.npz,_DIR15M_DATASET_NPZ=gs://jc-financial-models-prod/datasets/btc_features_15m_splits.npz,_MODEL_OUTPUT_GCS_PREFIX=gs://jc-financial-models-prod/15m"
```

Required substitutions:

- `_RET15M_DATASET_NPZ`: GCS (or local) path to the baseline regression splits (default points to the 15m NPZ)
- `_DIR15M_DATASET_NPZ`: direction dataset path (set equal to the regression NPZ if you reuse the same file)
- `_MODEL_OUTPUT_GCS_PREFIX`: destination bucket/prefix such as `gs://jc-financial-models-prod/15m`

Optional substitutions:

- `_RET15M_PARAMS_JSON`, `_DIR15M_PARAMS_JSON`: JSON overrides uploaded to GCS for the respective XGBoost jobs
- `_TRAIN_IMAGE`: override if you need to pin a different tag than `train-15m:latest`

Each Cloud Build step runs `train_ret15m` or `train_dir15m` inside the container and honors these env vars:

- `DATASET_NPZ_URI`: automatically set from the substitutions listed above
- `MODEL_OUTPUT_GCS_PREFIX`: base URI where the run uploads artifacts (the script appends `/<run_name>_<UTC timestamp>`)
- `RUN_NAME_OVERRIDE`: set per step (`ret15m` / `dir15m`) to keep directory names deterministic
- `PARAMS_JSON_URI`: optional pointer to hyperparameter overrides stored in GCS
- `LOCAL_OUTPUT_ROOT` / `DATASET_LOCAL_PATH`: override defaults (`/app/artifacts/models/15m` and `/tmp/train_dataset.npz`) when debugging locally

Verification checklist:

- Watch the Cloud Build logs for the `train-ret15m` and `train-dir15m` steps; both print the resolved dataset URI and the final `Uploaded artifacts to gs://...` line.
- After the build succeeds, list the bucket to confirm the timestamped folders: `gcloud storage ls ${_MODEL_OUTPUT_GCS_PREFIX}` and inspect the `summary.json` + `model_metadata*.json` files within each run directory.
- (Optional) confirm the image update with `gcloud artifacts docker images describe ${_TRAIN_IMAGE}`; the digest should match the build appended to the Cloud Build run.
- When rerunning ad-hoc training, clean up unused run folders or promote the desired directory to the locations referenced by downstream configs (for example the trade-ready scheduler) to avoid confusion.

Latest promotion (2026-01-11 23:12 UTC):

- Regression (ret_15m): `gs://jc-financial-models-prod/15m/xgb_ret15m_v1_20260111T231234Z/`
- Direction (direction_15m): `gs://jc-financial-models-prod/15m/xgb_dir15m_v1_20260111T231234Z/`
- Structured direction ensemble JSON: `gs://jc-financial-models-prod/15m/direction_models_active_20260111T231234Z.json`

Point Cloud Run / Cloud Build downloads at these URIs (or copy them into the canonical `artifacts/models/*` locations) when you need the latest 15m inference stack.
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

### 5.6 Hourly curated refresh automation

- See [docs/btc_features_refresh_scheduler.md](docs/btc_features_refresh_scheduler.md) for the Cloud Run + Cloud Scheduler job that keeps `btc_forecast_raw.spot_klines` and `btc_forecast_curated.btc_features_1h` fresh and emits Stackdriver alerts when the curated table lags more than three hours.


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