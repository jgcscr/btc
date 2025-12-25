# Trade-Ready Automation (Cloud Build + Scheduler)

This workflow runs hourly and produces a trade-ready JSON summary in Cloud Storage. The
pipeline executes `run-dataset-refresh`, backfills raw spot klines in BigQuery when
needed, generates multi-horizon signals, and writes a report to
`gs://jc-financial-466902-btc-forecast-data/reports/trade_ready/YYYYMMDD/HH.json`.

## 1. Prerequisites

1. **Enable required APIs** (once per project):
   ```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     cloudscheduler.googleapis.com \
     pubsub.googleapis.com \
     secretmanager.googleapis.com \
     bigquery.googleapis.com \
     run.googleapis.com
   ```
2. **Grant IAM roles** to the default Cloud Build service account
   (`$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")@cloudbuild.gserviceaccount.com`):
   - `roles/bigquery.dataEditor`
   - `roles/bigquery.jobUser`
   - `roles/storage.objectViewer`
   - `roles/storage.objectAdmin`
3. **Allow Cloud Build to read the trade service URL secret** (see next section).
4. Ensure the Cloud Run service (`btc-trading-service`) is deployed with the required
  environment variables and secrets (spot/futures buckets, Alpha Vantage and CryptoCompare keys).
  The hourly on-chain refresh expects `CRYPTOCOMPARE_API_KEY` to be available in the build
  environment or provided explicitly at runtime.

## 2. Secret Management

Store the public service endpoint in Secret Manager so Cloud Build can call the
private workflow without embedding URLs in the build config:

```bash
echo -n "https://btc-trading-service-1014392857490.us-central1.run.app" > /tmp/service-url.txt

# Create or update the secret
if gcloud secrets describe trade-service-url >/dev/null 2>&1; then
  gcloud secrets versions add trade-service-url --data-file=/tmp/service-url.txt
else
  gcloud secrets create trade-service-url --data-file=/tmp/service-url.txt
fi
rm /tmp/service-url.txt

# Grant Cloud Build access
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding trade-service-url \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

Store the CryptoCompare API key in Secret Manager so the refresh step can authenticate:

```bash
# Replace YOUR_CRYPTOCOMPARE_KEY with the actual key
echo -n "${YOUR_CRYPTOCOMPARE_KEY}" > /tmp/cryptocompare-key.txt

if gcloud secrets describe cryptocompare-api-key >/dev/null 2>&1; then
  gcloud secrets versions add cryptocompare-api-key --data-file=/tmp/cryptocompare-key.txt
else
  gcloud secrets create cryptocompare-api-key --data-file=/tmp/cryptocompare-key.txt
fi
rm /tmp/cryptocompare-key.txt

gcloud secrets add-iam-policy-binding cryptocompare-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```
```

## 3. Cloud Build Configuration

The pipeline definition lives at [`cloudbuild/trade_ready.yaml`](cloudbuild/trade_ready.yaml). Key steps:

1. Invoke `/run-dataset-refresh` with a 72-hour window and capture the JSON result.
2. Run `python -m src.scripts.ensure_spot_raw_sync` to backfill BigQuery raw klines if the curated table is ahead.
3. Generate classical technical indicators (RSI, stochastic, MACD, Bollinger, Keltner, ATR, Donchian) and persist them alongside macro/funding/on-chain features for downstream training and inference.
4. Invoke `/run-signal` with `--targets 1,4,8,12` and capture the response payload.
5. Assemble a structured report with durations and per-horizon metrics.
6. Upload the report to the hourly path under `reports/trade_ready/` in Cloud Storage.

The build uses Secret Manager to inject `SERVICE_URL` and relies on substitutions for
`PROJECT_ID`, `SPOT_GCS_BUCKET`, and the report bucket prefix.

## 4. Create Pub/Sub Trigger for Cloud Build

Create a Pub/Sub topic dedicated to the hourly workflow:

```bash
gcloud pubsub topics create trade-ready-trigger
```

Create the Cloud Build trigger that listens to the topic and runs the pipeline:

```bash
gcloud builds triggers create pubsub trade-ready-build \
  --description="Hourly trade-ready pipeline" \
  --topic=trade-ready-trigger \
  --build-config=cloudbuild/trade_ready.yaml \
  --substitutions=_PROJECT_ID=${PROJECT_ID},_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_REPORT_BUCKET=gs://jc-financial-466902-btc-forecast-data
```

Adjust the substitutions if you use different buckets or projects.

## 5. Cloud Scheduler Job

Create a Cloud Scheduler job that publishes to the topic at the top of every hour (UTC):

```bash
gcloud scheduler jobs create pubsub trade-ready-hourly \
  --schedule="0 * * * *" \
  --time-zone="Etc/UTC" \
  --topic=trade-ready-trigger \
  --message-body="{}"
```

The Scheduler job publishes an empty JSON message; Cloud Build ignores the content and
runs the pipeline.

## 6. Deployment Recap

1. Deploy/refresh the Cloud Run service so it exposes `/run-dataset-refresh` and `/run-signal`.
2. Configure the `trade-service-url` secret with the Cloud Run endpoint URL.
3. Grant Cloud Build the necessary IAM roles and secret access.
4. Create the Pub/Sub topic, Cloud Build trigger, and Cloud Scheduler job using the
   commands above.
5. Verify the first scheduled run in Cloud Build history and confirm the report shows up under
   `reports/trade_ready/YYYYMMDD/HH.json` in the forecast bucket.

With the scheduler active, the pipeline continuously maintains fresh datasets and signal reports that can be consumed by downstream trading systems.

## 7. Model Artifact Reference

The trade-ready workflow now deploys the TA-enhanced 1h ensemble bundle:

- Regression model: artifacts/models/xgb_ret1h_with_ta (replaces artifacts/models/xgb_ret1h_v1)
- Direction model: artifacts/models/xgb_dir1h_with_ta (replaces artifacts/models/xgb_dir1h_v1)

To mirror automation locally, invoke the refreshed CLI tooling:

- Generate signals across the standard horizons: `python -m src.scripts.run_signal_once --targets 1,4,8,12 --output artifacts/signals/run_signal_once_with_ta.json`
- Summarize the payload for reporting: `python -m src.scripts.evaluate_ensemble_signals --input artifacts/signals/run_signal_once_with_ta.json --summary artifacts/signals/run_signal_once_with_ta_summary.json`

Keep the legacy directories available until downstream consumers confirm the upgrade.

## 8. Monitoring and Alerts

- **Manual run:** `python -m src.check_pipeline_health --artifact-root artifacts/monitoring --staleness-hours 2 --max-missing-ratio 0.05`
- **Scheduler/cron:** create a Cloud Scheduler job (or cron entry) that runs the same command 10 minutes after the hourly refresh. Example cron: `10 * * * * python -m src.check_pipeline_health --artifact-root /workspace/artifacts/monitoring --staleness-hours 2 --max-missing-ratio 0.05`
- **Exit codes & logs:** exit `0` means all artifacts were refreshed within the staleness window and no missing-ratio breaches were detected. Exit `1` lists each failing artifact. Sample log output:
  ```
  Checked 4 artifact(s) with staleness <= 2.00h and missing_ratio <= 0.0500.
  Detected issues:
  - dataset_meta.json: stale by 3.25h (field generated_at, limit 2.00h)
  Pipeline health check failed.
  ```

Forward stdout/stderr to Cloud Logging or your alerting system so on-call can triage stale datasets quickly.

### Market Feature Refresh

- **CryptoCompare dependency:** the refresh step pulls hourly metrics (`active_addresses`, `new_addresses`,
  `transaction_count`, `hashrate`, `difficulty`) via `https://min-api.cryptocompare.com`. Provide a
  CryptoCompare API key through the `CRYPTOCOMPARE_API_KEY` environment variable or the CLI `--api-key`
  flag when executing manually. Cloud Build now injects this value from the `cryptocompare-api-key` Secret
  Manager entry via `availableSecrets`.
- **Cloud Build:** the hourly trade-ready pipeline now runs
  `python -m src.scripts.refresh_market_features --onchain-limit 720 --funding-limit 1000` before dataset and
  signal steps, so no separate cron is required unless you want a standalone rerun.
- **Manual rerun:** trigger the same command locally (or ad hoc via Scheduler) when you need to regenerate
  market features outside the hourly build.
- **Expected artifacts:** raw pulls land under `data/raw/onchain/cryptocompare/`, funding rates under
  `data/raw/funding/binance/`, and processed outputs update `data/processed/onchain/hourly_features.parquet`,
  `data/processed/funding/hourly_features.parquet`, and `artifacts/monitoring/*_summary.json`
