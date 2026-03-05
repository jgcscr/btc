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

### Slack / webhook alert secret

Create a dedicated secret for the monitoring webhook so alert rotations never require
checking YAML into source control. The Cloud Build steps reference the secret via the
`TRADE_READY_ALERT_WEBHOOK` environment variable.

```bash
# Replace with the Slack Incoming Webhook (or MS Teams/Discord URL).
echo -n "https://hooks.slack.com/services/T000/B000/XXXXX" > /tmp/trade-ready-alert-webhook.txt

if gcloud secrets describe trade-ready-alert-webhook >/dev/null 2>&1; then
  gcloud secrets versions add trade-ready-alert-webhook --data-file=/tmp/trade-ready-alert-webhook.txt
else
  gcloud secrets create trade-ready-alert-webhook --data-file=/tmp/trade-ready-alert-webhook.txt
fi
rm /tmp/trade-ready-alert-webhook.txt

gcloud secrets add-iam-policy-binding trade-ready-alert-webhook \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Secret Manager IAM matrix (Dec 2025)

The default Cloud Build identity for project `jc-financial-466902` is the service account
`1014392857490@cloudbuild.gserviceaccount.com`. Listing worker pools in `us-central1`
returned zero entries (`gcloud builds worker-pools list --region=us-central1`), so there is
no additional worker-pool service account to provision today. Manual (`gcloud builds
submit`) runs currently execute as the Compute Engine default service account
`1014392857490-compute@developer.gserviceaccount.com`, as confirmed via
`gcloud builds describe <BUILD_ID> --format='value(serviceAccount)'`. Grant both identities
the `roles/secretmanager.secretAccessor` role on every secret consumed by
[`cloudbuild/trade_ready.yaml`](cloudbuild/trade_ready.yaml):

| Secret name | Purpose | Bound members |
| --- | --- | --- |
| `trade-service-url` | Cloud Run endpoint used by `/run-dataset-refresh` + `/run-signal` | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `cryptocompare-api-key` | Funding + on-chain refresh auth | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `alpha-vantage-api-key` | Paid Alpha macro pulls | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `alpha-vantage-free-api-key` | Free tier fallback for macro pulls | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `twelvedata-api-key` | Alternative macro provider | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `tiingo-api-key` | Spot + macro Tiingo access | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `fred-api-key` | FRED macro feed | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `cryptoquant-api-key` | Flow/on-chain fallbacks | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |
| `trade-ready-alert-webhook` | Slack/webhook delivery for alerts | `serviceAccount:1014392857490@cloudbuild.gserviceaccount.com`, `serviceAccount:1014392857490-compute@developer.gserviceaccount.com` |

### Kaiko & Twelve Data Premium secrets

Create placeholder secrets now so production only needs to upload the live keys once the contracts are executed. The same names are consumed by [`env/load_alpha_vantage_secret.sh`](env/load_alpha_vantage_secret.sh) for local development and by [`cloudbuild/trade_ready.yaml`](cloudbuild/trade_ready.yaml) when the hourly build runs.

| Secret Manager name | Purpose | Exported env vars | Notes |
| --- | --- | --- | --- |
| `kaiko-api-key` | REST/WS key for Kaiko reference + market data | `KAIKO_API_KEY` | Provided to CLI tooling via `kaiko_api_key` in Cloud Build; also injected into the new Kaiko ingestors.
| `kaiko-api-secret` | Matching Kaiko secret used for HMAC signing (where required) | `KAIKO_API_SECRET` | Stored separately so ops can rotate the secret without touching the public key.
| `twelvedata-premium-api-key` | Premium tier macro + FX access | `TWELVEDATA_PREMIUM_API_KEY` | The free-tier `twelvedata-api-key` remains for legacy fallback; premium calls will default to this env var.

Bootstrap commands (safe to run today with placeholder values that signal "pending contract"):

```bash
for secret in kaiko-api-key kaiko-api-secret twelvedata-premium-api-key; do
  if ! gcloud secrets describe "$secret" >/dev/null 2>&1; then
    printf "pending-%s" "$secret" | gcloud secrets create "$secret" --data-file=-
  else
    printf "pending-%s" "$secret" | gcloud secrets versions add "$secret" --data-file=-
  fi
  for member in \
    serviceAccount:1014392857490@cloudbuild.gserviceaccount.com \
    serviceAccount:1014392857490-compute@developer.gserviceaccount.com; do
    gcloud secrets add-iam-policy-binding "$secret" \
      --member="$member" \
      --role="roles/secretmanager.secretAccessor" \
      --project=jc-financial-466902
  done
done
```

Once the real keys arrive, rerun the same loop with the production values (or use `gcloud secrets versions add`). No YAML edits are required—Cloud Build will auto-inject the next secret version and export `KAIKO_API_KEY`, `KAIKO_API_SECRET`, and `TWELVEDATA_PREMIUM_API_KEY` for every step.

One-liner to grant (or re-grant) the bindings:

```bash
for secret in \
  trade-service-url \
  cryptocompare-api-key \
  alpha-vantage-api-key \
  alpha-vantage-free-api-key \
  twelvedata-api-key \
  tiingo-api-key \
  fred-api-key \
  cryptoquant-api-key \
  trade-ready-alert-webhook; do
  for member in \
    serviceAccount:1014392857490@cloudbuild.gserviceaccount.com \
    serviceAccount:1014392857490-compute@developer.gserviceaccount.com; do
    gcloud secrets add-iam-policy-binding "${secret}" \
      --member="${member}" \
      --role="roles/secretmanager.secretAccessor" \
      --project=jc-financial-466902
  done
done
```

To rotate the webhook URL, repeat the snippet above (the new version becomes active
immediately). If you need to pause alert delivery without editing Cloud Build, add
`--substitutions=_ALERT_DRY_RUN=true` when running a manual build or temporarily update the
trigger substitution (see Section 8).
```

## 3. Cloud Build Configuration

The pipeline definition lives at [`cloudbuild/trade_ready.yaml`](cloudbuild/trade_ready.yaml). Key steps:

1. Invoke `/run-dataset-refresh` with a 72-hour window (now wrapped in exponential backoff, default 5 attempts) and capture the JSON result.
2. Run `python -m src.scripts.ensure_spot_raw_sync` to backfill BigQuery raw klines if the curated table is ahead.
3. Generate classical technical indicators (RSI, stochastic, MACD, Bollinger, Keltner, ATR, Donchian) and persist them alongside macro/funding/on-chain features for downstream training and inference.
4. Invoke `/run-signal` with `--targets 0.25,1,4,8,12` and capture the response payload.
5. Assemble a structured report with durations and per-horizon metrics.
6. Upload the report to the hourly path under `reports/trade_ready/` in Cloud Storage.
7. Execute `python -m src.scripts.check_pipeline_health --config configs/monitoring_sla_overrides.yaml --alert-output /workspace/tmp/health_alert.json --emit-alert-json --job-id $BUILD_ID` so the build itself produces the alert payload with run metadata.
8. Post the alert JSON to Slack (or another webhook) via `python -m src.scripts.post_alert_to_webhook`, honoring retries/backoff and the `_ALERT_DRY_RUN` substitution.

Fallback promotion defaults (Dec 2025): `_SPOT_PROVIDER`, `_ONCHAIN_SOURCE`, `_FUNDING_PROVIDER`, and the `_REG|DIR|LSTM|TRANSFORMER_MODEL_DIR_*` substitutions now resolve to the Tiingo/fallback bundle in `gs://jc-financial-models-prod/tiingo_fallback_20251229/`. Every build step exports these variables so local scripts, webhook invocations, and downstream CLIs load the promoted models without extra flags. Mirror the same values in the Cloud Run service (see [env/cloud_run_local.env](env/cloud_run_local.env)) or whichever Secret Manager entry backs your production deployment so `/run-dataset-refresh` and `/run-signal` stay aligned with the build jobs.

New substitution `_FALLBACK_MODE` (default `false`) lets you flip the Cloud Build trigger into keep-alive mode. When set to `true`, the vendor-heavy steps (`refresh-market-features`, `/run-*` calls, report build/publish) short-circuit and the build executes a local `python -m src.scripts.run_refresh_and_predict --config configs/run_refresh_and_predict.default.yaml --spot-provider binanceus --write-artifacts` run followed by the usual monitoring/export steps. Clear `_FALLBACK_MODE` (or set it back to `false`) to resume the full hourly workflow once live APIs stabilize.

With `_FALLBACK_MODE=true`, the monitoring phase now appends `--tolerate-known-critical` to `check_pipeline_health`, which tells the CLI to exit 0 only when every failing artifact is tagged as a vendor `degraded`/`maintenance` outage. The alert JSON is still written (and Slack/webhook delivery still occurs), but the build succeeds so long as the failures remain confined to known vendor outages.

### 1-minute spot ingestion & sync (Jan 2026)

Binance US minute candles follow the same bucket/table layout as the hourly feed. Run the daily spot ingestor for the dates you need, then sync those partitions with BigQuery via the updated raw sync CLI:

```bash
# 1) Write the 2026-01-10 minute bars to GCS
python -m data.ingestors.binance_spot_klines \
  --interval 1m \
  --date 2026-01-10 \
  --bucket jc-financial-466902-btc-forecast-data

# 2) Load the last 72 hours of 1m partitions into btc_forecast_raw.spot_klines
PROJECT_ID=jc-financial-466902 \ 
SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data \ 
python -m src.scripts.ensure_spot_raw_sync \
  --interval 1m \
  --hours 72
```

`--interval` overrides the `SPOT_INTERVAL` environment variable at runtime, and the script automatically filters the raw BigQuery table on that interval before deciding which partitions to load. Hourly jobs can continue to omit the flag and rely on the default `1h` setting.

#### Cloud Build automation (hourly)

A dedicated Cloud Build pipeline at [cloudbuild/binance_1m.yaml](cloudbuild/binance_1m.yaml) installs the repo requirements, calls `python -m data.ingestors.binance_us_spot --interval 1m --limit ${_BINANCE_LIMIT}` to keep the tidy parquet current, rewrites the last `${_BINANCE_UPLOAD_DAYS}` day partitions via `python -m data.ingestors.binance_spot_klines`, and finally runs `python -m src.scripts.ensure_spot_raw_sync --interval 1m --hours ${_SYNC_HOURS}` so the `btc_forecast_raw.spot_klines` table stays within a 96-hour catch-up window. All environment variables reuse the same `_PROJECT_ID`/`_SPOT_GCS_BUCKET` substitutions as the trade-ready pipeline, so no new secrets are required.

Manual dry-run:

```bash
gcloud builds submit \
  --config cloudbuild/binance_1m.yaml \
  --substitutions=_PROJECT_ID=jc-financial-466902,_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_SPOT_SYMBOL=BTCUSDT,_SPOT_INTERVAL=1m,_BINANCE_LIMIT=2880,_BINANCE_UPLOAD_DAYS=3,_SYNC_HOURS=96
```

Create a dedicated Pub/Sub topic + trigger so Cloud Scheduler can run the pipeline hourly without touching the trade-ready workflow:

```bash
gcloud pubsub topics create binance-1m-spot-trigger

gcloud builds triggers create pubsub binance-1m-spot-build \
  --description="Hourly Binance 1m ingest" \
  --topic=binance-1m-spot-trigger \
  --build-config=cloudbuild/binance_1m.yaml \
  --substitutions=_PROJECT_ID=jc-financial-466902,_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_SPOT_SYMBOL=BTCUSDT,_SPOT_INTERVAL=1m,_BINANCE_LIMIT=2880,_BINANCE_UPLOAD_DAYS=3,_SYNC_HOURS=96

gcloud scheduler jobs create pubsub binance-1m-spot-hourly \
  --schedule="0 * * * *" \
  --time-zone="Etc/UTC" \
  --topic=binance-1m-spot-trigger \
  --message-body="{}"
```

The ingestion loop rewrites the most recent three UTC partitions every hour, so duplicate uploads are harmless and the sync step only loads partitions that are missing from BigQuery. To verify freshness, run the following query (expect `lag_min < 60` for a healthy pipeline):

```bash
bq --project_id=jc-financial-466902 query --use_legacy_sql=false <<'SQL'
SELECT TIMESTAMP_SECONDS(DIV(MAX(ts), 1000000000)) AS max_ts_utc,
       TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), TIMESTAMP_SECONDS(DIV(MAX(ts), 1000000000)), MINUTE) AS lag_min,
       COUNT(*) AS row_count
FROM btc_forecast_raw.spot_klines
WHERE `interval` = '1m';
SQL
```

If `lag_min` exceeds one hour, inspect the latest Cloud Build run, confirm the Cloud Scheduler job fired, and re-run `gcloud builds submit --config cloudbuild/binance_1m.yaml ...` to backfill the missing window.

### Macro fallback workflow

US exchange holidays or Alpha/Twelve maintenance windows halt macro vendor bars for up to 72 hours. Keep monitoring green by synthesizing Tiingo/Twelve/FRED data and explicitly switching the refresh/prediction CLIs into fallback mode:

1. **Build (and promote) fallback macro features**
   ```bash
   python -m src.scripts.build_macro_fallback \
     --history-hours 720 \
     --promote-features-path data/processed/macro/hourly_features.parquet \
     --promote-summary-path artifacts/monitoring/macro_summary.json \
     --macro-chain-path artifacts/monitoring/macro_chain_comparison.json
   ```
  This ingests Tiingo ETFs (SPY, QQQ, GLD, HYG, USO), Twelve Data DXY/VIX, and FRED yields, forward-fills with an exponential decay (default half-life 72h), and writes both `data/processed/macro_fallback/hourly_features.parquet` and `artifacts/monitoring/macro_fallback_summary.json`. With the promote paths supplied, the canonical macro parquet/summary/chain JSONs are replaced by the fallback payload stamped with `"source": "fallback"` so the health checker sees a fresh artifact.
2. **Refresh market features with fallback inputs**
   ```bash
   python -m src.scripts.refresh_market_features \
     --macro-source fallback \
    --technical-price-source binanceus \
     --onchain-source fallback \
     --funding-provider cryptocompare
   ```
   The new `--macro-source` flag selects between vendor and fallback artifacts; the other switches keep spot/on-chain aligned with the Binance/CryptoCompare bundle already promoted for the holiday window.
3. **Regenerate predictions + monitoring exports** with the same overrides:
   ```bash
   python -m src.scripts.run_refresh_and_predict \
    --config configs/run_refresh_and_predict.default.yaml \
     --spot-provider binanceus \
     --macro-source fallback \
     --onchain-source fallback \
     --funding-provider cryptocompare \
     --write-artifacts
   ```
   `--write-artifacts` forces the script to refresh `trade_ready_summary.json`, `meta_baseline_summary.json`, and the per-component monitoring files using the fallback metadata so `check_pipeline_health` passes.
4. **Revert** once markets reopen by rerunning `refresh_market_features --macro-source vendor --onchain-source cryptocompare --technical-price-source binanceus` (or your preferred vendor stack) and executing `run_refresh_and_predict` without the fallback overrides. Regenerate `macro_chain_comparison.json` via the standard vendors to clear the `source: fallback` annotation.

`/run-dataset-refresh` automatically retries transient 5xx errors up to `_RUN_DATASET_MAX_RETRIES` times (default **5**) with exponential backoff starting at 5 seconds. Override the count by passing `_RUN_DATASET_MAX_RETRIES=<N>` in your substitutions (or setting `RUN_DATASET_MAX_RETRIES` in the build environment). Optionally tune the starting delay by defining `RUN_DATASET_BACKOFF_SECONDS` at invocation time if a longer cool-off is required.

The build uses Secret Manager to inject `SERVICE_URL`, the vendor API keys, and the
`TRADE_READY_ALERT_WEBHOOK`. Substitutions now cover `PROJECT_ID`, `SPOT_GCS_BUCKET`, the
report bucket prefix, `_ALERT_DRY_RUN` (default **false**), and `_RUN_DATASET_MAX_RETRIES`
(default **5**). To temporarily disable alert
posting for staging tests, run:

```bash
gcloud builds submit --config cloudbuild/trade_ready.yaml \
  --substitutions=_PROJECT_ID=${PROJECT_ID},_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_REPORT_BUCKET=gs://jc-financial-466902-btc-forecast-data,_ALERT_DRY_RUN=true,_RUN_DATASET_MAX_RETRIES=5,_SPOT_PROVIDER=binanceus,_ONCHAIN_SOURCE=fallback,_FUNDING_PROVIDER=cryptocompare,_REG_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_ret1h_with_ta,_DIR_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_dir1h_with_ta,_LSTM_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/lstm_dir1h,_TRANSFORMER_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/transformer_dir1h,_REG_MODEL_DIR_4H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_ret4h_v1,_DIR_MODEL_DIR_4H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_dir4h_v1
```

The monitoring step will still run (and fail the build on critical issues) but Slack delivery
is suppressed via the `--dry-run` flag.

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
  --substitutions=_PROJECT_ID=${PROJECT_ID},_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_REPORT_BUCKET=gs://jc-financial-466902-btc-forecast-data,_ALERT_DRY_RUN=false,_RUN_DATASET_MAX_RETRIES=5,_SPOT_PROVIDER=binanceus,_ONCHAIN_SOURCE=fallback,_FUNDING_PROVIDER=cryptocompare,_REG_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_ret1h_with_ta,_DIR_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_dir1h_with_ta,_LSTM_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/lstm_dir1h,_TRANSFORMER_MODEL_DIR_1H=gs://jc-financial-models-prod/tiingo_fallback_20251229/transformer_dir1h,_REG_MODEL_DIR_4H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_ret4h_v1,_DIR_MODEL_DIR_4H=gs://jc-financial-models-prod/tiingo_fallback_20251229/xgb_dir4h_v1
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

### Fallback bundle (Dec 2025)

Vendor outages (Alpha Vantage equities + CryptoCompare on-chain) require forcing the trade-ready stack to ingest Binance spot plus fallback on-chain/funding features. Promote bundle `artifacts/models/tiingo_fallback_20251229` (legacy name retained) with the following sequence whenever prod refreshes run in fallback mode:

1. Refresh market features so downstream datasets are built on the fallback feeds:
   ```bash
   python -m src.scripts.refresh_market_features \
     --technical-price-source binanceus \
     --onchain-source fallback \
     --funding-provider cryptocompare
   ```
2. Rebuild datasets, predictions, and monitoring artifacts with the Binance spot candles:
   ```bash
   python -m src.scripts.run_refresh_and_predict \
    --config configs/run_refresh_and_predict.default.yaml \
     --spot-provider binanceus
   ```
3. Load the new direction models (transformer + LSTM + XGB dir4h) from `artifacts/models/tiingo_fallback_20251229/` when deploying the trading service or Cloud Build workers.

The bundle keeps the legacy regression models for continuity and introduces Tiingo/fallback-trained direction heads:

- Regression model: artifacts/models/xgb_ret1h_with_ta (replaces artifacts/models/xgb_ret1h_v1)
- Direction model: artifacts/models/xgb_dir1h_with_ta (replaces artifacts/models/xgb_dir1h_v1)
- Fallback direction ensemble (current default): artifacts/models/tiingo_fallback_20251229/{xgb_dir4h_v1,transformer_dir1h,lstm_dir1h}

To mirror automation locally, invoke the refreshed CLI tooling:

- Generate signals across the standard horizons: `python -m src.scripts.run_signal_once --targets 0.25,1,4,8,12 --output artifacts/signals/run_signal_once_with_ta.json`
- Summarize the payload for reporting: `python -m src.scripts.evaluate_ensemble_signals --input artifacts/signals/run_signal_once_with_ta.json --summary artifacts/signals/run_signal_once_with_ta_summary.json`

Keep the legacy directories available until downstream consumers confirm the upgrade or vendor feeds recover (switch the commands back to `--spot-provider binanceus`, `--onchain-source cryptocompare`, and `--funding-provider binance`).

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

### Config-driven health checks & alerting

- **Per-artifact overrides:** place outage/SLA metadata in `configs/monitoring_sla_overrides.yaml` and run `python -m src.scripts.check_pipeline_health --artifact-root artifacts/monitoring --config configs/monitoring_sla_overrides.yaml`. Artifacts marked `vendor_status.state=degraded` are reported as warnings (instead of hard failures) until the supply recovers.
- **Structured alerts:** append `--emit-alert-json` to mirror the failure payload on stdout and/or `--alert-output /workspace/logs/monitoring_alert.json` to hand Cloud Build a Pub/Sub-friendly blob. Downstream automation can read the JSON, decide whether to email on-call, post to Slack, etc. The trade-ready Cloud Build now bakes this in via the `monitor-artifacts` step, which writes `/workspace/tmp/health_alert.json` and annotates it with job metadata (`BUILD_ID`, git commit, duration, host).
- **Example command:**
  ```bash
  python -m src.scripts.check_pipeline_health \
    --artifact-root artifacts/monitoring \
    --config configs/monitoring_sla_overrides.yaml \
    --alert-output /workspace/logs/monitoring_alert.json \
    --emit-alert-json \
    --job-id ${CLOUD_SCHEDULER_JOB_ID}
  ```
- **Cloud Scheduler ➜ Cloud Build ➜ Slack/email:**
-  Cloud Build already executes the same flow at the tail end of `trade_ready.yaml`:
  1. `monitor-artifacts` runs `check_pipeline_health` with overrides and populates `/workspace/tmp/health_alert.json`.
  2. `post-health-alert` calls `python -m src.scripts.post_alert_to_webhook --webhook-url $TRADE_READY_ALERT_WEBHOOK --max-retries 5 --initial-backoff 5 --timeout 15` and automatically adds `--dry-run` when `_ALERT_DRY_RUN=true`.
  3. Failures (non-zero exit) from either step abort the build, so on-call receives actionable Slack messages and Cloud Build shows the same log context.
-  To disable alerts temporarily (while keeping the monitoring gate), update the Cloud Build trigger:
  ```bash
  gcloud builds triggers update trade-ready-build \
    --set-substitutions=_PROJECT_ID=${PROJECT_ID},_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_REPORT_BUCKET=gs://jc-financial-466902-btc-forecast-data,_ALERT_DRY_RUN=true
  ```
  Re-enable by flipping `_ALERT_DRY_RUN` back to `false` or omitting it.
-  To rotate the webhook, push a new secret version (Section 2) and the next build will automatically pick it up—no YAML change required.
-  Swap the Slack webhook for SendGrid/SES by updating the secret value or pointing `_ALERT_DRY_RUN=true` and running a dry-run test with the new URL before enabling production traffic.

## 9. Live-only refresh (Jan 2026)

With vendor feeds down we must operate purely on data that can be refreshed right now. As of 2026‑01‑06, the only live source is Binance US hourly spot candles, so the pipeline runs on technical indicators derived from that feed alone. No forward-filling or fallback macro/on-chain artifacts are injected.

| Feed | Status (2026‑01‑06) | Failure mode | Bring-back checklist |
| --- | --- | --- | --- |
| Binance US spot | ✅ Healthy (latest candle 2026‑01‑06 21:00 UTC) | N/A | Keep `python -m data.ingestors.binance_us_spot --limit 720` on a cron so `data/raw/market/binanceus/…` always has fresh parquet. |
| Macro vendor blend (Alpha/Twelve/FRED) | ⚠️ Stale (rebuild stops at 2026‑01‑05 19:00 UTC) | Vendor feed not updating even though `python -m data.processed.compute_macro_features` completes | Resume once upstream timestamps are <2h old or switch back to the fallback macro workflow in Section 3. |
| Tiingo spot | ⛔ Blocked | `TiingoSpotIngestionError: TIINGO_API_KEY not set` | Rehydrate the `tiingo-api-key` secret (or export `TIINGO_API_KEY`) and rerun `python -m data.ingestors.tiingo_spot`. |
| TwelveData macro | ⛔ Blocked | `TwelveDataIngestionError: TWELVE_DATA_API_KEY not set` | Upload a fresh TwelveData key to Secret Manager and export `TWELVE_DATA_API_KEY` before rerunning the ingestor. |
| CryptoCompare on-chain | ⛔ Blocked | `CryptoCompareIngestionError` (auth now required) | Populate `CRYPTOCOMPARE_API_KEY` and rerun `python -m data.ingestors.cryptocompare_onchain`. |
| Binance futures funding | ⛔ Blocked | All live calls return HTTP 451 | Invoke from an allowed region (or proxy) or temporarily switch `--funding-provider cryptocompare` once those keys are restored. |
| CryptoQuant daily | ⛔ Blocked | `CQ_TOKEN` unset | Restore the `cryptoquant-api-key` secret so the ingestor can authenticate. |

### Live-only local run

1. **Fetch fresh spot candles** (defaults grab the latest 720 bars):
  ```bash
  python -m data.ingestors.binance_us_spot --limit 720
  ```
  The parquet lands under `data/raw/market/binanceus/entity=spot/symbol=BINANCEUS_SPOT_BTC_USDT/`.
2. **Build technical features from just those candles** (no macro/on-chain columns):
  ```bash
  python -m data.processed.compute_technical_features \
    --price-source binanceus \
    --output tmp/live_features/spot.parquet \
    --summary artifacts/monitoring/spot_only_summary.json \
    --history-limit 400
  ```
  Expect ~360 rows × 11 TA columns (RSI, MACD, bands, ATR, Donchian, etc.).
3. **Run predictions using only that parquet:**
  ```bash
  python -m src.scripts.run_refresh_and_predict \
    --config configs/run_refresh_and_predict.default.yaml \
    --use-local-features \
    --features-path tmp/live_features/spot.parquet \
    --dir-model-config-json artifacts/models/direction_models_latest.json \
    --thresholds-json artifacts/models/calibrated_thresholds.json \
    --write-artifacts
  ```
  The script produces `artifacts/monitoring/latest.json` + `artifacts/trade_ready_summary.json` stamped with `local_feature_overrides.features.path=tmp/live_features/spot.parquet`, proving only live data was used.

### Implications and reintroduction steps

- Models were trained with macro, funding, on-chain, and CryptoQuant signals, so feeding only spot-derived TA introduces a dataset shift. Expect low-confidence outputs (`signal_ensemble=0`). If the outage persists, plan a retraining pass on the reduced feature space.
- Tag the blocked feeds as `vendor_status.state=degraded` inside `configs/monitoring_sla_overrides.yaml` so health checks stay green while APIs are known-broken.
- When a feed recovers, rerun its ingestion command, execute `python -m src.scripts.refresh_market_features` with the vendor source, then rerun `run_refresh_and_predict` without `--use-local-features` to reattach the restored features.

#### Pub/Sub + Slack transport details

- **Topic & subscription (CLI):**
  ```bash
  gcloud pubsub topics create pipeline-alerts
  gcloud pubsub subscriptions create pipeline-alerts-slack \
    --topic pipeline-alerts \
    --push-endpoint=https://YOUR_REGION-YOUR_PROJECT.cloudfunctions.net/pipeline-alert-webhook \
    --ack-deadline=30
  ```
- **Terraform sample:**
  ```hcl
  resource "google_pubsub_topic" "pipeline_alerts" {
    name = "pipeline-alerts"
  }

  resource "google_pubsub_subscription" "pipeline_alerts_slack" {
    name  = "pipeline-alerts-slack"
    topic = google_pubsub_topic.pipeline_alerts.name

    push_config {
      push_endpoint = var.pipeline_alert_webhook
      oidc_token {
        service_account_email = var.pipeline_alert_service_account
      }
    }
    ack_deadline_seconds = 30
  }
  ```
- **Slack payload format:** `post_alert_to_webhook` sends `{ "text": "[CRITICAL] pipeline health …", "attachments": [...] }`. Configure the incoming webhook via `https://api.slack.com/messaging/webhooks` and store it as the `trade-ready-alert-webhook` Secret Manager entry referenced in Cloud Build.

## 9. Vertex AI Training

Use the Vertex AI custom job workflow to retrain the Tiingo/fallback direction head whenever the promoted models drift or new datasets land in `artifacts/datasets/`.

### Components

- Training entrypoint: [training/vertex_train.py](training/vertex_train.py) loads the latest multi-horizon dataset, fits the `xgb_dir4h` classifier, and uploads every artifact (model JSON, metadata, metrics) to `gs://jc-financial-models-prod/vertex_jobs/<job_id>/`.
- Training container: [cloudbuild/vertex_train.Dockerfile](cloudbuild/vertex_train.Dockerfile) installs the repo `requirements.txt`, copies the workspace, and sets the entrypoint to the script above.
- Submission helper: [scripts/submit_vertex_train.sh](scripts/submit_vertex_train.sh) rebuilds the container with Cloud Build, pushes it to `gcr.io/jc-financial-466902/vertex-train:latest`, and launches the custom job with the required environment (SPOT_PROVIDER=tiingo, MACRO_SOURCE=fallback, ONCHAIN_SOURCE=fallback, FUNDING_PROVIDER=cryptocompare).

### Runbook

1. Authenticate with `gcloud` and pick the target project/region:
  ```bash
  gcloud auth login
  gcloud config set project jc-financial-466902
  gcloud config set ai/region us-central1
  ```
2. Trigger a retrain (optional overrides shown as environment variables):
  ```bash
  PROJECT_ID=jc-financial-466902 \
  REGION=us-central1 \
  MACHINE_TYPE=n1-standard-8 \
  DATASET_PATH=artifacts/datasets/btc_features_multi_horizon_splits.npz \
  BASE_OUTPUT_DIR=gs://jc-financial-models-prod/vertex_jobs \
  scripts/submit_vertex_train.sh
  ```
  The script always rebuilds `gcr.io/${PROJECT_ID}/vertex-train:latest` using `cloudbuild/vertex_train.Dockerfile`, then submits a Vertex AI custom job whose display name is timestamped (for example `vertex-train-tiingo-fallback-20251229-0105`). Override the dataset path, output bucket, or machine shape by exporting the corresponding environment variables described at the top of the script.
3. Monitor the job:
  ```bash
  gcloud ai custom-jobs list --region us-central1 --filter="displayName~vertex-train-tiingo-fallback"
  gcloud ai custom-jobs describe --region us-central1 <JOB_ID>
  ```
  Logs stream to Cloud Logging under the Vertex Training resource type; the entrypoint also prints a JSON blob summarizing the metrics that were uploaded.

### Outputs

- Local artifacts land in `artifacts/models/xgb_dir4h_vertex/` inside the container before upload. When the upload step runs, the same files appear at `gs://jc-financial-models-prod/vertex_jobs/<job_id>/...` for downstream promotion.
- The `vertex_train.py` entrypoint writes `summary.json`, `metrics.json`, and `model_metadata_direction.json` alongside the `xgb_dir4h_model.json`, making it straightforward to diff checkpoints between jobs.
- The `VERTEX_GCS_BASE` and job ID used for uploads are logged to stdout; pinning a specific job for rollback simply requires copying the relevant GCS directory back into the deployment bucket or updating the `_DIR_MODEL_DIR_4H` substitution in the trade-ready pipeline.

### Optuna Tuning

- Tuning entrypoint: [training/vertex_tune.py](training/vertex_tune.py) runs an Optuna study with time-series cross validation, writes best-trial metadata into `artifacts/tuning/<run>_<timestamp>/`, and optionally uploads the folder to `gs://jc-financial-models-prod/vertex_jobs/<job_id>/`.
- Tuning container: [cloudbuild/vertex_tune.Dockerfile](cloudbuild/vertex_tune.Dockerfile) mirrors the training image but starts `vertex_tune.py` instead of the trainer.
- Submission helper: [scripts/submit_vertex_tune.sh](scripts/submit_vertex_tune.sh) builds/pushes `gcr.io/${PROJECT_ID}/vertex-tune:latest` and submits a custom job with study metadata (n-trials, folds, storage URI, timeout) plus the fallback environment variables.

Recommended loop:

1. Launch tuning with the helper script (override knobs via environment variables as needed):
  ```bash
  PROJECT_ID=jc-financial-466902 \
  REGION=us-central1 \
  MACHINE_TYPE=n1-standard-16 \
  N_TRIALS=80 \
  TIMEOUT_SECONDS=7200 \
  OPTUNA_STORAGE=sqlite:///artifacts/tuning/studies/tiingo_dir4h.db \
  BASE_OUTPUT_DIR=gs://jc-financial-models-prod/vertex_jobs \
  scripts/submit_vertex_tune.sh
  ```
  The script reuses the default Tiingo/fallback dataset and writes Optuna results to `artifacts/tuning/<run_name>_<timestamp>/` before uploading them to the Vertex output bucket under the active job ID. Pass `REUSE_STUDY=true` to append to an existing SQLite study or set `OPTUNA_STORAGE` to another SQLAlchemy URI (e.g., Cloud SQL) if you need concurrent workers.
2. Inspect study progress via Cloud Logging (resource type Vertex Training) or by tailing the Optuna study in storage:
  ```bash
  gcloud ai custom-jobs describe --region us-central1 <JOB_ID>
  python - <<'PY'
import optuna
study = optuna.load_study(study_name="vertex-xgb-dir4h-tiingo-fallback", storage="sqlite:///artifacts/tuning/studies/tiingo_dir4h.db")
print(study.best_trial.params)
PY
  ```
3. Promote the winning hyperparameters by pointing `train_xgb_dir4h_v1.py` (or a future automated retrain) at `best_params.json` from the tuning run, or copy the JSON into the promotion configs inside `configs/`.

Every tuning run emits `best_params.json`, `study_summary.json`, `trials.csv`, and `trials.json` alongside the Optuna storage (if enabled). These artifacts are versioned by timestamp so ops can diff studies across macro regimes before promoting a new checkpoint.

## 9. Vendor delays & manual overrides

Holiday trading hours (Dec 27–29, 2025) triggered multiple upstream pauses. Document the current status so on-call can manage expectations and know when to intervene:

| Feed | Status (Dec 28, 2025) | Expected recovery | Manual override |
| --- | --- | --- | --- |
| Alpha/Twelve/Tiingo macro chains (`alpha_vantage_catalog`, `macro_chain_comparison`, `macro_summary`) | **Degraded** – US equities closed, Alpha Vantage/Tiingo last bars at 2025‑12‑26 20:00Z. | Monday 2025‑12‑29 14:30Z when exchanges reopen. | If a downstream report must be generated before markets open, point `process_technical_features --price-source` at the latest Binance parquet and rerun `python -m data.ingestors.alpha_vantage_macro --run-catalog` once opening trades settle. |
| CryptoCompare on-chain (`onchain_summary`) | **Degraded** – `histo/day` endpoint still publishing 2025‑12‑24 data, vendor ticket CQ‑4182 open. | Vendor ETA pending; typical delay <48h during maintenance. | Run `python -m src.scripts.build_onchain_fallback` to synthesize surrogate metrics, then execute `python -m src.scripts.refresh_market_features --onchain-source fallback --funding-provider cryptocompare` so the summary reflects the fallback feed. Switch back to `--onchain-source cryptocompare` once vendor data resumes. |
| Technical summary (`technical_summary`) | **Degraded** – curated BigQuery table not updating past 2025‑12‑26 22:59:59Z. | Resumes automatically when macro refresh succeeds (same Monday open). | Run `python -m src.scripts.refresh_market_features --skip-onchain --funding-provider cryptocompare` followed by `python -m data.processed.compute_technical_features --price-source data/raw/market/binanceus/...` to seed local OHLCV until BigQuery syncs. |
| CryptoQuant fallback hourly (`cryptoquant_daily_summary`) | **Degraded** – hourly approvals pending support case CQ‑8842; rows stop at 2025‑12‑26 00:00Z. | Support SLA 24h after approval; expect catch-up by 2025‑12‑28 18:00Z. | Continue running `python -m data.processed.compute_cryptoquant_resampled`; if trading needs fresher flow metrics, pull directly from CryptoQuant UI and drop CSVs into `data/raw/cryptoquant/manual/` before rerunning the resampler. |
| Trade-ready predictions (`trade_ready_summary`) | **Healthy** – regenerated 2025‑12‑27T22:35Z via `run_refresh_and_predict`. | N/A | No manual action required. |

The same metadata is mirrored in `configs/monitoring_sla_overrides.yaml`; update that file whenever vendor SLAs change so the health checker can distinguish known vendor outages from genuine regressions.

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
- **Fallback surrogate:** when CryptoCompare lags, run `python -m src.scripts.build_onchain_fallback --history-hours 720 \
  --decay-half-life-hours 72` to extrapolate the missing window using Tiingo spot and funding proxies. Follow up with
  `python -m src.scripts.refresh_market_features --onchain-source fallback --technical-price-source binanceus --funding-provider cryptocompare`
  so `onchain_summary.json` captures the fallback metadata (including `source: fallback`, decay parameters, and last
  vendor timestamp). Toggle the flag back to `--onchain-source cryptocompare` once the upstream API resumes.

## 10. Kaiko & Twelve Data Premium rollout plan

- **DRI:** Market Data Platform (ops rotation) owns the Kaiko ingest hookup. Analytics Engineering provides backup on schema mapping, while SRE partners ensure Cloud Build + Cloud Run variables stay in sync.
- **Secrets:** `kaiko-api-key`, `kaiko-api-secret`, and `twelvedata-premium-api-key` live in Secret Manager and are surfaced to the build via `KAIKO_API_KEY`, `KAIKO_API_SECRET`, and `TWELVEDATA_PREMIUM_API_KEY` (Section 2). Local dev sources them through `env/load_alpha_vantage_secret.sh`.
- **Code scaffolding:**
  - [`src/data/ingestors/kaiko_reference.py`](src/data/ingestors/kaiko_reference.py) will own instrument catalogs, exchange metadata, and static symbol remaps.
  - [`src/data/ingestors/kaiko_market.py`](src/data/ingestors/kaiko_market.py) will own OHLCV + funding-depth pulls from Kaiko.
  - [`src/data/ingestors/twelvedata_premium.py`](src/data/ingestors/twelvedata_premium.py) will wrap the premium macro endpoints and track rate limits + authentication headers.

| Dataset / Artifact | Current provider | Future provider | Toggle once keys land |
| --- | --- | --- | --- |
| Spot klines (`data/raw/spot_klines/*.parquet`) | Tiingo | Kaiko Markets | Set `_SPOT_PROVIDER=kaiko` (Cloud Build) or run `python -m src.scripts.refresh_market_features --technical-price-source kaiko`. Use `python -m src.data.ingestors.kaiko_market --instrument btc-usd --exchange binance --granularity 1h` for live pulls, `--sample-path tests/data/kaiko_sample.json` for dry runs, and `--base-url https://mock.kaiko` when routing against requests-mock. |
| Reference metadata (`artifacts/monitoring/kaiko_reference_summary.json`) | N/A | Kaiko Reference | Run `python -m src.data.ingestors.kaiko_reference --instrument btc-usd --fields symbol,exchange,...` for live pulls. Supply `--sample-path tests/data/kaiko_sample.json` when credentials are unavailable and `--base-url https://mock.kaiko` to exercise the mock harness before promoting artifacts. |
| Macro FX / Rates (`data/processed/macro/hourly_features.parquet`) | Alpha Vantage + Twelve Data free mix | Twelve Data Premium | Export `TWELVEDATA_PREMIUM_API_KEY` and run `python -m src.data.ingestors.twelvedata_premium --instrument DXY --start ...` for live pulls.<br>Use `--sample-path tests/data/twelvedata_premium_sample.json` for offline validation, `--allow-free-fallback` to retry with `TWELVE_DATA_API_KEY`, and `--base-url https://mock.twelvedata` when pointing at the mock harness. Update `MACRO_PROVIDER_CHAIN` to prioritize `kaiko_reference,kaiko_market,twelvedata_premium` before the legacy fallbacks. |
| Funding metadata (`artifacts/monitoring/funding_summary.json`) | CryptoCompare | Kaiko Derivatives (phase 2) | After Kaiko depth endpoints stabilize, set `--funding-provider kaiko` on `refresh_market_features` and promote the derived artifacts. |

> **Live-call gate**: All Kaiko and Twelve Data ingestors now default to the bundled fixtures unless `LIVE_DATA_OK=1`. Export that flag (alongside the vendor API keys) whenever you genuinely want to hit the upstream APIs; leave it unset for offline QA and CI smoke runs.

**Kaiko CLI modes**

- Market data: omit `--sample-path` for live Kaiko REST pulls, pass `--sample-path tests/data/kaiko_sample.json` for offline QA, and override `--base-url https://mock.kaiko` when replaying requests-mock harnesses.
- Reference data: same flags apply; add `--fields symbol,exchange,...` to constrain payload sizes and keep monitoring artifacts small during dry runs.
- Funding-only rehearsals: `python -m src.data.ingestors.kaiko_market --sample-path ... --funding-output tmp/kaiko_funding.parquet` speeds up monitoring drills without hitting Kaiko.

**Twelve Data Premium CLI modes**

- Live mode: omit `--sample-path` so the ingestor calls Twelve Data REST with `TWELVEDATA_PREMIUM_API_KEY`.
- Sample mode: pass `--sample-path tests/data/twelvedata_premium_sample.json` (or another fixture) to validate transformations without credentials.
- Fallback mode: add `--allow-free-fallback` so the ingestor retries with `TWELVE_DATA_API_KEY` whenever the premium key rate-limits or is missing; combine with `--base-url https://mock.twelvedata` for requests-mock based rehearsals.

_Alias validation_: `python -m src.scripts.refresh_market_features --macro-source twelvedata_sample --technical-price-source kaiko_sample` replays [tmp/kaiko_dry_run/twelvedata_premium.parquet](tmp/kaiko_dry_run/twelvedata_premium.parquet) and [tmp/kaiko_dry_run/kaiko_ohlcv.parquet](tmp/kaiko_dry_run/kaiko_ohlcv.parquet) end-to-end, gracefully falling back when CryptoCompare returns 401s (see [logs/provider_audit_20251229/refresh_market_features_dry_run.log](logs/provider_audit_20251229/refresh_market_features_dry_run.log)). Ops can flip `_SPOT_PROVIDER=kaiko` / `_MACRO_PROVIDER_CHAIN=kaiko_reference,kaiko_market,twelvedata_premium,…` as soon as real keys land without additional plumbing.

### Activation checklist

1. **Secrets live** – push production Kaiko/Twelve keys into Secret Manager using the names listed above.
2. **Toggle build substitutions** – update the Cloud Build trigger to set `_SPOT_PROVIDER=kaiko` and export `MACRO_PROVIDER_CHAIN="kaiko_reference,kaiko_market,twelvedata_premium,alpha,alpha_free,twelve,tiingo,fred"` (the YAML already defaults to this chain unless a custom override is passed).
3. **Backfill datasets** – run the Kaiko market + reference ingestors locally (or via a one-off Cloud Build) to seed BigQuery + parquet history before flipping the hourly cron.
4. **Promote monitoring artifacts** – publish `artifacts/monitoring/kaiko_reference_summary.json` and `artifacts/monitoring/twelvedata_premium_macro_summary.json`, then mark their entries in [`configs/monitoring_sla_overrides.yaml`](configs/monitoring_sla_overrides.yaml) as `state: healthy` once the data is flowing.
5. **Enable runtime toggles** – redeploy Cloud Run with `KAIKO_API_KEY`, `KAIKO_API_SECRET`, and `TWELVEDATA_PREMIUM_API_KEY` set so `/run-dataset-refresh` and `/run-signal` can call the new providers.

#### Cutover checklist

Use this sequence the moment the production credentials land to promote Kaiko/Twelve into the hourly workflow:

```bash
# 1. Backfill Kaiko reference metadata (pairs/exchanges) for BTC spot.
python -m src.data.ingestors.kaiko_reference \
  --instrument btc-usd \
  --fields symbol,exchange,base_currency,quote_currency \
  --start 2024-12-01T00:00:00Z \
  --end $(date -u +"%Y-%m-%dT%H:%M:%SZ")

# 2. Backfill Kaiko spot/funding candles so datasets can pivot immediately.
python -m src.data.ingestors.kaiko_market \
  --instrument btc-usd \
  --exchange binance \
  --granularity 1h \
  --start 2024-12-01T00:00:00Z \
  --end $(date -u +"%Y-%m-%dT%H:%M:%SZ")

# 3. Refresh derived datasets with Kaiko spot + funding providers.
python -m src.scripts.refresh_market_features \
  --technical-price-source kaiko \
  --funding-provider kaiko \
  --onchain-source fallback

# 4. Regenerate predictions/monitoring artifacts with the new feeds.
python -m src.scripts.run_refresh_and_predict \
  --config configs/run_refresh_and_predict.default.yaml \
  --spot-provider kaiko \
  --funding-provider kaiko \
  --macro-source vendor \
  --write-artifacts

# 5. Switch Cloud Build's defaults to Kaiko + Twelve Premium.
gcloud builds triggers update trade-ready-build \
  --set-substitutions="_PROJECT_ID=jc-financial-466902,_SPOT_PROVIDER=kaiko,_ONCHAIN_SOURCE=fallback,_FUNDING_PROVIDER=kaiko,_MACRO_PROVIDER_CHAIN=kaiko_reference,kaiko_market,twelvedata_premium,alpha,alpha_free,twelve,tiingo,fred"

# 6. Redeploy Cloud Run with the same env overrides so /run-* endpoints stay in sync.
gcloud run deploy btc-trading-service \
  --image=gcr.io/jc-financial-466902/btc-trading-service:latest \
  --set-env-vars="SPOT_PROVIDER=kaiko,FUNDING_PROVIDER=kaiko,MACRO_PROVIDER_CHAIN=kaiko_reference,kaiko_market,twelvedata_premium,alpha,alpha_free,twelve,tiingo,fred"

# 7. Re-run monitoring to validate the cutover before enabling Scheduler.
python -m src.scripts.check_pipeline_health \
  --config configs/monitoring_sla_overrides.yaml \
  --emit-alert-json \
  --alert-output logs/provider_audit_20251229/health_alert_after_cutover.json
```

To roll back, unset `_SPOT_PROVIDER=kaiko`, drop Kaiko from the macro provider chain, and rotate the monitoring overrides back to Tiingo/Twelve Free while keeping the Kaiko artifacts tagged as `state: pending` for future use.

## 11. Fallback keep-alive automation

### Cron entry (runs every 6 hours by default)

Use the lightweight helper at [scripts/run_fallback_keepalive.sh](scripts/run_fallback_keepalive.sh) to run
`run_refresh_and_predict` with the Binance-only fallback defaults until premium vendors come back online. The
script only needs the workspace root path because it now ingests Binance spot klines directly; no Tiingo parquet
seed is required. Sample crontab entry:

```
0 */6 * * * WORKSPACE_ROOT=/workspaces/btc \
  /workspaces/btc/scripts/run_fallback_keepalive.sh \
  >> /workspaces/btc/logs/binance_keepalive/cron_driver.log 2>&1
```

- `WORKSPACE_ROOT` ensures the script runs from the repo root (defaults to `/workspaces/btc`).
- `KEEPALIVE_LOG_DIR` can be overridden if you want the per-run logs to land somewhere other than
  `logs/binance_keepalive/`.

Each invocation writes a timestamped log such as
`logs/binance_keepalive/run_refresh_and_predict_20251229T230000Z.log` plus the rolling cron driver log
referenced above.

### Monitoring keep-alive output

- Tail the latest run via `tail -F logs/binance_keepalive/run_refresh_and_predict_*.log` to confirm the
  fallback models regenerated artifacts successfully.
- Health checks run separately via `python -m src.scripts.check_pipeline_health` (Section 8). The build-system
  runs log files like [logs/binance_keepalive/check_pipeline_health_keepalive.log](logs/binance_keepalive/check_pipeline_health_keepalive.log)
  and alert payloads such as
  [logs/binance_keepalive/health_alert_keepalive.json](logs/binance_keepalive/health_alert_keepalive.json).
- When `_FALLBACK_MODE=true` (Section 3), Cloud Build executes the same fallback script automatically, adds
  `--tolerate-known-critical` to the monitoring step, and still ships alerts via `post_alert_to_webhook`. This keeps
  the notification channel updated without failing the build as long as every failing artifact is tagged as a
  vendor `degraded`/`maintenance` outage.

### Disabling keep-alive mode

Once Kaiko/Twelve resume service:

1. Remove or comment out the cron line above, or switch it to `@reboot` if you only need manual failover.
2. Update the Cloud Build trigger substitutions so `_FALLBACK_MODE=false` and the `_SPOT_PROVIDER` / `_ONCHAIN_SOURCE`
   values reflect the live vendors again (Section 4).
3. Redeploy Cloud Run (or rerun `run_refresh_and_predict` manually) with the vendor modes so `configs/monitoring_sla_overrides.yaml`
   can move the macro/on-chain artifacts back to `state: healthy`.
4. Archive the keep-alive logs under `logs/binance_keepalive/` for audit purposes, but stop tailing them in
   PagerDuty since the hourly Cloud Build pipeline will be primary again.

Document any new fallback incidents directly in the monitoring overrides file so the next keep-alive exercise has
fresh reference points.

## 12. Kaiko/Twelve Cutover readiness checklist

Run this sequence once Kaiko and Twelve Data Premium credentials hit Secret Manager so the data plane is staged before the Cloud Build trigger flips back to live vendors.

1. **Load secrets locally**
  ```bash
  source env/load_alpha_vantage_secret.sh
  printenv KAIKO_API_KEY | head -c 4 && echo "…"
  printenv TWELVEDATA_PREMIUM_API_KEY | head -c 4 && echo "…"
  ```
  - The loader exports `KAIKO_API_KEY`, `KAIKO_API_SECRET`, `TWELVEDATA_PREMIUM_API_KEY`, and legacy fallbacks into the shell. Override the loader path via `SECRET_LOADER=/custom/script.sh` when running the helper scripts below if needed.

2. **Backfill Kaiko reference listings**
  ```bash
  KAIKO_REFERENCE_INSTRUMENT=btc-usd \
  KAIKO_REFERENCE_START="2024-12-01T00:00:00Z" \
  KAIKO_REFERENCE_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  ./scripts/backfill_kaiko_reference.sh
  ```
  - Overrides: `KAIKO_REFERENCE_FIELDS`, `KAIKO_REFERENCE_PARQUET`, `KAIKO_REFERENCE_SUMMARY`, `KAIKO_REFERENCE_SAMPLE`.
  - Expected outputs: `data/processed/reference/kaiko_reference.parquet`, `artifacts/monitoring/kaiko_reference_summary.json`, log at `logs/kaiko_cutover/kaiko_reference_*.log`.

3. **Backfill Kaiko OHLCV + funding snapshots**
  ```bash
  KAIKO_MARKET_INSTRUMENT=btc-usd \
  KAIKO_MARKET_EXCHANGE=coinbase \
  KAIKO_MARKET_GRANULARITY=1h \
  ./scripts/backfill_kaiko_market.sh
  ```
  - Overrides: `KAIKO_MARKET_START/END/LIMIT`, `KAIKO_MARKET_OHLCV`, `KAIKO_MARKET_FUNDING`, `KAIKO_MARKET_SUMMARY`, `KAIKO_MARKET_SAMPLE`.
  - Expected outputs: `data/processed/spot/kaiko_ohlcv.parquet`, `data/processed/funding/kaiko_funding.parquet`, `artifacts/monitoring/kaiko_market_preview.json`, log at `logs/kaiko_cutover/kaiko_market_*.log`.

4. **Backfill Twelve Data Premium macro chains**
  ```bash
  TWELVEDATA_INSTRUMENT=DXY \
  TWELVEDATA_INTERVAL=1h \
  ./scripts/backfill_twelvedata.sh
  ```
  - Overrides: `TWELVEDATA_START/END/LIMIT`, `TWELVEDATA_PARQUET`, `TWELVEDATA_SUMMARY`, `TWELVEDATA_SAMPLE`.
  - Expected outputs: `data/processed/macro/twelvedata_premium.parquet`, `artifacts/monitoring/twelvedata_premium_macro_summary.json`, log at `logs/kaiko_cutover/twelvedata_premium_*.log`.

5. **Promote staged artifacts into the workflow**
  ```bash
  python -m src.scripts.refresh_market_features \
    --technical-price-source kaiko_sample \
    --macro-source twelvedata_sample \
    --funding-provider kaiko_sample \
    --onchain-source fallback

  python -m src.scripts.run_refresh_and_predict \
    --config configs/run_refresh_and_predict.default.yaml \
    --spot-provider kaiko \
    --funding-provider kaiko \
    --macro-source vendor \
    --write-artifacts
  ```
  - Until the live Kaiko APIs replace the sample transports, point the CLI at the staged parquet outputs so downstream datasets ingest the new structures.

6. **Run the cutover health gate**
  ```bash
  python -m src.scripts.check_pipeline_health \
    --config configs/monitoring_sla_overrides.yaml \
    --tolerate-known-critical \
    --emit-alert-json \
    --alert-output logs/tiingo_fallback_dryrun/health_alert_cutover_ready.json
  ```
  - Confirm the JSON reports `status: warning` (vendor degradations only). Attach this payload to the provider audit folder and keep the CLI logs in `logs/tiingo_fallback_dryrun/`.

7. **Flip Cloud Build + Scheduler once green**
  - Update the trigger substitutions to `_SPOT_PROVIDER=kaiko`, `_FUNDING_PROVIDER=kaiko`, `_MACRO_PROVIDER_CHAIN=kaiko_reference,kaiko_market,twelvedata_premium,…`, ensure `_FALLBACK_MODE=false`, then resume the hourly Scheduler job.
  - Redeploy Cloud Run with the same env vars so `/run-dataset-refresh` and `/run-signal` ingest the Kaiko/Twelve stack in lockstep with Cloud Build.
