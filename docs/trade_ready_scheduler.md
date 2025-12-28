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

To rotate the webhook URL, repeat the snippet above (the new version becomes active
immediately). If you need to pause alert delivery without editing Cloud Build, add
`--substitutions=_ALERT_DRY_RUN=true` when running a manual build or temporarily update the
trigger substitution (see Section 8).
```

## 3. Cloud Build Configuration

The pipeline definition lives at [`cloudbuild/trade_ready.yaml`](cloudbuild/trade_ready.yaml). Key steps:

1. Invoke `/run-dataset-refresh` with a 72-hour window and capture the JSON result.
2. Run `python -m src.scripts.ensure_spot_raw_sync` to backfill BigQuery raw klines if the curated table is ahead.
3. Generate classical technical indicators (RSI, stochastic, MACD, Bollinger, Keltner, ATR, Donchian) and persist them alongside macro/funding/on-chain features for downstream training and inference.
4. Invoke `/run-signal` with `--targets 1,4,8,12` and capture the response payload.
5. Assemble a structured report with durations and per-horizon metrics.
6. Upload the report to the hourly path under `reports/trade_ready/` in Cloud Storage.
7. Execute `python -m src.scripts.check_pipeline_health --config configs/monitoring_sla_overrides.yaml --alert-output /workspace/tmp/health_alert.json --emit-alert-json --job-id $BUILD_ID` so the build itself produces the alert payload with run metadata.
8. Post the alert JSON to Slack (or another webhook) via `python -m src.scripts.post_alert_to_webhook`, honoring retries/backoff and the `_ALERT_DRY_RUN` substitution.

The build uses Secret Manager to inject `SERVICE_URL`, the vendor API keys, and the
`TRADE_READY_ALERT_WEBHOOK`. Substitutions now cover `PROJECT_ID`, `SPOT_GCS_BUCKET`, the
report bucket prefix, and `_ALERT_DRY_RUN` (default **false**). To temporarily disable alert
posting for staging tests, run:

```bash
gcloud builds submit --config cloudbuild/trade_ready.yaml \
  --substitutions=_PROJECT_ID=${PROJECT_ID},_SPOT_GCS_BUCKET=jc-financial-466902-btc-forecast-data,_REPORT_BUCKET=gs://jc-financial-466902-btc-forecast-data,_ALERT_DRY_RUN=true
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

## 9. Vendor delays & manual overrides

Holiday trading hours (Dec 27–29, 2025) triggered multiple upstream pauses. Document the current status so on-call can manage expectations and know when to intervene:

| Feed | Status (Dec 28, 2025) | Expected recovery | Manual override |
| --- | --- | --- | --- |
| Alpha/Twelve/Tiingo macro chains (`alpha_vantage_catalog`, `macro_chain_comparison`, `macro_summary`) | **Degraded** – US equities closed, Alpha Vantage/Tiingo last bars at 2025‑12‑26 20:00Z. | Monday 2025‑12‑29 14:30Z when exchanges reopen. | If a downstream report must be generated before markets open, point `process_technical_features --price-source` at the latest Binance parquet and rerun `python -m data.ingestors.alpha_vantage_macro --run-catalog` once opening trades settle. |
| CryptoCompare on-chain (`onchain_summary`) | **Degraded** – `histo/day` endpoint still publishing 2025‑12‑24 data, vendor ticket CQ‑4182 open. | Vendor ETA pending; typical delay <48h during maintenance. | Switch `refresh_market_features` to `--skip-onchain` and ingest manual CSVs in `data/raw/onchain/manual/` via `python -m data.processed.compute_onchain_features --raw-root data/raw/onchain/manual`. |
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
