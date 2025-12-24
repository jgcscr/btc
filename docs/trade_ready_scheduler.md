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
```

## 3. Cloud Build Configuration

The pipeline definition lives at [`cloudbuild/trade_ready.yaml`](cloudbuild/trade_ready.yaml). Key steps:

1. Invoke `/run-dataset-refresh` with a 72-hour window and capture the JSON result.
2. Run `python -m src.scripts.ensure_spot_raw_sync` to backfill BigQuery raw klines if the curated table is ahead.
3. Invoke `/run-signal` with `--targets 1,4,8,12` and capture the response payload.
4. Assemble a structured report with durations and per-horizon metrics.
5. Upload the report to the hourly path under `reports/trade_ready/` in Cloud Storage.

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
