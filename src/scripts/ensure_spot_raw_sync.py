"""Ensure BigQuery spot_klines raw table matches curated coverage.

This utility compares the max timestamps between the curated 1h feature table and
raw spot klines. When raw data lags the curated table, it loads the missing
partitions from GCS parquet files into BigQuery.

Environment variables:
    PROJECT_ID: GCP project id (required)
    RAW_TABLE: BigQuery table (dataset.table) for raw spot klines.
               Defaults to "btc_forecast_raw.spot_klines".
    CURATED_TABLE: BigQuery table (dataset.table) for curated features.
                   Defaults to "btc_forecast_curated.btc_features_1h".
    SPOT_GCS_BUCKET: Bucket containing spot parquet files (required).
    SPOT_INTERVAL: Interval label used in the object path. Defaults to "1h".
    MAX_BACKFILL_DAYS: Optional safety bound on days to load (default 7).

Usage:
    python -m src.scripts.ensure_spot_raw_sync

This script intentionally relies on google-cloud-bigquery and assumes
application default credentials are available (e.g., Cloud Build service
account with BigQuery Data Editor and Storage Object Viewer roles).
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional

from google.cloud import bigquery


def _get_env(name: str, *, required: bool = False, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _max_ts(client: bigquery.Client, fq_table: str, column: str = "ts") -> Optional[datetime]:
    query = f"SELECT MAX({column}) AS max_ts FROM `{client.project}.{fq_table}`"
    query_job = client.query(query)
    result = list(query_job.result())
    if not result:
        return None
    max_value = result[0]["max_ts"]
    if isinstance(max_value, datetime):
        if max_value.tzinfo is None:
            return max_value.replace(tzinfo=timezone.utc)
        return max_value.astimezone(timezone.utc)
    return None


def _inclusive_date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_partition_uris(bucket: str, interval_label: str, dates: Iterable[date]) -> List[str]:
    return [
        f"gs://{bucket}/raw/spot_klines/interval={interval_label}/yyyy={dt:%Y}/mm={dt:%m}/dd={dt:%d}/*.parquet"
        for dt in dates
    ]


def main() -> None:
    project_id = _get_env("PROJECT_ID", required=True)
    raw_table = _get_env("RAW_TABLE", default="btc_forecast_raw.spot_klines")
    curated_table = _get_env("CURATED_TABLE", default="btc_forecast_curated.btc_features_1h")
    bucket = _get_env("SPOT_GCS_BUCKET", required=True)
    interval_label = _get_env("SPOT_INTERVAL", default="1h")
    max_backfill_days = int(_get_env("MAX_BACKFILL_DAYS", default="7"))

    client = bigquery.Client(project=project_id)

    curated_max = _max_ts(client, curated_table)
    raw_max = _max_ts(client, raw_table)

    if curated_max is None:
        print("No rows found in curated table; nothing to sync.")
        return

    if raw_max is None:
        start_date = (curated_max - timedelta(days=max_backfill_days)).date()
        print("Raw table empty; backfilling recent window", start_date, curated_max.date())
    elif raw_max >= curated_max:
        print("Raw table up-to-date (raw_max >= curated_max); nothing to load.")
        return
    else:
        start_date = (raw_max + timedelta(hours=1)).date()
        print(f"Raw table lags curated ({raw_max.isoformat()} < {curated_max.isoformat()}); triggering load.")

    end_date = curated_max.date()

    if (end_date - start_date).days > max_backfill_days:
        start_date = end_date - timedelta(days=max_backfill_days)
        print("Limiting backfill window", start_date, end_date)

    uris = build_partition_uris(bucket, interval_label, _inclusive_date_range(start_date, end_date))

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    table_id = f"{project_id}.{raw_table}"
    load_job = client.load_table_from_uri(uris, table_id, job_config=job_config)
    load_job.result()
    print(f"Loaded {len(uris)} URIs into {table_id}:\n" + "\n".join(uris))


if __name__ == "__main__":
    main()
