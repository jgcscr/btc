from typing import Optional

import pandas as pd
from google.cloud import bigquery, bigquery_storage

from src.config import (
    PROJECT_ID,
    BQ_DATASET_CURATED,
    BQ_TABLE_FEATURES_1H,
    BQ_TABLE_FEATURES_15M,
)


def _load_features_table(
    *,
    project_id: str,
    dataset_id: str,
    table_id: str,
    where_clause: Optional[str] = None,
) -> pd.DataFrame:
    """Helper that loads an arbitrary curated features table from BigQuery."""
    client = bigquery.Client(project=project_id)
    table_fq = f"`{project_id}.{dataset_id}.{table_id}`"

    query = f"SELECT * FROM {table_fq}"
    if where_clause:
        query += f" WHERE {where_clause}"
    query += " ORDER BY ts"

    job = client.query(query)

    with bigquery_storage.BigQueryReadClient() as bqstorage_client:
        df = job.to_dataframe(bqstorage_client=bqstorage_client)

    return df


def load_btc_features_1h(
    project_id: str = PROJECT_ID,
    dataset_id: str = BQ_DATASET_CURATED,
    table_id: str = BQ_TABLE_FEATURES_1H,
    where_clause: Optional[str] = None,
) -> pd.DataFrame:
    """Load the curated 1h BTC features table from BigQuery into a DataFrame."""

    return _load_features_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        where_clause=where_clause,
    )


def load_btc_features_15m(
    project_id: str = PROJECT_ID,
    dataset_id: str = BQ_DATASET_CURATED,
    table_id: str = BQ_TABLE_FEATURES_15M,
    where_clause: Optional[str] = None,
) -> pd.DataFrame:
    """Load the curated 15m BTC features table from BigQuery into a DataFrame."""

    return _load_features_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        where_clause=where_clause,
    )
