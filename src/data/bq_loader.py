from typing import Optional

import pandas as pd
from google.cloud import bigquery

from src.config import PROJECT_ID, BQ_DATASET_CURATED, BQ_TABLE_FEATURES_1H


def load_btc_features_1h(
    project_id: str = PROJECT_ID,
    dataset_id: str = BQ_DATASET_CURATED,
    table_id: str = BQ_TABLE_FEATURES_1H,
    where_clause: Optional[str] = None,
) -> pd.DataFrame:
    """Load the curated 1h BTC features table from BigQuery into a DataFrame.

    Parameters
    ----------
    project_id: GCP project id.
    dataset_id: BigQuery dataset name containing the features table.
    table_id: BigQuery table name of the features table.
    where_clause: Optional SQL condition without the 'WHERE' keyword,
        e.g. "ts >= '2023-01-01'".
    """
    client = bigquery.Client(project=project_id)
    table_fq = f"`{project_id}.{dataset_id}.{table_id}`"

    query = f"SELECT * FROM {table_fq}"
    if where_clause:
        query += f" WHERE {where_clause}"
    query += " ORDER BY ts"

    job = client.query(query)
    df = job.to_dataframe()
    return df
