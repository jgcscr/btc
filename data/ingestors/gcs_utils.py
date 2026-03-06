from google.cloud import storage


def upload_file_to_gcs(local_path: str, bucket_name: str, blob_path: str) -> None:
    """Upload a local file to GCS at gs://bucket_name/blob_path."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")
