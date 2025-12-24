import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator, Iterable, List, Optional, Tuple

from google.cloud import storage

_CLIENT: Optional[storage.Client] = None


def _get_client() -> storage.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = storage.Client()
    return _CLIENT


def is_gcs_uri(path: Optional[str]) -> bool:
    return bool(path and path.startswith("gs://"))


def split_gcs_uri(uri: str) -> Tuple[str, str]:
    if not is_gcs_uri(uri):
        raise ValueError(f"Not a GCS URI: {uri}")
    path = uri[5:]
    if "/" in path:
        bucket, blob = path.split("/", 1)
    else:
        bucket, blob = path, ""
    return bucket, blob


def join_uri(base_uri: str, relative_path: str) -> str:
    if is_gcs_uri(base_uri):
        bucket, prefix = split_gcs_uri(base_uri)
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        relative = relative_path.lstrip("/")
        combined = f"{prefix}{relative}" if prefix else relative
        return f"gs://{bucket}/{combined}"
    return str(Path(base_uri) / relative_path)


def gcs_blob_exists(uri: str) -> bool:
    bucket_name, blob_name = split_gcs_uri(uri)
    if not blob_name:
        return False
    client = _get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.exists()


def _download_blob(uri: str, local_path: Path) -> None:
    bucket_name, blob_name = split_gcs_uri(uri)
    client = _get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(local_path)


def _upload_blob(local_path: Path, uri: str) -> None:
    bucket_name, blob_name = split_gcs_uri(uri)
    client = _get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(local_path)


def resolve_to_local(path: Optional[str]) -> Tuple[str, Optional[Callable[[], None]]]:
    if not path or not is_gcs_uri(path):
        return path or "", None

    bucket_name, blob_name = split_gcs_uri(path)
    if not blob_name:
        raise ValueError(f"GCS URI must include object key: {path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="btc_cloud_io_"))
    local_path = temp_dir / Path(blob_name).name
    client = _get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if blob.exists():
        blob.download_to_filename(local_path)
    else:
        raise FileNotFoundError(f"Blob not found: {path}")

    def cleanup() -> None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return str(local_path), cleanup


@contextmanager
def local_artifact(target_uri: str) -> Generator[str, None, None]:
    if not is_gcs_uri(target_uri):
        path = Path(target_uri)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        yield str(path)
        return

    bucket_name, blob_name = split_gcs_uri(target_uri)
    client = _get_client()
    temp_dir = Path(tempfile.mkdtemp(prefix="btc_cloud_io_"))
    local_path = temp_dir / Path(blob_name).name
    if blob_name:
        blob = client.bucket(bucket_name).blob(blob_name)
        if blob.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        yield str(local_path)
        if local_path.exists():
            _upload_blob(local_path, target_uri)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def local_directory(base_uri: str, subdir: str) -> Generator[Tuple[Path, str], None, None]:
    subdir = subdir.strip("/") or "output"
    if not is_gcs_uri(base_uri):
        path = Path(base_uri) / subdir
        path.mkdir(parents=True, exist_ok=True)
        yield path, str(path)
        return

    temp_root = Path(tempfile.mkdtemp(prefix="btc_governance_"))
    local_dir = temp_root / subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = join_uri(base_uri, subdir)
    try:
        yield local_dir, remote_prefix
        upload_directory(local_dir, remote_prefix)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def upload_directory(local_dir: Path, destination_uri: str) -> None:
    if not is_gcs_uri(destination_uri):
        target_dir = Path(destination_uri)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in local_dir.rglob('*'):
            if item.is_file():
                relative = item.relative_to(local_dir)
                dest = target_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        return

    bucket_name, base_blob = split_gcs_uri(destination_uri)
    client = _get_client()
    prefix = base_blob.rstrip('/')
    for item in local_dir.rglob('*'):
        if not item.is_file():
            continue
        relative = item.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{relative}" if prefix else relative
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(item)


def list_subdirectories(base_uri: str) -> List[str]:
    if not is_gcs_uri(base_uri):
        base_path = Path(base_uri)
        if not base_path.exists():
            return []
        return sorted([p.name for p in base_path.iterdir() if p.is_dir()])

    bucket_name, prefix = split_gcs_uri(base_uri)
    client = _get_client()
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    iterator = client.list_blobs(bucket_name, prefix=prefix, delimiter='/')
    prefixes: List[str] = []
    for page in iterator.pages:
        prefixes.extend(page.prefixes)
    cleaned: List[str] = []
    for p in prefixes:
        if prefix:
            suffix = p[len(prefix):]
        else:
            suffix = p
        suffix = suffix.rstrip('/')
        if suffix:
            cleaned.append(suffix)
    return sorted(cleaned)