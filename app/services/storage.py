"""Object-storage abstraction — LocalStorage (filesystem) and S3Storage (boto3).

The compression engine is path-based: it reads a source file and writes the
compressed artifact to a local directory. So the worker stages I/O through a
local temp work dir and uses this backend only at the boundaries — pulling the
uploaded source down before the pipeline, pushing the produced artifact up
after. Download endpoints stream straight from the backend.

LocalStorage preserves the original on-disk behavior (files under storage_dir),
so tests and single-box local dev are byte-identical. S3Storage targets any
S3-compatible store (AWS S3, MinIO) via an endpoint URL + path-style addressing.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import get_settings

_CHUNK = 1 << 20  # 1 MiB streaming chunks


class StorageBackend(Protocol):
    def upload_file(self, local_path: str, key: str) -> None: ...
    def download_to(self, key: str, local_path: str) -> None: ...
    def open_stream(self, key: str) -> tuple[Iterator[bytes], int]: ...
    def read_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...


class StorageError(Exception):
    """Raised when an object is missing or the backend is unreachable."""


# ── local filesystem ──────────────────────────────────────────────────────────

class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def upload_file(self, local_path: str, key: str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local_path).resolve() != dest.resolve():
            shutil.copyfile(local_path, dest)

    def download_to(self, key: str, local_path: str) -> None:
        src = self._path(key)
        if not src.exists():
            raise StorageError(f"object not found: {key}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)

    def open_stream(self, key: str) -> tuple[Iterator[bytes], int]:
        path = self._path(key)
        if not path.exists():
            raise StorageError(f"object not found: {key}")
        size = path.stat().st_size

        def gen() -> Iterator[bytes]:
            with path.open("rb") as fh:
                while chunk := fh.read(_CHUNK):
                    yield chunk

        return gen(), size

    def read_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise StorageError(f"object not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        base = self._path(prefix)
        if base.is_dir():
            shutil.rmtree(base, ignore_errors=True)
        else:
            for p in self.root.glob(f"{prefix}*"):
                p.unlink(missing_ok=True)

    def ping(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        return True


# ── S3 / MinIO ────────────────────────────────────────────────────────────────

class S3Storage:
    def __init__(
        self, *, endpoint_url: str | None, bucket: str, region: str,
        access_key: str | None, secret_key: str | None, force_path_style: bool,
    ) -> None:
        import boto3
        from botocore.client import Config

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(s3={"addressing_style": "path" if force_path_style else "auto"}),
        )

    def upload_file(self, local_path: str, key: str) -> None:
        self._client.upload_file(local_path, self.bucket, key)

    def download_to(self, key: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, key, local_path)
        except Exception as exc:  # noqa: BLE001 — normalize to StorageError
            raise StorageError(f"object not found: {key}") from exc

    def open_stream(self, key: str) -> tuple[Iterator[bytes], int]:
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"object not found: {key}") from exc
        size = int(obj["ContentLength"])
        body = obj["Body"]

        def gen() -> Iterator[bytes]:
            try:
                yield from body.iter_chunks(_CHUNK)
            finally:
                body.close()

        return gen(), size

    def read_bytes(self, key: str) -> bytes:
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"object not found: {key}") from exc
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001 — create if missing
            self._client.create_bucket(Bucket=self.bucket)

    def ping(self) -> bool:
        self._client.head_bucket(Bucket=self.bucket)
        return True


# ── factory ───────────────────────────────────────────────────────────────────

@lru_cache
def get_storage() -> StorageBackend:
    s = get_settings()
    if s.storage_backend == "s3":
        store = S3Storage(
            endpoint_url=s.s3_endpoint_url, bucket=s.s3_bucket, region=s.s3_region,
            access_key=s.s3_access_key, secret_key=s.s3_secret_key,
            force_path_style=s.s3_force_path_style,
        )
        store.ensure_bucket()
        return store
    return LocalStorage(s.storage_dir)


def reset_storage() -> None:
    get_storage.cache_clear()


# ── key helpers (single source of truth for the layout) ───────────────────────

def artifact_key(model_id: str, suffix: str) -> str:
    # basename keeps the model id so the download filename stays
    # "{model_id}_compressed.{ext}" (preserves the SPA/SDK contract).
    return f"artifacts/{model_id}/{model_id}_compressed{suffix}"


def artifact_prefix(model_id: str) -> str:
    return f"artifacts/{model_id}/"


def ingested_key(model_id: str) -> str:
    """Post-ingestion ONNX — the exact graph trial configs re-apply onto.
    Lives under artifact_prefix so model deletion cleans it up for free."""
    return f"artifacts/{model_id}/{model_id}_ingested.onnx"


def trial_artifact_key(model_id: str, trial_number: int) -> str:
    """Per-Pareto-trial exported artifact (materialized on demand)."""
    return f"artifacts/{model_id}/pareto/{model_id}_trial{trial_number:03d}.onnx"


def source_key(token: str, file_name: str) -> str:
    return f"uploads/{token}_{Path(file_name).name}"
