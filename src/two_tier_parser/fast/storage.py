"""Optional S3-compatible object storage reads for fast-parser claim-check requests.

Multipart uploads never use this module, so the service runs with no object store
at all. JSON requests provide a bucket and key; the async HTTP handler schedules
these synchronous SDK reads in a thread. The parser reads caller-owned objects
without storing extraction results.

Settings (environment variables; the ``MINIO_*`` names are accepted as fallbacks
so earlier deployments keep working unchanged):

    OBJECT_STORE_ENDPOINT    host[:port] without a scheme (default minio:9000 once
                             any other setting marks storage as configured)
    OBJECT_STORE_ACCESS_KEY  access key id (fallback MINIO_ACCESS_KEY, MINIO_ROOT_USER)
    OBJECT_STORE_SECRET_KEY  secret key (fallback MINIO_SECRET_KEY, MINIO_ROOT_PASSWORD)
    OBJECT_STORE_SECURE      ``true`` selects TLS (fallback MINIO_SECURE)
    OBJECT_STORE_REGION      region name for stores that require one (optional)

Any S3-compatible service works: AWS S3, Cloudflare R2, Ceph, MinIO, or Google
Cloud Storage through its S3 interoperability endpoint.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from opentelemetry import trace
from urllib3 import PoolManager, Timeout

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")

DEFAULT_ENDPOINT = "minio:9000"
_SETTING_NAMES = (
    "OBJECT_STORE_ENDPOINT",
    "OBJECT_STORE_ACCESS_KEY",
    "OBJECT_STORE_SECRET_KEY",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
)


class StorageNotConfigured(RuntimeError):
    """Raised when a claim-check read is requested but no object store is configured."""


@dataclass(frozen=True)
class StoreSettings:
    """Connection settings resolved from the environment."""

    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    region: Optional[str]


def _first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def storage_configured() -> bool:
    """Return True when the environment names an object store.

    Nothing set means a standalone deployment that only accepts direct uploads.
    """
    return any(os.getenv(name) for name in _SETTING_NAMES)


def load_settings() -> StoreSettings:
    """Resolve connection settings, preferring ``OBJECT_STORE_*`` over ``MINIO_*``."""
    secure = _first("OBJECT_STORE_SECURE", "MINIO_SECURE") or "false"
    return StoreSettings(
        endpoint=_first("OBJECT_STORE_ENDPOINT", "MINIO_ENDPOINT") or DEFAULT_ENDPOINT,
        access_key=_first("OBJECT_STORE_ACCESS_KEY", "MINIO_ACCESS_KEY", "MINIO_ROOT_USER") or "",
        secret_key=_first("OBJECT_STORE_SECRET_KEY", "MINIO_SECRET_KEY", "MINIO_ROOT_PASSWORD")
        or "",
        secure=secure.strip().lower() == "true",
        region=_first("OBJECT_STORE_REGION"),
    )


_client: Any = None


def get_client() -> Any:
    """Return the shared client, initializing it from the environment on demand.

    Creating the client does not establish connectivity or bucket access.

    Raises:
        StorageNotConfigured: No object-store setting is present.
    """
    global _client
    if not storage_configured():
        raise StorageNotConfigured("No object store is configured")
    if _client is None:
        from minio import Minio

        settings = load_settings()
        _client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
            region=settings.region,
        )
        logger.info(
            "S3-compatible object storage client initialized",
            extra={
                "service_name": "fast-parser",
                "endpoint": settings.endpoint,
                "secure": settings.secure,
            },
        )
    return _client


def read_object(bucket: str, key: str) -> bytes:
    """Read object bytes synchronously and release the response connection.

    Args:
        bucket: Caller-owned S3-compatible bucket name.
        key: Object key validated by the HTTP request handler.

    Returns:
        Complete object contents as bytes.

    Raises:
        StorageNotConfigured: No object store is configured.
        Exception: SDK configuration, retrieval, or response reading failed.
            The HTTP handler translates storage failures to HTTP 502.
    """
    # Span and attribute names predate the provider-neutral settings; existing trace
    # queries and dashboards key on them, so they are kept.
    with tracer.start_as_current_span(
        "minio.get_object",
        attributes={
            "service_name": "fast-parser",
            "minio.bucket": bucket,
            "minio.key": key,
        },
    ) as span:
        client = get_client()
        response = client.get_object(bucket, key)
        try:
            data = response.read()
            span.set_attribute("file.size_bytes", len(data))
            return data
        finally:
            response.close()
            response.release_conn()


def check_connectivity(timeout_seconds: float = 1.0) -> bool:
    """Verify the configured object store with a bounded, non-retrying client.

    Parser reads keep the shared SDK client's normal retry behavior. Health
    probes use separate connection/read timeouts instead of the normal SDK retry
    sequence. The async health route also imposes its own budget around this call.

    Args:
        timeout_seconds: Per-connection and per-read socket timeout, not a strict
            total deadline for DNS, connection setup, and response handling.

    Returns:
        True if listing buckets succeeds; False if no store is configured or the
        probe raises an exception.
    """
    if not storage_configured():
        return False
    from minio import Minio

    settings = load_settings()
    http_client = PoolManager(
        timeout=Timeout(connect=timeout_seconds, read=timeout_seconds),
        retries=False,
    )
    try:
        client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
            region=settings.region,
            http_client=http_client,
        )
        client.list_buckets()
        return True
    except Exception:
        return False
    finally:
        http_client.clear()
