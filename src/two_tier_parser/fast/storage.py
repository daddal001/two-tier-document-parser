"""Optional S3-compatible object storage reads for fast-parser claim-check requests.

Multipart uploads bypass this adapter. JSON requests provide a bucket and key;
the async HTTP handler schedules these synchronous SDK reads in a thread.
The parser reads caller-owned objects without storing extraction results.
"""

import logging
import os

from opentelemetry import trace
from urllib3 import PoolManager, Timeout

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")

_minio_client = None


def get_minio_client():
    """Return the shared client, initializing it from MINIO_* variables on demand.

    MINIO_ENDPOINT defaults to minio:9000. Access and secret keys use
    MINIO_ACCESS_KEY/MINIO_SECRET_KEY, with MINIO_ROOT_USER and
    MINIO_ROOT_PASSWORD as compatibility fallbacks. MINIO_SECURE=true selects TLS.
    Creating the client does not establish connectivity or bucket access.
    """
    global _minio_client
    if _minio_client is None:
        from minio import Minio

        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "")
        secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        _minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        logger.info(
            "S3-compatible object storage client initialized",
            extra={"service_name": "fast-parser", "endpoint": endpoint, "secure": secure},
        )
    return _minio_client


def read_file_from_minio(bucket: str, key: str) -> bytes:
    """Read object bytes synchronously and release the response connection.

    Args:
        bucket: Caller-owned S3-compatible bucket name.
        key: Object key validated by the HTTP request handler.

    Returns:
        Complete object contents as bytes.

    Raises:
        Exception: SDK configuration, retrieval, or response reading failed.
            The HTTP handler translates storage failures to HTTP 502.
    """
    with tracer.start_as_current_span(
        "minio.get_object",
        attributes={
            "service_name": "fast-parser",
            "minio.bucket": bucket,
            "minio.key": key,
        },
    ) as span:
        client = get_minio_client()
        response = client.get_object(bucket, key)
        try:
            data = response.read()
            span.set_attribute("file.size_bytes", len(data))
            return data
        finally:
            response.close()
            response.release_conn()


def check_minio_connectivity(timeout_seconds: float = 1.0) -> bool:
    """Verify S3-compatible object storage with a bounded, non-retrying health-check client.

    Parser reads keep the shared SDK client's normal retry behavior. Health
    probes use separate connection/read timeouts instead of the normal SDK retry
    sequence. The async health route also imposes its own budget around this call.

    Args:
        timeout_seconds: Per-connection and per-read socket timeout, not a strict
            total deadline for DNS, connection setup, and response handling.

    Returns:
        True if listing buckets succeeds; False if the probe raises an exception.
    """
    from minio import Minio

    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    http_client = PoolManager(
        timeout=Timeout(connect=timeout_seconds, read=timeout_seconds),
        retries=False,
    )
    try:
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            http_client=http_client,
        )
        client.list_buckets()
        return True
    except Exception:
        return False
    finally:
        http_client.clear()
