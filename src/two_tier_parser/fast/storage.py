"""S3-compatible object storage storage client for Fast Parser (Claim-Check Pattern).

Reads PDF files directly from S3-compatible object storage instead of receiving bytes via HTTP.
"""

import logging
import os

from opentelemetry import trace
from urllib3 import PoolManager, Timeout

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")

_minio_client = None


def get_minio_client():
    """Lazy-initialized S3-compatible object storage client."""
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
    """Read file bytes from S3-compatible object storage with OTel span."""
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
    probes need a tighter contract: an unavailable S3-compatible object storage endpoint must return
    ``False`` before Kubernetes' three-second readiness timeout instead of
    blocking the server's event loop through the SDK retry sequence.
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
