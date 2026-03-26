"""MinIO storage client for Fast Parser (ADR-0039 Claim-Check Pattern).

Reads PDF files directly from MinIO instead of receiving bytes via HTTP.
"""
import logging
import os

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")

_minio_client = None


def get_minio_client():
    """Lazy-initialized MinIO client."""
    global _minio_client
    if _minio_client is None:
        from minio import Minio
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "")
        secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        _minio_client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        logger.info(
            "MinIO client initialized",
            extra={"service_name": "fast-parser", "endpoint": endpoint, "secure": secure},
        )
    return _minio_client


def read_file_from_minio(bucket: str, key: str) -> bytes:
    """Read file bytes from MinIO with OTel span."""
    with tracer.start_as_current_span("minio.get_object", attributes={
        "service_name": "fast-parser",
        "minio.bucket": bucket,
        "minio.key": key,
    }) as span:
        client = get_minio_client()
        response = client.get_object(bucket, key)
        try:
            data = response.read()
            span.set_attribute("file.size_bytes", len(data))
            return data
        finally:
            response.close()
            response.release_conn()


def check_minio_connectivity() -> bool:
    """Health check: verify MinIO is reachable."""
    try:
        client = get_minio_client()
        client.list_buckets()
        return True
    except Exception:
        return False
