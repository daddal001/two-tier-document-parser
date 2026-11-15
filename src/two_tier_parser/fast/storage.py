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

Read limits (each read ends on these instead of the SDK's five-minute defaults; a read that
exceeds them fails with a 502, frees its admission slot, and does not affect the next read):

    OBJECT_STORE_CONNECT_TIMEOUT_SECONDS   connecting to the store (default 5)
    OBJECT_STORE_READ_TIMEOUT_SECONDS      waiting for the next bytes on the socket (default 20)
    OBJECT_STORE_RETRIES                   retries after a connection error or a 5xx (default 2)
    OBJECT_STORE_READ_BUDGET_SECONDS       the whole read, request to last byte (default 100)

The budget bounds how long a read can hold its admission slot. The two socket timeouts are
reduced when needed so that every attempt fits inside it (a store that sends a 503 with a long
Retry-After is not waited for either), and once bytes are arriving the budget is checked as
each piece comes in, so a stall at that point can overshoot it by at most one read timeout.

Any S3-compatible service works: AWS S3, Cloudflare R2, Ceph, MinIO, or Google
Cloud Storage through its S3 interoperability endpoint.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import certifi
from opentelemetry import trace
from urllib3 import PoolManager, Timeout
from urllib3.util import Retry

from ..common.env import env_number

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


class ObjectTooLarge(RuntimeError):
    """Raised when an object exceeds the size limit before it is fully buffered."""


class StorageTimeout(RuntimeError):
    """Raised when a read outlives its budget while still delivering bytes."""


# The body is read in pieces this big, checking the budget between pieces.
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ReadLimits:
    """How long a claim-check read may take."""

    connect_timeout: float
    read_timeout: float
    retries: int
    budget: float


def read_limits() -> ReadLimits:
    """Read the limits from the environment; a malformed value stops the read, naming it.

    Each of the ``retries + 1`` attempts may spend its connect timeout and then wait for the
    response headers, so the two timeouts are scaled down together when that would not fit in
    90 percent of the budget (the rest covers the retry back-off). The budget is never changed.
    """
    limits = ReadLimits(
        connect_timeout=env_number("OBJECT_STORE_CONNECT_TIMEOUT_SECONDS", 5.0, float, minimum=0.1),
        read_timeout=env_number("OBJECT_STORE_READ_TIMEOUT_SECONDS", 20.0, float, minimum=0.1),
        retries=env_number("OBJECT_STORE_RETRIES", 2, int, minimum=0),
        budget=env_number("OBJECT_STORE_READ_BUDGET_SECONDS", 100.0, float, minimum=0.1),
    )
    share = 0.9 * limits.budget / (limits.retries + 1)
    if limits.connect_timeout + limits.read_timeout <= share:
        return limits
    scale = share / (limits.connect_timeout + limits.read_timeout)
    return ReadLimits(
        connect_timeout=limits.connect_timeout * scale,
        read_timeout=limits.read_timeout * scale,
        retries=limits.retries,
        budget=limits.budget,
    )


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
        limits = read_limits()
        configured = (
            env_number("OBJECT_STORE_CONNECT_TIMEOUT_SECONDS", 5.0, float, minimum=0.1),
            env_number("OBJECT_STORE_READ_TIMEOUT_SECONDS", 20.0, float, minimum=0.1),
        )
        if (limits.connect_timeout, limits.read_timeout) != configured:
            logger.warning(
                "Object store timeouts reduced to fit the read budget",
                extra={
                    "service_name": "fast-parser",
                    "connect_timeout": round(limits.connect_timeout, 3),
                    "read_timeout": round(limits.read_timeout, 3),
                    "retries": limits.retries,
                    "budget_seconds": limits.budget,
                },
            )
        # The SDK's own client waits up to five minutes per socket operation and retries five
        # times, with no limit on the whole read. This one is the same client with bounded values.
        http_client = PoolManager(
            timeout=Timeout(connect=limits.connect_timeout, read=limits.read_timeout),
            maxsize=10,
            cert_reqs="CERT_REQUIRED",
            ca_certs=os.environ.get("SSL_CERT_FILE") or certifi.where(),
            retries=Retry(
                total=limits.retries,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
                respect_retry_after_header=False,  # a store asking for minutes is not waited for
                # All the back-off sleeps together stay inside the tenth of the budget that
                # read_limits() leaves for them; urllib3 would otherwise double each sleep.
                backoff_max=max(0.1 * limits.budget / max(limits.retries, 1), 0.01),
            ),
        )
        _client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
            region=settings.region,
            http_client=http_client,
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


def read_object(bucket: str, key: str, max_bytes: Optional[int] = None) -> bytearray:
    """Read object bytes synchronously and release the response connection.

    Args:
        bucket: Caller-owned S3-compatible bucket name.
        key: Object key validated by the HTTP request handler.
        max_bytes: Largest object to buffer. The response ``Content-Length`` is checked
            first, so an oversized object is rejected without reading its body; a
            response without that header is read at most ``max_bytes + 1`` bytes.

    Returns:
        Complete object contents, as a ``bytearray`` built in place so no second copy is held.

    Raises:
        StorageNotConfigured: No object store is configured.
        ObjectTooLarge: The object is larger than ``max_bytes``.
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
        limits = read_limits()
        deadline = time.monotonic() + limits.budget
        client = get_client()
        # shortcut: with no OBJECT_STORE_REGION the SDK first asks where the bucket is, one small
        # request per bucket that it caches, and that answer is outside the read budget. Set the
        # region to skip it; revisit if a store is ever seen to stall on it.
        response = client.get_object(bucket, key)
        try:
            data: bytearray
            if max_bytes is not None:
                declared = response.headers.get("Content-Length", "")
                if declared.isascii() and declared.isdigit() and int(declared) > max_bytes:
                    raise ObjectTooLarge(f"object is {declared} bytes; limit is {max_bytes}")
            data = _read_within(response, deadline, max_bytes, limits.budget)
            span.set_attribute("file.size_bytes", len(data))
            return data
        finally:
            response.close()
            response.release_conn()


def _read_within(
    response: Any, deadline: float, max_bytes: Optional[int], budget: float
) -> bytearray:
    """Read a response in pieces, giving up at ``deadline`` even if bytes are still arriving.

    Each socket read is bounded by the client's read timeout, so a stalled server ends there;
    this adds the limit for a server that keeps sending slowly. ``read1`` returns whatever one
    socket read delivered, so the deadline is checked as bytes arrive; ``read(n)`` would wait
    for all ``n`` bytes and let a trickling server hold the call far past the budget. Raising
    closes the connection in the caller's ``finally``, so the half-read response never goes
    back to the pool.
    """
    # Grown in place and returned as it is: collecting pieces and joining them holds the whole
    # object twice at the end, which at a 100 MiB document and several in flight is real memory
    # (measured on an 80 MiB object: 167 MiB at peak that way, 87 MiB this way).
    buffer = bytearray()
    while True:
        if time.monotonic() > deadline:
            raise StorageTimeout(f"object read exceeded its {budget:g} second budget")
        piece = response.read1(_CHUNK_BYTES)
        if not piece:
            return buffer
        if max_bytes is not None and len(buffer) + len(piece) > max_bytes:
            raise ObjectTooLarge(f"object exceeds the {max_bytes} byte limit")
        buffer += piece


def check_connectivity(timeout_seconds: float = 1.0) -> bool:
    """Verify the configured object store with a bounded, non-retrying client.

    Parser reads keep the shared SDK client's normal retry behavior. Health
    probes use separate connection/read timeouts instead of the normal SDK retry
    sequence. The async health route also imposes its own budget around this call.

    Args:
        timeout_seconds: Per-connection and per-read socket timeout, not a strict
            total deadline for DNS, connection setup, and response handling.

    Returns:
        True if listing buckets succeeds, or if the store answers "access denied" (it is
        reachable and has authenticated the caller; listing every bucket is a broader
        permission than reading objects, which is all this service needs). False if no store
        is configured, the store cannot be reached, or it rejects the credentials or fails.
    """
    if not storage_configured():
        return False
    from minio import Minio
    from minio.error import S3Error

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
    except S3Error as error:
        return error.code == "AccessDenied"
    except Exception:
        return False
    finally:
        http_client.clear()
