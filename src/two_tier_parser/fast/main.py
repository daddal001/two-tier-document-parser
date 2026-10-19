"""Fast Parser Service - FastAPI application with PyMuPDF4LLM.

Entry point following industry standard naming convention (main.py, not app.py).
This prevents module/package name collisions with FastAPI's app object.

Observability:
- OpenTelemetry API-only instrumentation (no-op without SDK)
- Structured JSON logging with NullHandler fallback
- Zero overhead for users who don't need tracing
- prometheus_client metrics exposed at /metrics (gauge + histogram per )
"""

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace

# Prometheus client is optional (defence-in-depth: never crash startup if the
# package is missing in a particular build). We publish two
# metrics: `pages_executor_queue_depth` (Gauge) and `parse_pages_total_ms_seconds`
# (Histogram). Falls back to no-op stand-ins so the route logic stays clean.
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_REGISTRY = CollectorRegistry()
    _PROM_OK = True
except ImportError:  # pragma: no cover - exercised only on stripped builds
    _PROMETHEUS_REGISTRY = None
    _PROM_OK = False

    class _NoopMetric:  # noqa: D401 - shim
        def labels(self, *a, **kw):
            return self

        def inc(self, *a, **kw):
            return None

        def dec(self, *a, **kw):
            return None

        def set(self, *a, **kw):
            return None

        def observe(self, *a, **kw):
            return None

    Gauge = Histogram = lambda *a, **kw: _NoopMetric()  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    def generate_latest(_registry=None):  # type: ignore[no-redef]
        return b""


from .models import HealthResponse, PageRangeParseResponse, ParseResponse
from .service import parse_pdf, parse_pdf_page_range
from .storage import check_minio_connectivity, read_file_from_minio

# Configure structured logging (applications can override)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")

# Initialize FastAPI app
app = FastAPI(
    title="Fast Parser Service",
    description="Ultra-fast PDF parsing using PyMuPDF4LLM",
    version="1.0.0",
)

# Optional OpenTelemetry instrumentation (API-only pattern)
# Zero overhead if SDK not installed - API calls are no-ops
try:
    from .core.telemetry import OTEL_AVAILABLE, setup_telemetry

    if OTEL_AVAILABLE:
        setup_telemetry(app)
        logger.info(
            "OpenTelemetry instrumentation enabled",
            extra={"service_name": "fast-parser", "component": "telemetry"},
        )
except ImportError:
    logger.debug(
        "Telemetry core not available, running without tracing",
        extra={"service_name": "fast-parser", "component": "telemetry"},
    )


# ProcessPoolExecutor for concurrent parsing (PyMuPDF doesn't support threading)
# See: https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
# max_tasks_per_child=50 recycles workers after 50 parse tasks to release accumulated
# memory from PyMuPDF internal caches while retaining the established general-
# service default. Deployments may override this with WORKER_MAX_TASKS.
# default worker count = container vCPUs. Prior hard-coded 2 was the
# largest single contributor to the 10× throughput gap. Respects cgroup CPU
# limits via sched_getaffinity on Linux; falls back to os.cpu_count elsewhere.
def _default_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS / Windows
        return max(1, os.cpu_count() or 2)


WORKERS = int(os.getenv("WORKERS", str(_default_workers())))
MAX_TASKS = int(os.getenv("WORKER_MAX_TASKS", "50"))
executor = ProcessPoolExecutor(max_workers=WORKERS, max_tasks_per_child=MAX_TASKS)

# dedicated pool for the latency-sensitive `/parse-pages` route.
# Eliminates priority inversion against multi-minute `/parse` calls on the
# shared `executor`. The established default keeps at least two page workers;
# deployments with tighter capacity can override this explicitly.
PAGES_WORKERS = int(os.getenv("PAGES_WORKERS", str(max(2, WORKERS // 2))))
pages_executor = ProcessPoolExecutor(
    max_workers=PAGES_WORKERS,
    max_tasks_per_child=MAX_TASKS,
)
_executor_reset_lock = threading.Lock()


def _recover_broken_executor(
    failed_executor: ProcessPoolExecutor,
    *,
    stage1: bool,
) -> HTTPException:
    """Replace a failed parser pool and return a retryable HTTP error.

    A child OOM or abrupt exit breaks the entire ProcessPoolExecutor. Keeping
    that object would make every later request fail even after memory pressure
    clears. The identity check ensures concurrent failures replace a given
    pool only once.
    """
    global executor, pages_executor

    with _executor_reset_lock:
        current = pages_executor if stage1 else executor
        if current is failed_executor:
            workers = PAGES_WORKERS if stage1 else WORKERS
            replacement = ProcessPoolExecutor(
                max_workers=workers,
                max_tasks_per_child=MAX_TASKS,
            )
            if stage1:
                pages_executor = replacement
            else:
                executor = replacement
            failed_executor.shutdown(wait=False, cancel_futures=True)

    return HTTPException(
        status_code=503,
        detail="parser_worker_unavailable",
        headers={"Retry-After": "2"},
    )


# backpressure threshold: 503 when queue depth > BACKPRESSURE_FACTOR × workers.
PAGES_BACKPRESSURE_FACTOR = int(os.getenv("PAGES_BACKPRESSURE_FACTOR", "2"))

# Stage-2 `/parse` admission control. Symmetric to the `/parse-pages`
# backpressure but on the SHARED `executor`. Without this, `/parse` had no
# admission control: an upstream caller that raises its concurrency could queue work inside
# the shared process pool, growing latency and memory instead of shedding load.
# A 503 lets callers apply bounded backoff. This pool-specific check falls open
# if the executor's private queue attribute is unavailable.
PARSE_BACKPRESSURE_FACTOR = int(os.getenv("PARSE_BACKPRESSURE_FACTOR", "2"))

# Bound the complete document working set (uploads/storage reads plus executor work), not
# only the private ProcessPoolExecutor queue. This prevents concurrent requests
# from each passing a stale queue-depth check and buffering unbounded PDFs.
# Default to the configured process slots for standalone deployments. Callers
# can set a different positive bound for their memory and latency requirements.
MAX_INFLIGHT_DOCUMENTS = int(os.getenv("MAX_INFLIGHT_DOCUMENTS", str(WORKERS + PAGES_WORKERS)))
MINIO_HEALTH_TIMEOUT_SECONDS = float(os.getenv("MINIO_HEALTH_TIMEOUT_SECONDS", "1.5"))
if WORKERS < 1 or PAGES_WORKERS < 1 or MAX_TASKS < 1 or MAX_INFLIGHT_DOCUMENTS < 1:
    raise ValueError(
        "WORKERS, PAGES_WORKERS, WORKER_MAX_TASKS, and MAX_INFLIGHT_DOCUMENTS must be positive"
    )
_inflight_documents = 0
_inflight_lock = asyncio.Lock()


@app.middleware("http")
async def bound_parser_inflight_documents(request: Request, call_next):
    """Reject excess parser work before reading a document into memory.

    The bound is shared by the Stage 1 and Stage 2 endpoints within the service
    process. A 503 is retriable and deliberately includes ``Retry-After``.
    """
    global _inflight_documents
    if request.url.path not in {"/parse", "/parse-pages"}:
        return await call_next(request)

    async with _inflight_lock:
        if _inflight_documents >= MAX_INFLIGHT_DOCUMENTS:
            logger.warning(
                "Rejecting parser request with 503: document admission saturated",
                extra={
                    "service_name": "fast-parser",
                    "component": "backpressure",
                    "inflight_documents": _inflight_documents,
                    "max_inflight_documents": MAX_INFLIGHT_DOCUMENTS,
                    "endpoint": request.url.path,
                },
            )
            return JSONResponse(
                status_code=503,
                content={"detail": "document_admission_saturated"},
                headers={"Retry-After": "2"},
            )
        _inflight_documents += 1

    try:
        return await call_next(request)
    finally:
        async with _inflight_lock:
            _inflight_documents -= 1


# Prometheus metrics for . Available to any Prometheus-compatible monitoring system.
if _PROM_OK:
    PAGES_EXECUTOR_QUEUE_DEPTH = Gauge(
        "pages_executor_queue_depth",
        "Current pending-work-item count for the /parse-pages dedicated ProcessPoolExecutor .",
        registry=_PROMETHEUS_REGISTRY,
    )
    PARSE_PAGES_TOTAL_MS = Histogram(
        "parse_pages_total_ms_seconds",
        "End-to-end wall-clock latency of /parse-pages, route-handler entry → return.",
        registry=_PROMETHEUS_REGISTRY,
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    PARSE_PAGES_BACKPRESSURE_503 = Gauge(
        "parse_pages_backpressure_503_total",
        "Cumulative count of /parse-pages requests rejected with HTTP 503 due to pool saturation.",
        registry=_PROMETHEUS_REGISTRY,
    )
    EXECUTOR_QUEUE_DEPTH = Gauge(
        "executor_queue_depth",
        "Current pending-work-item count for the shared /parse (Stage 2) ProcessPoolExecutor.",
        registry=_PROMETHEUS_REGISTRY,
    )
    PARSE_BACKPRESSURE_503 = Gauge(
        "parse_backpressure_503_total",
        "Cumulative count of /parse requests rejected with HTTP 503 due to shared-pool saturation.",
        registry=_PROMETHEUS_REGISTRY,
    )
else:  # pragma: no cover
    PAGES_EXECUTOR_QUEUE_DEPTH = Gauge()
    PARSE_PAGES_TOTAL_MS = Histogram()
    PARSE_PAGES_BACKPRESSURE_503 = Gauge()
    EXECUTOR_QUEUE_DEPTH = Gauge()
    PARSE_BACKPRESSURE_503 = Gauge()


def _pages_executor_queue_depth() -> int:
    """Read pending-work-item count from the dedicated `/parse-pages` pool.

    `_pending_work_items` is a private CPython attribute used by `concurrent.futures`
    internally; The check wraps it in a try/except so backpressure fails *open*
    (never blocks legitimate traffic) if a future Python release renames or
    removes it. In that case, shared admission and request timeouts still apply.
    """
    try:
        return len(pages_executor._pending_work_items)  # type: ignore[attr-defined]
    except AttributeError:
        return 0


def _executor_queue_depth() -> int:
    """Read pending-work-item count from the shared `/parse` (Stage 2) pool.

    Same private-attribute access + fail-open contract as
    `_pages_executor_queue_depth`: if `_pending_work_items` is renamed in a
    future CPython release this returns 0, so Stage-2 backpressure simply
    disengages rather than rejecting all traffic.
    """
    try:
        return len(executor._pending_work_items)  # type: ignore[attr-defined]
    except AttributeError:
        return 0


# Check if no-GIL mode is enabled
NO_GIL = os.getenv("PYTHON_GIL") == "0"


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up both process pools on shutdown ."""
    logger.info(
        "Shutting down, cleaning up process pools",
        extra={
            "service_name": "fast-parser",
            "component": "shutdown",
            "executor_workers": WORKERS,
            "pages_executor_workers": PAGES_WORKERS,
        },
    )
    executor.shutdown(wait=True)
    pages_executor.shutdown(wait=True)


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint .

    Exposes `pages_executor_queue_depth`, `parse_pages_total_ms_seconds`, and
    `parse_pages_backpressure_503_total`. Returns an empty body when
    `prometheus_client` is unavailable so scrapers see HTTP 200, not 5xx.
    """
    # Refresh the gauge on every scrape so we always publish a current
    # snapshot even if the route hasn't been called recently.
    PAGES_EXECUTOR_QUEUE_DEPTH.set(_pages_executor_queue_depth())
    EXECUTOR_QUEUE_DEPTH.set(_executor_queue_depth())
    return Response(
        content=generate_latest(_PROMETHEUS_REGISTRY) if _PROM_OK else b"",
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with worker and S3-compatible object storage status."""
    with tracer.start_as_current_span("health", attributes={"service_name": "fast-parser"}):
        try:
            minio_ok = await asyncio.wait_for(
                asyncio.to_thread(check_minio_connectivity),
                timeout=MINIO_HEALTH_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "S3-compatible object storage health check exceeded probe budget",
                extra={
                    "service_name": "fast-parser",
                    "timeout_seconds": MINIO_HEALTH_TIMEOUT_SECONDS,
                },
            )
            minio_ok = False
        return HealthResponse(
            status="degraded" if not minio_ok else "healthy",
            workers=WORKERS,
            no_gil=NO_GIL,
            minio_connected=minio_ok,
        )


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Security: User-supplied filenames can contain path traversal sequences
    like '../' or absolute paths. This function extracts only the basename.
    """
    if not filename:
        return "document.pdf"
    # Extract basename to prevent path traversal
    safe_name = os.path.basename(filename)
    safe_name = safe_name.replace("/", "_").replace("\\", "_")
    if not safe_name.lower().endswith(".pdf"):
        safe_name = "document.pdf"
    return safe_name


def _image_opts_from_payload(payload: dict, *, stage1: bool = False) -> dict:
    """Extract image-extraction options from a JSON request payload.

    Stage 1 (/parse-pages) defaults are tighter than Stage 2 (/parse) because
    Stage 1 is latency-critical (≤ 1.5s soft / ≤ 5s hard SLO per ).
    OWASP ASVS V12 + pixel-flood issue #1740 caps.
    """
    default_max_count = 50 if stage1 else 500
    default_max_bytes = 10 * 1024 * 1024 if stage1 else 50 * 1024 * 1024
    default_max_pixels = 50_000_000

    extract = payload.get("extract_images", True)
    if isinstance(extract, str):
        extract = extract.strip().lower() in {"1", "true", "yes"}
    try:
        max_count = int(payload.get("image_max_count", default_max_count))
    except (TypeError, ValueError):
        max_count = default_max_count
    try:
        max_bytes = int(payload.get("image_max_bytes", default_max_bytes))
    except (TypeError, ValueError):
        max_bytes = default_max_bytes
    try:
        max_pixels = int(payload.get("image_max_pixels", default_max_pixels))
    except (TypeError, ValueError):
        max_pixels = default_max_pixels

    return {
        "extract_images_flag": bool(extract),
        "image_max_count": max(0, max_count),
        "image_max_bytes": max(0, max_bytes),
        "image_max_pixels": max(0, max_pixels),
    }


async def _read_pdf_from_request(
    request: Request,
    *,
    stage1: bool = False,
) -> tuple[bytes, str, str, Optional[str], int, dict]:
    """Extract PDF bytes and filename from either JSON (claim-check) or multipart request.

    Returns:
        Tuple of (pdf_bytes, safe_filename, parse_source, task_id, start_page, image_opts).
        ``task_id`` and ``start_page`` are only populated from JSON payloads
        (preemption/resume); multipart uploads always return
        (None, 0, <defaults>).
        ``image_opts`` is the image-extraction options dict.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        # Claim-check path: read from S3-compatible object storage via bucket/key reference
        body = await request.body()
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        bucket = payload.get("bucket")
        key = payload.get("key")
        filename = payload.get("filename", "document.pdf")
        task_id = payload.get("task_id")
        try:
            start_page = max(0, int(payload.get("start_page", 0)))
        except (TypeError, ValueError):
            start_page = 0
        image_opts = _image_opts_from_payload(payload, stage1=stage1)

        if not bucket or not key:
            raise HTTPException(status_code=400, detail="JSON body must contain 'bucket' and 'key'")

        # Security: prevent path traversal in key
        if ".." in key:
            raise HTTPException(status_code=400, detail="Invalid key: path traversal not allowed")
        if not key.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        safe_filename = _sanitize_filename(filename)
        logger.info(
            "Claim-check parse request",
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "bucket": bucket,
                "key": key,
                "task_id": task_id,
                "start_page": start_page,
            },
        )

        try:
            pdf_bytes = read_file_from_minio(bucket, key)
        except Exception as e:
            logger.error(
                "Failed to read file from S3-compatible object storage",
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "bucket": bucket,
                    "key": key,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise HTTPException(
                status_code=502, detail=f"Failed to read file from storage: {type(e).__name__}"
            )

        return pdf_bytes, safe_filename, "claim-check", task_id, start_page, image_opts

    elif "multipart/form-data" in content_type:
        # Legacy multipart path: file uploaded directly
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(status_code=400, detail="No file field in multipart form")

        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        safe_filename = _sanitize_filename(file.filename)
        logger.info(
            "Multipart parse request",
            extra={"service_name": "fast-parser", "document_name": safe_filename},
        )

        try:
            pdf_bytes = await file.read()
        except Exception as e:
            logger.error(
                "Failed to read uploaded file",
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise HTTPException(status_code=400, detail="Failed to read file")

        # Multipart uploads use defaults for image extraction.
        image_opts = _image_opts_from_payload({}, stage1=stage1)
        return pdf_bytes, safe_filename, "multipart", None, 0, image_opts

    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported content type. Use application/json (claim-check) or multipart/form-data (file upload)",
        )


@app.post("/parse", response_model=ParseResponse)
async def parse(request: Request) -> ParseResponse:
    """Parse PDF file to markdown.

    Accepts either:
    - application/json with {"bucket", "key", "filename"} (claim-check pattern)
    - multipart/form-data with file field (legacy upload)

    Returns:
        ParseResponse with markdown content and metadata
    """
    request_start = time.perf_counter()
    parse_span_ctx = tracer.start_as_current_span(
        "parse", attributes={"service_name": "fast-parser", "endpoint": "/parse"}
    )
    parse_span_ctx.__enter__()

    try:
        # Stage-2 admission control (symmetric to /parse-pages). Sample
        # the shared-pool depth BEFORE the S3-compatible object storage read so a saturated pool sheds
        # load fast (503) instead of buffering an unbounded backlog of in-flight
        # full parses. Callers own retry budgets and deployment scaling policy.
        queue_depth = _executor_queue_depth()
        EXECUTOR_QUEUE_DEPTH.set(queue_depth)
        backpressure_limit = PARSE_BACKPRESSURE_FACTOR * WORKERS
        if queue_depth > backpressure_limit:
            PARSE_BACKPRESSURE_503.inc()
            span = trace.get_current_span()
            span.set_attribute("parse.backpressure", True)
            span.set_attribute("executor.queue_depth", queue_depth)
            span.set_attribute("executor.backpressure_limit", backpressure_limit)
            logger.warning(
                "Rejecting /parse with 503: shared pool saturated",
                extra={
                    "service_name": "fast-parser",
                    "component": "backpressure",
                    "executor_queue_depth": queue_depth,
                    "executor_workers": WORKERS,
                    "backpressure_limit": backpressure_limit,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="parse_pool_saturated",
            )

        pdf_bytes, safe_filename, parse_source, task_id, start_page, image_opts = (
            await _read_pdf_from_request(request, stage1=False)
        )
    except HTTPException:
        parse_span_ctx.__exit__(None, None, None)
        raise

    # Set source on span
    span = trace.get_current_span()
    span.set_attribute("parse.source", parse_source)
    span.set_attribute("parse.extract_images", image_opts["extract_images_flag"])
    if task_id:
        span.set_attribute("parse.task_id", task_id)
        span.set_attribute("parse.start_page", start_page)

    file_size = len(pdf_bytes)
    logger.debug(
        "File read complete",
        extra={
            "service_name": "fast-parser",
            "document_name": safe_filename,
            "size_bytes": file_size,
        },
    )

    # Validate file size (100MB limit)
    if file_size > 100 * 1024 * 1024:
        logger.warning(
            "File too large",
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "size_mb": file_size / (1024 * 1024),
            },
        )
        parse_span_ctx.__exit__(None, None, None)
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    # Parse PDF in process pool
    try:
        loop = asyncio.get_event_loop()
        logger.debug(
            "Submitting to process pool",
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "workers": WORKERS,
            },
        )

        active_executor = executor
        result = await loop.run_in_executor(
            active_executor,
            parse_pdf,
            pdf_bytes,
            safe_filename,  # SECURITY: Use sanitized filename
            task_id,
            start_page,
            0.8,  # progress_ceiling (default)
            image_opts["extract_images_flag"],
            image_opts["image_max_count"],
            image_opts["image_max_bytes"],
            image_opts["image_max_pixels"],
        )

        # Check if result is None (shouldn't happen with fast parser, but safety check)
        if result is None:
            logger.error(
                "Parsing returned None",
                extra={"service_name": "fast-parser", "document_name": safe_filename},
            )
            raise HTTPException(status_code=500, detail="Parsing failed: No result returned")

        # Check if result contains an error
        if "error" in result:
            logger.error(
                "Parsing failed",
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "error_message": result.get("error", "Unknown error"),
                },
            )
            raise HTTPException(
                status_code=500, detail=f"Parsing failed: {result.get('error', 'Unknown error')}"
            )

        # translate preemption into HTTP 409. The caller
        # chooses how to schedule a subsequent resume request.
        if result.get("metadata", {}).get("preempted"):
            last_page = result["metadata"].get("last_page_parsed", -1)
            total_pages = result["metadata"].get("pages", 0)
            logger.info(
                "Parse preempted, returning 409",
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "task_id": task_id,
                    "last_page_parsed": last_page,
                    "total_pages": total_pages,
                },
            )
            span.set_attribute("parse.preempted", True)
            span.set_attribute("parse.last_page_parsed", last_page)
            parse_span_ctx.__exit__(None, None, None)
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "preempted",
                    "last_page_parsed": last_page,
                    "pages_total": total_pages,
                    "markdown_prefix_length": len(result.get("markdown", "")),
                },
            )

        total_time_ms = int((time.perf_counter() - request_start) * 1000)

        # Structured log with all relevant metrics
        logger.info(
            "Parse complete",
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "parse_source": parse_source,
                "page_count": result["metadata"]["pages"],
                "parse_time_ms": result["metadata"]["processing_time_ms"],
                "total_time_ms": total_time_ms,
                "size_bytes": file_size,
                "throughput_pages_per_sec": (
                    result["metadata"]["pages"] / (result["metadata"]["processing_time_ms"] / 1000)
                    if result["metadata"]["processing_time_ms"] > 0
                    else 0
                ),
            },
        )

        parse_span_ctx.__exit__(None, None, None)
        return ParseResponse(**result)

    except HTTPException:
        parse_span_ctx.__exit__(None, None, None)
        raise
    except BrokenProcessPool as e:
        logger.error(
            "Parser worker process exited; replacing pool",
            exc_info=True,
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "error_type": type(e).__name__,
            },
        )
        parse_span_ctx.__exit__(type(e), e, e.__traceback__)
        raise _recover_broken_executor(active_executor, stage1=False)
    except Exception as e:
        logger.error(
            "Parsing failed",
            exc_info=True,
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        parse_span_ctx.__exit__(type(e), e, e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse-pages", response_model=PageRangeParseResponse)
async def parse_pages(
    request: Request,
    start_page: int = Query(0, ge=0, description="Start page (0-indexed, inclusive)"),
    end_page: Optional[int] = Query(
        None, ge=1, description="End page (exclusive). None = all remaining pages"
    ),
) -> PageRangeParseResponse:
    """Parse specific page range of a PDF document.

    Accepts either:
    - application/json with {"bucket", "key", "filename"} (claim-check pattern)
    - multipart/form-data with file field (legacy upload)

    Uses Python slice semantics for page ranges:
    - start_page: 0-indexed, inclusive
    - end_page: exclusive (like Python slicing). None means parse to end.

    Example: start_page=0, end_page=2 parses pages 0 and 1 (first 2 pages)
    """
    request_start = time.perf_counter()

    with tracer.start_as_current_span(
        "parse_pages", attributes={"service_name": "fast-parser", "endpoint": "/parse-pages"}
    ) as span:
        # backpressure: refuse fast and let the caller apply its retry policy. Threshold is sampled
        # *before* PDF read so we don't waste S3-compatible object storage bandwidth on a request
        # we're about to reject.
        queue_depth = _pages_executor_queue_depth()
        PAGES_EXECUTOR_QUEUE_DEPTH.set(queue_depth)
        backpressure_limit = PAGES_BACKPRESSURE_FACTOR * PAGES_WORKERS
        if queue_depth > backpressure_limit:
            PARSE_PAGES_BACKPRESSURE_503.inc()
            span.set_attribute("parse.backpressure", True)
            span.set_attribute("pages_executor.queue_depth", queue_depth)
            span.set_attribute("pages_executor.backpressure_limit", backpressure_limit)
            logger.warning(
                "Rejecting /parse-pages with 503: pool saturated",
                extra={
                    "service_name": "fast-parser",
                    "component": "backpressure",
                    "pages_executor_queue_depth": queue_depth,
                    "pages_executor_workers": PAGES_WORKERS,
                    "backpressure_limit": backpressure_limit,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="parse_pages_pool_saturated",
            )

        try:
            pdf_bytes, safe_filename, parse_source, _task_id, _start_page, image_opts = (
                await _read_pdf_from_request(request, stage1=True)
            )
        except HTTPException:
            raise

        span.set_attribute("parse.extract_images", image_opts["extract_images_flag"])

        span.set_attribute("parse.source", parse_source)
        span.set_attribute("pages_executor.queue_depth", queue_depth)

        file_size = len(pdf_bytes)
        logger.info(
            "Page range parse request",
            extra={
                "service_name": "fast-parser",
                "document_name": safe_filename,
                "start_page": start_page,
                "end_page": end_page,
                "pages_executor_queue_depth": queue_depth,
            },
        )

        # Validate file size (100MB limit)
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")

        # submit to the dedicated `pages_executor`, not the shared
        # `executor`. Refresh queue-depth after submit so the gauge reflects
        # the new pending count for any concurrent scrape.
        try:
            loop = asyncio.get_event_loop()
            active_executor = pages_executor
            future = loop.run_in_executor(
                active_executor,
                parse_pdf_page_range,
                pdf_bytes,
                safe_filename,  # SECURITY: Use sanitized filename
                start_page,
                end_page,
                image_opts["extract_images_flag"],
                image_opts["image_max_count"],
                image_opts["image_max_bytes"],
                image_opts["image_max_pixels"],
            )
            PAGES_EXECUTOR_QUEUE_DEPTH.set(_pages_executor_queue_depth())
            result = await future

            total_time_s = time.perf_counter() - request_start
            total_time_ms = int(total_time_s * 1000)
            PARSE_PAGES_TOTAL_MS.observe(total_time_s)
            # Refresh after completion so a low-traffic scrape sees the
            # post-completion depth, not the stale pre-submit value.
            PAGES_EXECUTOR_QUEUE_DEPTH.set(_pages_executor_queue_depth())

            logger.info(
                "Page range parse complete",
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "parse_source": parse_source,
                    "pages_parsed": result["metadata"]["pages_parsed"],
                    "page_count": result["metadata"]["total_pages"],
                    "start_page": result["metadata"]["start_page"],
                    "end_page": result["metadata"]["end_page"],
                    "parse_time_ms": result["metadata"]["processing_time_ms"],
                    "total_time_ms": total_time_ms,
                    "pages_executor_queue_depth_at_submit": queue_depth,
                },
            )

            return PageRangeParseResponse(**result)

        except BrokenProcessPool as e:
            logger.error(
                "Page-range parser worker process exited; replacing pool",
                exc_info=True,
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "error_type": type(e).__name__,
                },
            )
            raise _recover_broken_executor(active_executor, stage1=True)
        except Exception as e:
            logger.error(
                "Page range parsing failed",
                exc_info=True,
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
