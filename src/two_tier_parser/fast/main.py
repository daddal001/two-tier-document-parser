"""Fast Parser Service - FastAPI application with PyMuPDF4LLM.

Entry point following industry standard naming convention (main.py, not app.py).
This prevents module/package name collisions with FastAPI's app object.

Observability:
- OpenTelemetry API-only instrumentation (no-op without SDK)
- Structured JSON logging with NullHandler fallback
- Zero overhead for users who don't need tracing
"""
import asyncio
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from opentelemetry import trace

from .models import ParseResponse, HealthResponse, PageRangeParseResponse
from .service import parse_pdf, parse_pdf_page_range
from .storage import read_file_from_minio, check_minio_connectivity

# Configure structured logging (applications can override)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")

# Initialize FastAPI app
app = FastAPI(
    title="Fast Parser Service",
    description="Ultra-fast PDF parsing using PyMuPDF4LLM",
    version="1.0.0"
)

# Optional OpenTelemetry instrumentation (API-only pattern)
# Zero overhead if SDK not installed - API calls are no-ops
try:
    from .core.telemetry import setup_telemetry, OTEL_AVAILABLE
    if OTEL_AVAILABLE:
        setup_telemetry(app)
        logger.info(
            "OpenTelemetry instrumentation enabled",
            extra={"service_name": "fast-parser", "component": "telemetry"}
        )
except ImportError:
    logger.debug(
        "Telemetry core not available, running without tracing",
        extra={"service_name": "fast-parser", "component": "telemetry"}
    )

# ProcessPoolExecutor for concurrent parsing (PyMuPDF doesn't support threading)
# See: https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
# max_tasks_per_child=50 recycles workers after 50 parse tasks to release accumulated
# memory from PyMuPDF internal caches (Meta TorchServe pattern).
# ADR-0049: default worker count = container vCPUs. Prior hard-coded 2 was the
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

# Check if no-GIL mode is enabled
NO_GIL = os.getenv("PYTHON_GIL") == "0"


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up process pool on shutdown."""
    logger.info(
        "Shutting down, cleaning up process pool",
        extra={"service_name": "fast-parser", "component": "shutdown"}
    )
    executor.shutdown(wait=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with worker and MinIO status."""
    with tracer.start_as_current_span("health", attributes={"service_name": "fast-parser"}):
        minio_ok = check_minio_connectivity()
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
    if not safe_name.lower().endswith('.pdf'):
        safe_name = "document.pdf"
    return safe_name


def _image_opts_from_payload(payload: dict, *, stage1: bool = False) -> dict:
    """Extract ADR-0049 image-extraction options from a JSON request payload.

    Stage 1 (/parse-pages) defaults are tighter than Stage 2 (/parse) because
    Stage 1 is latency-critical (≤ 1.5s soft / ≤ 5s hard SLO per ADR-0047).
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
        (ADR-0047 preemption/resume); multipart uploads always return
        (None, 0, <defaults>).
        ``image_opts`` is the ADR-0049 image-extraction options dict.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        # Claim-check path: read from MinIO via bucket/key reference
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
                "Failed to read file from MinIO",
                extra={
                    "service_name": "fast-parser",
                    "document_name": safe_filename,
                    "bucket": bucket,
                    "key": key,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise HTTPException(status_code=502, detail=f"Failed to read file from storage: {type(e).__name__}")

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
        logger.info("Multipart parse request", extra={"service_name": "fast-parser", "document_name": safe_filename})

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
    parse_span_ctx = tracer.start_as_current_span("parse", attributes={"service_name": "fast-parser", "endpoint": "/parse"})
    parse_span_ctx.__enter__()

    try:
        pdf_bytes, safe_filename, parse_source, task_id, start_page, image_opts = await _read_pdf_from_request(request, stage1=False)
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
    logger.debug("File read complete", extra={"service_name": "fast-parser", "document_name": safe_filename, "size_bytes": file_size})

    # Validate file size (100MB limit)
    if file_size > 100 * 1024 * 1024:
        logger.warning("File too large", extra={"service_name": "fast-parser", "document_name": safe_filename, "size_mb": file_size / (1024*1024)})
        parse_span_ctx.__exit__(None, None, None)
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    # Parse PDF in process pool
    try:
        loop = asyncio.get_event_loop()
        logger.debug("Submitting to process pool", extra={"service_name": "fast-parser", "document_name": safe_filename, "workers": WORKERS})

        result = await loop.run_in_executor(
            executor,
            parse_pdf,
            pdf_bytes,
            safe_filename,  # SECURITY: Use sanitized filename
            task_id,
            start_page,
            0.8,  # progress_ceiling (ADR-0047 default)
            image_opts["extract_images_flag"],
            image_opts["image_max_count"],
            image_opts["image_max_bytes"],
            image_opts["image_max_pixels"],
        )

        # Check if result is None (shouldn't happen with fast parser, but safety check)
        if result is None:
            logger.error("Parsing returned None", extra={"service_name": "fast-parser", "document_name": safe_filename})
            raise HTTPException(status_code=500, detail="Parsing failed: No result returned")

        # Check if result contains an error
        if "error" in result:
            logger.error(
                "Parsing failed",
                extra={"service_name": "fast-parser", "document_name": safe_filename, "error_message": result.get('error', 'Unknown error')}
            )
            raise HTTPException(
                status_code=500,
                detail=f"Parsing failed: {result.get('error', 'Unknown error')}"
            )

        # ADR-0047: translate preemption into HTTP 409. The Stage 2 Celery task
        # catches this and RPUSHes a resume payload to parser:deferred.
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
                "throughput_pages_per_sec": result["metadata"]["pages"] / (result["metadata"]["processing_time_ms"] / 1000) if result["metadata"]["processing_time_ms"] > 0 else 0,
            }
        )

        parse_span_ctx.__exit__(None, None, None)
        return ParseResponse(**result)

    except HTTPException:
        parse_span_ctx.__exit__(None, None, None)
        raise
    except Exception as e:
        logger.error(
            "Parsing failed",
            exc_info=True,
            extra={"service_name": "fast-parser", "document_name": safe_filename, "error_type": type(e).__name__, "error_message": str(e)}
        )
        parse_span_ctx.__exit__(type(e), e, e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse-pages", response_model=PageRangeParseResponse)
async def parse_pages(
    request: Request,
    start_page: int = Query(0, ge=0, description="Start page (0-indexed, inclusive)"),
    end_page: Optional[int] = Query(None, ge=1, description="End page (exclusive). None = all remaining pages"),
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

    with tracer.start_as_current_span("parse_pages", attributes={"service_name": "fast-parser", "endpoint": "/parse-pages"}) as span:
        try:
            pdf_bytes, safe_filename, parse_source, _task_id, _start_page, image_opts = await _read_pdf_from_request(request, stage1=True)
        except HTTPException:
            raise

        span.set_attribute("parse.extract_images", image_opts["extract_images_flag"])

        span.set_attribute("parse.source", parse_source)

        file_size = len(pdf_bytes)
        logger.info(
            "Page range parse request",
            extra={"service_name": "fast-parser", "document_name": safe_filename, "start_page": start_page, "end_page": end_page}
        )

        # Validate file size (100MB limit)
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")

        # Parse PDF page range in process pool
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
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

            total_time_ms = int((time.perf_counter() - request_start) * 1000)

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
                }
            )

            return PageRangeParseResponse(**result)

        except Exception as e:
            logger.error(
                "Page range parsing failed",
                exc_info=True,
                extra={"service_name": "fast-parser", "document_name": safe_filename, "error_type": type(e).__name__, "error_message": str(e)}
            )
            raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
