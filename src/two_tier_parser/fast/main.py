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
WORKERS = int(os.getenv("WORKERS", "2"))  # Reduced default for process overhead
executor = ProcessPoolExecutor(max_workers=WORKERS)

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


async def _read_pdf_from_request(request: Request) -> tuple[bytes, str, str]:
    """Extract PDF bytes and filename from either JSON (claim-check) or multipart request.

    Returns:
        Tuple of (pdf_bytes, safe_filename, parse_source).
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
            extra={"service_name": "fast-parser", "document_name": safe_filename, "bucket": bucket, "key": key},
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

        return pdf_bytes, safe_filename, "claim-check"

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

        return pdf_bytes, safe_filename, "multipart"

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
        pdf_bytes, safe_filename, parse_source = await _read_pdf_from_request(request)
    except HTTPException:
        parse_span_ctx.__exit__(None, None, None)
        raise

    # Set source on span
    span = trace.get_current_span()
    span.set_attribute("parse.source", parse_source)

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
            safe_filename  # SECURITY: Use sanitized filename
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
            pdf_bytes, safe_filename, parse_source = await _read_pdf_from_request(request)
        except HTTPException:
            raise

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
