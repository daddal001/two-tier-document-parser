"""FastAPI entry point for MinerU multimodal PDF extraction.

CUDA selects MinerU high effort; MPS and CPU use its medium pipeline. A thread pool keeps
model inference in the serving process; periodic cleanup reclaims Python and
CUDA caches. The entrypoint prepares model files before HTTP startup.

Routes:
    POST /parse: Multipart PDF extraction with a 500 MiB input limit checked
        after reading the upload; returns Markdown, images, tables, and formulas.
        At most MAX_INFLIGHT_DOCUMENTS documents are admitted (running plus queued);
        further requests get an immediate 503 with ``Retry-After``.
    GET /health: Service status and CUDA availability, not an inference probe.

Logging is local JSON unless the host configured handlers. Trace export requires
explicit OTLP configuration. The app.py module remains a compatibility wrapper.
"""

import asyncio
import gc
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from opentelemetry import trace
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Keep the established config-path variable usable while MinerU 4 migrates
# from magic-pdf JSON keys to its YAML model-base-dir configuration.
if os.getenv("MINERU_TOOLS_CONFIG_JSON") and not os.getenv("MINERU_CONFIG"):
    os.environ["MINERU_CONFIG"] = os.environ["MINERU_TOOLS_CONFIG_JSON"]

# Configure structured logging (applications can override)
from .. import API_VERSION
from ..common.json_logging import configure_json_logging
from .models import HealthResponse, ParseResponse
from .service import detect_device, parse_pdf

configure_json_logging(os.getenv("LOG_LEVEL", "INFO"))

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.accurate")

# Initialize FastAPI app
app = FastAPI(
    title="Accurate Parser Service",
    description="High-quality PDF parsing using MinerU with multimodal extraction",
    version=API_VERSION,
)

# Optional OpenTelemetry instrumentation (API-only pattern)
# No trace export if SDK not installed - API calls are no-ops
try:
    from .core.telemetry import OTEL_AVAILABLE, setup_telemetry

    if OTEL_AVAILABLE:
        enabled = setup_telemetry(app)
        logger.debug(
            "OpenTelemetry setup complete",
            extra={"service_name": "accurate-parser", "component": "telemetry", "enabled": enabled},
        )
except ImportError:
    logger.debug(
        "Telemetry core not available, running without tracing",
        extra={"service_name": "accurate-parser", "component": "telemetry"},
    )


# ThreadPoolExecutor for concurrent parsing
# Cannot use ProcessPoolExecutor because GPU models (PyTorch/MinerU) must stay in-process.
# Instead, periodic gc.collect() + torch.cuda.empty_cache() reclaims accumulated memory.
def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer setting, treating an empty value as unset.

    A malformed or out-of-range value stops startup with a message naming the variable.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {raw!r}")
    return value


WORKERS = _env_int("WORKERS", 2)
MAX_TASKS_BEFORE_CLEANUP = _env_int("WORKER_MAX_TASKS", 50)
# Documents admitted at once, running plus queued. A document can take minutes, so the
# queue stays short and callers are told to retry instead of waiting behind it.
MAX_INFLIGHT_DOCUMENTS = _env_int("MAX_INFLIGHT_DOCUMENTS", WORKERS * 2)
executor = ThreadPoolExecutor(max_workers=WORKERS)

# Admission state. Only the event loop thread touches the counter, so it needs no lock.
_inflight_documents = 0
_PARSER_PATHS = {"/parse"}


def _route_path(scope: Scope) -> str:
    """Return the request path the router sees, without the ASGI ``root_path`` prefix.

    Behind a path-prefixing proxy, servers such as uvicorn (``--root-path``) put the prefix
    in ``scope["path"]`` and the router strips it again; comparing the raw path would
    miss ``/parse`` and skip admission control.
    """
    path = scope["path"]
    root = (scope.get("root_path") or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root) :]
    return path.rstrip("/") or "/"


class AdmissionMiddleware:
    """Reject excess parse requests before their upload is read.

    A slot is taken when the request is admitted and normally released when the
    response ends. Once the handler has queued work it takes over the slot (see
    ``_handoff``) and releases it when the worker thread finishes, so a client that
    disconnects mid-parse cannot let the number of documents in flight exceed the bound.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        global _inflight_documents
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or _route_path(scope) not in _PARSER_PATHS
        ):
            await self.app(scope, receive, send)
            return

        if _inflight_documents >= MAX_INFLIGHT_DOCUMENTS:
            logger.warning(
                "Rejecting parse request with 503: document admission saturated",
                extra={
                    "service_name": "accurate-parser",
                    "component": "backpressure",
                    "inflight_documents": _inflight_documents,
                    "max_inflight_documents": MAX_INFLIGHT_DOCUMENTS,
                },
            )
            response = JSONResponse(
                status_code=503,
                content={"detail": "document_admission_saturated"},
                headers={"Retry-After": "5"},
            )
            await response(scope, receive, send)
            return

        _inflight_documents += 1
        scope.setdefault("state", {})["admission_slot_owned_by_handler"] = False
        try:
            await self.app(scope, receive, send)
        finally:
            if not scope["state"]["admission_slot_owned_by_handler"]:
                _release_slot()


def _release_slot() -> None:
    """Return one admission slot; must run on the event loop thread."""
    global _inflight_documents
    _inflight_documents -= 1
    logger.debug(
        "Parse admission released",
        extra={"service_name": "accurate-parser", "inflight_documents": _inflight_documents},
    )


def _handoff(request: Request) -> None:
    """Make the handler responsible for releasing this request's admission slot."""
    request.scope["state"]["admission_slot_owned_by_handler"] = True


app.add_middleware(AdmissionMiddleware)


def _internal_error(message: str, safe_filename: str, detail: str, exc: Optional[BaseException]):
    """Log the full failure and return a generic 500 that carries only a correlation id.

    Exception and MinerU error text can contain file paths or library internals, so it
    stays in the server log. The id in the response lets an operator find the log line.
    """
    correlation_id = uuid.uuid4().hex[:12]
    logger.error(
        message,
        exc_info=exc,
        extra={
            "service_name": "accurate-parser",
            "document_name": safe_filename,
            "error_type": type(exc).__name__ if exc else "ParseError",
            "error_message": detail,
            "correlation_id": correlation_id,
        },
    )
    return HTTPException(
        status_code=500,
        detail=f"Parsing failed (correlation_id={correlation_id})",
    )


# Task counter for periodic memory cleanup (thread-safe)
_task_counter = 0
_task_counter_lock = threading.Lock()


def _parse_pdf_with_cleanup(pdf_bytes: bytes, filename: str) -> tuple[Optional[dict], float]:
    """Wrapper around parse_pdf that triggers periodic GC and CUDA cache cleanup.

    Returns the parse result and its approximate JSON size in MiB. The size is measured
    here, on the worker thread, so serializing a large result never blocks the event loop.

    Every WORKER_MAX_TASKS requests, runs gc.collect() and torch.cuda.empty_cache()
    to release accumulated memory from PIL images, pdfium handles, and CUDA caches.
    Follows Meta TorchServe's worker recycling pattern adapted for ThreadPoolExecutor.
    """
    global _task_counter
    try:
        result: Optional[dict] = parse_pdf(pdf_bytes, filename)
        size_mb = len(json.dumps(result)) / (1024 * 1024) if result else 0.0
        return result, size_mb
    finally:
        with _task_counter_lock:
            _task_counter += 1
            if _task_counter >= MAX_TASKS_BEFORE_CLEANUP:
                tasks_since_cleanup = _task_counter
                _task_counter = 0
                logger.info(
                    "Periodic memory cleanup after %d tasks",
                    tasks_since_cleanup,
                    extra={
                        "service_name": "accurate-parser",
                        "component": "memory",
                        "tasks_since_cleanup": tasks_since_cleanup,
                    },
                )
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    # CUDA cleanup is best effort and must not discard a parsed result.
                    pass


# Device selection includes native Apple MPS while keeping the existing health
# field name and response shape stable.
DEVICE_MODE, GPU_NAME = detect_device()
GPU_AVAILABLE = DEVICE_MODE in {"cuda", "mps"}
GPU_MEMORY = 0
if DEVICE_MODE == "cuda":
    try:
        import torch

        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception as exc:
        logger.info(
            "CUDA memory detection unavailable",
            extra={"service_name": "accurate-parser", "error_type": type(exc).__name__},
        )


@app.on_event("startup")
async def startup_event() -> None:
    """Set MinerU's supported device override without rewriting user config."""
    device_mode = DEVICE_MODE
    if os.getenv("MINERU_DEVICE_MODE", "auto").lower() in {"", "auto"}:
        os.environ["MINERU_DEVICE_MODE"] = device_mode

    if device_mode in {"cuda", "mps"}:
        import torch

        available = (
            torch.cuda.is_available()
            if device_mode == "cuda"
            else torch.backends.mps.is_available()
        )
        if not available:
            raise RuntimeError(
                f"MINERU_DEVICE_MODE={device_mode} was requested, but that device is unavailable"
            )

    logger.info(
        "MinerU startup configuration",
        extra={
            "service_name": "accurate-parser",
            "component": "startup",
            "gpu_available": GPU_AVAILABLE,
            "gpu_name": GPU_NAME,
            "gpu_memory_gb": GPU_MEMORY,
            "device_mode": device_mode,
            "workers": WORKERS,
        },
    )

    if DEVICE_MODE == "cuda":
        try:
            import torch

            logger.info(
                "CUDA details",
                extra={
                    "service_name": "accurate-parser",
                    "component": "startup",
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "device_name": torch.cuda.get_device_name(0),
                },
            )
        except Exception as e:
            logger.warning(
                "Could not get detailed GPU info",
                extra={
                    "service_name": "accurate-parser",
                    "component": "startup",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

    logger.info(
        "MinerU device configured",
        extra={
            "service_name": "accurate-parser",
            "component": "startup",
            "device_mode": os.environ["MINERU_DEVICE_MODE"],
        },
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up thread pool on shutdown."""
    logger.info(
        "Shutting down, cleaning up thread pool",
        extra={"service_name": "accurate-parser", "component": "shutdown"},
    )
    executor.shutdown(wait=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with GPU status."""
    with tracer.start_as_current_span("health", attributes={"service_name": "accurate-parser"}):
        return HealthResponse(status="healthy", workers=WORKERS, gpu_available=GPU_AVAILABLE)


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Security: User-supplied filenames can contain path traversal sequences
    like '../' or absolute paths. This function extracts only the basename
    and removes any potentially dangerous characters.
    """
    if not filename:
        return "document.pdf"
    # Extract basename to prevent path traversal
    safe_name = os.path.basename(filename)
    # Remove any remaining path separators (extra safety)
    safe_name = safe_name.replace("/", "_").replace("\\", "_")
    # Ensure it still ends with .pdf
    if not safe_name.lower().endswith(".pdf"):
        safe_name = "document.pdf"
    return safe_name


@app.post("/parse", response_model=ParseResponse)
async def parse(request: Request, file: UploadFile = File(...)) -> ParseResponse:
    """Parse PDF file to markdown with images, tables, and formulas.

    Uses MinerU for multimodal extraction:
    - CUDA mode: high effort with the selected VLM engine
    - CPU mode: medium effort with small models

    Args:
        request: The request, used to hand its admission slot to the worker.
        file: PDF file upload

    Returns:
        ParseResponse with markdown content, images, tables, formulas, and metadata
    """
    request_start = time.perf_counter()
    with tracer.start_as_current_span(
        "parse", attributes={"service_name": "accurate-parser", "endpoint": "/parse"}
    ):

        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            logger.warning(
                "Rejected non-PDF file",
                extra={
                    "service_name": "accurate-parser",
                    "document_name": file.filename[:50] if file.filename else "unknown",
                },
            )
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # SECURITY: Sanitize filename to prevent path traversal attacks
        safe_filename = _sanitize_filename(file.filename)
        logger.info(
            "Parse request received",
            extra={
                "service_name": "accurate-parser",
                "document_name": safe_filename,
                "gpu_available": GPU_AVAILABLE,
            },
        )

        # Read file content
        try:
            pdf_bytes = await file.read()
            file_size = len(pdf_bytes)
            logger.debug(
                "File read complete",
                extra={
                    "service_name": "accurate-parser",
                    "document_name": safe_filename,
                    "size_bytes": file_size,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to read uploaded file",
                extra={
                    "service_name": "accurate-parser",
                    "document_name": safe_filename,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise HTTPException(status_code=400, detail="Failed to read file")

        # Validate file size (500MB limit for accurate parser)
        if file_size > 500 * 1024 * 1024:
            logger.warning(
                "File too large",
                extra={
                    "service_name": "accurate-parser",
                    "document_name": safe_filename,
                    "size_mb": file_size / (1024 * 1024),
                },
            )
            raise HTTPException(status_code=413, detail="File too large (max 500MB)")

        # Parse PDF in thread pool
        try:
            loop = asyncio.get_running_loop()
            logger.info(
                "Submitting to thread pool",
                extra={
                    "service_name": "accurate-parser",
                    "document_name": safe_filename,
                    "workers": WORKERS,
                    "parser_mode": "vlm" if DEVICE_MODE == "cuda" else "pipeline",
                },
            )

            # The admission slot is held until the worker finishes, even if this request
            # is cancelled by a client disconnect: the thread keeps running regardless.
            worker = executor.submit(
                _parse_pdf_with_cleanup,
                pdf_bytes,
                safe_filename,  # SECURITY: Use sanitized filename
            )
            _handoff(request)
            worker.add_done_callback(lambda _: loop.call_soon_threadsafe(_release_slot))
            parsed = await asyncio.wrap_future(worker)
            result, response_size_mb = parsed

            # Check if result is None or contains an error
            if result is None:
                raise _internal_error(
                    "Parsing returned no result", safe_filename, "No result returned", None
                )

            if "error" in result:
                raise _internal_error(
                    "Parsing failed",
                    safe_filename,
                    str(result.get("error", "Unknown error")),
                    None,
                )

            total_time_ms = int((time.perf_counter() - request_start) * 1000)

            # Structured log with all relevant metrics
            logger.info(
                "Parse complete",
                extra={
                    "service_name": "accurate-parser",
                    "document_name": safe_filename,
                    "page_count": result["metadata"]["pages"],
                    "parse_time_ms": result["metadata"]["processing_time_ms"],
                    "total_time_ms": total_time_ms,
                    "size_bytes": file_size,
                    "response_size_mb": round(response_size_mb, 2),
                    "images_count": len(result["images"]),
                    "tables_count": len(result["tables"]),
                    "formulas_count": len(result["formulas"]),
                    "parser_mode": result["metadata"].get("backend", "unknown"),
                    "gpu_used": result["metadata"].get("gpu_used", False),
                    "accuracy_tier": result["metadata"].get("accuracy_tier", "unknown"),
                },
            )

            return ParseResponse(**result)

        except HTTPException:
            raise
        except Exception as e:
            raise _internal_error("Parsing failed", safe_filename, str(e), e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
