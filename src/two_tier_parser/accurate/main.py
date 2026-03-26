"""Accurate Parser Service - FastAPI application with MinerU.

Entry point following industry standard naming convention (main.py, not app.py).
This prevents module/package name collisions with FastAPI's app object.

MinerU provides high-accuracy PDF parsing with GPU acceleration:
- VLM backend (GPU): 95%+ accuracy, ~10-30s/page
- Pipeline backend (CPU): 80-85% accuracy, ~5-15s/page

Observability:
- OpenTelemetry API-only instrumentation (no-op without SDK)
- Structured JSON logging with NullHandler fallback
- MinerU operations wrapped with tracing spans
"""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile
from opentelemetry import trace

from .models import ParseResponse, HealthResponse
from .service import parse_pdf

# Configure structured logging (applications can override)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.accurate")

# Initialize FastAPI app
app = FastAPI(
    title="Accurate Parser Service",
    description="High-quality PDF parsing using MinerU with multimodal extraction",
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
            extra={"service_name": "accurate-parser", "component": "telemetry"}
        )
except ImportError:
    logger.debug(
        "Telemetry core not available, running without tracing",
        extra={"service_name": "accurate-parser", "component": "telemetry"}
    )

# ThreadPoolExecutor for concurrent parsing
WORKERS = int(os.getenv("WORKERS", "2"))
executor = ThreadPoolExecutor(max_workers=WORKERS)

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(
            "GPU detected",
            extra={"service_name": "accurate-parser", "gpu_name": GPU_NAME, "gpu_memory_gb": round(GPU_MEMORY, 1)}
        )
    else:
        GPU_NAME = None
        GPU_MEMORY = 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    GPU_MEMORY = 0


@app.on_event("startup")
async def startup_event():
    """Configure MinerU based on hardware availability.

    Generates magic-pdf.json dynamically to set device mode.
    """
    import json
    from pathlib import Path

    span_ctx = tracer.start_as_current_span("startup", attributes={"service_name": "accurate-parser"})
    span_ctx.__enter__()

    config_path = Path("/root/magic-pdf.json")
    device_mode = "cuda" if GPU_AVAILABLE else "cpu"

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
        }
    )

    if GPU_AVAILABLE:
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
                }
            )
        except Exception as e:
            logger.warning(
                "Could not get detailed GPU info",
                extra={"service_name": "accurate-parser", "component": "startup", "error_type": type(e).__name__, "error_message": str(e)}
            )

    try:
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = {
                "bucket_info": {
                    "bucket-name": "bucket_name",
                    "access-key": "ak",
                    "secret-key": "sk",
                    "endpoint": "http://127.0.0.1:9000"
                },
                "models-dir": "/root/.cache/huggingface/hub",
                "table-config": {
                    "model": "TableMaster",
                    "is_table_recog_enable": True,
                    "max_time": 1200
                }
            }

        config["device-mode"] = device_mode
        config_path.write_text(json.dumps(config, indent=4))
        logger.info(
            "Updated magic-pdf.json",
            extra={"service_name": "accurate-parser", "component": "startup", "device_mode": device_mode}
        )

    except Exception as e:
        logger.error(
            "Failed to update magic-pdf.json",
            extra={"service_name": "accurate-parser", "component": "startup", "error_type": type(e).__name__, "error_message": str(e)}
        )

    span_ctx.__exit__(None, None, None)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up thread pool on shutdown."""
    logger.info(
        "Shutting down, cleaning up thread pool",
        extra={"service_name": "accurate-parser", "component": "shutdown"}
    )
    executor.shutdown(wait=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with GPU status."""
    with tracer.start_as_current_span("health", attributes={"service_name": "accurate-parser"}):
        return HealthResponse(
            status="healthy",
            workers=WORKERS,
            gpu_available=GPU_AVAILABLE
        )


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
    if not safe_name.lower().endswith('.pdf'):
        safe_name = "document.pdf"
    return safe_name


@app.post("/parse", response_model=ParseResponse)
async def parse(file: UploadFile = File(...)) -> ParseResponse:
    """Parse PDF file to markdown with images, tables, and formulas.

    Uses MinerU for high-accuracy extraction:
    - GPU mode: VLM backend, 95%+ accuracy
    - CPU mode: Pipeline backend, 80-85% accuracy

    Args:
        file: PDF file upload

    Returns:
        ParseResponse with markdown content, images, tables, formulas, and metadata
    """
    request_start = time.perf_counter()
    parse_span_ctx = tracer.start_as_current_span("parse", attributes={"service_name": "accurate-parser", "endpoint": "/parse"})
    parse_span_ctx.__enter__()

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(
            "Rejected non-PDF file",
            extra={"service_name": "accurate-parser", "document_name": file.filename[:50] if file.filename else "unknown"}
        )
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # SECURITY: Sanitize filename to prevent path traversal attacks
    safe_filename = _sanitize_filename(file.filename)
    logger.info(
        "Parse request received",
        extra={"service_name": "accurate-parser", "document_name": safe_filename, "gpu_available": GPU_AVAILABLE}
    )

    # Read file content
    try:
        pdf_bytes = await file.read()
        file_size = len(pdf_bytes)
        logger.debug("File read complete", extra={"service_name": "accurate-parser", "document_name": safe_filename, "size_bytes": file_size})
    except Exception as e:
        logger.error(
            "Failed to read uploaded file",
            extra={"service_name": "accurate-parser", "document_name": safe_filename, "error_type": type(e).__name__, "error_message": str(e)}
        )
        raise HTTPException(status_code=400, detail="Failed to read file")

    # Validate file size (500MB limit for accurate parser)
    if file_size > 500 * 1024 * 1024:
        logger.warning("File too large", extra={"service_name": "accurate-parser", "document_name": safe_filename, "size_mb": file_size / (1024*1024)})
        raise HTTPException(status_code=413, detail="File too large (max 500MB)")

    # Parse PDF in thread pool
    try:
        loop = asyncio.get_event_loop()
        logger.info(
            "Submitting to thread pool",
            extra={"service_name": "accurate-parser", "document_name": safe_filename, "workers": WORKERS, "parser_mode": "vlm" if GPU_AVAILABLE else "pipeline"}
        )

        result = await loop.run_in_executor(
            executor,
            parse_pdf,
            pdf_bytes,
            safe_filename  # SECURITY: Use sanitized filename
        )

        # Check if result is None or contains an error
        if result is None:
            logger.error("Parsing returned None", extra={"service_name": "accurate-parser", "document_name": safe_filename})
            raise HTTPException(status_code=500, detail="Parsing failed: No result returned")

        if "error" in result:
            logger.error(
                "Parsing failed",
                extra={"service_name": "accurate-parser", "document_name": safe_filename, "error_message": result.get('error', 'Unknown error')}
            )
            raise HTTPException(
                status_code=500,
                detail=f"Parsing failed: {result.get('error', 'Unknown error')}"
            )

        total_time_ms = int((time.perf_counter() - request_start) * 1000)

        # Calculate approximate response size
        import json
        response_size_mb = len(json.dumps(result)) / (1024 * 1024)

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
            extra={"service_name": "accurate-parser", "document_name": safe_filename, "error_type": type(e).__name__, "error_message": str(e)}
        )
        parse_span_ctx.__exit__(type(e), e, e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
