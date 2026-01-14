"""
Fast Parser Service - FastAPI application with PyMuPDF4LLM.
"""
import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .models import ParseResponse, HealthResponse, PageRangeParseResponse
from .service import parse_pdf, parse_pdf_page_range

# Configure structured logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fast Parser Service",
    description="Ultra-fast PDF parsing using PyMuPDF4LLM",
    version="1.0.0"
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
    executor.shutdown(wait=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        workers=WORKERS,
        no_gil=NO_GIL
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


@app.post("/parse", response_model=ParseResponse)
async def parse(file: UploadFile = File(...)) -> ParseResponse:
    """
    Parse PDF file to markdown.

    Args:
        file: PDF file upload

    Returns:
        ParseResponse with markdown content and metadata
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # SECURITY: Sanitize filename for logging and metadata
    safe_filename = _sanitize_filename(file.filename)

    # Read file content
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")

    # Validate file size (100MB limit)
    if len(pdf_bytes) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    # Parse PDF in thread pool
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            parse_pdf,
            pdf_bytes,
            safe_filename  # SECURITY: Use sanitized filename
        )

        # Check if result is None (shouldn't happen with fast parser, but safety check)
        if result is None:
            logger.error(f"Parsing returned None for {safe_filename}")
            raise HTTPException(status_code=500, detail="Parsing failed: No result returned")

        # Check if result contains an error
        if "error" in result:
            logger.error(f"Parsing failed for {safe_filename}: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Parsing failed: {result.get('error', 'Unknown error')}"
            )

        # Now safe to access result attributes
        logger.info(
            f'{{"filename": "{safe_filename}", "pages": {result["metadata"]["pages"]}, '
            f'"processing_time_ms": {result["metadata"]["processing_time_ms"]}}}'
        )

        return ParseResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parsing failed for {safe_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse-pages", response_model=PageRangeParseResponse)
async def parse_pages(
    file: UploadFile = File(...),
    start_page: int = Query(0, ge=0, description="Start page (0-indexed, inclusive)"),
    end_page: Optional[int] = Query(
        None, ge=1, description="End page (exclusive). None = all remaining pages"
    ),
) -> PageRangeParseResponse:
    """
    Parse specific page range of a PDF document.

    Uses Python slice semantics for page ranges:
    - start_page: 0-indexed, inclusive
    - end_page: exclusive (like Python slicing). None means parse to end.

    Example: start_page=0, end_page=2 parses pages 0 and 1 (first 2 pages)

    Args:
        file: PDF file upload
        start_page: Start page (0-indexed, inclusive). Default: 0
        end_page: End page (exclusive). None = parse all remaining pages.

    Returns:
        PageRangeParseResponse with markdown content and page range metadata
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # SECURITY: Sanitize filename
    safe_filename = _sanitize_filename(file.filename)

    # Read file content
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")

    # Validate file size (100MB limit)
    if len(pdf_bytes) > 100 * 1024 * 1024:
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

        logger.info(
            f'{{"filename": "{safe_filename}", "pages_parsed": {result["metadata"]["pages_parsed"]}, '
            f'"total_pages": {result["metadata"]["total_pages"]}, '
            f'"start_page": {result["metadata"]["start_page"]}, '
            f'"end_page": {result["metadata"]["end_page"]}, '
            f'"processing_time_ms": {result["metadata"]["processing_time_ms"]}}}'
        )

        return PageRangeParseResponse(**result)

    except Exception as e:
        logger.error(f"Page range parsing failed for {safe_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
