"""PyMuPDF4LLM wrapper for fast PDF parsing.

Provides ultra-fast markdown extraction (~1s/page) with page-by-page error recovery.

Span hierarchy:
    parse_pdf (root)
        ├── tempfile.create
        ├── pdf.load
        ├── pdf.fast_parse (attempt full doc)
        │   └── pymupdf4llm.to_markdown
        ├── pdf.page_by_page_parse (fallback on error)
        │   └── loop.pages
        │       ├── pdf.page[0].parse
        │       ├── pdf.page[1].parse
        │       └── ... (per page)
        ├── pdf.result_join
        └── tempfile.cleanup
"""
import gc
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pymupdf4llm

logger = logging.getLogger(__name__)

# Optional OpenTelemetry tracing (API-only, no-op without SDK)
try:
    from .core.telemetry import traced_operation, OTEL_AVAILABLE
except ImportError:
    OTEL_AVAILABLE = False

    def traced_operation(name, attributes=None):
        """No-op context manager when telemetry not available."""
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            logger.debug("Starting %s", name, extra=attributes or {})
            try:
                yield None
            except Exception:
                logger.error("%s failed", name, exc_info=True)
                raise
            logger.debug("Completed %s", name)

        return _noop()


def parse_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parse PDF using PyMuPDF4LLM with page-by-page error recovery.

    If individual pages cause errors (e.g., table detection bug), skip those pages
    and continue parsing the rest of the document.

    All operations are wrapped with tracing spans for observability.

    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Dictionary with markdown content and metadata.
        Metadata includes 'skipped_pages' list if any pages failed to parse.
    """
    start_time = time.time()

    # Create temporary file for PDF processing
    with traced_operation("tempfile.create", {"file_name": filename}) as span:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.write(pdf_bytes)
        tmp_file.close()  # Close but don't delete yet

        if span:
            span.set_attribute("temp_path", str(tmp_path))
            span.set_attribute("pdf_size_bytes", len(pdf_bytes))

    try:
        import pymupdf

        # Get page count first
        with traced_operation("pdf.load", {"file_name": filename}) as span:
            with pymupdf.open(tmp_path) as doc:
                total_pages = len(doc)
            logger.debug("PDF loaded: %d pages", total_pages)
            if span:
                span.set_attribute("total_pages", total_pages)

        # Try parsing all pages at once first (fastest path)
        skipped_pages = []
        try:
            with traced_operation("pdf.fast_parse", {
                "file_name": filename,
                "total_pages": total_pages,
                "mode": "full_document",
            }) as span:
                markdown_text = pymupdf4llm.to_markdown(str(tmp_path))
                logger.info("Fast parse succeeded for %s (%d pages)", filename, total_pages)
                if span:
                    span.set_attribute("parse_mode", "full_document")
                    span.set_attribute("markdown_length", len(markdown_text))

        except (AttributeError, ValueError) as e:
            # Table detection bugs - parse page by page
            # - AttributeError: "'NoneType' object has no attribute 'tables'"
            # - ValueError: "not a textpage of this page" (threading issue)
            error_msg = str(e)
            if "'NoneType' object has no attribute 'tables'" in error_msg or \
               "not a textpage" in error_msg:

                logger.warning(
                    "Fast parse failed, falling back to page-by-page: %s",
                    error_msg[:100],
                    extra={"file_name": filename, "error_type": type(e).__name__}
                )

                with traced_operation("pdf.page_by_page_parse", {
                    "file_name": filename,
                    "total_pages": total_pages,
                    "mode": "page_by_page",
                    "fallback_reason": error_msg[:100],
                }) as fallback_span:
                    markdown_parts = []

                    for page_num in range(total_pages):
                        with traced_operation(f"pdf.page[{page_num}].parse", {
                            "page": page_num,
                            "total_pages": total_pages,
                        }) as page_span:
                            try:
                                # Parse single page
                                page_md = pymupdf4llm.to_markdown(
                                    str(tmp_path),
                                    pages=[page_num]  # Single page
                                )
                                markdown_parts.append(page_md)

                                if page_span:
                                    page_span.set_attribute("status", "success")
                                    page_span.set_attribute("markdown_length", len(page_md))

                            except (AttributeError, ValueError) as page_error:
                                # Skip this problematic page
                                page_error_msg = str(page_error)
                                if "'NoneType' object has no attribute 'tables'" in page_error_msg or \
                                   "not a textpage" in page_error_msg:
                                    skipped_pages.append(page_num + 1)  # 1-indexed for user
                                    # Add placeholder for skipped page
                                    markdown_parts.append(
                                        f"\n\n---\n**[Page {page_num + 1} skipped due to parsing error]**\n---\n\n"
                                    )
                                    logger.warning(
                                        "Skipped page %d: %s",
                                        page_num + 1,
                                        page_error_msg[:50],
                                        extra={"file_name": filename}
                                    )

                                    if page_span:
                                        page_span.set_attribute("status", "skipped")
                                        page_span.set_attribute("error", page_error_msg[:100])
                                else:
                                    raise  # Different error, re-raise

                    markdown_text = "\n\n".join(markdown_parts)

                    if fallback_span:
                        fallback_span.set_attribute("pages_parsed", total_pages - len(skipped_pages))
                        fallback_span.set_attribute("pages_skipped", len(skipped_pages))
                        fallback_span.set_attribute("markdown_length", len(markdown_text))
            else:
                raise  # Different AttributeError, re-raise

        processing_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "markdown": markdown_text,
            "metadata": {
                "pages": total_pages,
                "processing_time_ms": processing_time_ms,
                "parser": "pymupdf4llm",
                "version": "0.2.0",
                "filename": filename
            }
        }

        # Add skipped pages info if any
        if skipped_pages:
            result["metadata"]["skipped_pages"] = skipped_pages
            result["metadata"]["warning"] = f"Skipped {len(skipped_pages)} page(s) due to parsing errors"
            logger.warning(
                "Parse completed with skipped pages",
                extra={
                    "file_name": filename,
                    "skipped_pages": skipped_pages,
                    "skipped_count": len(skipped_pages),
                    "total_pages": total_pages,
                }
            )

        logger.info(
            "Parse complete",
            extra={
                "file_name": filename,
                "pages": total_pages,
                "processing_time_ms": processing_time_ms,
                "skipped_count": len(skipped_pages),
                "markdown_length": len(markdown_text),
            }
        )

        return result

    finally:
        # Clean up temporary file
        with traced_operation("tempfile.cleanup", {"file_name": filename}) as span:
            gc.collect()  # Force garbage collection to release file handles
            try:
                tmp_path.unlink(missing_ok=True)
                if span:
                    span.set_attribute("cleanup_status", "success")
            except PermissionError:
                # File may still be locked on Windows
                logger.debug("Temp file cleanup deferred (file locked): %s", tmp_path)
                if span:
                    span.set_attribute("cleanup_status", "deferred")


def parse_pdf_page_range(
    pdf_bytes: bytes,
    filename: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse specific page range of PDF using PyMuPDF4LLM.

    Uses Python slice semantics: start_page is inclusive, end_page is exclusive.
    Example: start_page=0, end_page=2 parses pages 0 and 1 (first 2 pages).

    All operations are wrapped with tracing spans for observability.

    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for metadata
        start_page: 0-indexed start page (inclusive). Clamped to valid range.
        end_page: End page (exclusive). None means parse to end of document.

    Returns:
        Dictionary with markdown content and metadata including:
        - total_pages: Total pages in document
        - pages_parsed: Number of pages actually parsed
        - start_page: Actual start page used (after clamping)
        - end_page: Actual end page used (after clamping)
    """
    start_time = time.time()

    # Create temporary file for PDF processing
    with traced_operation("tempfile.create", {"file_name": filename, "mode": "page_range"}) as span:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.write(pdf_bytes)
        tmp_file.close()

        if span:
            span.set_attribute("temp_path", str(tmp_path))
            span.set_attribute("pdf_size_bytes", len(pdf_bytes))

    try:
        import pymupdf

        # Get total page count
        with traced_operation("pdf.load", {"file_name": filename, "mode": "page_range"}) as span:
            with pymupdf.open(tmp_path) as doc:
                total_pages = len(doc)
            if span:
                span.set_attribute("total_pages", total_pages)

        # Clamp start_page to valid range [0, total_pages]
        start_page = max(0, min(start_page, total_pages))

        # Handle end_page: None means parse to end, otherwise clamp
        if end_page is None:
            end_page = total_pages
        else:
            end_page = max(start_page, min(end_page, total_pages))

        # Build list of pages to parse (0-indexed)
        pages_to_parse = list(range(start_page, end_page))

        # Handle empty range
        if not pages_to_parse:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Empty page range",
                extra={
                    "file_name": filename,
                    "start_page": start_page,
                    "end_page": end_page,
                    "total_pages": total_pages,
                }
            )
            return {
                "markdown": "",
                "metadata": {
                    "filename": filename,
                    "total_pages": total_pages,
                    "pages_parsed": 0,
                    "start_page": start_page,
                    "end_page": end_page,
                    "processing_time_ms": processing_time_ms,
                    "parser": "pymupdf4llm",
                    "version": "0.2.0",
                },
            }

        # Parse specified pages using document object to ensure proper cleanup
        with traced_operation("pdf.page_range_parse", {
            "file_name": filename,
            "start_page": start_page,
            "end_page": end_page,
            "pages_count": len(pages_to_parse),
            "total_pages": total_pages,
        }) as span:
            doc = pymupdf.open(tmp_path)
            try:
                markdown_text = pymupdf4llm.to_markdown(
                    doc,
                    pages=pages_to_parse,
                    write_images=False,  # Skip images for partial parse (faster)
                )
                if span:
                    span.set_attribute("markdown_length", len(markdown_text))
            finally:
                doc.close()

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Page range parse complete",
            extra={
                "file_name": filename,
                "pages_parsed": len(pages_to_parse),
                "total_pages": total_pages,
                "start_page": start_page,
                "end_page": end_page,
                "processing_time_ms": processing_time_ms,
                "markdown_length": len(markdown_text),
            }
        )

        return {
            "markdown": markdown_text,
            "metadata": {
                "filename": filename,
                "total_pages": total_pages,
                "pages_parsed": len(pages_to_parse),
                "start_page": start_page,
                "end_page": end_page,
                "processing_time_ms": processing_time_ms,
                "parser": "pymupdf4llm",
                "version": "0.2.0",
            },
        }

    finally:
        # Clean up temporary file (retry on Windows file locking)
        with traced_operation("tempfile.cleanup", {"file_name": filename}) as span:
            gc.collect()  # Force garbage collection to release file handles
            try:
                tmp_path.unlink(missing_ok=True)
                if span:
                    span.set_attribute("cleanup_status", "success")
            except PermissionError:
                logger.debug("Temp file cleanup deferred (file locked): %s", tmp_path)
                if span:
                    span.set_attribute("cleanup_status", "deferred")
