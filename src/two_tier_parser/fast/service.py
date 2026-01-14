"""
PyMuPDF4LLM wrapper for fast PDF parsing.
"""
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pymupdf4llm


def parse_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Parse PDF using PyMuPDF4LLM with page-by-page error recovery.

    If individual pages cause errors (e.g., table detection bug), skip those pages
    and continue parsing the rest of the document.

    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Dictionary with markdown content and metadata.
        Metadata includes 'skipped_pages' list if any pages failed to parse.
    """
    start_time = time.time()

    # Create temporary file for PDF processing
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(pdf_bytes)

    try:
        import pymupdf

        # Get page count first
        with pymupdf.open(tmp_path) as doc:
            total_pages = len(doc)

        # Try parsing all pages at once first (fastest path)
        try:
            markdown_text = pymupdf4llm.to_markdown(str(tmp_path))
            skipped_pages = []

        except (AttributeError, ValueError) as e:
            # Table detection bugs - parse page by page
            # - AttributeError: "'NoneType' object has no attribute 'tables'"
            # - ValueError: "not a textpage of this page" (threading issue)
            error_msg = str(e)
            if "'NoneType' object has no attribute 'tables'" in error_msg or \
               "not a textpage" in error_msg:
                markdown_parts = []
                skipped_pages = []

                for page_num in range(total_pages):
                    try:
                        # Parse single page
                        page_md = pymupdf4llm.to_markdown(
                            str(tmp_path),
                            pages=[page_num]  # Single page
                        )
                        markdown_parts.append(page_md)

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
                        else:
                            raise  # Different error, re-raise

                markdown_text = "\n\n".join(markdown_parts)
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

        return result

    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)


def parse_pdf_page_range(
    pdf_bytes: bytes,
    filename: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse specific page range of PDF using PyMuPDF4LLM.

    Uses Python slice semantics: start_page is inclusive, end_page is exclusive.
    Example: start_page=0, end_page=2 parses pages 0 and 1 (first 2 pages).

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
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(pdf_bytes)

    try:
        import pymupdf

        # Get total page count
        with pymupdf.open(tmp_path) as doc:
            total_pages = len(doc)

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
        doc = pymupdf.open(tmp_path)
        try:
            markdown_text = pymupdf4llm.to_markdown(
                doc,
                pages=pages_to_parse,
                write_images=False,  # Skip images for partial parse (faster)
            )
        finally:
            doc.close()

        processing_time_ms = int((time.time() - start_time) * 1000)

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
        import gc
        gc.collect()  # Force garbage collection to release file handles
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # File may still be locked on Windows, will be cleaned up later
