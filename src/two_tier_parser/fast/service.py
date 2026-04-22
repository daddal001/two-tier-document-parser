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
import base64
import gc
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ADR-0049: activate PyMuPDF-Layout GNN BEFORE importing pymupdf4llm.
# Auto-bundled in pymupdf4llm >= 0.3.0; module is import-order-sensitive.
# FAST_PARSER_DISABLE_LAYOUT=1 skips the import as an emergency escape hatch.
if os.getenv("FAST_PARSER_DISABLE_LAYOUT", "").strip().lower() not in {"1", "true", "yes"}:
    try:
        import pymupdf.layout  # noqa: F401
    except Exception:
        # Missing module (older pymupdf4llm) or platform incompatibility; run
        # stock pymupdf4llm without layout rather than failing the whole parser.
        pass

import pymupdf4llm
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.fast")


_redis_client = None


def _get_preempt_redis():
    """Lazy Redis client for reading parse_cancel:{task_id} flags.

    Runs inside the ProcessPoolExecutor worker, so each worker initializes
    its own connection on first use. Returns None on any failure — cancel
    checks become no-ops, which is the safe default (work runs to completion).
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    url = os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL")
    if not url:
        return None
    try:
        import redis

        _redis_client = redis.from_url(
            url,
            decode_responses=False,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
            health_check_interval=30,
        )
        return _redis_client
    except Exception:
        logger.warning("Preempt Redis init failed; cancel checks disabled", exc_info=True)
        return None


def _is_cancelled(task_id: Optional[str]) -> bool:
    """Non-raising check for parse_cancel:{task_id} flag.

    ADR-0047. Callers check between page boundaries; returning False on error
    means the parse runs to completion (safe default).
    """
    if not task_id:
        return False
    client = _get_preempt_redis()
    if client is None:
        return False
    try:
        return bool(client.exists(f"parse_cancel:{task_id}"))
    except Exception:
        return False

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
            logger.debug(
                "Starting operation",
                extra={"service_name": "fast-parser", "operation": name, **(attributes or {})}
            )
            try:
                yield None
            except Exception:
                logger.error(
                    "Operation failed",
                    exc_info=True,
                    extra={"service_name": "fast-parser", "operation": name}
                )
                raise
            logger.debug(
                "Completed operation",
                extra={"service_name": "fast-parser", "operation": name}
            )

        return _noop()


# ADR-0049: image extraction via direct PyMuPDF API (not pymupdf4llm.write_images).
# Bypasses open upstream bug pymupdf4llm#352 (image_path ignored under layout).
# OWASP ASVS V12 caps enforced: count, bytes, pixels.

# Sentinel for "extraction truncated by cap" — emitted in result metadata so
# Prometheus + caller can alert on runaway-image PDFs.
IMAGE_TRUNCATION_REASONS = {"count", "bytes", "pixels"}


def extract_images(
    doc,
    *,
    page_range: Optional[Iterable[int]] = None,
    max_count: int = 500,
    max_bytes: int = 50 * 1024 * 1024,
    max_pixels: int = 50_000_000,
) -> Dict[str, Any]:
    """Extract images from an open pymupdf.Document via the native xref API.

    Returns a dict: ``{"images": [ImageData-shaped dicts], "truncated": <reason|None>, "total_bytes": int}``.
    Deduplicates by SHA-256 content hash — the same PDF xref referenced across
    multiple pages (e.g. a header logo) uploads once with ``pages=[0, 1, 2, ...]``.

    Caps (OWASP ASVS V12 + issue #1740 pixel-flood):
        max_count: hard cap on distinct images returned
        max_bytes: cap on total decoded image bytes
        max_pixels: cap on any single image's pixel area (width × height)

    When a cap is hit, extraction stops and ``truncated`` is set to the reason.
    Callers MUST surface this to the user (metadata flag + Prometheus counter).
    """
    if page_range is None:
        page_range = range(len(doc))
    pages = list(page_range)

    seen: Dict[str, Dict[str, Any]] = {}
    total_bytes = 0
    truncated: Optional[str] = None

    for page_num in pages:
        if truncated:
            break
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            # Dedup: same xref on multiple pages → one record with pages=[...]
            try:
                info = doc.extract_image(xref)
            except Exception as exc:
                logger.warning(
                    "extract_image failed for xref=%d on page %d: %s",
                    xref, page_num, exc,
                    extra={"service_name": "fast-parser", "xref": xref, "page": page_num},
                )
                continue

            raw: bytes = info.get("image") or b""
            if not raw:
                continue

            # OWASP ASVS V12.2 / #1740 pixel-flood check BEFORE expensive ops
            width = int(info.get("width", 0))
            height = int(info.get("height", 0))
            if width * height > max_pixels:
                truncated = "pixels"
                logger.warning(
                    "image exceeds max_pixels cap",
                    extra={
                        "service_name": "fast-parser",
                        "xref": xref,
                        "width": width,
                        "height": height,
                        "max_pixels": max_pixels,
                    },
                )
                break

            image_id = hashlib.sha256(raw).hexdigest()[:16]

            if image_id in seen:
                # Same image re-appears; record the page but don't re-count bytes.
                seen_pages = seen[image_id].setdefault("pages", [seen[image_id]["page"]])
                if page_num not in seen_pages:
                    seen_pages.append(page_num)
                continue

            if len(seen) >= max_count:
                truncated = "count"
                break
            if total_bytes + len(raw) > max_bytes:
                truncated = "bytes"
                break

            try:
                bbox_rect = page.get_image_bbox(img_info)
                if bbox_rect.is_empty or bbox_rect.is_infinite:
                    bbox = None
                else:
                    bbox = [float(bbox_rect.x0), float(bbox_rect.y0), float(bbox_rect.x1), float(bbox_rect.y1)]
            except Exception:
                bbox = None

            ext = (info.get("ext") or "png").lower()
            seen[image_id] = {
                "image_id": image_id,
                "image_base64": base64.b64encode(raw).decode("ascii"),
                "page": page_num,
                "pages": [page_num],
                "bbox": bbox,
                "mime": f"image/{ext}",
                "width": width,
                "height": height,
            }
            total_bytes += len(raw)

    return {
        "images": list(seen.values()),
        "truncated": truncated,
        "total_bytes": total_bytes,
    }


def parse_pdf(
    pdf_bytes: bytes,
    filename: str,
    task_id: Optional[str] = None,
    start_page: int = 0,
    progress_ceiling: float = 0.8,
    extract_images_flag: bool = True,
    image_max_count: int = 500,
    image_max_bytes: int = 50 * 1024 * 1024,
    image_max_pixels: int = 50_000_000,
) -> Dict[str, Any]:
    """Parse PDF using PyMuPDF4LLM with page-by-page error recovery.

    If individual pages cause errors (e.g., table detection bug), skip those pages
    and continue parsing the rest of the document.

    Preemption (ADR-0047): when ``task_id`` is provided, parsing iterates
    page-by-page and between each page checks the Redis ``parse_cancel:{task_id}``
    flag. If the flag is set and fewer than ``progress_ceiling`` of pages have
    been parsed, the function returns early with ``preempted=True`` and the
    partial markdown. The caller (main.py) translates this into an HTTP 409
    Preempted response. If ``task_id`` is None the fast full-document path is
    used (unchanged behavior).

    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for logging
        task_id: Optional Celery task ID for cooperative preemption.
        start_page: 0-indexed resume cursor (inclusive). Used when resuming a
            previously preempted parse from ADR-0046's ``parser:deferred``.
        progress_ceiling: Fraction of total pages above which preemption is
            ignored (near-finished work is not discarded). Default 0.8.

    Returns:
        Dictionary with markdown content and metadata.
        Metadata includes ``skipped_pages`` list if any pages failed to parse,
        plus ``preempted`` and ``last_page_parsed`` when preempted.
    """
    start_time = time.time()
    parse_span_ctx = tracer.start_as_current_span("parse_pdf", attributes={"service_name": "fast-parser", "document_name": filename})
    parse_span_ctx.__enter__()

    import pymupdf

    # ADR-0049: BytesIO input — skip tempfile round-trip. pymupdf accepts
    # stream=bytes since 1.14.13. Single open (previously this function opened
    # the file 3 times: page count, full parse, fallback). Reuse one handle
    # for the entire request to save ~2× open overhead on small docs.
    with traced_operation("pdf.load", {"file_name": filename, "pdf_size_bytes": len(pdf_bytes)}) as span:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        if span:
            span.set_attribute("total_pages", total_pages)
            span.set_attribute("pdf_size_bytes", len(pdf_bytes))
        logger.debug(
            "PDF loaded",
            extra={"service_name": "fast-parser", "page_count": total_pages, "document_name": filename}
        )

    try:
        # Preemptable path (ADR-0047): task_id present means Stage 2 resume/cancel
        # semantics. We iterate page-by-page and check parse_cancel:{task_id}
        # between pages so long parses can be preempted by Stage 1 partial parses.
        skipped_pages = []
        preempted = False
        last_page_parsed = -1

        if task_id is not None:
            with traced_operation("pdf.preemptable_parse", {
                "file_name": filename,
                "total_pages": total_pages,
                "mode": "page_by_page_preemptable",
                "task_id": task_id,
                "start_page": start_page,
            }) as preempt_span:
                markdown_parts = []
                effective_start = max(0, min(start_page, total_pages))
                for page_num in range(effective_start, total_pages):
                    progress = (page_num - effective_start) / max(total_pages - effective_start, 1)
                    if progress < progress_ceiling and _is_cancelled(task_id):
                        preempted = True
                        logger.info(
                            "Parse preempted at page boundary",
                            extra={
                                "service_name": "fast-parser",
                                "document_name": filename,
                                "task_id": task_id,
                                "last_page_parsed": last_page_parsed,
                                "total_pages": total_pages,
                            },
                        )
                        break
                    try:
                        page_md = pymupdf4llm.to_markdown(doc, pages=[page_num])
                        markdown_parts.append(page_md)
                        last_page_parsed = page_num
                    except (AttributeError, ValueError) as page_error:
                        page_error_msg = str(page_error)
                        if "'NoneType' object has no attribute 'tables'" in page_error_msg or \
                           "not a textpage" in page_error_msg:
                            skipped_pages.append(page_num + 1)
                            markdown_parts.append(
                                f"\n\n---\n**[Page {page_num + 1} skipped due to parsing error]**\n---\n\n"
                            )
                            last_page_parsed = page_num
                        else:
                            raise

                markdown_text = "\n\n".join(markdown_parts)
                if preempt_span:
                    preempt_span.set_attribute("preempted", preempted)
                    preempt_span.set_attribute("last_page_parsed", last_page_parsed)
                    preempt_span.set_attribute("pages_parsed", last_page_parsed - effective_start + 1 if last_page_parsed >= 0 else 0)
                    preempt_span.set_attribute("markdown_length", len(markdown_text))
        else:
            # Full-document fast path. Layout GNN (if enabled) kicks in here.
            try:
                with traced_operation("pdf.fast_parse", {
                    "file_name": filename,
                    "total_pages": total_pages,
                    "mode": "full_document",
                }) as span:
                    markdown_text = pymupdf4llm.to_markdown(doc)
                    logger.info(
                        "Fast parse succeeded",
                        extra={"service_name": "fast-parser", "document_name": filename, "page_count": total_pages}
                    )
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
                        "Fast parse failed, falling back to page-by-page",
                        extra={"service_name": "fast-parser", "document_name": filename, "error_type": type(e).__name__, "error_message": error_msg[:100]}
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
                                    # Parse single page — reuse single doc handle.
                                    page_md = pymupdf4llm.to_markdown(
                                        doc,
                                        pages=[page_num]
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
                                            "Skipped page due to parsing error",
                                            extra={
                                                "service_name": "fast-parser",
                                                "document_name": filename,
                                                "page": page_num + 1,
                                                "error_message": page_error_msg[:50],
                                            }
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

        # ADR-0047 preemption: report partial-progress state so main.py can
        # translate into HTTP 409. last_page_parsed is -1 when nothing parsed.
        if preempted or task_id is not None:
            result["metadata"]["preempted"] = preempted
            result["metadata"]["last_page_parsed"] = last_page_parsed
            result["metadata"]["start_page"] = start_page

        # Add skipped pages info if any
        if skipped_pages:
            result["metadata"]["skipped_pages"] = skipped_pages
            result["metadata"]["warning"] = f"Skipped {len(skipped_pages)} page(s) due to parsing errors"
            logger.warning(
                "Parse completed with skipped pages",
                extra={
                    "service_name": "fast-parser",
                    "document_name": filename,
                    "skipped_pages": skipped_pages,
                    "skipped_count": len(skipped_pages),
                    "page_count": total_pages,
                }
            )

        # ADR-0049: extract images from the same open doc handle before close.
        images: List[Dict[str, Any]] = []
        images_truncated: Optional[str] = None
        if extract_images_flag:
            with traced_operation("pdf.extract_images", {
                "file_name": filename,
                "max_count": image_max_count,
                "max_bytes": image_max_bytes,
                "max_pixels": image_max_pixels,
            }) as img_span:
                # Only extract from pages that were actually parsed; respects
                # ADR-0047 preempt (we don't want to pay for pages we bailed on).
                if task_id is not None and last_page_parsed >= 0:
                    image_page_range = range(effective_start, last_page_parsed + 1)
                elif task_id is not None:
                    image_page_range = range(effective_start, effective_start)  # empty
                else:
                    image_page_range = range(total_pages)
                extract_result = extract_images(
                    doc,
                    page_range=image_page_range,
                    max_count=image_max_count,
                    max_bytes=image_max_bytes,
                    max_pixels=image_max_pixels,
                )
                images = extract_result["images"]
                images_truncated = extract_result["truncated"]
                if img_span:
                    img_span.set_attribute("image_count", len(images))
                    img_span.set_attribute("image_total_bytes", extract_result["total_bytes"])
                    if images_truncated:
                        img_span.set_attribute("truncated_reason", images_truncated)

        result["images"] = images
        if images_truncated:
            result["metadata"]["images_truncated"] = images_truncated

        logger.info(
            "Parse complete",
            extra={
                "service_name": "fast-parser",
                "document_name": filename,
                "page_count": total_pages,
                "duration_ms": processing_time_ms,
                "skipped_count": len(skipped_pages),
                "markdown_length": len(markdown_text),
                "image_count": len(images),
                "images_truncated": images_truncated,
            }
        )

        return result

    finally:
        # ADR-0049: no tempfile; single doc handle closes here. gc.collect()
        # still useful in long-lived ProcessPool worker to release pymupdf
        # internal caches — kept for parity with max_tasks_per_child pattern.
        try:
            doc.close()
        except Exception:
            pass
        gc.collect()
        parse_span_ctx.__exit__(None, None, None)


def parse_pdf_page_range(
    pdf_bytes: bytes,
    filename: str,
    start_page: int = 0,
    end_page: Optional[int] = None,
    extract_images_flag: bool = True,
    image_max_count: int = 50,
    image_max_bytes: int = 10 * 1024 * 1024,
    image_max_pixels: int = 50_000_000,
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
    range_span_ctx = tracer.start_as_current_span("parse_pdf_page_range", attributes={"service_name": "fast-parser", "document_name": filename})
    range_span_ctx.__enter__()

    import pymupdf

    # ADR-0049: BytesIO input, single doc handle (was: tempfile + 2 opens).
    with traced_operation("pdf.load", {"file_name": filename, "mode": "page_range", "pdf_size_bytes": len(pdf_bytes)}) as span:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        if span:
            span.set_attribute("total_pages", total_pages)
            span.set_attribute("pdf_size_bytes", len(pdf_bytes))

    try:
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
                    "service_name": "fast-parser",
                    "document_name": filename,
                    "start_page": start_page,
                    "end_page": end_page,
                    "page_count": total_pages,
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

        with traced_operation("pdf.page_range_parse", {
            "file_name": filename,
            "start_page": start_page,
            "end_page": end_page,
            "pages_count": len(pages_to_parse),
            "total_pages": total_pages,
        }) as span:
            markdown_text = pymupdf4llm.to_markdown(
                doc,
                pages=pages_to_parse,
                write_images=False,  # pymupdf4llm image path has layout-bug #352; we extract via direct API in extract_images() instead.
            )
            if span:
                span.set_attribute("markdown_length", len(markdown_text))

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Page range parse complete",
            extra={
                "service_name": "fast-parser",
                "document_name": filename,
                "pages_parsed": len(pages_to_parse),
                "page_count": total_pages,
                "start_page": start_page,
                "end_page": end_page,
                "duration_ms": processing_time_ms,
                "markdown_length": len(markdown_text),
            }
        )

        # ADR-0049: image extraction for Stage 1 partial parses — tighter caps
        # than Stage 2 to protect the 1.5s soft SLO under pathological PDFs.
        images: List[Dict[str, Any]] = []
        images_truncated: Optional[str] = None
        if extract_images_flag:
            with traced_operation("pdf.extract_images", {
                "file_name": filename,
                "mode": "page_range",
                "max_count": image_max_count,
            }) as img_span:
                extract_result = extract_images(
                    doc,
                    page_range=pages_to_parse,
                    max_count=image_max_count,
                    max_bytes=image_max_bytes,
                    max_pixels=image_max_pixels,
                )
                images = extract_result["images"]
                images_truncated = extract_result["truncated"]
                if img_span:
                    img_span.set_attribute("image_count", len(images))
                    img_span.set_attribute("image_total_bytes", extract_result["total_bytes"])
                    if images_truncated:
                        img_span.set_attribute("truncated_reason", images_truncated)

        result_metadata = {
            "filename": filename,
            "total_pages": total_pages,
            "pages_parsed": len(pages_to_parse),
            "start_page": start_page,
            "end_page": end_page,
            "processing_time_ms": processing_time_ms,
            "parser": "pymupdf4llm",
            "version": "0.2.0",
        }
        if images_truncated:
            result_metadata["images_truncated"] = images_truncated

        return {
            "markdown": markdown_text,
            "metadata": result_metadata,
            "images": images,
        }

    finally:
        try:
            doc.close()
        except Exception:
            pass
        gc.collect()
        range_span_ctx.__exit__(None, None, None)
