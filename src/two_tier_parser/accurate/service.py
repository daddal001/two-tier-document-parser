"""MinerU wrapper for accurate PDF parsing with multimodal extraction.

Automatically falls back to pipeline backend (CPU) if no GPU is detected.

This module wraps MinerU (external library) with tracing spans to provide
visibility into GPU/CPU operations without modifying MinerU itself.

Span hierarchy:
    parse_pdf (root)
        ├── gpu.detection
        ├── backend.select
        ├── mineru.load_images
        ├── [VLM PATH]
        │   └── mineru.vlm_analyze
        ├── [PIPELINE PATH]
        │   ├── mineru.tempdir_create
        │   ├── mineru.do_parse
        │   └── mineru.middle_json_load
        ├── mineru.union_make
        └── multimodal.extraction
            ├── page[N].extract
            │   ├── image.extract[N_M]
            │   ├── table.extract[N_M]
            │   └── formula.extract[N_M]
"""
import base64
import io
import json
import logging
import os
import time
import traceback
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Optional OpenTelemetry tracing (API-only, no-op without SDK)
try:
    from .core.telemetry import traced_operation, get_tracer, OTEL_AVAILABLE
except ImportError:
    OTEL_AVAILABLE = False

    def traced_operation(name, attributes=None):
        """No-op context manager when telemetry not available."""
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            logger.info("Starting %s", name, extra=attributes or {})
            try:
                yield None
            except Exception:
                logger.error("%s failed", name, exc_info=True)
                raise
            logger.info("Completed %s", name)

        return _noop()

    def get_tracer():
        return None


def parse_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parse PDF using MinerU with automatic GPU fallback.

    - GPU available: VLM backend (transformers) - 95%+ accuracy
    - No GPU: Pipeline backend (CPU-only) - 80-85% accuracy

    All MinerU operations are wrapped with tracing spans for observability.

    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for logging

    Returns:
        Dictionary with markdown, images, tables, formulas, and metadata
    """
    start_time = time.time()

    # Get tracer for manual span creation
    tracer = get_tracer()

    try:
        # GPU Detection
        with traced_operation("gpu.detection", {"file_name": filename}) as span:
            gpu_available = False
            gpu_name = None
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info("GPU detected: %s", gpu_name)
                    if span:
                        span.set_attribute("gpu.name", gpu_name)
                        span.set_attribute("gpu.available", True)
            except (ImportError, Exception) as e:
                logger.info("GPU check failed: %s", e)
                if span:
                    span.set_attribute("gpu.available", False)
                    span.set_attribute("gpu.check_error", str(e))

        # Backend Selection
        with traced_operation("backend.select", {"gpu_available": gpu_available}) as span:
            if gpu_available:
                backend = 'transformers'  # VLM backend (95%+ accuracy, requires GPU)
                use_vlm = True
                logger.info("Using VLM backend for highest accuracy (GPU-accelerated)")
                logger.info("Expected processing time: ~10-11 mins for 10 pages on T4")
            else:
                backend = 'pipeline'  # Pipeline backend (80-85% accuracy, CPU-only)
                use_vlm = False
                logger.info("No GPU detected - using pipeline backend (CPU-only, 80-85% accuracy)")
                logger.info("Expected processing time: ~2-3 mins for 10 pages on CPU")

            if span:
                span.set_attribute("backend.name", backend)
                span.set_attribute("backend.use_vlm", use_vlm)

        # Import MinerU components
        from mineru.utils.enum_class import MakeMode, ContentType, ImageType
        from mineru.utils.pdf_image_tools import load_images_from_pdf, get_crop_img
        from mineru.version import __version__ as mineru_version

        # Load images into memory for manual cropping
        with traced_operation("mineru.load_images", {"file_name": filename, "image_type": "PIL"}) as span:
            images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
            page_count_from_images = len(images_list)
            logger.info("Loaded %d page images for %s", page_count_from_images, filename)
            if span:
                span.set_attribute("pages.loaded", page_count_from_images)

        logger.info("Starting MinerU processing for %s (backend: %s)", filename, backend)

        # Process based on backend type
        if use_vlm:
            # VLM Backend (GPU-accelerated)
            with traced_operation("mineru.vlm_analyze", {
                "backend": backend,
                "gpu_used": True,
                "file_name": filename,
            }) as span:
                from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
                from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

                # Call vlm_doc_analyze directly with image_writer=None to skip file writes
                middle_json, infer_result = vlm_doc_analyze(
                    pdf_bytes=pdf_bytes,
                    image_writer=None,  # Critical: None means no file writes!
                    backend=backend,
                    server_url=None
                )
                logger.info("VLM processing completed. Extracting results from middle_json...")

                if span:
                    pdf_info = middle_json.get("pdf_info", [])
                    span.set_attribute("pages.processed", len(pdf_info))
        else:
            # Pipeline Backend (CPU-only)
            from mineru.cli.common import do_parse
            from mineru.backend.pipeline.middle_json_mkcontent import union_make as pipeline_union_make
            import tempfile

            # Pipeline backend requires file-based processing
            with traced_operation("mineru.tempdir_create", {"file_name": filename}) as span:
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir = temp_dir_obj.name
                pdf_path = os.path.join(temp_dir, filename)
                output_dir = os.path.join(temp_dir, "output")

                # Write PDF to temp file
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_bytes)

                if span:
                    span.set_attribute("temp_dir", temp_dir)
                    span.set_attribute("pdf_size_bytes", len(pdf_bytes))

            try:
                with traced_operation("mineru.do_parse", {
                    "backend": backend,
                    "gpu_used": False,
                    "file_name": filename,
                }) as span:
                    logger.info("Running pipeline backend (layout detection + OCR)...")

                    # Run pipeline processing
                    do_parse(
                        pdf_path=pdf_path,
                        output_dir=output_dir,
                        output_image_dir=output_dir,
                        backend=backend,
                        model_json_path=None,
                        start_page_id=0,
                        end_page_id=None,
                        image_writer="disk"
                    )

                    if span:
                        span.set_attribute("output_dir", output_dir)

                with traced_operation("mineru.middle_json_load", {"file_name": filename}) as span:
                    # Load middle.json from output
                    middle_json_path = os.path.join(output_dir, filename.replace('.pdf', ''), "middle.json")
                    with open(middle_json_path, 'r', encoding='utf-8') as f:
                        middle_json = json.load(f)

                    logger.info("Pipeline processing completed. Extracting results from middle_json...")

                    if span:
                        pdf_info = middle_json.get("pdf_info", [])
                        span.set_attribute("pages.processed", len(pdf_info))

                # Use pipeline's union_make instead of VLM's
                vlm_union_make = pipeline_union_make

            finally:
                # Cleanup temp directory
                try:
                    temp_dir_obj.cleanup()
                except Exception as e:
                    logger.warning("Failed to cleanup temp dir: %s", e)

        # Extract pdf_info from middle_json (results are in memory)
        pdf_info = middle_json.get("pdf_info", [])
        page_count = len(pdf_info)

        logger.info("Found %d pages in results", page_count)

        # Generate markdown from middle_json
        with traced_operation("mineru.union_make", {
            "page_count": page_count,
            "mode": "MM_MD",
        }) as span:
            # Note: vlm_union_make takes positional arguments: pdf_info, make_mode, image_dir
            markdown_text = vlm_union_make(
                pdf_info,
                MakeMode.MM_MD,
                ""  # No image directory needed (images are base64 in memory)
            )
            logger.info("Generated markdown (%d characters)", len(markdown_text))

            if span:
                span.set_attribute("markdown.length", len(markdown_text))

        # Helper to convert PIL image to base64
        def pil_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Helper for recursive block traversal
        def _traverse_blocks(blocks):
            """Recursively yield all blocks and spans from nested structure"""
            if not blocks:
                return
            for block in blocks:
                yield block
                if "blocks" in block:
                    yield from _traverse_blocks(block["blocks"])
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            yield from _traverse_blocks(line["spans"])

        # Extract images, tables, and formulas from pdf_info spans
        images = []
        tables = []
        formulas = []

        with traced_operation("multimodal.extraction", {
            "page_count": page_count,
            "file_name": filename,
        }) as extraction_span:
            for page_idx, page_info in enumerate(pdf_info):
                with traced_operation(f"page[{page_idx}].extract", {"page": page_idx}) as page_span:
                    # Get page image for cropping
                    image_dict = images_list[page_idx]
                    page_pil_img = image_dict["img_pil"]
                    scale = image_dict["scale"]

                    # Check para_blocks first (VLM), then preproc_blocks (Pipeline)
                    spans = page_info.get("para_blocks", []) or page_info.get("preproc_blocks", [])

                    page_images = 0
                    page_tables = 0
                    page_formulas = 0

                    for span_data in _traverse_blocks(spans):
                        span_type = span_data.get("type", "")

                        # Extract images
                        if span_type == ContentType.IMAGE or span_type == "image":
                            bbox = span_data.get("bbox", [])
                            if bbox:
                                try:
                                    crop = get_crop_img(bbox, page_pil_img, scale)
                                    img_b64 = pil_to_base64(crop)
                                    images.append({
                                        "image_id": f"page_{page_idx}_img_{len(images)}",
                                        "image_base64": img_b64,
                                        "page": page_idx,
                                        "bbox": bbox
                                    })
                                    page_images += 1
                                except Exception as e:
                                    logger.warning("Failed to crop image on page %d: %s", page_idx, e)

                        # Extract tables
                        elif span_type == ContentType.TABLE or span_type == "table":
                            table_content = span_data.get("content", "")
                            html_content = span_data.get("html", "")
                            bbox = span_data.get("bbox", [])

                            # Use content as markdown (it's usually markdown or plain text)
                            # If no content, use HTML as fallback
                            markdown_content = table_content if table_content else html_content

                            if markdown_content:
                                tables.append({
                                    "table_id": f"page_{page_idx}_table_{len(tables)}",
                                    "markdown": markdown_content,
                                    "page": page_idx,
                                    "bbox": bbox
                                })
                                page_tables += 1

                        # Extract formulas
                        elif span_type == ContentType.INTERLINE_EQUATION or span_type == "interline_equation":
                            formula_content = span_data.get("content", "")
                            bbox = span_data.get("bbox", [])
                            if formula_content:
                                formulas.append({
                                    "formula_id": f"page_{page_idx}_formula_{len(formulas)}",
                                    "latex": formula_content,
                                    "page": page_idx,
                                    "bbox": bbox
                                })
                                page_formulas += 1

                    if page_span:
                        page_span.set_attribute("images.extracted", page_images)
                        page_span.set_attribute("tables.extracted", page_tables)
                        page_span.set_attribute("formulas.extracted", page_formulas)

            if extraction_span:
                extraction_span.set_attribute("total.images", len(images))
                extraction_span.set_attribute("total.tables", len(tables))
                extraction_span.set_attribute("total.formulas", len(formulas))

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Extraction complete",
            extra={
                "file_name": filename,
                "pages": page_count,
                "images": len(images),
                "tables": len(tables),
                "formulas": len(formulas),
                "processing_time_ms": processing_time_ms,
                "backend": backend,
                "gpu_used": gpu_available,
            }
        )

        return {
            "markdown": markdown_text,
            "images": images,
            "tables": tables,
            "formulas": formulas,
            "metadata": {
                "pages": page_count,
                "processing_time_ms": processing_time_ms,
                "parser": "mineru",
                "backend": backend,
                "gpu_used": gpu_available,
                "accuracy_tier": "very-high" if use_vlm else "high",
                "version": mineru_version,
                "filename": filename
            }
        }

    except Exception as e:
        logger.error("MinerU parsing failed: %s", e, exc_info=True)
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
