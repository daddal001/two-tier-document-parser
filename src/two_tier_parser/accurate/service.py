"""
MinerU wrapper for accurate PDF parsing with multimodal extraction.
Automatically falls back to pipeline backend (CPU) if no GPU is detected.
"""
import base64
import json
import time
import io
import traceback
from typing import Dict, Any
import os
from loguru import logger

def parse_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Parse PDF using MinerU with automatic GPU fallback.
    - GPU available: VLM backend (transformers) - 95%+ accuracy
    - No GPU: Pipeline backend (CPU-only) - 80-85% accuracy
    
    Args:
        pdf_bytes: PDF file content as bytes
        filename: Original filename for logging
        
    Returns:
        Dictionary with markdown, images, tables, formulas, and metadata
    """
    start_time = time.time()

    try:
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        except (ImportError, Exception) as e:
            logger.info(f"GPU check failed: {e}")
            gpu_available = False
        
        # Select backend based on GPU availability
        if gpu_available:
            backend = 'transformers'  # VLM backend (95%+ accuracy, requires GPU)
            use_vlm = True
            logger.info("🚀 Using VLM backend for highest accuracy (GPU-accelerated)")
            logger.info("Expected processing time: ~10-11 mins for 10 pages on T4")
        else:
            backend = 'pipeline'  # Pipeline backend (80-85% accuracy, CPU-only)
            use_vlm = False
            logger.info("💻 No GPU detected - using pipeline backend (CPU-only, 80-85% accuracy)")
            logger.info("Expected processing time: ~2-3 mins for 10 pages on CPU")
        
        # Import MinerU components
        from mineru.utils.enum_class import MakeMode, ContentType, ImageType
        from mineru.utils.pdf_image_tools import load_images_from_pdf, get_crop_img
        from mineru.version import __version__
        
        # Load images into memory for manual cropping
        logger.info(f"Loading images for {filename}...")
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        
        logger.info(f"Starting MinerU processing for {filename} (backend: {backend})")
        
        # Process based on backend type
        if use_vlm:
            # VLM Backend (GPU-accelerated)
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
        else:
            # Pipeline Backend (CPU-only)
            from mineru.cli.common import do_parse
            from mineru.backend.pipeline.middle_json_mkcontent import union_make as pipeline_union_make
            import tempfile
            
            # Pipeline backend requires file-based processing
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = os.path.join(temp_dir, filename)
                output_dir = os.path.join(temp_dir, "output")
                
                # Write PDF to temp file
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_bytes)
                
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
                
                # Load middle.json from output
                middle_json_path = os.path.join(output_dir, filename.replace('.pdf', ''), "middle.json")
                with open(middle_json_path, 'r', encoding='utf-8') as f:
                    middle_json = json.load(f)
                
                logger.info("Pipeline processing completed. Extracting results from middle_json...")
                
                # Use pipeline's union_make instead of VLM's
                vlm_union_make = pipeline_union_make
        
        # Extract pdf_info from middle_json (results are in memory)
        pdf_info = middle_json.get("pdf_info", [])
        page_count = len(pdf_info)
        
        logger.info(f"Found {page_count} pages in results")
        
        # Generate markdown from middle_json using vlm_union_make
        # Note: vlm_union_make takes positional arguments: pdf_info, make_mode, image_dir
        markdown_text = vlm_union_make(
            pdf_info,
            MakeMode.MM_MD,
            ""  # No image directory needed (images are base64 in memory)
        )
        
        logger.info(f"Generated markdown ({len(markdown_text)} characters)")
        
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
        
        for page_idx, page_info in enumerate(pdf_info):
            # Get page image for cropping
            image_dict = images_list[page_idx]
            page_pil_img = image_dict["img_pil"]
            scale = image_dict["scale"]

            # Check para_blocks first (VLM), then preproc_blocks (Pipeline)
            spans = page_info.get("para_blocks", []) or page_info.get("preproc_blocks", [])
            
            for span in _traverse_blocks(spans):
                span_type = span.get("type", "")
                
                # Extract images
                if span_type == ContentType.IMAGE or span_type == "image":
                    bbox = span.get("bbox", [])
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
                        except Exception as e:
                            logger.warning(f"Failed to crop image: {e}")
                
                # Extract tables
                elif span_type == ContentType.TABLE or span_type == "table":
                    table_content = span.get("content", "")
                    html_content = span.get("html", "")
                    bbox = span.get("bbox", [])
                    
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
                
                # Extract formulas
                elif span_type == ContentType.INTERLINE_EQUATION or span_type == "interline_equation":
                    formula_content = span.get("content", "")
                    bbox = span.get("bbox", [])
                    if formula_content:
                        formulas.append({
                            "formula_id": f"page_{page_idx}_formula_{len(formulas)}",
                            "latex": formula_content,
                            "page": page_idx,
                            "bbox": bbox
                        })
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Extracted results: {len(images)} images, {len(tables)} tables, {len(formulas)} formulas")
        logger.info(f"Total processing time: {processing_time_ms}ms")
        
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
                "version": __version__,
                "filename": filename
            }
        }

    except Exception as e:
        logger.error(f"MinerU parsing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
