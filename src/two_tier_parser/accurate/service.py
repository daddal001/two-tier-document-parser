"""MinerU-backed PDF parsing with the public API response kept stable."""

import gc
import logging
import os
import time
import traceback
from typing import Any

from opentelemetry import trace

from .mineru_adapter import analyze

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("parser.accurate")


def detect_device() -> tuple[str, str | None]:
    """Honor MINERU_DEVICE_MODE or detect CUDA, native Apple MPS, then CPU.

    MinerU's VLM tier is CUDA-only in this service. MPS accelerates the
    medium-effort pipeline; a failed selected backend is never retried on CPU.
    """
    configured_device = os.getenv("MINERU_DEVICE_MODE", "auto").lower()
    if configured_device not in {"", "auto"}:
        if configured_device == "mps":
            return configured_device, "Apple Metal (MPS)"
        if configured_device == "cuda":
            try:
                import torch

                return configured_device, torch.cuda.get_device_name(0)
            except Exception:
                return configured_device, None
        return configured_device, None
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            return "mps", "Apple Metal (MPS)"
    except Exception as exc:
        logger.info(
            "Accelerator detection unavailable; selecting the CPU model tier",
            extra={"service_name": "accurate-parser", "error_type": type(exc).__name__},
        )
    return "cpu", None


def parse_pdf(pdf_bytes: bytes, filename: str) -> dict[str, Any]:
    """Use CUDA VLM when available; run MinerU's pipeline on MPS or CPU.

    MinerU 4 returns typed, in-memory document results. The adapter renders
    Markdown and maps its content list into the service's established response
    fields. A failed selected inference propagates without CPU/GPU retry.
    """
    started = time.perf_counter()
    device, device_name = detect_device()
    try:
        if device == "cuda" and device_name is None:
            import torch

            device_name = torch.cuda.get_device_name(0)
        elif device == "mps" and device_name is None:
            device_name = "Apple Metal (MPS)"

        # MinerU's supplied high-effort VLM image targets CUDA. On Apple Silicon,
        # its small-model pipeline can use PyTorch MPS without selecting vLLM.
        use_vlm = device == "cuda"
        effort = "high" if use_vlm else "medium"
        logger.info(
            "MinerU analysis selected",
            extra={
                "service_name": "accurate-parser",
                "effort": effort,
                "device": device,
                "device_name": device_name,
            },
        )
        with tracer.start_as_current_span("mineru.analyze") as span:
            span.set_attribute("mineru.effort", effort)
            result = analyze(pdf_bytes, use_vlm=use_vlm)

        processing_time_ms = int((time.perf_counter() - started) * 1000)
        from mineru.version import __version__ as mineru_version

        response = {
            "markdown": result["markdown"],
            "images": result["images"],
            "tables": result["tables"],
            "formulas": result["formulas"],
            "metadata": {
                "pages": result["pages"],
                "processing_time_ms": processing_time_ms,
                "parser": "mineru",
                "backend": "vlm" if use_vlm else "pipeline",
                "gpu_used": device in {"cuda", "mps"},
                "accuracy_tier": "very-high" if use_vlm else "high",
                "version": mineru_version,
                "filename": filename,
            },
        }
        logger.info(
            "Extraction complete",
            extra={
                "service_name": "accurate-parser",
                "document_name": filename,
                "page_count": result["pages"],
                "images": len(result["images"]),
                "tables": len(result["tables"]),
                "formulas": len(result["formulas"]),
                "duration_ms": processing_time_ms,
                "parser_mode": response["metadata"]["backend"],
                "gpu_used": device in {"cuda", "mps"},
                "gpu_name": device_name,
            },
        )
        return response
    except Exception as exc:
        logger.error(
            "MinerU parsing failed",
            exc_info=True,
            extra={
                "service_name": "accurate-parser",
                "document_name": filename,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return {"error": str(exc), "traceback": traceback.format_exc()}
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug(
                "CUDA cache cleanup unavailable",
                extra={"service_name": "accurate-parser", "error_type": type(exc).__name__},
            )
