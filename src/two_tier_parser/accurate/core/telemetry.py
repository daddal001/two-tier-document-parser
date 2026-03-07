"""Optional OpenTelemetry instrumentation for accurate parser (MinerU wrapper).

Uses API-only pattern: no-op if SDK not installed.
This maintains portability for OSS users who don't need tracing.

The accurate parser wraps MinerU (external library) with tracing spans
to provide visibility into GPU/CPU operations without modifying MinerU itself.

Span hierarchy for MinerU operations:
    accurate-parser (service)
        ├── mineru.load_images (span)
        ├── mineru.vlm_analyze OR mineru.pipeline_parse (span)
        ├── mineru.union_make (span)
        └── mineru.get_crop_img (repeated spans)

See: https://opentelemetry.io/docs/concepts/instrumentation/libraries/
"""
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Optional OpenTelemetry (graceful fallback)
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
    tracer = trace.get_tracer("two_tier_parser.accurate", "1.0.0")
except ImportError:
    OTEL_AVAILABLE = False
    tracer = None
    logger.debug("OpenTelemetry not installed - telemetry disabled")


def setup_telemetry(app) -> bool:
    """Initialize telemetry if available.

    Args:
        app: FastAPI application instance

    Returns:
        True if telemetry was enabled, False otherwise
    """
    if OTEL_AVAILABLE:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry instrumentation enabled for accurate-parser")
        return True
    return False


def get_tracer():
    """Get the tracer instance (or None if not available)."""
    return tracer


@contextmanager
def traced_operation(
    name: str,
    attributes: Optional[Dict[str, Any]] = None
) -> Generator[Any, None, None]:
    """Context manager for tracing external library calls (MinerU).

    Creates a span if OpenTelemetry is available, otherwise just logs.
    This is the primary mechanism for wrapping MinerU calls without
    modifying MinerU itself.

    Args:
        name: Span name (e.g., "mineru.vlm_analyze", "mineru.load_images")
        attributes: Optional span attributes for context

    Yields:
        Span object if available, None otherwise

    Example:
        with traced_operation("mineru.vlm_analyze", {"backend": "transformers"}) as span:
            middle_json, infer_result = vlm_doc_analyze(pdf_bytes, ...)
            if span:
                span.set_attribute("pages", len(middle_json.get("pdf_info", [])))

    Key MinerU operations to wrap:
        - mineru.load_images: PDF to PIL images (~1s)
        - mineru.vlm_analyze: GPU VLM analysis (10-30s/page)
        - mineru.pipeline_parse: CPU fallback (5-15s/page)
        - mineru.union_make: Markdown generation (~1s)
        - mineru.get_crop_img: Image cropping (~0.1s/image)
    """
    if tracer:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    # Handle non-string/number types
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(k, v)
                    else:
                        span.set_attribute(k, str(v))
            logger.info("Starting %s", name, extra=attributes or {})
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.recoverable", False)
                logger.error("%s failed: %s", name, e, exc_info=True)
                raise
            logger.info("Completed %s", name)
    else:
        logger.info("Starting %s", name, extra=attributes or {})
        try:
            yield None
        except Exception as e:
            logger.error("%s failed: %s", name, e, exc_info=True)
            raise
        logger.info("Completed %s", name)
