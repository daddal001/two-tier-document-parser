"""Optional OpenTelemetry instrumentation for fast parser.

Uses API-only pattern: no-op if SDK not installed.
This maintains portability for OSS users who don't need tracing.

Per OpenTelemetry docs:
- Libraries use API only (opentelemetry-api) - never the SDK
- API is a no-op without SDK - zero overhead for users who don't need tracing
- Applications own the SDK - they choose when/how to initialize

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
    tracer = trace.get_tracer("two_tier_parser.fast", "1.0.0")
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
        logger.info("OpenTelemetry instrumentation enabled for fast-parser")
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
    """Context manager for tracing operations.

    Creates a span if OpenTelemetry is available, otherwise just logs.

    Args:
        name: Span name (e.g., "pdf.load", "pymupdf4llm.to_markdown")
        attributes: Optional span attributes

    Yields:
        Span object if available, None otherwise

    Example:
        with traced_operation("pdf.parse", {"filename": "doc.pdf"}) as span:
            result = parse_document()
            if span:
                span.set_attribute("pages", result.pages)
    """
    if tracer:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            logger.debug("Starting %s", name, extra=attributes or {})
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                logger.error("%s failed: %s", name, e, exc_info=True)
                raise
            logger.debug("Completed %s", name)
    else:
        logger.debug("Starting %s", name, extra=attributes or {})
        try:
            yield None
        except Exception as e:
            logger.error("%s failed: %s", name, e, exc_info=True)
            raise
        logger.debug("Completed %s", name)
