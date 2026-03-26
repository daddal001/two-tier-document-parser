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
    logger.debug("OpenTelemetry not installed - telemetry disabled", extra={"service_name": "two-tier-parser"})


def setup_telemetry(app) -> bool:
    """Initialize telemetry if available.

    If the SDK is installed (telemetry-sdk extra), configures OTLP export
    using standard OTEL_* environment variables. Without the SDK, the API
    stubs remain no-ops (zero overhead).

    Args:
        app: FastAPI application instance

    Returns:
        True if telemetry was enabled, False otherwise
    """
    if not OTEL_AVAILABLE:
        return False

    # If the SDK is installed, auto-configure TracerProvider from env vars
    # (OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME, etc.)
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        import os
        service_name = os.environ.get("OTEL_SERVICE_NAME", "fast-parser")
        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        logger.info(
            "OpenTelemetry SDK configured with OTLP exporter",
            extra={"service_name": "fast-parser", "component": "telemetry"}
        )
    except ImportError:
        logger.debug(
            "OpenTelemetry SDK not installed - using API-only (no-op) mode",
            extra={"service_name": "fast-parser", "component": "telemetry"}
        )

    FastAPIInstrumentor.instrument_app(app)
    logger.info(
        "OpenTelemetry instrumentation enabled",
        extra={"service_name": "fast-parser", "component": "telemetry"}
    )
    return True


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
            logger.debug(
                "Starting operation",
                extra={"service_name": "fast-parser", "operation": name, **(attributes or {})}
            )
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                logger.error(
                    "Operation failed",
                    exc_info=True,
                    extra={"service_name": "fast-parser", "operation": name, "error_type": type(e).__name__, "error_message": str(e)}
                )
                raise
            logger.debug(
                "Completed operation",
                extra={"service_name": "fast-parser", "operation": name}
            )
    else:
        logger.debug(
            "Starting operation",
            extra={"service_name": "fast-parser", "operation": name, **(attributes or {})}
        )
        try:
            yield None
        except Exception as e:
            logger.error(
                "Operation failed",
                exc_info=True,
                extra={"service_name": "fast-parser", "operation": name, "error_type": type(e).__name__, "error_message": str(e)}
            )
            raise
        logger.debug(
            "Completed operation",
            extra={"service_name": "fast-parser", "operation": name}
        )
