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
        service_name = os.environ.get("OTEL_SERVICE_NAME", "accurate-parser")
        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        logger.info(
            "OpenTelemetry SDK configured with OTLP exporter",
            extra={"service_name": "accurate-parser", "component": "telemetry"}
        )
    except ImportError:
        logger.debug(
            "OpenTelemetry SDK not installed - using API-only (no-op) mode",
            extra={"service_name": "accurate-parser", "component": "telemetry"}
        )

    FastAPIInstrumentor.instrument_app(app)
    logger.info(
        "OpenTelemetry instrumentation enabled",
        extra={"service_name": "accurate-parser", "component": "telemetry"}
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
            logger.info(
                "Starting operation",
                extra={"service_name": "accurate-parser", "operation": name, **(attributes or {})}
            )
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.recoverable", False)
                logger.error(
                    "Operation failed",
                    exc_info=True,
                    extra={"service_name": "accurate-parser", "operation": name, "error_type": type(e).__name__, "error_message": str(e)}
                )
                raise
            logger.info(
                "Completed operation",
                extra={"service_name": "accurate-parser", "operation": name}
            )
    else:
        logger.info(
            "Starting operation",
            extra={"service_name": "accurate-parser", "operation": name, **(attributes or {})}
        )
        try:
            yield None
        except Exception as e:
            logger.error(
                "Operation failed",
                exc_info=True,
                extra={"service_name": "accurate-parser", "operation": name, "error_type": type(e).__name__, "error_message": str(e)}
            )
            raise
        logger.info(
            "Completed operation",
            extra={"service_name": "accurate-parser", "operation": name}
        )
