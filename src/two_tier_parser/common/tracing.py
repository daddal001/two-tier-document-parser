"""Bounded settings for the OTLP trace exporter.

Left alone, the exporter waits up to ten seconds for a collector and retries with backoff, and
the batch processor allows thirty seconds for an export. At exit the SDK flushes pending spans,
so an unreachable collector kept the process alive for ten seconds or more, in every worker of
a pod that was being replaced. These defaults keep that to a few seconds. Anything an operator
sets through the standard OTEL_* variables still wins.
"""

import os
from typing import Any, Dict

DEFAULT_TIMEOUT_SECONDS = 3.0

_EXPORTER_TIMEOUT_VARIABLES = ("OTEL_EXPORTER_OTLP_TIMEOUT", "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT")
_PROCESSOR_TIMEOUT_VARIABLE = "OTEL_BSP_EXPORT_TIMEOUT"


def exporter_kwargs() -> Dict[str, Any]:
    """Arguments for ``OTLPSpanExporter``: a short timeout unless the operator set one."""
    if any(os.getenv(name) for name in _EXPORTER_TIMEOUT_VARIABLES):
        return {}
    return {"timeout": DEFAULT_TIMEOUT_SECONDS}


def processor_kwargs() -> Dict[str, Any]:
    """Arguments for ``BatchSpanProcessor``: a short export timeout unless the operator set one."""
    if os.getenv(_PROCESSOR_TIMEOUT_VARIABLE):
        return {}
    return {"export_timeout_millis": int(DEFAULT_TIMEOUT_SECONDS * 1000)}


def configure_tracer_provider(default_service_name: str) -> Any:
    """Install an SDK tracer provider that exports to the OTLP endpoint named in the environment.

    The parent process and every pool worker call this, so a worker started with ``spawn`` or
    ``forkserver`` (which inherits no provider) traces like one started with ``fork`` (which
    inherits the parent's). Returns the new provider, or ``None`` when tracing was not asked for
    (no endpoint, or ``OTEL_SDK_DISABLED=true``) or a provider is already installed.

    Raises:
        ImportError: The OpenTelemetry SDK or the OTLP exporter is not installed.
    """
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return None
    if not (
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    ):
        return None

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return None  # inherited from the parent by fork, or installed by the host

    service_name = os.environ.get("OTEL_SERVICE_NAME", default_service_name)
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs()), **processor_kwargs())
    )
    trace.set_tracer_provider(provider)
    return provider
