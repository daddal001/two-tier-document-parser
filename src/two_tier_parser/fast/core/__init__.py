"""Optional tracing helpers for the fast parser.

Exports setup_telemetry, get_tracer, and OTEL_AVAILABLE. Import traced_operation from
core.telemetry when wrapping individual operations.
Local logging lives in common.json_logging and requires no collector.
Trace export requires an explicitly configured OTLP endpoint and the optional
SDK/exporter dependencies; OTEL_SDK_DISABLED=true disables setup.
"""

from .telemetry import OTEL_AVAILABLE, get_tracer, setup_telemetry

__all__ = ["setup_telemetry", "get_tracer", "OTEL_AVAILABLE"]
