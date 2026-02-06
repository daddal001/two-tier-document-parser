"""Core utilities for the fast parser.

This package provides optional observability features:
- telemetry: OpenTelemetry tracing (no-op without SDK)
- logging: Structured logging with NullHandler fallback

The API-only pattern ensures zero overhead for users without tracing SDK installed.
"""
from .telemetry import setup_telemetry, get_tracer, OTEL_AVAILABLE

__all__ = ["setup_telemetry", "get_tracer", "OTEL_AVAILABLE"]
