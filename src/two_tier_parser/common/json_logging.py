"""Self-contained JSON logging configured only through environment variables.

Every record becomes one JSON object on one line, which log shippers (Loki,
Fluent Bit, CloudWatch, Datadog, ``docker logs | jq``) parse without extra
configuration. No collector or logging service is required.

Environment variables:
    LOG_LEVEL: Level name in any case (``info``, ``INFO``) or a number. Emitted
        ``level`` values are always upper case. Unknown values fall back to INFO.
    LOG_FORMAT: ``json`` (default) or ``text`` for a human-readable local format.
    SERVICE_NAME: Value of the ``service_name`` field when a record does not
        carry its own. ``OTEL_SERVICE_NAME`` is the fallback.

Fields written on every record: ``timestamp`` (UTC, ``YYYY-MM-DDTHH:MM:SSZ``),
``level``, ``logger``, ``message`` and ``service_name``. ``trace_id`` and
``span_id`` are added while an OpenTelemetry span is active. Fields passed with
``extra=`` (``task_id``, ``duration_ms``, ``error_type`` and so on) are added
beside them, but can never replace the five canonical fields. ``exception``
holds the formatted traceback when ``exc_info`` is set.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Union

_RESERVED = {"timestamp", "level", "logger", "message", "exception"}
_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_DEFAULT_SERVICE = "two-tier-parser"


def safe_log_value(value: Any) -> Optional[str]:
    """Escape caller-controlled line breaks so a value cannot forge log lines.

    JSON output already escapes them; this protects hosts that route records
    through a plain-text handler.
    """
    if value is None:
        return None
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def resolve_level(value: Union[str, int, None], default: int = logging.INFO) -> int:
    """Convert a level name (any case) or number to a logging level.

    Returns ``default`` for empty or unknown values so a typo never prevents the
    service from starting.
    """
    if value is None:
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    if text.isdigit():
        return int(text)
    return logging.getLevelNamesMapping().get(text.upper(), default)


def _service_name(default: str = _DEFAULT_SERVICE) -> str:
    return os.getenv("SERVICE_NAME") or os.getenv("OTEL_SERVICE_NAME") or default


def _trace_fields() -> Dict[str, str]:
    """Return the active OpenTelemetry trace and span ids, or nothing."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return {
                "trace_id": format(context.trace_id, "032x"),
                "span_id": format(context.span_id, "016x"),
            }
    except Exception:  # nosec B110 - logging must never fail because tracing is unavailable
        pass
    return {}


class JsonFormatter(logging.Formatter):
    """Serialize each log record as a single-line JSON object."""

    def __init__(self, default_service: str = _DEFAULT_SERVICE) -> None:
        """Set the ``service_name`` used when neither the environment nor the record gives one."""
        super().__init__()
        self._default_service = default_service

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as JSON; a value that cannot be encoded never drops the line."""
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD}
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service_name": _service_name(self._default_service),
            **_trace_fields(),
        }
        # Extras are added last and skipped when they would replace a canonical field;
        # a record-supplied service_name is the one deliberate override.
        for key, value in extras.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            # Circular references or non-string keys: keep the line, drop the extras.
            canonical = {k: payload[k] for k in payload if k in _RESERVED | {"service_name"}}
            canonical["log_error"] = "extra fields could not be serialized"
            return json.dumps(canonical, default=str)


def configure_json_logging(
    level: Union[str, int, None] = None, service_name: str = _DEFAULT_SERVICE
) -> None:
    """Install a stderr handler on the root logger unless the host already did.

    Args:
        level: Level name or number. Defaults to the ``LOG_LEVEL`` variable, then INFO.
        service_name: Default ``service_name`` field; ``SERVICE_NAME`` overrides it.

    An embedding application's existing logging configuration takes precedence
    (``logging.basicConfig`` does nothing when the root logger has handlers).
    """
    requested = level if level is not None else os.getenv("LOG_LEVEL")
    resolved = resolve_level(requested)
    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "json").strip().lower() == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    else:
        handler.setFormatter(JsonFormatter(service_name))
    logging.basicConfig(level=resolved, handlers=[handler])
    if requested and not str(requested).strip().isdigit():
        if str(requested).strip().upper() not in logging.getLevelNamesMapping():
            logging.getLogger(__name__).warning(
                "Unrecognized LOG_LEVEL %s; using INFO", safe_log_value(requested)
            )
