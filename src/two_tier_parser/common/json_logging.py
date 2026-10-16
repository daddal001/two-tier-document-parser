"""JSON logging with escaped messages and structured application fields."""

import json
import logging
from datetime import datetime, timezone

_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def safe_log_value(value):
    """Escape caller-controlled line breaks even with a plain-text host handler."""
    if value is None:
        return None
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


class JsonFormatter(logging.Formatter):
    """Serialize each log record as one JSON object with escaped message fields."""

    def format(self, record):
        """Return JSON containing the message, extra fields, and exception details."""
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_logging(level="INFO"):
    """Configure stderr JSON logging unless the host already installed handlers.

    Args:
        level: Standard-library logging level name or numeric level.

    Existing application logging configuration takes precedence. No network
    collector or external logging service is required.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    # Honour an embedding application's existing logging configuration.
    logging.basicConfig(level=level, handlers=[handler])
