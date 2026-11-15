"""Send loguru records, which MinerU uses, through standard logging.

loguru writes plain text to stderr by default. In a service that logs JSON, those lines mix
with the JSON records, and a line written while another is unfinished can corrupt a record.
Forwarding them to the standard logger gives them the service's JSON format and level.
"""

import logging
from typing import Any

_FALLBACK_LOGGER = "mineru"


def forward_loguru(loguru_logger: Any) -> None:
    """Replace loguru's sinks with one that logs through the standard ``logging`` module.

    Does nothing when loguru is not installed (``loguru_logger`` is None). The sink accepts
    every level; the standard logger's own level and handlers decide what is emitted.
    """
    if loguru_logger is None:
        return

    def sink(message: Any) -> None:
        record = message.record
        exception = record.get("exception")
        exc_info = (
            (exception.type, exception.value, exception.traceback)
            if exception is not None
            else None
        )
        logging.getLogger(record.get("name") or _FALLBACK_LOGGER).log(
            record["level"].no, "%s", record["message"], exc_info=exc_info
        )

    loguru_logger.remove()
    loguru_logger.add(sink, level=0, format="{message}")
