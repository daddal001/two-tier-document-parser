import sys

from loguru import logger

from .config import settings


def configure_logging():
    """Configure loguru logging."""
    logger.remove()

    # Add stderr sink with structured formatting if needed, or colorized
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )


# Auto-configure on import or explicit call?
# Explicit call is better for control, but for simplicity here:
configure_logging()
