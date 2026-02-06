"""Backwards-compatible module for accurate parser.

This file exists for backwards compatibility. The entry point has been
renamed from app.py to main.py following industry standards.

New code should use:
    from two_tier_parser.accurate.main import app

This module will be removed in a future version.
"""
import warnings

warnings.warn(
    "two_tier_parser.accurate.app is deprecated. Use two_tier_parser.accurate.main instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from main for backwards compatibility
from .main import *  # noqa: F401, F403
from .main import app  # noqa: F401
