"""Backwards-compatible module for fast parser.

This file exists for backwards compatibility. The entry point has been
renamed from app.py to main.py following industry standards.

New code should use:
    from two_tier_parser.fast.main import app

This module will be removed in a future version.
"""
import warnings

warnings.warn(
    "two_tier_parser.fast.app is deprecated. Use two_tier_parser.fast.main instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export public names from main for backwards compatibility.
from . import main as _main

__all__ = [name for name in dir(_main) if not name.startswith("_")]
globals().update({name: getattr(_main, name) for name in __all__})
