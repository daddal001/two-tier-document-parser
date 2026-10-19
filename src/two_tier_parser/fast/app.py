"""Backwards-compatible module for fast parser.

This file exists for backwards compatibility. The entry point has been
moved from app.py to main.py; existing imports remain supported.

New code should use:
    from two_tier_parser.fast.main import app

Importing this wrapper emits a DeprecationWarning. No removal release is scheduled.
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
