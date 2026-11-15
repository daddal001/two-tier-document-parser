"""Settings read from the environment."""

import os
from typing import Any, Callable, Optional


def env_number(
    name: str, default: Any, kind: Callable[[str], Any], minimum: Optional[float] = None
) -> Any:
    """Read a numeric setting, treating an empty value as unset.

    An empty variable is what templated deployments produce for an unset value. A
    malformed or out-of-range value stops startup with a message naming the variable.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {raw!r}")
    return value
