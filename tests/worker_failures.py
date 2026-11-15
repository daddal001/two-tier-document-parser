"""Functions that run in pool workers and fail in the ways that matter. Importable by workers."""


class NeedsTwoArguments(Exception):
    """Pickles fine, but cannot be rebuilt from its own ``args``: unpickling raises TypeError."""

    def __init__(self, first, second):
        super().__init__(f"{first} then {second}")


class Plain(Exception):
    """An ordinary exception that crosses the process boundary without trouble."""


def fine(value=1):
    return value


def raises_unrebuildable():
    raise NeedsTwoArguments("a", "b")


def raises_plain():
    raise Plain("an ordinary failure")


def raises_unrebuildable_with_args(*_args, **_kwargs):
    raise NeedsTwoArguments("x", "y")


def provider_name():
    """The kind of tracer provider this worker process ended up with."""
    from opentelemetry import trace

    return type(trace.get_tracer_provider()).__name__


def service_name():
    """The service name on the worker's tracer provider resource, or None without an SDK."""
    from opentelemetry import trace

    resource = getattr(trace.get_tracer_provider(), "resource", None)
    return resource.attributes.get("service.name") if resource else None


def root_formatter_name():
    """The formatter on the worker's root logger handler (JSON logging is set up by the pool)."""
    import logging

    handlers = logging.getLogger().handlers
    return type(handlers[0].formatter).__name__ if handlers and handlers[0].formatter else None
