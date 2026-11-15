"""Run work in a pool worker so a failure cannot break the pool.

A process pool sends a worker's exception back by pickling it. Some exceptions pickle fine but
cannot be rebuilt in the parent (their ``__init__`` needs arguments that ``args`` does not
hold, as with some native-library errors). Rebuilding fails inside the pool's own result thread,
which marks the whole pool broken: that job and every later one fail, and the service has to
replace the pool and drop the work in flight. Wrapping the call turns such an exception into a
plain one that names the original, so one bad document stays one failed request.
"""

import pickle
from typing import Any, Callable

from .json_logging import configure_json_logging
from .tracing import configure_tracer_provider


class WorkerError(RuntimeError):
    """A worker's exception that could not cross the process boundary intact.

    The message names the original type and carries its text; the original traceback is in the
    worker's log.
    """


def run_safely(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn``; if it raises something the parent could not rebuild, raise ``WorkerError``.

    An exception that survives a pickle round trip (the one the pool performs) is re-raised
    unchanged, so callers still see the original type.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as error:
        try:
            # Rebuilding an object this process just pickled, the same step the pool performs on
            # the parent side; no outside data is involved.
            pickle.loads(pickle.dumps(error))  # nosec B301
        except Exception:
            kind = f"{type(error).__module__}.{type(error).__qualname__}"
            raise WorkerError(f"{kind}: {error}") from None
        raise


def init_worker(service_name: str) -> None:
    """Prepare a pool worker: JSON logging and, if requested, tracing, as in the parent.

    Workers started with ``spawn`` or ``forkserver`` inherit neither, so each applies the same
    environment-driven setup when it starts. A forked worker already has the parent's provider,
    which is left alone. This module is light on purpose: a worker must not import the
    application to run it.
    """
    configure_json_logging(None, service_name)
    try:
        configure_tracer_provider(service_name)
    except ImportError:
        pass  # no SDK installed: tracing stays the API's no-op, as in the parent
