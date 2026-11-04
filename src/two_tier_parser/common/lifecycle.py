"""Bounded shutdown for the parser worker pools.

Waiting for a pool with ``shutdown(wait=True)`` inside an async handler blocks the event
loop and never returns if a worker is stuck. Stopping a pool here has a time budget: queued
work is dropped, running work gets the budget to finish, and worker processes still running
afterwards are terminated.
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Sequence

DRAIN_SECONDS_ENV = "SHUTDOWN_DRAIN_SECONDS"
DEFAULT_DRAIN_SECONDS = 10.0
# How long a terminated worker process gets to exit before it is killed.
_TERMINATE_GRACE_SECONDS = 2.0


def drain_budget() -> float:
    """Read the shutdown drain budget in seconds; an empty value means the default.

    The budget should be shorter than the platform's termination grace period (30 seconds
    on Kubernetes unless set), because the process is killed when that period ends.
    """
    raw = (os.getenv(DRAIN_SECONDS_ENV) or "").strip()
    if not raw:
        return DEFAULT_DRAIN_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{DRAIN_SECONDS_ENV} must be a number, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{DRAIN_SECONDS_ENV} must be at least 0, got {raw!r}")
    return value


def stop_pool(pool: Executor, budget_s: float) -> bool:
    """Stop ``pool`` within ``budget_s`` seconds; return True if it stopped on its own.

    Queued work is cancelled. Work already running gets the budget to finish. A process
    pool whose workers are still running after that has them terminated, then killed; a
    thread pool cannot be forced, so the caller decides what to do when this returns False.
    """
    # The wait runs in a helper thread so the budget can be enforced with public APIs only.
    waiter = threading.Thread(
        target=pool.shutdown, kwargs={"wait": True, "cancel_futures": True}, daemon=True
    )
    waiter.start()
    waiter.join(budget_s)
    if not waiter.is_alive():
        return True
    if isinstance(pool, ProcessPoolExecutor):
        _terminate_workers(pool)
        waiter.join(_TERMINATE_GRACE_SECONDS + 1)
    return False


def _terminate_workers(pool: ProcessPoolExecutor) -> None:
    # shortcut: ProcessPoolExecutor has no public way to stop running workers, so this uses
    # its private process map; revisit if CPython adds ProcessPoolExecutor.terminate_workers.
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()


async def drain_pools(
    pools: Sequence[Executor], *, budget_s: float, logger: logging.Logger, service_name: str
) -> list[Executor]:
    """Stop every pool within one shared budget without blocking the event loop.

    Returns the pools that had work still running when the budget ran out.
    """

    def stop_all() -> list[Executor]:
        deadline = time.monotonic() + budget_s
        stuck = []
        for pool in pools:
            if not stop_pool(pool, max(0.0, deadline - time.monotonic())):
                stuck.append(pool)
        return stuck

    stuck = await asyncio.to_thread(stop_all)
    if stuck:
        logger.error(
            "Worker pool did not drain within the shutdown budget; running work was abandoned",
            extra={
                "service_name": service_name,
                "component": "shutdown",
                "budget_seconds": budget_s,
                "stuck_pools": len(stuck),
            },
        )
    return stuck
