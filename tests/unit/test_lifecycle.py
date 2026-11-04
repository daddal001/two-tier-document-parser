"""Bounded shutdown of the worker pools."""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from two_tier_parser.accurate import main as accurate_main
from two_tier_parser.common import lifecycle
from two_tier_parser.fast import main as fast_main

LOG = logging.getLogger("lifecycle-test")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _worker_pids(pool: ProcessPoolExecutor) -> list[int]:
    return list(pool._processes)  # the test needs the pids to prove the workers are gone


def _start_stuck_pool() -> ProcessPoolExecutor:
    pool = ProcessPoolExecutor(max_workers=1)
    pool.submit(time.sleep, 120)
    deadline = time.monotonic() + 20
    while not pool._processes:  # wait until the worker has started
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(0.5)  # and has picked the task up
    return pool


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 10.0), ("", 10.0), ("  ", 10.0), ("2.5", 2.5), ("0", 0.0)],
)
def test_drain_budget_reads_the_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(lifecycle.DRAIN_SECONDS_ENV, raising=False)
    else:
        monkeypatch.setenv(lifecycle.DRAIN_SECONDS_ENV, value)
    assert lifecycle.drain_budget() == expected


@pytest.mark.parametrize("value", ["soon", "-1"])
def test_drain_budget_rejects_bad_values_by_name(monkeypatch, value):
    monkeypatch.setenv(lifecycle.DRAIN_SECONDS_ENV, value)
    with pytest.raises(ValueError, match=lifecycle.DRAIN_SECONDS_ENV):
        lifecycle.drain_budget()


def test_an_idle_pool_stops_cleanly_and_quickly():
    pool = ProcessPoolExecutor(max_workers=1)
    pool.submit(int).result()
    started = time.monotonic()
    assert lifecycle.stop_pool(pool, budget_s=10) is True
    assert time.monotonic() - started < 5


def test_work_that_finishes_inside_the_budget_is_allowed_to_finish():
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(time.sleep, 0.3)
    assert lifecycle.stop_pool(pool, budget_s=10) is True
    assert future.done() and future.exception() is None


def test_a_stuck_process_pool_is_terminated_within_the_budget():
    pool = _start_stuck_pool()
    pids = _worker_pids(pool)
    started = time.monotonic()
    assert lifecycle.stop_pool(pool, budget_s=0.5) is False
    assert time.monotonic() - started < 10
    deadline = time.monotonic() + 10
    while any(_alive(pid) for pid in pids):  # a terminated worker is reaped, not left running
        assert time.monotonic() < deadline, "worker processes survived the forced stop"
        time.sleep(0.1)


def test_queued_work_is_dropped_not_run():
    pool = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    pool.submit(release.wait, 30)
    queued = pool.submit(lambda: "ran")
    try:
        assert lifecycle.stop_pool(pool, budget_s=0.3) is False  # the first task is still running
        assert queued.cancelled()
    finally:
        release.set()


def test_a_stuck_thread_pool_is_reported_not_forced():
    pool = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    pool.submit(release.wait, 30)
    try:
        assert lifecycle.stop_pool(pool, budget_s=0.3) is False
    finally:
        release.set()


def test_draining_does_not_block_the_event_loop():
    pool = _start_stuck_pool()
    ticks = []

    async def exercise():
        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        task = asyncio.create_task(ticker())
        stuck = await lifecycle.drain_pools(
            [pool], budget_s=1.0, logger=LOG, service_name="lifecycle-test"
        )
        task.cancel()
        return stuck

    started = time.monotonic()
    assert asyncio.run(exercise()) == [pool]
    elapsed = time.monotonic() - started
    assert len(ticks) >= elapsed / 0.05 * 0.5  # the loop kept running while workers drained


def test_drain_pools_share_one_budget_across_pools():
    first, second = _start_stuck_pool(), _start_stuck_pool()
    started = time.monotonic()
    stuck = asyncio.run(
        lifecycle.drain_pools(
            [first, second], budget_s=1.0, logger=LOG, service_name="lifecycle-test"
        )
    )
    assert len(stuck) == 2
    assert time.monotonic() - started < 15  # not budget per pool times the worker grace


def test_fast_shutdown_terminates_stuck_workers(monkeypatch):
    stuck, idle = _start_stuck_pool(), ProcessPoolExecutor(max_workers=1)
    pids = _worker_pids(stuck)
    monkeypatch.setattr(fast_main, "executor", stuck)
    monkeypatch.setattr(fast_main, "pages_executor", idle)
    monkeypatch.setattr(fast_main, "SHUTDOWN_DRAIN_SECONDS", 0.5)
    started = time.monotonic()
    asyncio.run(fast_main.shutdown_event())
    assert time.monotonic() - started < 15
    deadline = time.monotonic() + 10
    while any(_alive(pid) for pid in pids):
        assert time.monotonic() < deadline
        time.sleep(0.1)


def test_accurate_shutdown_exits_when_inference_is_stuck(monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    pool.submit(release.wait, 30)
    exits = []
    monkeypatch.setattr(accurate_main, "executor", pool)
    monkeypatch.setattr(accurate_main, "SHUTDOWN_DRAIN_SECONDS", 0.3)
    monkeypatch.setattr(accurate_main, "_exit", exits.append)
    try:
        asyncio.run(accurate_main.shutdown_event())
        assert exits == [1]
    finally:
        release.set()


def test_accurate_shutdown_does_not_exit_when_idle(monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(int).result()
    exits = []
    monkeypatch.setattr(accurate_main, "executor", pool)
    monkeypatch.setattr(accurate_main, "_exit", exits.append)
    asyncio.run(accurate_main.shutdown_event())
    assert exits == []
