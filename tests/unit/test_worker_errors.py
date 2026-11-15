"""An exception that cannot cross the process boundary must not take the whole pool down."""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest
from fastapi.testclient import TestClient

from tests import worker_failures as failures
from two_tier_parser.common.worker import WorkerError, run_safely
from two_tier_parser.fast import main as fast_main


@pytest.fixture
def pool():
    executor = ProcessPoolExecutor(max_workers=1)
    yield executor
    executor.shutdown(wait=True, cancel_futures=True)


def test_without_the_wrapper_such_an_exception_breaks_the_pool(pool):
    """The problem this guards against: one failure takes every later request down with it."""
    with pytest.raises(BrokenProcessPool):
        pool.submit(failures.raises_unrebuildable).result(timeout=60)
    with pytest.raises(BrokenProcessPool):
        pool.submit(failures.fine).result(timeout=60)


def test_with_the_wrapper_the_failure_is_reported_and_the_pool_survives(pool):
    with pytest.raises(WorkerError) as raised:
        pool.submit(run_safely, failures.raises_unrebuildable).result(timeout=60)

    assert "NeedsTwoArguments" in str(raised.value) and "a then b" in str(raised.value)
    assert pool.submit(run_safely, failures.fine, 7).result(timeout=60) == 7  # still healthy


def test_an_ordinary_exception_arrives_unchanged(pool):
    with pytest.raises(failures.Plain, match="an ordinary failure"):
        pool.submit(run_safely, failures.raises_plain).result(timeout=60)


def test_the_result_of_a_successful_call_is_passed_through():
    assert run_safely(failures.fine, 5) == 5
    assert run_safely(lambda a, b=0: a + b, 1, b=2) == 3  # also callable without a pool


def test_an_exception_that_pickles_is_re_raised_as_it_was():
    with pytest.raises(failures.Plain):
        run_safely(failures.raises_plain)


def test_the_original_type_is_named_in_the_replacement():
    with pytest.raises(WorkerError, match=r"tests\.worker_failures\.NeedsTwoArguments"):
        run_safely(failures.raises_unrebuildable)


def test_the_service_answers_500_and_keeps_serving_after_such_a_failure(monkeypatch):
    """Through the real /parse route with a real process pool."""
    workers = ProcessPoolExecutor(max_workers=1)
    monkeypatch.setattr(fast_main, "executor", workers)
    monkeypatch.setattr(fast_main, "_inflight_documents", 0)
    monkeypatch.setattr(fast_main, "parse_pdf", failures.raises_unrebuildable_with_args)
    client = TestClient(fast_main.app, raise_server_exceptions=False)
    try:
        failed = client.post("/parse", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})

        assert failed.status_code == 500  # an ordinary failure, not "worker unavailable"
        assert failed.json()["detail"].startswith("Parsing failed (correlation_id=")
        assert fast_main.executor is workers  # the pool was not replaced

        monkeypatch.setattr(fast_main, "parse_pdf", failures.fine)
        again = client.post("/parse", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
        assert again.status_code != 503  # the same pool took the next job
    finally:
        workers.shutdown(wait=True, cancel_futures=True)


def test_asyncio_gather_over_the_pool_still_isolates_failures(pool):
    async def run():
        loop = asyncio.get_running_loop()
        return await asyncio.gather(
            loop.run_in_executor(pool, run_safely, failures.raises_unrebuildable),
            loop.run_in_executor(pool, run_safely, failures.fine, 3),
            return_exceptions=True,
        )

    first, second = asyncio.run(run())

    assert isinstance(first, WorkerError)
    assert second == 3
