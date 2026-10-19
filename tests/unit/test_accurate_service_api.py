"""HTTP behavior of the accurate service with MinerU replaced by a controllable fake."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from two_tier_parser.accurate import main

PDF = {"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")}


def _result(pages=1):
    return {
        "markdown": "text",
        "images": [],
        "tables": [],
        "formulas": [],
        "metadata": {
            "pages": pages,
            "processing_time_ms": 5,
            "parser": "mineru",
            "backend": "pipeline",
            "gpu_used": False,
            "accuracy_tier": "high",
            "version": "test",
            "filename": "doc.pdf",
        },
    }


@pytest.fixture
def fresh_pool(monkeypatch):
    """Give each test its own worker pool and a clean admission counter."""
    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(main, "executor", pool)
    monkeypatch.setattr(main, "_inflight_documents", 0)
    monkeypatch.setattr(main, "MAX_INFLIGHT_DOCUMENTS", 2)
    yield pool
    pool.shutdown(wait=True)


def _client(root_path=""):
    transport = httpx.ASGITransport(app=main.app, root_path=root_path)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class _BlockingParser:
    """A parse_pdf stand-in that holds every call until released."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Semaphore(0)
        self.calls = 0

    def __call__(self, pdf_bytes, filename):
        self.calls += 1
        self.started.release()
        assert self.release.wait(10)
        return _result()

    async def wait_started(self, count):
        for _ in range(count):
            while not self.started.acquire(blocking=False):
                await asyncio.sleep(0.01)


def test_requests_beyond_the_bound_get_503_while_health_stays_available(monkeypatch, fresh_pool):
    parser = _BlockingParser()
    monkeypatch.setattr(main, "parse_pdf", parser)

    async def exercise():
        async with _client() as client:
            running = [asyncio.create_task(client.post("/parse", files=PDF)) for _ in range(2)]
            await parser.wait_started(2)

            rejected = await client.post("/parse", files=PDF)
            health = await client.get("/health")
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "5"
            assert rejected.json() == {"detail": "document_admission_saturated"}
            assert health.status_code == 200
            assert parser.calls == 2  # the rejected upload never reached a worker

            parser.release.set()
            assert [r.status_code for r in await asyncio.gather(*running)] == [200, 200]
            assert (await client.post("/parse", files=PDF)).status_code == 200

    asyncio.run(exercise())
    assert main._inflight_documents == 0


def test_a_queued_document_counts_against_the_bound(monkeypatch):
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main, "executor", pool)
    monkeypatch.setattr(main, "_inflight_documents", 0)
    monkeypatch.setattr(main, "MAX_INFLIGHT_DOCUMENTS", 2)
    parser = _BlockingParser()
    monkeypatch.setattr(main, "parse_pdf", parser)

    async def exercise():
        async with _client() as client:
            first = asyncio.create_task(client.post("/parse", files=PDF))
            await parser.wait_started(1)
            queued = asyncio.create_task(client.post("/parse", files=PDF))
            while main._inflight_documents < 2:
                await asyncio.sleep(0.01)

            assert (await client.post("/parse", files=PDF)).status_code == 503

            parser.release.set()
            assert (await first).status_code == 200
            assert (await queued).status_code == 200

    try:
        asyncio.run(exercise())
    finally:
        pool.shutdown(wait=True)
    assert main._inflight_documents == 0


def test_a_disconnecting_client_does_not_free_the_slot_while_the_worker_runs(
    monkeypatch, fresh_pool
):
    parser = _BlockingParser()
    monkeypatch.setattr(main, "parse_pdf", parser)

    async def exercise():
        async with _client() as client:
            request = asyncio.create_task(client.post("/parse", files=PDF))
            await parser.wait_started(1)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request

            assert main._inflight_documents == 1  # the thread is still parsing

            parser.release.set()
            for _ in range(200):
                if main._inflight_documents == 0:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(exercise())
    assert main._inflight_documents == 0


def test_rejected_requests_before_the_handler_release_their_slot(fresh_pool):
    async def exercise():
        async with _client() as client:
            for _ in range(5):  # more than the bound of 2
                bad = await client.post("/parse", files={"file": ("notes.txt", b"x", "text/plain")})
                assert bad.status_code == 400

    asyncio.run(exercise())
    assert main._inflight_documents == 0


def test_admission_applies_behind_a_path_prefix(monkeypatch, fresh_pool):
    monkeypatch.setattr(main, "_inflight_documents", 2)

    async def exercise():
        async with _client(root_path="/parser") as client:
            return await client.post("/parser/parse", files=PDF)

    assert asyncio.run(exercise()).status_code == 503


@pytest.mark.parametrize(
    ("path", "root", "expected"),
    [
        ("/parse", "", "/parse"),
        ("/parse/", "", "/parse"),
        ("/parser/parse", "/parser", "/parse"),
        ("/parser/parse", "/parser/", "/parse"),
        ("/parserx/parse", "/parser", "/parserx/parse"),
        ("/parser", "/parser", "/"),
    ],
)
def test_route_path_strips_only_a_whole_segment_prefix(path, root, expected):
    assert main._route_path({"path": path, "root_path": root}) == expected


def test_a_parse_failure_returns_a_generic_500_and_logs_the_detail(monkeypatch, fresh_pool, caplog):
    secret = "/srv/models/layout.onnx missing"
    monkeypatch.setattr(main, "parse_pdf", lambda *_: {"error": secret, "traceback": "tb"})

    async def exercise():
        async with _client() as client:
            return await client.post("/parse", files=PDF)

    with caplog.at_level(logging.ERROR, logger=main.logger.name):
        response = asyncio.run(exercise())

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Parsing failed (correlation_id=")
    assert secret not in response.text
    logged = [r for r in caplog.records if getattr(r, "error_message", "") == secret]
    assert logged
    assert f"correlation_id={logged[0].correlation_id}" in detail
    assert main._inflight_documents == 0


def test_an_unexpected_exception_is_generic_in_the_response_and_full_in_the_log(
    monkeypatch, fresh_pool, caplog
):
    def explode(*_):
        raise RuntimeError("CUDA out of memory at /secret/path")

    monkeypatch.setattr(main, "parse_pdf", explode)

    async def exercise():
        async with _client() as client:
            return await client.post("/parse", files=PDF)

    with caplog.at_level(logging.ERROR, logger=main.logger.name):
        response = asyncio.run(exercise())

    assert response.status_code == 500
    assert "CUDA" not in response.text and "/secret/path" not in response.text
    record = next(r for r in caplog.records if r.levelno == logging.ERROR)
    assert record.error_type == "RuntimeError"
    assert "/secret/path" in record.error_message
    assert record.exc_info is not None


def test_a_none_result_is_a_generic_500(monkeypatch, fresh_pool):
    monkeypatch.setattr(main, "parse_pdf", lambda *_: None)

    async def exercise():
        async with _client() as client:
            return await client.post("/parse", files=PDF)

    response = asyncio.run(exercise())
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Parsing failed (correlation_id=")


def test_successful_parse_keeps_the_response_shape(monkeypatch, fresh_pool):
    monkeypatch.setattr(main, "parse_pdf", lambda *_: _result(pages=3))

    async def exercise():
        async with _client() as client:
            return await client.post("/parse", files=PDF)

    response = asyncio.run(exercise())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"markdown", "metadata", "images", "tables", "formulas"}
    assert body["metadata"]["pages"] == 3


def test_response_size_is_measured_off_the_event_loop(monkeypatch, fresh_pool):
    monkeypatch.setattr(main, "parse_pdf", lambda *_: _result())
    worker_threads = []
    real = main._parse_pdf_with_cleanup

    def recording(*args):
        worker_threads.append(threading.current_thread())
        return real(*args)

    monkeypatch.setattr(main, "_parse_pdf_with_cleanup", recording)

    async def exercise():
        async with _client() as client:
            return await client.post("/parse", files=PDF)

    loop_thread = threading.current_thread()
    assert asyncio.run(exercise()).status_code == 200
    # The size is computed inside _parse_pdf_with_cleanup, which must not run on the loop.
    assert worker_threads and loop_thread not in worker_threads


def test_periodic_cleanup_logs_the_count_it_performed(monkeypatch, caplog):
    monkeypatch.setattr(main, "parse_pdf", lambda *_: _result())
    monkeypatch.setattr(main, "MAX_TASKS_BEFORE_CLEANUP", 3)
    monkeypatch.setattr(main, "_task_counter", 0)

    with caplog.at_level(logging.INFO, logger=main.logger.name):
        for _ in range(3):
            main._parse_pdf_with_cleanup(b"pdf", "doc.pdf")

    cleanup = [r for r in caplog.records if getattr(r, "component", "") == "memory"]
    assert len(cleanup) == 1
    assert cleanup[0].tasks_since_cleanup == 3
    assert main._task_counter == 0


def test_numeric_settings_treat_empty_as_unset_and_reject_garbage(monkeypatch):
    monkeypatch.setenv("TEST_SETTING", "")
    assert main._env_int("TEST_SETTING", 7) == 7
    monkeypatch.setenv("TEST_SETTING", " 4 ")
    assert main._env_int("TEST_SETTING", 7) == 4
    monkeypatch.setenv("TEST_SETTING", "many")
    with pytest.raises(ValueError, match="TEST_SETTING must be an integer"):
        main._env_int("TEST_SETTING", 7)
    monkeypatch.setenv("TEST_SETTING", "0")
    with pytest.raises(ValueError, match="TEST_SETTING must be at least 1"):
        main._env_int("TEST_SETTING", 7)
