"""Unit tests for fast parser service."""

import asyncio
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
from starlette.requests import Request
from starlette.responses import Response

# Add the submodule's src-layout root (tests/unit -> repository root -> src).
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from two_tier_parser.fast import main as fast_main
from two_tier_parser.fast.models import ParseResponse
from two_tier_parser.fast.service import parse_pdf


def _request(path: str) -> Request:
    """Build the minimum HTTP scope needed by the admission middleware."""
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("fast-parser", 8004),
            "root_path": "",
        }
    )


def test_document_admission_rejects_before_route_when_saturated(monkeypatch):
    """A bounded working set returns a retriable 503 without calling the route."""
    monkeypatch.setattr(
        fast_main,
        "_inflight_documents",
        fast_main.MAX_INFLIGHT_DOCUMENTS,
    )
    route_called = False

    async def call_next(_request):
        nonlocal route_called
        route_called = True
        return Response(status_code=200)

    response = asyncio.run(fast_main.bound_parser_inflight_documents(_request("/parse"), call_next))

    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert not route_called


@pytest.mark.parametrize("override,expected", [(None, 10), ("12", 12)])
def test_admission_configuration_at_startup(override, expected):
    """Standalone defaults follow worker slots; explicit limits take precedence."""
    env = dict(os.environ, WORKERS="8", PAGES_WORKERS="2", OTEL_SDK_DISABLED="true")
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    env.pop("MAX_INFLIGHT_DOCUMENTS", None)
    if override is not None:
        env["MAX_INFLIGHT_DOCUMENTS"] = override
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from two_tier_parser.fast import main; "
            f"assert main.MAX_INFLIGHT_DOCUMENTS == {expected}; "
            "main.executor.shutdown(); main.pages_executor.shutdown()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("route", ["/parse", "/parse-pages"])
def test_multipart_routes_without_minio(monkeypatch, route):
    """HTTP uploads reach either parser without a storage dependency."""

    def storage_forbidden(*_args, **_kwargs):
        pytest.fail("A direct upload must not access MinIO")

    def parsed(pdf_bytes, filename, *_args):
        assert pdf_bytes == b"%PDF-fixture"
        assert filename == "document.pdf"
        return {
            "markdown": "",
            "images": [],
            "metadata": {
                "filename": filename,
                "total_pages": 1,
                "pages_parsed": 1,
                "pages": 1,
                "start_page": 0,
                "end_page": 1,
                "processing_time_ms": 1,
            },
        }

    monkeypatch.setattr(fast_main, "read_file_from_minio", storage_forbidden)
    monkeypatch.setattr(fast_main, "check_minio_connectivity", storage_forbidden)
    monkeypatch.setattr(fast_main, "parse_pdf", parsed)
    monkeypatch.setattr(fast_main, "parse_pdf_page_range", parsed)
    monkeypatch.setattr(fast_main, "_inflight_documents", 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(fast_main, "executor", pool)
        monkeypatch.setattr(fast_main, "pages_executor", pool)

        async def upload():
            # Exercise the complete ASGI HTTP stack without a test-client
            # dependency or any external service.
            body = (
                b'--fixture\r\nContent-Disposition: form-data; name="file"; '
                b'filename="document.pdf"\r\nContent-Type: application/pdf\r\n'
                b"\r\n%PDF-fixture\r\n--fixture--\r\n"
            )
            scope = dict(_request(route).scope)
            scope["headers"] = [
                (b"content-type", b"multipart/form-data; boundary=fixture"),
                (b"content-length", str(len(body)).encode()),
            ]
            sent = False
            messages = []

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)

            await asyncio.wait_for(fast_main.app(scope, receive, send), timeout=5)
            return messages

        response = asyncio.run(upload())
    import json

    assert (
        next(m["status"] for m in response if m["type"] == "http.response.start") == 200
    ), response
    payload = b"".join(m.get("body", b"") for m in response if m["type"] == "http.response.body")
    assert json.loads(payload)["markdown"] == ""
    assert fast_main._inflight_documents == 0


def test_document_admission_releases_slot_after_request(monkeypatch):
    """Successful and failed handlers cannot leak an inflight admission slot."""
    monkeypatch.setattr(fast_main, "_inflight_documents", 0)

    async def call_next(_request):
        assert fast_main._inflight_documents == 1
        return Response(status_code=204)

    response = asyncio.run(
        fast_main.bound_parser_inflight_documents(
            _request("/parse-pages"),
            call_next,
        )
    )

    assert response.status_code == 204
    assert fast_main._inflight_documents == 0


def test_health_reports_degraded_without_blocking_event_loop(monkeypatch):
    """An unavailable MinIO check runs off-loop and returns degraded."""
    caller_thread = threading.get_ident()
    check_thread = None

    def unavailable_minio():
        nonlocal check_thread
        check_thread = threading.get_ident()
        return False

    monkeypatch.setattr(fast_main, "check_minio_connectivity", unavailable_minio)

    response = asyncio.run(fast_main.health())

    assert response.status == "degraded"
    assert response.minio_connected is False
    assert check_thread is not None
    assert check_thread != caller_thread


def test_health_bounds_slow_minio_check(monkeypatch):
    """A stalled dependency check returns before the probe budget expires."""

    async def stalled_to_thread(_function):
        await asyncio.sleep(60)

    monkeypatch.setattr(asyncio, "to_thread", stalled_to_thread)
    monkeypatch.setattr(fast_main, "MINIO_HEALTH_TIMEOUT_SECONDS", 0.01)

    response = asyncio.run(fast_main.health())

    assert response.status == "degraded"
    assert response.minio_connected is False


@pytest.mark.parametrize(
    ("stage1", "attribute", "workers"),
    [
        (False, "executor", fast_main.WORKERS),
        (True, "pages_executor", fast_main.PAGES_WORKERS),
    ],
)
def test_broken_worker_pool_is_replaced_and_reported_as_retryable(
    monkeypatch, stage1, attribute, workers
):
    """A dead pool is replaced once and surfaced as HTTP 503."""

    class FailedPool:
        def shutdown(self, **_kwargs):
            return None

    failed = FailedPool()
    replacement = object()
    constructor_calls = []

    def build_executor(**kwargs):
        constructor_calls.append(kwargs)
        return replacement

    monkeypatch.setattr(fast_main, attribute, failed)
    monkeypatch.setattr(fast_main, "ProcessPoolExecutor", build_executor)

    error = fast_main._recover_broken_executor(failed, stage1=stage1)

    assert error.status_code == 503
    assert error.detail == "parser_worker_unavailable"
    assert error.headers == {"Retry-After": "2"}
    assert getattr(fast_main, attribute) is replacement
    assert constructor_calls == [
        {"max_workers": workers, "max_tasks_per_child": fast_main.MAX_TASKS}
    ]


def test_broken_process_pool_is_the_worker_loss_signal():
    """Pin the standard exception type used by failed process futures."""
    assert issubclass(BrokenProcessPool, RuntimeError)


def test_full_parse_maps_worker_loss_to_retryable_503(monkeypatch):
    """The Stage-2 route must not convert a dead worker into terminal 500."""

    async def read_fixture(_request, *, stage1):
        assert stage1 is False
        return (
            b"pdf",
            "fixture.pdf",
            "fixture",
            None,
            0,
            {
                "extract_images_flag": False,
                "image_max_count": 0,
                "image_max_bytes": 0,
                "image_max_pixels": 0,
            },
        )

    class BrokenLoop:
        def run_in_executor(self, *_args):
            async def fail():
                raise BrokenProcessPool("worker exited")

            return fail()

    recovered = []

    def recover(failed_executor, *, stage1):
        recovered.append((failed_executor, stage1))
        return fast_main.HTTPException(
            status_code=503,
            detail="parser_worker_unavailable",
            headers={"Retry-After": "2"},
        )

    monkeypatch.setattr(fast_main, "_executor_queue_depth", lambda: 0)
    monkeypatch.setattr(fast_main, "_read_pdf_from_request", read_fixture)
    monkeypatch.setattr(fast_main.asyncio, "get_event_loop", lambda: BrokenLoop())
    monkeypatch.setattr(fast_main, "_recover_broken_executor", recover)

    with pytest.raises(fast_main.HTTPException) as raised:
        asyncio.run(fast_main.parse(_request("/parse")))

    assert raised.value.status_code == 503
    assert raised.value.detail == "parser_worker_unavailable"
    assert raised.value.headers == {"Retry-After": "2"}
    assert recovered == [(fast_main.executor, False)]


def test_parse_pdf_basic(sample_pdf_path):
    """Test basic PDF parsing."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf(pdf_bytes, sample_pdf_path.name)

    assert "markdown" in result
    assert "metadata" in result
    assert isinstance(result["markdown"], str)
    assert len(result["markdown"]) > 0


def test_parse_pdf_metadata(sample_pdf_path):
    """Test that metadata is correctly structured."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf(pdf_bytes, sample_pdf_path.name)

    metadata = result["metadata"]
    assert "pages" in metadata
    assert "processing_time_ms" in metadata
    assert isinstance(metadata["pages"], int)
    assert metadata["pages"] > 0


def test_parse_response_model(sample_pdf_path):
    """Test that result matches ParseResponse model."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf(pdf_bytes, sample_pdf_path.name)

    # Should be able to create ParseResponse from result
    response = ParseResponse(**result)
    assert response.markdown is not None
    assert response.metadata is not None


def test_parse_empty_pdf():
    """Test handling of empty PDF."""
    # Create minimal PDF bytes (this is a placeholder - actual test would need valid PDF)
    empty_bytes = b""

    # This should raise an error or return empty result
    # Adjust based on actual implementation
    with pytest.raises(Exception):
        parse_pdf(empty_bytes, "empty.pdf")
