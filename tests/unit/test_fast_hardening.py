"""Fast service hardening: generic 500s, early 413s, bucket allow-list, cleanup on failure."""

import asyncio
import logging
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from two_tier_parser.fast import main as fast_main
from two_tier_parser.fast import service, storage


def _request(body: bytes, content_type: str, headers=()) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/parse",
            "headers": [(b"content-type", content_type.encode()), *headers],
            "query_string": b"",
        },
        receive,
    )


def _claim(bucket="b", key="k.pdf") -> Request:
    return _request(('{"bucket": "%s", "key": "%s"}' % (bucket, key)).encode(), "application/json")


# --- generic 500 ---------------------------------------------------------------------------


def test_internal_error_hides_exception_text_but_logs_it(caplog):
    secret = "/srv/private/path/leak.pdf failed"
    with caplog.at_level(logging.ERROR):
        error = fast_main._internal_error("Parsing failed", "doc.pdf", RuntimeError(secret))
    assert error.status_code == 500
    assert secret not in error.detail
    correlation_id = error.detail.split("correlation_id=")[1].rstrip(")")
    record = caplog.records[-1]
    assert record.correlation_id == correlation_id
    assert record.error_message == secret
    assert record.error_type == "RuntimeError"


def test_parse_route_returns_generic_500(monkeypatch):
    def explode(*_args):
        raise ValueError("internal /tmp/secret detail")

    monkeypatch.setattr(fast_main, "parse_pdf", explode)
    monkeypatch.setattr(fast_main, "read_object", lambda *a: b"%PDF-1.4")
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(fast_main, "executor", pool)
        with pytest.raises(HTTPException) as caught:
            asyncio.run(fast_main.parse(_claim()))
    assert caught.value.status_code == 500
    assert "secret" not in caught.value.detail
    assert "correlation_id=" in caught.value.detail


# --- early 413 -----------------------------------------------------------------------------


def test_multipart_declared_length_over_limit_is_rejected_before_parsing():
    request = _request(
        b"",
        "multipart/form-data; boundary=x",
        [(b"content-length", str(fast_main.MAX_PDF_BYTES * 2).encode())],
    )
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main._read_pdf_from_request(request))
    assert caught.value.status_code == 413


def test_object_over_limit_is_413_from_content_length(monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")

    class Response:
        headers = {"Content-Length": str(fast_main.MAX_PDF_BYTES + 1)}
        closed = released = False

        def read(self, *_a):
            pytest.fail("an oversized body must not be read")

        def close(self):
            self.closed = True

        def release_conn(self):
            self.released = True

    response = Response()

    class Client:
        def get_object(self, *_a):
            return response

    monkeypatch.setattr(storage, "_client", Client())
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main._read_pdf_from_request(_claim()))
    assert caught.value.status_code == 413
    assert response.closed and response.released


def test_read_object_without_length_header_is_still_bounded(monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")

    class Response:
        headers: dict = {}

        def read(self, amount=None):
            return b"x" * amount

        def close(self):
            pass

        def release_conn(self):
            pass

    class Client:
        def get_object(self, *_a):
            return Response()

    monkeypatch.setattr(storage, "_client", Client())
    with pytest.raises(storage.ObjectTooLarge):
        storage.read_object("b", "k.pdf", max_bytes=10)


def test_read_object_within_limit_returns_bytes(monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")

    class Response:
        headers = {"Content-Length": "5"}

        def read(self, amount=None):
            return b"%PDF-"

        def close(self):
            pass

        def release_conn(self):
            pass

    class Client:
        def get_object(self, *_a):
            return Response()

    monkeypatch.setattr(storage, "_client", Client())
    assert storage.read_object("b", "k.pdf", max_bytes=10) == b"%PDF-"


def test_storage_failure_stays_502(monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")

    def broken(*_a):
        raise ConnectionError("no route")

    monkeypatch.setattr(fast_main, "read_object", broken)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main._read_pdf_from_request(_claim()))
    assert caught.value.status_code == 502


# --- bucket allow-list ---------------------------------------------------------------------


def test_allow_list_rejects_other_buckets(monkeypatch):
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", "docs, uploads")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main._read_pdf_from_request(_claim(bucket="secrets")))
    assert caught.value.status_code == 403


def test_allow_list_permits_listed_bucket(monkeypatch):
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", "docs, uploads")
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr(fast_main, "read_object", lambda *a: b"%PDF-")
    pdf_bytes, *_rest = asyncio.run(fast_main._read_pdf_from_request(_claim(bucket="uploads")))
    assert pdf_bytes == b"%PDF-"


@pytest.mark.parametrize("value", [None, "", " , "])
def test_allow_list_unset_or_empty_is_unrestricted(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PARSER_ALLOWED_BUCKETS", raising=False)
    else:
        monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", value)
    assert fast_main._allowed_buckets() is None


# --- cleanup on corrupt input --------------------------------------------------------------


class _RecordingTracer:
    """Counts span entries and exits so a leaked span is observable without the SDK."""

    def __init__(self):
        self.entered = 0
        self.exited = 0

    def start_as_current_span(self, *_a, **_kw):
        tracer = self

        class Span:
            def __enter__(self):
                tracer.entered += 1
                return self

            def __exit__(self, *_exc):
                tracer.exited += 1
                return False

            def set_attribute(self, *_a):
                pass

        return Span()


@pytest.mark.parametrize(
    "call",
    [
        lambda: service.parse_pdf(b"not a pdf", "bad.pdf"),
        lambda: service.parse_pdf_page_range(b"not a pdf", "bad.pdf"),
    ],
)
def test_corrupt_pdf_raises_and_closes_its_span(monkeypatch, call):
    recorder = _RecordingTracer()
    monkeypatch.setattr(service, "tracer", recorder)
    with pytest.raises(Exception):
        call()
    assert recorder.entered == 1
    assert recorder.exited == 1
