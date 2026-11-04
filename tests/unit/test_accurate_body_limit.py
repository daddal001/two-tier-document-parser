"""The accurate service refuses an oversized upload while it streams, before reading it all."""

import pytest

from two_tier_parser.accurate import main

from ..asgi_harness import post

MULTIPART = (b"content-type", b"multipart/form-data; boundary=xyz")
PREAMBLE = (
    b'--xyz\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
    b"Content-Type: application/pdf\r\n\r\n"
)


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_PDF_BYTES", 1000)
    monkeypatch.setattr(main, "_MULTIPART_OVERHEAD_BYTES", 100)
    monkeypatch.setattr(main, "_inflight_documents", 0)

    def must_not_parse(*_args):
        raise AssertionError("an oversized upload must never reach the parser")

    monkeypatch.setattr(main, "parse_pdf", must_not_parse)


def _upload_chunks(count):
    """A well-formed multipart start followed by file bytes that never reach a closing boundary."""
    return [PREAMBLE + b"x" * 400] + [b"x" * 400] * count


def test_streamed_upload_without_content_length_is_stopped_early(small_limit):
    consumed = []
    status, body = post(main.app, "/parse", [MULTIPART], _upload_chunks(50), consumed)
    assert status == 413
    assert b"File too large (max 500MB)" in body
    assert len(consumed) <= 3  # the other chunks are never read


def test_understated_content_length_is_still_stopped(small_limit):
    consumed = []
    status, _ = post(
        main.app, "/parse", [MULTIPART, (b"content-length", b"10")], _upload_chunks(50), consumed
    )
    assert status == 413
    assert len(consumed) <= 3


def test_declared_oversize_is_refused_without_reading_the_body(small_limit):
    consumed = []
    status, _ = post(
        main.app,
        "/parse",
        [MULTIPART, (b"content-length", b"5000")],
        [b"x" * 400] * 5,
        consumed,
    )
    assert status == 413
    assert consumed == []


def test_a_non_multipart_body_is_never_streamed_past_the_limit(small_limit, monkeypatch):
    monkeypatch.setattr(main, "_JSON_BODY_LIMIT_BYTES", 1000)
    consumed = []
    status, _ = post(
        main.app, "/parse", [(b"content-type", b"application/json")], [b"x" * 400] * 50, consumed
    )
    assert status in (413, 422)  # the route takes only uploads
    assert sum(consumed) <= 1000 + 400  # at most the limit plus the chunk that crossed it


def test_declared_oversize_json_is_refused_without_reading(small_limit, monkeypatch):
    monkeypatch.setattr(main, "_JSON_BODY_LIMIT_BYTES", 1000)
    consumed = []
    status, _ = post(
        main.app,
        "/parse",
        [(b"content-type", b"application/json"), (b"content-length", b"5000")],
        [b"x" * 400] * 5,
        consumed,
    )
    assert status == 413
    assert consumed == []


def test_other_routes_are_not_limited(small_limit):
    consumed = []
    status, _ = post(main.app, "/health", [], [b"x" * 400] * 3, consumed)
    assert status != 413


def test_the_default_limit_is_500_mib():
    assert main.MAX_PDF_BYTES == 500 * 1024 * 1024
