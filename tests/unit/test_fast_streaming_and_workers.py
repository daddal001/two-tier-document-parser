"""Streaming body limits and worker-process logging."""

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from two_tier_parser.fast import main as fast_main

SRC = str(Path(__file__).parents[2] / "src")


def _post(path, headers, chunks, consumed, complete=False):
    """Drive the ASGI app with a body that arrives in chunks; record how many were read."""
    sent = []
    iterator = iter(chunks)
    consumed_count = [0]

    async def receive():
        chunk = next(iterator, None)
        if chunk is None:
            return {"type": "http.disconnect"}
        consumed.append(len(chunk))
        consumed_count[0] += 1
        more = consumed_count[0] < len(chunks) or not complete
        return {"type": "http.request", "body": chunk, "more_body": more}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "server": ("test", 80),
        "client": ("test", 1),
        "scheme": "http",
    }
    asyncio.run(fast_main.app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(fast_main, "MAX_PDF_BYTES", 1000)
    monkeypatch.setattr(fast_main, "_MULTIPART_OVERHEAD_BYTES", 100)


MULTIPART = (b"content-type", b"multipart/form-data; boundary=xyz")
PREAMBLE = (
    b'--xyz\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
    b"Content-Type: application/pdf\r\n\r\n"
)


def _upload_chunks(count):
    """A well-formed multipart start followed by file bytes that never reach a closing boundary."""
    return [PREAMBLE + b"x" * 400] + [b"x" * 400] * count


@pytest.mark.parametrize("path", ["/parse", "/parse-pages"])
def test_streamed_upload_without_content_length_is_stopped_early(small_limit, path):
    consumed = []
    status, body = _post(path, [MULTIPART], _upload_chunks(50), consumed)
    assert status == 413
    assert b"File too large" in body
    # 1100 byte limit: the third 400 byte chunk crosses it, so the other 47 are never read.
    assert len(consumed) <= 3


def test_understated_content_length_is_still_stopped(small_limit):
    consumed = []
    status, _ = _post(
        "/parse", [MULTIPART, (b"content-length", b"10")], _upload_chunks(50), consumed
    )
    assert status == 413
    assert len(consumed) <= 3


def test_non_ascii_content_length_is_not_a_server_error(small_limit):
    consumed = []
    status, _ = _post(
        "/parse",
        [MULTIPART, (b"content-length", b"\xb2")],
        [PREAMBLE.replace(b"a.pdf", b"a.txt") + b"x" * 10 + b"\r\n--xyz--\r\n"],
        consumed,
        complete=True,
    )
    assert status == 400  # reached the handler, which rejects the non-PDF name


def test_declared_oversize_is_refused_without_reading_the_body(small_limit):
    consumed = []
    status, _ = _post(
        "/parse", [MULTIPART, (b"content-length", b"5000")], [b"x" * 400] * 5, consumed
    )
    assert status == 413
    assert consumed == []


def test_endless_json_body_is_stopped(monkeypatch):
    monkeypatch.setattr(fast_main, "_JSON_BODY_LIMIT_BYTES", 1000)
    consumed = []
    status, _ = _post(
        "/parse", [(b"content-type", b"application/json")], [b"x" * 400] * 50, consumed
    )
    assert status == 413
    assert len(consumed) <= 3


def test_other_routes_are_not_limited(small_limit):
    consumed = []
    status, _ = _post("/health", [], [b"x" * 400] * 3, consumed)
    assert status != 413


_PROBE = textwrap.dedent("""
    import logging, multiprocessing, sys

    def run(pool):
        logging.getLogger("probe").info("probe-info")
        pool.submit(logging.getLogger("probe").info, "worker-info").result()
        pool.submit(logging.getLogger("probe").warning, "worker-warning").result()

    if __name__ == "__main__":
        # Imported only here, as under uvicorn: a spawned child re-imports this script but
        # must not get its logging from importing the application.
        from two_tier_parser.fast import main

        multiprocessing.set_start_method("spawn")
        run(main.executor)
        run(main.pages_executor)
        # A replacement pool must be configured the same way.
        main._recover_broken_executor(main.executor, stage1=False)
        main.executor.submit(logging.getLogger("probe").info, "replacement-info").result()
        main.executor.shutdown()
        main.pages_executor.shutdown()
    """)


def test_spawned_workers_emit_structured_logs_including_replacement_pools(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(_PROBE)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": SRC, "LOG_LEVEL": "info", "WORKERS": "1", "PAGES_WORKERS": "1"},
    )
    assert result.returncode == 0, result.stderr
    records = []
    for line in result.stderr.splitlines():
        if '"probe"' in line:
            records.append(json.loads(line))  # a plain-text line would fail here
    messages = {r["message"]: r for r in records}
    for expected in ("worker-info", "worker-warning", "replacement-info"):
        assert expected in messages, result.stderr
    assert messages["worker-warning"]["level"] == "WARNING"
    assert messages["worker-info"]["service_name"] == "fast-parser"
