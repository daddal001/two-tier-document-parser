"""Behavior under the real server: path-prefixing proxies and the container log stream."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SRC = str(Path(__file__).parents[2] / "src")

sys.path.insert(0, SRC)

from two_tier_parser.fast import main as fast_main


def test_route_path_strips_the_root_path_prefix():
    assert fast_main._route_path({"path": "/parser/parse", "root_path": "/parser"}) == "/parse"
    assert fast_main._route_path({"path": "/parser/parse/", "root_path": "/parser/"}) == "/parse"
    assert fast_main._route_path({"path": "/parse", "root_path": ""}) == "/parse"
    assert fast_main._route_path({"path": "/parse"}) == "/parse"
    # Only a real path-segment prefix is stripped.
    assert fast_main._route_path({"path": "/parserx/parse", "root_path": "/parser"}) == (
        "/parserx/parse"
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:  # nosec B310 - local test URL
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _post_ignoring_early_close(base: str, path: str, body: bytes) -> int:
    """POST a body; a server that answers 413 and closes early still yields its response."""
    import http.client

    connection = http.client.HTTPConnection(base.split("//")[1], timeout=90)
    try:
        connection.putrequest("POST", path)
        connection.putheader("content-type", "application/json")
        connection.putheader("content-length", str(len(body)))
        connection.endheaders()
        try:
            connection.send(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        return connection.getresponse().status
    finally:
        connection.close()


@pytest.fixture(params=["", "/parser"], ids=["no-prefix", "root-path"])
def server(request, tmp_path):
    port = _free_port()
    log = tmp_path / "server.log"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "two_tier_parser.fast.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if request.param:
        command += ["--root-path", request.param]
    env = {**os.environ, "PYTHONPATH": SRC, "WORKERS": "1", "PAGES_WORKERS": "1"}
    for name in ("MINIO_ENDPOINT", "OBJECT_STORE_ENDPOINT", "LOG_FORMAT", "LOG_LEVEL"):
        env.pop(name, None)
    with log.open("wb") as sink:
        process = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, env=env
        )  # nosec B603
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(240):
                try:
                    if _request(base + "/health") == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.5)
            else:
                pytest.fail(log.read_text())
            yield base, log
        finally:
            process.terminate()
            process.wait(timeout=30)


def test_body_limit_applies_behind_a_path_prefix(server):
    base, _ = server
    body = json.dumps({"bucket": "b", "key": "a.pdf", "filename": "a.pdf", "pad": "x" * 2_100_000})
    status = _post_ignoring_early_close(base, "/parse", body.encode())
    assert status == 413


def test_every_server_log_line_is_json(server):
    base, log = server
    _request(base + "/health")
    _request(base + "/parse", b"{}", {"content-type": "application/json"})
    time.sleep(0.5)
    lines = [line for line in log.read_text().splitlines() if line.strip()]
    assert lines
    for line in lines:
        json.loads(line)  # raises on any plain-text line, including uvicorn's access log
    messages = " ".join(json.loads(line)["message"] for line in lines)
    assert "GET" in messages and "/health" in messages  # the access log is present, as JSON
