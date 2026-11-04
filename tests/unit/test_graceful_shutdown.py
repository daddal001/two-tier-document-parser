"""What a SIGTERM does to the real server while a request is running."""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[2]
# Current uvicorn finishes its graceful shutdown, then re-raises the signal so the exit status
# says how the process was stopped; older releases exit 0. Both are a clean stop.
CLEAN_EXIT = (0, -signal.SIGTERM)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def small_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "shutdown test")
    data = document.tobytes()
    document.close()
    return data


def upload(port: int, pdf: bytes):
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + pdf
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/parse",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def start_server(extra_env=None):
    port = free_port()
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}",
        "SLOW_PARSE_SECONDS": "4",
        "SHUTDOWN_DRAIN_SECONDS": "1",  # far shorter than the parse that is running
        "OTEL_SDK_DISABLED": "true",
        "WORKERS": "2",
        **(extra_env or {}),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.slow_fast_app:app", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            assert process.poll() is None, "the server exited during startup"
            assert time.monotonic() < deadline, "the server never became healthy"
            time.sleep(0.3)
    return port, process


@pytest.fixture
def server():
    port, process = start_server()
    yield port, process
    if process.poll() is None:
        process.kill()
    process.wait(10)


def test_a_request_already_running_finishes_before_the_server_exits(server):
    port, process = server
    result = {}
    worker = threading.Thread(target=lambda: result.update(status=upload(port, small_pdf())))
    worker.start()
    time.sleep(1.0)  # the parse has started and needs about three more seconds

    stopped = time.monotonic()
    process.send_signal(signal.SIGTERM)
    worker.join(30)
    exit_code = process.wait(30)

    assert result["status"] == 200, "the drain budget must not cut off a request in flight"
    assert exit_code in CLEAN_EXIT
    assert time.monotonic() - stopped >= 2.0  # it really waited for the request to finish


def test_an_idle_server_exits_promptly_on_sigterm(server):
    _, process = server
    started = time.monotonic()
    process.send_signal(signal.SIGTERM)

    assert process.wait(15) in CLEAN_EXIT
    assert time.monotonic() - started < 10


def test_a_request_that_outlives_the_server_drain_budget_is_cut_off_on_request():
    """UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN bounds how long uvicorn waits for running requests."""
    port, process = start_server(
        {"UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN": "1", "SLOW_PARSE_SECONDS": "30"}
    )
    try:
        result = {}
        worker = threading.Thread(target=lambda: result.update(status=upload(port, small_pdf())))
        worker.start()
        time.sleep(1.0)

        stopped = time.monotonic()
        process.send_signal(signal.SIGTERM)
        process.wait(20)

        assert time.monotonic() - stopped < 15  # not the 30 seconds the parse would take
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(10)
