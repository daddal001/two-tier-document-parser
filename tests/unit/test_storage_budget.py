"""A slow or stalled object-store read must end on a budget, free its slot, and not poison the next."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from two_tier_parser.fast import storage

BODY = b"%PDF-1.4 " + b"x" * 20_000


class Behaviour:
    """What the fake object store does; a test sets ``mode`` before reading."""

    mode = "normal"
    requests = 0
    stop = threading.Event()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        Behaviour.requests += 1
        mode = Behaviour.mode
        if mode == "unavailable":  # a 503 that asks the client to come back much later
            self.send_response(503)
            self.send_header("Retry-After", "120")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode == "silent":  # accepts the request and never answers
            Behaviour.stop.wait(60)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.send_header("ETag", '"abc"')
        self.end_headers()
        if mode == "normal":
            self.wfile.write(BODY)
        elif mode == "trickle":  # a byte every 0.1 s: always progressing, never finishing
            for index in range(len(BODY)):
                if Behaviour.stop.is_set():
                    return
                try:
                    self.wfile.write(BODY[index : index + 1])
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.1)
        elif mode == "stall_after_headers":  # sends the headers, then nothing
            Behaviour.stop.wait(60)


@pytest.fixture
def store(monkeypatch):
    Behaviour.mode = "normal"
    Behaviour.requests = 0
    Behaviour.stop = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", f"127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY", "key")
    monkeypatch.setenv("OBJECT_STORE_SECRET_KEY", "secret")
    monkeypatch.setenv("OBJECT_STORE_REGION", "us-east-1")  # skips the bucket-location lookup
    monkeypatch.setenv("OBJECT_STORE_CONNECT_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("OBJECT_STORE_READ_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("OBJECT_STORE_RETRIES", "0")
    monkeypatch.setattr(storage, "_client", None)
    yield Behaviour
    Behaviour.stop.set()
    server.shutdown()
    server.server_close()


def test_a_normal_read_returns_the_object(store):
    assert storage.read_object("bucket", "doc.pdf", 1_000_000) == BODY


def test_a_trickling_read_ends_at_the_budget_instead_of_running_on(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "1")
    store.mode = "trickle"  # 20,000 bytes at 10 per second would take over half an hour

    started = time.monotonic()
    with pytest.raises(storage.StorageTimeout):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert time.monotonic() - started < 4


def test_a_server_that_sends_headers_then_stalls_ends_at_the_read_timeout(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "30")
    store.mode = "stall_after_headers"

    started = time.monotonic()
    with pytest.raises(Exception):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert time.monotonic() - started < 6  # the 1 second socket timeout, not the 300 of the SDK


def test_a_server_that_never_answers_ends_at_the_read_timeout(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "30")
    store.mode = "silent"

    started = time.monotonic()
    with pytest.raises(Exception):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert time.monotonic() - started < 6


def test_the_next_read_succeeds_after_a_stalled_one(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "1")
    store.mode = "trickle"
    with pytest.raises(storage.StorageTimeout):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    store.mode = "normal"

    assert storage.read_object("bucket", "doc.pdf", 1_000_000) == BODY


def test_retries_are_bounded_by_the_setting(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_RETRIES", "1")
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "30")
    store.mode = "silent"

    with pytest.raises(Exception):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert store.requests <= 2  # the first attempt and at most one retry, not the SDK's five


def test_the_size_limit_still_applies_to_a_streamed_body(store):
    with pytest.raises(storage.ObjectTooLarge):
        storage.read_object("bucket", "doc.pdf", 1_000)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("OBJECT_STORE_READ_BUDGET_SECONDS", "soon", "OBJECT_STORE_READ_BUDGET_SECONDS"),
        ("OBJECT_STORE_READ_TIMEOUT_SECONDS", "0", "OBJECT_STORE_READ_TIMEOUT_SECONDS"),
        ("OBJECT_STORE_RETRIES", "-1", "OBJECT_STORE_RETRIES"),
    ],
)
def test_bad_settings_are_rejected_by_name(store, monkeypatch, name, value, match):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=match):
        storage.read_object("bucket", "doc.pdf", 1_000_000)


def test_the_defaults_are_far_below_the_sdks(monkeypatch):
    for name in (
        "OBJECT_STORE_CONNECT_TIMEOUT_SECONDS",
        "OBJECT_STORE_READ_TIMEOUT_SECONDS",
        "OBJECT_STORE_RETRIES",
        "OBJECT_STORE_READ_BUDGET_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = storage.read_limits()

    assert settings.connect_timeout <= 10 and settings.read_timeout <= 60
    assert settings.retries <= 3 and settings.budget <= 300  # the SDK's: 300 s and five retries


# ---- the budget bounds the phase before any bytes arrive, too ----


def test_the_socket_timeouts_are_clamped_so_attempts_fit_inside_the_budget(monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_CONNECT_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("OBJECT_STORE_READ_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OBJECT_STORE_RETRIES", "2")
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "10")

    limits = storage.read_limits()

    # three attempts, each connecting and then waiting for headers, must fit in the budget
    assert (limits.connect_timeout + limits.read_timeout) * (limits.retries + 1) <= limits.budget
    assert limits.budget == 10  # the budget itself is never changed


def test_the_defaults_fit_the_budget_without_any_clamping(monkeypatch):
    for name in (
        "OBJECT_STORE_CONNECT_TIMEOUT_SECONDS",
        "OBJECT_STORE_READ_TIMEOUT_SECONDS",
        "OBJECT_STORE_RETRIES",
        "OBJECT_STORE_READ_BUDGET_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    limits = storage.read_limits()

    assert (limits.connect_timeout, limits.read_timeout, limits.retries, limits.budget) == (
        5.0,
        20.0,
        2,
        100.0,
    )


def test_a_store_that_stalls_before_the_headers_ends_inside_a_small_budget(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_CONNECT_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OBJECT_STORE_READ_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OBJECT_STORE_RETRIES", "1")
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "2")
    store.mode = "silent"

    started = time.monotonic()
    with pytest.raises(Exception):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert time.monotonic() - started < 4  # not the 30 second socket timeouts


def test_a_long_retry_after_from_the_store_is_not_waited_for(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_RETRIES", "2")
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "30")
    store.mode = "unavailable"

    started = time.monotonic()
    with pytest.raises(Exception):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert time.monotonic() - started < 6  # the store asked for 120 seconds


def test_the_object_is_returned_without_a_second_copy_of_it(store):
    data = storage.read_object("bucket", "doc.pdf", 1_000_000)

    assert isinstance(data, bytearray)  # built in place, so the peak is one copy plus a chunk
    assert data == BODY


def test_retry_back_off_fits_in_the_budget_even_with_many_retries(store, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_RETRIES", "10")
    monkeypatch.setenv("OBJECT_STORE_READ_BUDGET_SECONDS", "2")
    store.mode = "unavailable"  # immediate 503s: nothing but the back-off sleeps takes time

    started = time.monotonic()
    with pytest.raises(Exception):
        storage.read_object("bucket", "doc.pdf", 1_000_000)

    # ten retries with urllib3's default exponential back-off would sleep for minutes
    assert time.monotonic() - started < 4
    assert store.requests == 11  # every retry still happened; only the sleeping was bounded
