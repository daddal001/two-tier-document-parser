"""An unreachable trace collector must not hold the process open at exit."""

import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from two_tier_parser.common import tracing

NO_SETTINGS = [
    "OTEL_EXPORTER_OTLP_TIMEOUT",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    "OTEL_BSP_EXPORT_TIMEOUT",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in NO_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_the_exporter_gets_a_short_timeout_unless_the_operator_set_one():
    assert tracing.exporter_kwargs() == {"timeout": tracing.DEFAULT_TIMEOUT_SECONDS}
    assert tracing.DEFAULT_TIMEOUT_SECONDS <= 5  # the SDK's own default is 10 s, with retries


@pytest.mark.parametrize(
    "name", ["OTEL_EXPORTER_OTLP_TIMEOUT", "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"]
)
def test_an_exporter_timeout_set_by_the_operator_is_left_alone(monkeypatch, name):
    monkeypatch.setenv(name, "20")

    assert tracing.exporter_kwargs() == {}  # the SDK reads the variable itself


def test_the_batch_processor_gets_a_short_export_timeout_unless_set():
    assert tracing.processor_kwargs() == {
        "export_timeout_millis": int(tracing.DEFAULT_TIMEOUT_SECONDS * 1000)
    }


def test_a_batch_timeout_set_by_the_operator_is_left_alone(monkeypatch):
    monkeypatch.setenv("OTEL_BSP_EXPORT_TIMEOUT", "30000")

    assert tracing.processor_kwargs() == {}


CHILD = """
import os
os.environ["OTEL_SERVICE_NAME"] = "probe"
from {module} import main
from opentelemetry import trace
with trace.get_tracer("probe").start_as_current_span("work"):
    pass
print("exiting", flush=True)
"""


def exit_time(service, collector):
    """Seconds the process takes to start, trace once, and exit with the collector as given."""
    listener = None
    if collector == "silent":  # accepts connections and never speaks
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        accepted = []
        threading.Thread(
            target=lambda: [accepted.append(listener.accept()) for _ in iter(int, 1)], daemon=True
        ).start()
    else:  # a port nothing listens on
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
    env = {
        **os.environ,
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{port}",
        "WORKERS": "1",
    }
    env.pop("OTEL_SDK_DISABLED", None)
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", CHILD.format(module=service)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    elapsed = time.monotonic() - started
    if listener:
        listener.close()
    assert done.returncode == 0, done.stderr[-500:]
    return elapsed


@pytest.mark.parametrize("collector", ["silent", "refused"])
@pytest.mark.parametrize("service", ["two_tier_parser.fast", "two_tier_parser.accurate"])
def test_exit_is_prompt_when_the_collector_cannot_be_reached(service, collector):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")

    assert exit_time(service, collector) < 6.5  # the baseline is about 1 s; it was 8 to 11 s
