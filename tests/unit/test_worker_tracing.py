"""Tracing in pool workers, whatever way the platform starts them.

Linux forks workers, which then inherit the parent's tracer provider. Workers started with
spawn or forkserver (macOS, and the default of newer Pythons) start with none, so their spans
were silently dropped. Each worker now sets one up the way the parent does.
"""

import multiprocessing
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")

from tests import worker_failures as probes  # noqa: E402
from two_tier_parser.common.worker import init_worker  # noqa: E402

ENDPOINT = "http://127.0.0.1:9"  # nothing listens; setting it up is what is tested


def pool(method, service="fast-parser"):
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context(method),
        initializer=init_worker,
        initargs=(service,),
    )


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_a_worker_started_without_inheriting_the_parent_gets_a_provider(monkeypatch, method):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", ENDPOINT)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    with pool(method) as workers:
        assert workers.submit(probes.provider_name).result(timeout=120) == "TracerProvider"
        assert workers.submit(probes.service_name).result(timeout=120) == "fast-parser"


def test_the_service_name_setting_is_honoured_in_workers(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "custom-name")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    with pool("spawn") as workers:
        assert workers.submit(probes.service_name).result(timeout=120) == "custom-name"


def test_without_an_endpoint_workers_stay_untraced(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    with pool("spawn") as workers:
        assert workers.submit(probes.provider_name).result(timeout=120) == "ProxyTracerProvider"


def test_a_disabled_sdk_stays_disabled_in_workers(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    with pool("spawn") as workers:
        assert workers.submit(probes.provider_name).result(timeout=120) == "ProxyTracerProvider"


def test_workers_still_log_as_json(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with pool("spawn") as workers:
        assert workers.submit(probes.root_formatter_name).result(timeout=120) == "JsonFormatter"


def run_python(program, **env):
    import os

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    assert result.returncode == 0, result.stderr[-600:]
    return result.stdout.strip().splitlines()[-1]


def test_a_provider_that_is_already_installed_is_not_replaced():
    outcome = run_python(
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from two_tier_parser.common.tracing import configure_tracer_provider

        mine = TracerProvider()
        trace.set_tracer_provider(mine)
        print(configure_tracer_provider("x") is None and trace.get_tracer_provider() is mine)
        """,
        OTEL_EXPORTER_OTLP_ENDPOINT=ENDPOINT,
    )
    assert outcome == "True"  # a forked worker inherits the parent's provider and keeps it


def test_the_provider_is_only_created_when_tracing_was_asked_for():
    outcome = run_python(
        """
        from two_tier_parser.common.tracing import configure_tracer_provider
        print(configure_tracer_provider("x"))
        """,
        OTEL_EXPORTER_OTLP_ENDPOINT="",
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="",
    )
    assert outcome == "None"
