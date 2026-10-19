"""Versions, interpreter-mode reporting, OTLP gating and pool backpressure."""

import asyncio
import importlib
import sys
import tomllib
from pathlib import Path

import pymupdf
import pymupdf4llm
import pytest
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

import two_tier_parser
from two_tier_parser.fast import main as fast_main
from two_tier_parser.fast import service
from two_tier_parser.fast.core import telemetry


def _reload_main():
    """Reload the app module, first releasing the pools the previous load created."""
    fast_main.executor.shutdown(wait=False, cancel_futures=True)
    fast_main.pages_executor.shutdown(wait=False, cancel_futures=True)
    return importlib.reload(fast_main)


def test_package_version_matches_pyproject():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert two_tier_parser.__version__ == declared


def test_app_version_is_the_api_version():
    assert fast_main.app.version == two_tier_parser.API_VERSION


def test_metadata_version_is_the_installed_pymupdf4llm():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello")
    data = doc.tobytes()
    doc.close()
    expected = pymupdf4llm.__version__
    assert service.parse_pdf(data, "d.pdf")["metadata"]["version"] == expected
    assert service.parse_pdf_page_range(data, "d.pdf")["metadata"]["version"] == expected


@pytest.mark.parametrize(("gil_enabled", "no_gil"), [(True, False), (False, True)])
def test_no_gil_follows_the_interpreter(monkeypatch, gil_enabled, no_gil):
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: gil_enabled, raising=False)
    try:
        module = _reload_main()
        assert module.NO_GIL is no_gil
    finally:
        monkeypatch.undo()
        _reload_main()


def test_no_gil_is_false_when_the_interpreter_cannot_say(monkeypatch):
    monkeypatch.delattr(sys, "_is_gil_enabled", raising=False)
    try:
        assert _reload_main().NO_GIL is False
    finally:
        monkeypatch.undo()
        _reload_main()


# --- OTLP export is opt-in -------------------------------------------------------------------


@pytest.fixture
def instrumented(monkeypatch):
    calls = []

    class Instrumentor:
        @staticmethod
        def instrument_app(app):
            calls.append(app)

    monkeypatch.setattr(telemetry, "FastAPIInstrumentor", Instrumentor, raising=False)
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    for name in (
        "OTEL_SDK_DISABLED",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_no_exporter_is_configured_without_an_endpoint(instrumented, monkeypatch):
    from opentelemetry import trace

    def forbidden(_provider):
        pytest.fail("a tracer provider must not be installed without an endpoint")

    monkeypatch.setattr(trace, "set_tracer_provider", forbidden)
    assert telemetry.setup_telemetry(FastAPI()) is True
    assert len(instrumented) == 1


def test_sdk_disabled_skips_all_setup(instrumented, monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    assert telemetry.setup_telemetry(FastAPI()) is False
    assert instrumented == []


@pytest.mark.parametrize(
    "variable", ["OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
)
def test_explicit_endpoint_installs_a_provider(instrumented, monkeypatch, variable):
    pytest.importorskip("opentelemetry.sdk.trace")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    from opentelemetry import trace

    installed = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    monkeypatch.setenv(variable, "http://collector:4317")
    assert telemetry.setup_telemetry(FastAPI()) is True
    assert len(installed) == 1


# --- pool backpressure -----------------------------------------------------------------------


def _blank_request():
    from starlette.requests import Request

    return Request(
        {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    )


def test_parse_pages_sheds_load_when_its_pool_is_deep(monkeypatch):
    limit = fast_main.PAGES_BACKPRESSURE_FACTOR * fast_main.PAGES_WORKERS
    monkeypatch.setattr(fast_main, "_pages_executor_queue_depth", lambda: limit + 1)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main.parse_pages(_blank_request(), start_page=0, end_page=None))
    assert caught.value.status_code == 503
    assert caught.value.detail == "parse_pages_pool_saturated"


def test_parse_sheds_load_when_the_shared_pool_is_deep(monkeypatch):
    limit = fast_main.PARSE_BACKPRESSURE_FACTOR * fast_main.WORKERS
    monkeypatch.setattr(fast_main, "_executor_queue_depth", lambda: limit + 1)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main.parse(_blank_request()))
    assert caught.value.status_code == 503
    assert caught.value.detail == "parse_pool_saturated"


@pytest.mark.parametrize("offset", [0, -1])
def test_parse_admits_work_at_or_below_the_limit(monkeypatch, offset):
    limit = fast_main.PARSE_BACKPRESSURE_FACTOR * fast_main.WORKERS
    monkeypatch.setattr(fast_main, "_executor_queue_depth", lambda: limit + offset)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main.parse(_blank_request()))
    # Admitted: it fails later on the missing content type, not on backpressure.
    assert caught.value.status_code == 415


def test_queue_depth_probe_fails_open(monkeypatch):
    class NoQueue:
        pass

    monkeypatch.setattr(fast_main, "executor", NoQueue())
    monkeypatch.setattr(fast_main, "pages_executor", NoQueue())
    assert fast_main._executor_queue_depth() == 0
    assert fast_main._pages_executor_queue_depth() == 0


def test_service_starts_and_serves_empty_metrics_without_prometheus(monkeypatch):
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    try:
        module = _reload_main()
        assert module._PROM_OK is False
        module.PARSE_BACKPRESSURE_503.inc()
        body = asyncio.run(module.metrics())
        assert body.status_code == 200
        assert body.body == b""
    finally:
        monkeypatch.undo()
        _reload_main()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"WORKERS": "abc"}, "WORKERS must be a number"),
        ({"WORKERS": "0"}, "WORKERS must be at least 1"),
        ({"PAGES_WORKERS": "x"}, "PAGES_WORKERS must be a number"),
        ({"OBJECT_STORE_HEALTH_TIMEOUT_SECONDS": "soon"}, "OBJECT_STORE_HEALTH_TIMEOUT_SECONDS"),
    ],
)
def test_bad_sizing_settings_name_the_variable(monkeypatch, env, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    try:
        with pytest.raises(ValueError, match=message):
            _reload_main()
    finally:
        monkeypatch.undo()
        _reload_main()


def test_empty_sizing_settings_mean_unset(monkeypatch):
    monkeypatch.setenv("WORKERS", "")
    monkeypatch.setenv("MAX_INFLIGHT_DOCUMENTS", " ")
    try:
        module = _reload_main()
        assert module.WORKERS >= 1 and module.MAX_INFLIGHT_DOCUMENTS >= 2
    finally:
        monkeypatch.undo()
        _reload_main()
