"""JSON logging: line shape, protected fields, level handling and failure safety."""

import json
import logging
import re

import pytest

from two_tier_parser.common.json_logging import (
    JsonFormatter,
    configure_json_logging,
    resolve_level,
    safe_log_value,
)


def _record(message="hello", level=logging.INFO, exc_info=None, **extra):
    record = logging.LogRecord("test.logger", level, __file__, 1, message, None, exc_info)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def _format(record):
    line = JsonFormatter().format(record)
    assert "\n" not in line
    return json.loads(line)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("LOG_LEVEL", "LOG_FORMAT", "SERVICE_NAME", "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)


def test_canonical_fields_and_timestamp_shape():
    payload = _format(_record("hello %s" % "world"))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["service_name"] == "two-tier-parser"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", payload["timestamp"])


def test_extra_fields_are_included():
    payload = _format(_record(task_id="t-1", duration_ms=12))
    assert payload["task_id"] == "t-1"
    assert payload["duration_ms"] == 12


def test_extras_cannot_overwrite_canonical_fields():
    forged = _record("real", level=logging.ERROR)
    forged.__dict__.update({"timestamp": "forged", "level": "DEBUG", "message": "forged"})
    payload = _format(forged)
    assert payload["level"] == "ERROR"
    assert payload["message"] == "real"
    assert payload["timestamp"] != "forged"


def test_service_name_from_environment_and_record(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-otel")
    assert _format(_record())["service_name"] == "from-otel"
    monkeypatch.setenv("SERVICE_NAME", "explicit")
    assert _format(_record())["service_name"] == "explicit"
    assert _format(_record(service_name="fast-parser"))["service_name"] == "explicit"


def test_record_service_name_is_used_when_no_environment_override():
    assert _format(_record(service_name="fast-parser"))["service_name"] == "fast-parser"


def test_extras_cannot_forge_trace_ids():
    payload = _format(_record(trace_id="forged", span_id="forged"))
    assert "trace_id" not in payload and "span_id" not in payload


def test_exception_is_one_escaped_field():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        payload = _format(_record("failed", level=logging.ERROR, exc_info=sys.exc_info()))
    assert "ValueError: boom" in payload["exception"]


def test_unserializable_extra_does_not_drop_the_line():
    payload = _format(_record(blob=object()))
    assert payload["blob"].startswith("<object object")


def test_unencodable_extras_fall_back_to_canonical_fields():
    circular = []
    circular.append(circular)
    payload = _format(_record("kept", data={1: circular}))
    assert payload["message"] == "kept"
    assert "log_error" in payload
    assert "data" not in payload


def test_newlines_in_message_stay_on_one_line():
    line = JsonFormatter().format(_record("a\nb\r\nc"))
    assert "\n" not in line and "\r" not in line
    assert json.loads(line)["message"] == "a\nb\r\nc"


def test_trace_ids_present_only_inside_a_span():
    from opentelemetry import trace

    assert "trace_id" not in _format(_record())
    context = trace.SpanContext(
        trace_id=0xABC123, span_id=0xDEF456, is_remote=False, trace_flags=trace.TraceFlags(1)
    )
    with trace.use_span(trace.NonRecordingSpan(context)):
        payload = _format(_record())
    assert payload["trace_id"] == format(0xABC123, "032x")
    assert payload["span_id"] == format(0xDEF456, "016x")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("info", logging.INFO),
        ("INFO", logging.INFO),
        (" Debug ", logging.DEBUG),
        ("warning", logging.WARNING),
        ("20", 20),
        (10, 10),
        ("", logging.INFO),
        (None, logging.INFO),
        ("nonsense", logging.INFO),
    ],
)
def test_resolve_level(value, expected):
    assert resolve_level(value) == expected


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield root
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


@pytest.mark.parametrize("spelling", ["info", "INFO", "Info"])
def test_log_level_is_case_insensitive_and_output_is_upper_case(clean_root, monkeypatch, spelling):
    clean_root.handlers.clear()
    monkeypatch.setenv("LOG_LEVEL", spelling)
    configure_json_logging()
    assert clean_root.level == logging.INFO
    assert isinstance(clean_root.handlers[0].formatter, JsonFormatter)
    assert _format(_record())["level"] == "INFO"


def test_unknown_log_level_falls_back_to_info(clean_root, monkeypatch):
    clean_root.handlers.clear()
    monkeypatch.setenv("LOG_LEVEL", "loud")
    configure_json_logging()
    assert clean_root.level == logging.INFO


def test_text_format_is_opt_in(clean_root, monkeypatch):
    clean_root.handlers.clear()
    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_json_logging()
    assert not isinstance(clean_root.handlers[0].formatter, JsonFormatter)


def test_existing_host_configuration_wins(clean_root):
    clean_root.handlers.clear()
    host_handler = logging.NullHandler()
    clean_root.addHandler(host_handler)
    configure_json_logging("debug")
    assert clean_root.handlers == [host_handler]


def test_safe_log_value_escapes_line_breaks():
    assert safe_log_value("a\nb\rc") == "a\\nb\\rc"
    assert safe_log_value(None) is None
    assert safe_log_value(5) == "5"
