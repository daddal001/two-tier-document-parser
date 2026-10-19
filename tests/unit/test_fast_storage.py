"""Object-store settings, standalone health and the unconfigured claim-check path."""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from two_tier_parser.fast import main as fast_main
from two_tier_parser.fast import storage

ALL_SETTINGS = (
    "OBJECT_STORE_ENDPOINT",
    "OBJECT_STORE_ACCESS_KEY",
    "OBJECT_STORE_SECRET_KEY",
    "OBJECT_STORE_SECURE",
    "OBJECT_STORE_REGION",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    "MINIO_SECURE",
)


@pytest.fixture(autouse=True)
def _no_store(monkeypatch):
    for name in ALL_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(storage, "_client", None)


def test_nothing_set_means_standalone():
    assert storage.storage_configured() is False


@pytest.mark.parametrize("name", ["OBJECT_STORE_ENDPOINT", "MINIO_ENDPOINT", "MINIO_ROOT_USER"])
def test_any_store_setting_marks_storage_configured(monkeypatch, name):
    monkeypatch.setenv(name, "x")
    assert storage.storage_configured() is True


def test_legacy_minio_names_still_work(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "legacy:9000")
    monkeypatch.setenv("MINIO_ROOT_USER", "user")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "password")
    monkeypatch.setenv("MINIO_SECURE", "TRUE")
    settings = storage.load_settings()
    assert (settings.endpoint, settings.access_key, settings.secret_key) == (
        "legacy:9000",
        "user",
        "password",
    )
    assert settings.secure is True


def test_object_store_names_take_precedence(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "legacy:9000")
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "s3.example.com")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "old")
    monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY", "new")
    monkeypatch.setenv("OBJECT_STORE_REGION", "auto")
    settings = storage.load_settings()
    assert settings.endpoint == "s3.example.com"
    assert settings.access_key == "new"
    assert settings.region == "auto"


def test_default_endpoint_applies_once_credentials_are_set(monkeypatch):
    monkeypatch.setenv("MINIO_ROOT_USER", "user")
    assert storage.load_settings().endpoint == storage.DEFAULT_ENDPOINT


def test_read_without_store_raises_specific_error():
    with pytest.raises(storage.StorageNotConfigured):
        storage.read_object("bucket", "key.pdf")


def test_connectivity_check_is_false_without_store():
    assert storage.check_connectivity() is False


def test_health_is_healthy_without_a_store(monkeypatch):
    def forbidden(*_a, **_kw):
        pytest.fail("No probe should run when no store is configured")

    monkeypatch.setattr(fast_main, "check_connectivity", forbidden)
    response = asyncio.run(fast_main.health())
    assert response.status == "healthy"
    assert response.storage_configured is False
    assert response.minio_connected is False


@pytest.mark.parametrize(("reachable", "status"), [(True, "healthy"), (False, "degraded")])
def test_health_with_a_store_reflects_reachability(monkeypatch, reachable, status):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.example.com")
    monkeypatch.setattr(fast_main, "check_connectivity", lambda: reachable)
    response = asyncio.run(fast_main.health())
    assert response.status == status
    assert response.storage_configured is True
    assert response.minio_connected is reachable


def _json_request(body: bytes) -> Request:
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
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive,
    )


def test_claim_check_without_store_is_501_with_guidance():
    request = _json_request(b'{"bucket": "b", "key": "k.pdf"}')
    with pytest.raises(HTTPException) as caught:
        asyncio.run(fast_main._read_pdf_from_request(request))
    assert caught.value.status_code == 501
    assert "multipart" in caught.value.detail
