"""The claim-check bucket allow-list's explicit open mode and warning, and the store health probe."""

import logging
import types

import pytest
from fastapi.testclient import TestClient
from minio.error import S3Error

from two_tier_parser.fast import main as fast_main
from two_tier_parser.fast import storage

# ---- the allow-list ----


@pytest.mark.parametrize("value", ["*", " * ", "docs,*", "*,docs", "  ,*, "])
def test_a_star_is_the_explicit_open_mode(monkeypatch, value):
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", value)

    assert fast_main._allowed_buckets() is None


def test_named_buckets_are_still_a_restriction(monkeypatch):
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", "docs, uploads")

    assert fast_main._allowed_buckets() == {"docs", "uploads"}


def _startup_warnings(caplog):
    return [r for r in caplog.records if "not restricted to specific buckets" in r.getMessage()]


def test_an_unrestricted_store_is_warned_about_at_startup(monkeypatch, caplog):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.delenv("PARSER_ALLOWED_BUCKETS", raising=False)

    with caplog.at_level(logging.WARNING), TestClient(fast_main.app):
        pass

    warnings = _startup_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].setting == "PARSER_ALLOWED_BUCKETS"
    assert "*" in warnings[0].remedy  # the message says how to choose on purpose


@pytest.mark.parametrize("value", ["", "   ", ",,"])
def test_an_empty_setting_counts_as_unset(monkeypatch, caplog, value):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", value)

    with caplog.at_level(logging.WARNING):
        fast_main._warn_if_unrestricted()

    assert len(_startup_warnings(caplog)) == 1


@pytest.mark.parametrize("value", ["docs", "docs,uploads", "*"])
def test_a_deliberate_choice_is_not_warned_about(monkeypatch, caplog, value):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", value)

    with caplog.at_level(logging.WARNING):
        fast_main._warn_if_unrestricted()

    assert _startup_warnings(caplog) == []


def test_a_standalone_service_has_nothing_to_restrict_and_says_nothing(monkeypatch, caplog):
    monkeypatch.delenv("PARSER_ALLOWED_BUCKETS", raising=False)  # and no object store configured

    with caplog.at_level(logging.WARNING):
        fast_main._warn_if_unrestricted()

    assert _startup_warnings(caplog) == []


def test_the_warning_does_not_repeat_bucket_names(monkeypatch, caplog):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY", "key-id-123")
    monkeypatch.delenv("PARSER_ALLOWED_BUCKETS", raising=False)

    with caplog.at_level(logging.WARNING):
        fast_main._warn_if_unrestricted()

    text = " ".join(str(vars(r)) for r in _startup_warnings(caplog))
    assert "key-id-123" not in text and "store.invalid" not in text


# ---- the health probe ----


def s3_error(code):
    return S3Error(code, code, "/", "request", "host", types.SimpleNamespace(status=403))


class ListingClient:
    """Stands in for the MinIO client's ``list_buckets``."""

    outcome = None

    def __init__(self, *args, **kwargs):
        pass

    def list_buckets(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return []


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr("minio.Minio", ListingClient)

    def run(outcome):
        ListingClient.outcome = outcome
        return storage.check_connectivity()

    return run


def test_a_store_that_lists_its_buckets_is_connected(probe):
    assert probe(None) is True


def test_a_credential_that_may_not_list_buckets_still_means_the_store_is_reachable(probe):
    # Reading an object and listing every bucket are different permissions. A store that answers
    # "access denied" is up and has authenticated the caller; only the listing is refused.
    assert probe(s3_error("AccessDenied")) is True


@pytest.mark.parametrize(
    "code", ["InvalidAccessKeyId", "SignatureDoesNotMatch", "InternalError", "SlowDown"]
)
def test_other_store_errors_are_not_connected(probe, code):
    assert probe(s3_error(code)) is False  # wrong credentials, or the store is unwell


@pytest.mark.parametrize("error", [ConnectionError("refused"), TimeoutError("slow"), OSError("x")])
def test_a_store_that_cannot_be_reached_is_not_connected(probe, error):
    assert probe(error) is False


def test_without_a_configured_store_it_is_not_connected(monkeypatch):
    monkeypatch.delenv("OBJECT_STORE_ENDPOINT", raising=False)

    assert storage.check_connectivity() is False
