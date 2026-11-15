"""The between-pages cancellation check must not cost a timeout per page when Redis is down."""

import time

import pytest

from two_tier_parser.fast import service


class FakeRedis:
    def __init__(self, cancelled=(), fail=False, delay=0.0):
        self.cancelled = set(cancelled)
        self.fail = fail
        self.delay = delay
        self.queries = []

    def exists(self, key):
        self.queries.append(key)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise ConnectionError("redis is down")
        return int(key in self.cancelled)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(service, "_now", fake)
    monkeypatch.setattr(service._redis_backoff, "until", 0.0)
    return fake


def use(monkeypatch, client):
    monkeypatch.setattr(service, "_get_preempt_redis", lambda: client)
    return client


def test_a_cancelled_task_is_reported(clock, monkeypatch):
    use(monkeypatch, FakeRedis(cancelled={"parse_cancel:t1"}))

    assert service._is_cancelled("t1") is True
    assert service._is_cancelled("t2") is False


def test_a_healthy_redis_is_asked_before_every_page(clock, monkeypatch):
    client = use(monkeypatch, FakeRedis())

    for _ in range(5):
        service._is_cancelled("t1")

    assert len(client.queries) == 5


def test_no_task_id_never_touches_redis(clock, monkeypatch):
    client = use(monkeypatch, FakeRedis())

    assert service._is_cancelled(None) is False
    assert client.queries == []


def test_a_failure_means_not_cancelled_and_does_not_raise(clock, monkeypatch):
    use(monkeypatch, FakeRedis(fail=True))

    assert service._is_cancelled("t1") is False


def test_after_one_failure_redis_is_left_alone_for_a_while(clock, monkeypatch):
    client = use(monkeypatch, FakeRedis(fail=True))

    results = [service._is_cancelled("t1") for _ in range(10)]

    assert results == [False] * 10
    assert len(client.queries) == 1  # one page paid for the failure, the other nine did not


def test_redis_is_tried_again_once_the_backoff_has_passed(clock, monkeypatch):
    client = use(monkeypatch, FakeRedis(fail=True))
    service._is_cancelled("t1")
    clock.now += service._REDIS_BACKOFF_SECONDS - 1
    service._is_cancelled("t1")
    assert len(client.queries) == 1

    clock.now += 2
    client.fail = False
    client.cancelled.add("parse_cancel:t1")

    assert service._is_cancelled("t1") is True  # recovered: a cancel is honoured again


def test_a_second_failure_starts_a_fresh_backoff(clock, monkeypatch):
    client = use(monkeypatch, FakeRedis(fail=True))
    service._is_cancelled("t1")
    clock.now += service._REDIS_BACKOFF_SECONDS + 1
    service._is_cancelled("t1")
    service._is_cancelled("t1")

    assert len(client.queries) == 2


def test_a_slow_redis_costs_one_timeout_not_one_per_page(clock, monkeypatch):
    use(monkeypatch, FakeRedis(fail=True, delay=0.15))

    started = time.monotonic()
    for _ in range(10):  # ten pages of a document while Redis is unreachable
        service._is_cancelled("t1")

    assert time.monotonic() - started < 0.6  # without the backoff this is about 1.5 seconds


def test_the_backoff_is_logged_once(clock, monkeypatch, caplog):
    use(monkeypatch, FakeRedis(fail=True))

    with caplog.at_level("WARNING"):
        for _ in range(5):
            service._is_cancelled("t1")

    assert len([r for r in caplog.records if "cancellation checks paused" in r.getMessage()]) == 1
