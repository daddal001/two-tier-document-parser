"""Numeric settings read from the environment."""

import subprocess
import sys

import pytest

from two_tier_parser.common.env import env_number

NAME = "TEST_SETTING"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_unset_or_empty_value_means_the_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(NAME, raising=False)
    else:
        monkeypatch.setenv(NAME, value)  # what a templated deployment produces for "no value"

    assert env_number(NAME, 7, int) == 7


@pytest.mark.parametrize(("kind", "text", "expected"), [(int, "12", 12), (float, " 2.5 ", 2.5)])
def test_a_valid_value_is_converted(monkeypatch, kind, text, expected):
    monkeypatch.setenv(NAME, text)

    assert env_number(NAME, 0, kind) == expected


@pytest.mark.parametrize("text", ["soon", "1.5", "0x10"])
def test_a_malformed_integer_names_the_variable(monkeypatch, text):
    monkeypatch.setenv(NAME, text)

    with pytest.raises(ValueError, match=NAME):
        env_number(NAME, 1, int)


def test_a_value_below_the_minimum_names_the_variable(monkeypatch):
    monkeypatch.setenv(NAME, "0")

    with pytest.raises(ValueError, match=f"{NAME} must be at least 1"):
        env_number(NAME, 5, int, minimum=1)


def test_a_value_at_the_minimum_is_accepted(monkeypatch):
    monkeypatch.setenv(NAME, "1")

    assert env_number(NAME, 5, int, minimum=1) == 1


def import_service(value):
    """Import the fast service in a fresh interpreter with the deadline setting as given."""
    env = {"PATH": "", "OTEL_SDK_DISABLED": "true"}
    if value is not None:
        env["FAST_PARSER_WALL_DEADLINE_S"] = value
    program = "from two_tier_parser.fast import service; print(service.FAST_PARSER_WALL_DEADLINE_S)"
    return subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )


@pytest.mark.parametrize(("value", "expected"), [(None, "1500.0"), ("", "1500.0"), ("90", "90.0")])
def test_the_wall_clock_deadline_setting(value, expected):
    result = import_service(value)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


def test_a_zero_deadline_disables_it():
    assert import_service("0").stdout.strip().splitlines()[-1] == "0.0"


def test_a_malformed_deadline_stops_startup_with_the_variable_name():
    result = import_service("soon")

    assert result.returncode != 0
    assert "FAST_PARSER_WALL_DEADLINE_S must be a number" in result.stderr
