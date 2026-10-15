"""Tests for the ``two-tier-parser-demo`` command line client."""

import json
from pathlib import Path

import pytest
import requests

from two_tier_parser import cli


class _Response:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 test")
    return path


FAST_PAYLOAD = {"markdown": "# Title", "metadata": {"pages": 2, "parser": "pymupdf4llm"}}
ACCURATE_PAYLOAD = {
    "markdown": "x" * 600,
    "metadata": {"pages": 1, "parser": "mineru"},
    "images": [{}, {}],
    "tables": [{}],
    "formulas": [],
}


@pytest.mark.parametrize(
    ("mode", "payload", "url"),
    [
        ("fast", FAST_PAYLOAD, cli.FAST_API_URL),
        ("accurate", ACCURATE_PAYLOAD, cli.ACCURATE_API_URL),
    ],
)
def test_parse_file_success_posts_to_mode_endpoint_and_saves_result(
    monkeypatch, pdf, mode, payload, url
):
    calls = []

    def fake_post(target, files, timeout):
        calls.append((target, sorted(files), timeout))
        return _Response(200, payload)

    monkeypatch.setattr(cli.requests, "post", fake_post)

    assert cli.parse_file(pdf, mode, timeout=7) == 0
    assert calls == [(url, ["file"], 7)]
    saved = json.loads(pdf.with_suffix(f".{mode}.json").read_text())
    assert saved == payload


def test_non_200_response_returns_one_and_survives_markup_in_body(monkeypatch, pdf, capsys):
    monkeypatch.setattr(
        cli.requests, "post", lambda *a, **k: _Response(503, text="busy [bold]retry[/bold] [/nope]")
    )

    assert cli.parse_file(pdf, "fast") == 1
    out = capsys.readouterr().out
    assert "503" in out
    assert "[bold]retry[/bold] [/nope]" in out
    assert not pdf.with_suffix(".fast.json").exists()


@pytest.mark.parametrize(
    ("error", "needle"),
    [
        (requests.exceptions.Timeout(), "timed out"),
        (requests.exceptions.ConnectionError("refused [/nope]"), "Connection failed"),
    ],
)
def test_transport_errors_return_one(monkeypatch, pdf, capsys, error, needle):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli.requests, "post", boom)

    assert cli.parse_file(pdf, "fast") == 1
    assert needle in capsys.readouterr().out


def test_main_reports_missing_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", ["prog", str(tmp_path / "missing.pdf")])

    assert cli.main() == 1
    assert "File not found" in capsys.readouterr().out


def test_main_defaults_to_fast_mode_with_600s_timeout(monkeypatch, pdf):
    seen = {}

    def fake_parse(path, mode, timeout):
        seen.update(path=path, mode=mode, timeout=timeout)
        return 0

    monkeypatch.setattr(cli, "parse_file", fake_parse)
    monkeypatch.setattr("sys.argv", ["prog", str(pdf)])

    assert cli.main() == 0
    assert seen == {"path": pdf, "mode": "fast", "timeout": 600}


def test_main_rejects_unknown_mode(monkeypatch, pdf):
    monkeypatch.setattr("sys.argv", ["prog", str(pdf), "--mode", "slow"])

    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
