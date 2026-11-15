"""The commit security gate must fail on invalid downloads or failed analysis."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "codeql_hook", Path(__file__).resolve().parents[2] / "scripts/codeql_precommit.py"
)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def test_bundle_checksum_mismatch_never_publishes_installation(tmp_path, monkeypatch):
    manifest = tmp_path / "bundle.json"
    name = "codeql-bundle-osx64.tar.gz"
    manifest.write_text(
        json.dumps(
            {
                "version": "test",
                "assets": {name: {"url": "https://example.com/bundle", "sha256": "0" * 64}},
            }
        )
    )
    monkeypatch.setattr(hook, "MANIFEST", manifest)
    monkeypatch.setattr(hook, "bundle_name", lambda: name)
    monkeypatch.setattr(hook, "urlopen", lambda *a, **kw: io.BytesIO(b"invalid download"))
    monkeypatch.setenv("CODEQL_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        hook.install_codeql()
    assert not list((tmp_path / "cache").rglob("codeql"))


@pytest.mark.parametrize(
    "runs",
    [[], [{"results": []}], [{"invocations": [{"executionSuccessful": False}], "results": []}]],
)
def test_empty_or_failed_analysis_cannot_pass(tmp_path, runs):
    report = tmp_path / "scan.sarif"
    report.write_text(json.dumps({"runs": runs}))
    with pytest.raises(RuntimeError):
        hook.sarif_findings(report)


def test_warning_findings_are_not_discarded(tmp_path):
    report = tmp_path / "scan.sarif"
    finding = {"ruleId": "py/example", "level": "warning", "message": {"text": "Issue"}}
    report.write_text(
        json.dumps(
            {"runs": [{"invocations": [{"executionSuccessful": True}], "results": [finding]}]}
        )
    )
    assert hook.sarif_findings(report) == [finding]
