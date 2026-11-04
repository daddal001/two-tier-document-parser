"""The script that records model revisions."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pin_model_revisions.py"
spec = importlib.util.spec_from_file_location("pin_model_revisions", SCRIPT)
pin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin)

SHAS = {"a/one": "1" * 40, "b/two": "2" * 40}


def test_the_manifest_lists_each_repository_once_in_a_stable_order():
    manifest = pin.build_manifest(["b/two", "a/one", "a/one"], SHAS.__getitem__)

    assert manifest == {"huggingface": {"a/one": "1" * 40, "b/two": "2" * 40}}
    assert list(manifest["huggingface"]) == ["a/one", "b/two"]


def test_drift_reports_only_pins_that_are_no_longer_the_newest():
    manifest = {"huggingface": {"a/one": "1" * 40, "b/two": "0" * 40}}

    assert pin.drift(manifest, SHAS.__getitem__) == [("b/two", "0" * 40, "2" * 40)]


def test_no_drift_when_every_pin_is_the_newest():
    manifest = {"huggingface": dict(SHAS)}

    assert pin.drift(manifest, SHAS.__getitem__) == []


def test_the_bundled_manifest_is_in_the_format_the_script_writes():
    bundled = json.loads(pin.MANIFEST.read_text())

    assert list(bundled) == ["huggingface"]
    assert list(bundled["huggingface"]) == sorted(bundled["huggingface"])
