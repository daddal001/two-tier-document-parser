"""Declared runtime dependencies stay limited to what the code and its images use."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]

# Declared once, imported nowhere, started by nothing: each one added packages to the images and
# to every dependency scan for no purpose. MinerU brings loguru and pydantic-settings along
# itself, where the accurate image needs them.
UNUSED = {"loguru", "pydantic-settings", "gunicorn"}


def runtime_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return {canonicalize_name(Requirement(text).name) for text in data["dependencies"]}


def test_unused_packages_are_not_declared_as_runtime_dependencies():
    assert runtime_dependencies() & UNUSED == set()


def test_nothing_in_the_source_imports_the_removed_packages():
    import re

    pattern = re.compile(r"^\s*(?:import|from)\s+(loguru|pydantic_settings|gunicorn)\b", re.M)
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        # the accurate service imports loguru only to forward MinerU's records, guarded by
        # ImportError, because MinerU (not this package) brings it in
        if path.name != "main.py" and pattern.search(path.read_text())
    ]
    assert offenders == []


def test_the_images_do_not_start_the_removed_server():
    for dockerfile in (ROOT / "deploy").glob("Dockerfile*"):
        assert "gunicorn" not in dockerfile.read_text()
    assert "gunicorn" not in (ROOT / "deploy" / "entrypoint-accurate.sh").read_text()
