"""The version-sync check: pyproject, the package and the changelog name the same release."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_versions.py"


def load():
    spec = importlib.util.spec_from_file_location("check_versions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PYPROJECT = '[project]\nname = "x"\nversion = "{v}"\n'
INIT = '"""Doc."""\n__version__ = "{v}"\nAPI_VERSION = "1.0.0"\n'
CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n- pending\n\n## [{v}] - 2026-01-01\n\n- released\n"


def project(tmp_path, *, pyproject="1.2.3", package="1.2.3", changelog="1.2.3"):
    (tmp_path / "src" / "two_tier_parser").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(PYPROJECT.format(v=pyproject))
    (tmp_path / "src" / "two_tier_parser" / "__init__.py").write_text(INIT.format(v=package))
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG.format(v=changelog))
    return tmp_path


def test_matching_versions_pass(tmp_path):
    assert load().problems(project(tmp_path)) == []


def test_the_real_repository_is_consistent():
    assert load().problems(ROOT) == []


@pytest.mark.parametrize(
    ("mismatch", "needle"),
    [
        ({"package": "1.2.4"}, "__version__"),
        ({"changelog": "1.2.2"}, "CHANGELOG"),
        ({"pyproject": "1.3.0"}, "pyproject"),
    ],
)
def test_each_mismatch_is_reported_by_name(tmp_path, mismatch, needle):
    found = load().problems(project(tmp_path, **mismatch))

    assert found and any(needle in line for line in found)


def test_an_unreleased_section_alone_is_not_a_release(tmp_path):
    root = project(tmp_path)
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n- pending\n")

    assert any("no released version" in line for line in load().problems(root))


def test_the_api_version_is_not_tied_to_the_package_version(tmp_path):
    root = project(tmp_path)  # API_VERSION is 1.0.0 while the package is 1.2.3

    assert load().problems(root) == []


def run(root):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)], capture_output=True, text=True
    )


def test_the_command_fails_on_a_mismatch_and_says_why(tmp_path):
    result = run(project(tmp_path, package="9.9.9"))

    assert result.returncode == 1
    assert "__version__" in result.stdout + result.stderr


def test_the_command_passes_on_a_consistent_project(tmp_path):
    assert run(project(tmp_path)).returncode == 0
