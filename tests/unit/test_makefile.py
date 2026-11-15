"""The Makefile drives the profile-based Compose file; its service names must exist."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def dry_run(*arguments):
    result = subprocess.run(
        ["make", "-n", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def compose_services():
    return set(yaml.safe_load((ROOT / "deploy" / "docker-compose.yml").read_text())["services"])


@pytest.mark.parametrize("profile", ["cuda", "cpu"])
def test_build_accurate_names_a_service_that_exists(profile):
    output = dry_run("build-accurate", f"ACCURATE_PROFILE={profile}")

    assert f"--profile {profile}" in output
    service = output.split(" build ")[-1].split()[0]
    assert service == f"accurate-parser-{profile}"
    assert service in compose_services()


@pytest.mark.parametrize("target", ["build", "up", "logs"])
def test_compose_targets_select_the_accurate_profile(target):
    assert "--profile cpu" in dry_run(target, "ACCURATE_PROFILE=cpu")


@pytest.mark.parametrize("target", ["down", "clean"])
def test_teardown_targets_cover_both_profiles(target):
    output = dry_run(target, "ACCURATE_PROFILE=cpu")

    assert "--profile cuda" in output and "--profile cpu" in output


def test_the_default_profile_follows_the_presence_of_an_nvidia_gpu():
    output = dry_run("build-accurate")

    assert ("--profile cuda" in output) == (shutil.which("nvidia-smi") is not None)
