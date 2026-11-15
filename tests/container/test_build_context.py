"""What Docker really sends as the build context, against what .dockerignore is meant to send."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.dockerignore import excluded, parse

from .conftest import ROOT, docker

pytestmark = pytest.mark.container


def git_visible_files():
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    # a submodule shows up as one entry; its contents are checked through the path rules
    return sorted(name for name in out.decode().split("\0") if name and (ROOT / name).is_file())


def context_listing():
    """The files Docker puts in the build context, read back from a throwaway image."""
    with tempfile.TemporaryDirectory() as directory:
        dockerfile = Path(directory) / "Dockerfile"
        dockerfile.write_text(
            "FROM python:3.12-slim\nCOPY . /ctx\nRUN cd /ctx && find . -type f | sort > /files.txt\n"
        )
        tag = "two-tier-parser-context-probe"
        docker("build", "-q", "-f", str(dockerfile), "-t", tag, str(ROOT), timeout=900)
        try:
            return set(docker("run", "--rm", "--entrypoint", "cat", tag, "/files.txt").split())
        finally:
            docker("rmi", "-f", tag, check=False)


def test_docker_sends_exactly_what_the_rules_say():
    rules = parse((ROOT / ".dockerignore").read_text())
    actual = {name.removeprefix("./") for name in context_listing()}
    visible = git_visible_files()

    should_be_there = [name for name in visible if not excluded(name, rules)]
    should_not_be_there = [name for name in visible if excluded(name, rules)]

    missing = [name for name in should_be_there if name not in actual]
    leaked = [name for name in should_not_be_there if name in actual]
    assert missing == [], f"left out of the context but needed: {missing[:5]}"
    assert leaked == [], f"sent to the daemon despite the rules: {leaked[:5]}"
    assert "deploy/Dockerfile.fast" in actual and "uv.lock" in actual and "src" in str(actual)


def test_the_context_is_small():
    actual = context_listing()

    assert not any(
        name.startswith(("./.git/", "./.codeql-cache/", "./tests/", "./docs/")) for name in actual
    )
