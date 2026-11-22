"""Trivy scans of the repository files and the images, and OSV scans of the pinned files.

They run with the rest of the suite, so a bare ``pytest`` or ``make test`` includes them. They
need a Docker daemon and a network (the scanners download their vulnerability databases), and a
newly published advisory can fail them without any change in this repository, which is why the
pre-commit test hook leaves them out (``-m "not security"``) and GitHub Actions does not run
them. ``make test-security`` runs only these; ``pytest -m "not security"`` leaves them out.

Environment:
    TRIVY_IMAGE         The Trivy image to scan with (default: the version pinned in test_trivy.py).
    OSV_IMAGE           The OSV-Scanner image to scan with (default: the version pinned in test_osv.py).
    FAST_IMAGE          Scan this prebuilt fast image instead of building deploy/Dockerfile.fast.
    ACCURATE_IMAGE      Scan this prebuilt CPU image instead of building deploy/Dockerfile.accurate.cpu.
    GPU_IMAGE           Also scan this CUDA image (about 45 GB; never built here).
"""

import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_docker():
    try:
        subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("a Docker daemon is required")
