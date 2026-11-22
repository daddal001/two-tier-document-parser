"""No known vulnerability in the pinned dependency files, according to OSV.

Trivy (test_trivy.py) looks inside the built images; OSV looks at what the files pin, so a
vulnerable pin is caught before an image is built. Advisories that have to be accepted go in
``osv-scanner.toml`` with a reason and an ``ignoreUntil`` date.
"""

import datetime
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.security

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "osv-scanner.toml"
# The one place the scanner version lives; Dependabot cannot see it, so raise it by hand.
OSV_IMAGE = os.getenv("OSV_IMAGE", "ghcr.io/google/osv-scanner:v2.7.1")


def scan(*lockfiles):
    """Scan the files and return one readable line per advisory.

    ``requirements.txt:`` marks a pinned list that is not named like one. Resolution is turned
    off because these pins are already the resolved set: resolving them again would apply the
    Pillow cap that the image deliberately lifts, and the scanner would skip the file.
    """
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{ROOT}:/src:ro",
        OSV_IMAGE,
        "scan",
        "source",
        "--no-resolve",
        "--config",
        "/src/osv-scanner.toml",
        "--format",
        "json",
    ]
    for lockfile in lockfiles:
        kind, _, path = lockfile.rpartition(":")
        command += ["-L", f"{kind}:/src/{path}" if kind else f"/src/{path}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    # 0 is "nothing found" and 1 is "found something"; anything else means it did not scan.
    assert result.returncode in (0, 1), f"OSV-Scanner did not run: {result.stderr[-1500:]}"
    # A file it could not read is skipped without failing the run; that must not look clean.
    assert "Error during extraction" not in result.stderr, result.stderr[-1500:]
    lines = []
    for scanned in json.loads(result.stdout or "{}").get("results") or []:
        for package in scanned.get("packages") or []:
            for advisory in package.get("vulnerabilities") or []:
                lines.append(
                    f"{advisory['id']}: {package['package']['name']} "
                    f"{package['package']['version']} in {scanned['source']['path']}"
                )
    return lines


def test_the_lock_file_pins_no_vulnerable_package():
    found = scan("uv.lock")

    assert not found, "\n".join(found)


def test_the_pinned_image_sets_pin_no_vulnerable_package():
    found = scan(
        "requirements.txt:deploy/constraints/accurate-cpu.txt",
        "requirements.txt:deploy/constraints/accurate-gpu.txt",
    )

    assert not found, "\n".join(found)


def test_every_ignored_advisory_has_a_reason_and_a_future_date():
    ignored = tomllib.loads(CONFIG.read_text()).get("IgnoredVulns") or []
    today = datetime.date.today()
    for entry in ignored:
        assert entry.get("reason"), f"{entry.get('id')}: no reason saying why it is accepted"
        until = entry.get("ignoreUntil")
        assert until, f"{entry['id']}: no ignoreUntil date"
        assert until > today, f"{entry['id']}: the exception ended on {until}; fix or renew it"
