"""No fixable HIGH or CRITICAL finding in the repository files or in the images.

Findings with no published fix are not counted: nothing can be done about them yet, and they
would fail every run until a fix appears. A finding that has to be accepted goes in
``.trivyignore.yaml`` with a reason and an expiry date.
"""

import datetime
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.security

ROOT = Path(__file__).resolve().parents[2]
IGNORE_FILE = ROOT / ".trivyignore.yaml"
# The one place the scanner version lives; Dependabot cannot see it, so raise it by hand.
TRIVY_IMAGE = os.getenv("TRIVY_IMAGE", "aquasec/trivy:0.76.2")
SEVERITY = "HIGH,CRITICAL"
# Not first-party files: MinerU is its own project, the rest are tool caches and local notes.
SKIPPED_DIRECTORIES = (
    "MinerU",
    ".git",
    ".venv",
    ".venv.nosync",
    ".codeql-cache",
    "docs/standards",
    "research_notes",
)


def trivy(*arguments, mount_repository=False):
    """Run Trivy in its container and return its JSON report."""
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",  # lets it read images from the local daemon
        "-v",
        "trivy-cache:/root/.cache",  # keeps the vulnerability database between runs
        "-v",
        f"{IGNORE_FILE}:/trivyignore.yaml:ro",
    ]
    if mount_repository:
        command += ["-v", f"{ROOT}:/src:ro"]
    command += [
        TRIVY_IMAGE,
        *arguments,
        "--quiet",
        "--severity",
        SEVERITY,
        "--ignore-unfixed",
        "--ignorefile",
        "/trivyignore.yaml",
        "--format",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
    assert result.returncode == 0, f"Trivy did not run: {result.stderr[-1500:]}"
    return json.loads(result.stdout or "{}")


def findings(report):
    """One readable line per vulnerability or failed configuration check."""
    lines = []
    for result in report.get("Results") or []:
        for item in result.get("Vulnerabilities") or []:
            lines.append(
                f"{item['VulnerabilityID']} {item['Severity']}: {item['PkgName']} "
                f"{item['InstalledVersion']} (fixed in {item.get('FixedVersion', '?')}) "
                f"in {result['Target']}"
            )
        for item in result.get("Misconfigurations") or []:
            if item.get("Status") == "FAIL":
                lines.append(
                    f"{item['ID']} {item['Severity']}: {item['Title']} in {result['Target']}"
                )
    return lines


def build(dockerfile, tag):
    subprocess.run(
        ["docker", "build", "-q", "-f", dockerfile, "-t", tag, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=3600,
        check=True,
    )
    return tag


def test_the_repository_files_have_no_fixable_high_findings():
    skips = [argument for path in SKIPPED_DIRECTORIES for argument in ("--skip-dirs", path)]
    report = trivy("fs", "--scanners", "vuln,misconfig", *skips, "/src", mount_repository=True)

    assert not findings(report), "\n".join(findings(report))


def test_the_fast_image_has_no_fixable_high_findings():
    image = os.getenv("FAST_IMAGE") or build(
        "deploy/Dockerfile.fast", "two-tier-fast:security-scan"
    )

    report = trivy("image", "--scanners", "vuln", image)

    assert not findings(report), "\n".join(findings(report))


def test_the_accurate_cpu_image_has_no_fixable_high_findings():
    image = os.getenv("ACCURATE_IMAGE") or build(
        "deploy/Dockerfile.accurate.cpu", "two-tier-accurate-cpu:security-scan"
    )

    report = trivy("image", "--scanners", "vuln", image)

    assert not findings(report), "\n".join(findings(report))


def test_the_cuda_image_has_no_fixable_high_findings():
    image = os.getenv("GPU_IMAGE")
    if not image:
        pytest.skip("set GPU_IMAGE to scan the CUDA image (about 45 GB, so never built here)")

    report = trivy("image", "--scanners", "vuln", image)

    assert not findings(report), "\n".join(findings(report))


def test_every_accepted_finding_has_a_reason_and_has_not_expired():
    entries = (yaml.safe_load(IGNORE_FILE.read_text()) or {}).get("vulnerabilities") or []
    today = datetime.date.today()
    for entry in entries:
        assert entry.get("statement"), f"{entry.get('id')}: no statement saying why it is accepted"
        expiry = entry.get("expired_at")
        assert expiry, f"{entry['id']}: no expired_at date"
        assert expiry > today, f"{entry['id']}: the exception expired on {expiry}; fix or renew it"
