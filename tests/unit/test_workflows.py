"""Workflow hygiene that does not need the network: pinned actions, least privilege."""

import re
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
SHA = re.compile(r"^[0-9a-f]{40}$")


def steps(workflow):
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])


def load(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_full_commit(path):
    for step in steps(load(path)):
        reference = step.get("uses")
        if not reference or reference.startswith("./"):
            continue
        _, _, version = reference.partition("@")
        assert SHA.match(version), f"{path.name}: {reference} is a movable reference"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_each_workflow_declares_its_permissions(path):
    workflow = load(path)

    assert "permissions" in workflow, f"{path.name} falls back to the repository default"
    assert workflow["permissions"].get("contents") in ("read", "none")


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_checkouts_do_not_leave_credentials_behind(path):
    for step in steps(load(path)):
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False, path.name


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_has_a_timeout(path):
    for name, job in load(path)["jobs"].items():
        assert "timeout-minutes" in job, f"{path.name}: job {name} can run for six hours"


def test_scheduled_scans_exist_for_dependencies_and_code():
    security = load(ROOT / ".github" / "workflows" / "security.yml")

    assert set(security["jobs"]) == {"dependencies", "code"}
    assert "schedule" in security[True]  # YAML reads the key "on" as True


def test_trivy_and_osv_are_not_run_in_github_actions():
    # They are tests (tests/security) that run with the rest of the suite, not workflow jobs.
    for path in WORKFLOWS:
        for job in load(path)["jobs"].values():
            for step in job.get("steps", []):
                text = f"{step.get('uses', '')} {step.get('run', '')}".lower()
                assert "trivy" not in text and "osv-scanner" not in text, path.name
                assert "tests/security" not in text, path.name


def test_the_vulnerability_scans_are_left_out_of_pre_commit():
    # A newly published advisory can fail them without any change here, so they must not stand
    # between a developer and a commit.
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}

    assert not [name for name in hooks if "trivy" in name or "osv" in name]
    assert "not security" in hooks["pytest"]["entry"]


def test_a_failed_audit_does_not_hide_the_next_one():
    security = load(ROOT / ".github" / "workflows" / "security.yml")
    audits = [
        s for s in security["jobs"]["dependencies"]["steps"] if "pip-audit" in s.get("run", "")
    ]

    assert len(audits) == 3
    # the second runs even when the first fails, so both sets are always reported
    assert all("cancelled" in audit.get("if", "") for audit in audits[1:])


def test_ci_skips_the_local_codeql_hook_because_codeql_has_its_own_workflow():
    steps_with_hooks = [
        step
        for step in steps(load(ROOT / ".github" / "workflows" / "ci.yml"))
        if "pre-commit run --all-files" in step.get("run", "")
    ]

    assert steps_with_hooks
    assert all("codeql" in step.get("env", {}).get("SKIP", "") for step in steps_with_hooks)
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    ids = [hook["id"] for repo in config["repos"] for hook in repo["hooks"]]
    assert "codeql" in ids  # and the hook it skips really exists


@pytest.mark.parametrize("event", ["push", "pull_request"])
@pytest.mark.parametrize("workflow", ["container.yml", "accurate-image.yml"])
@pytest.mark.parametrize(
    "path", [".dockerignore", "pyproject.toml", "uv.lock", "README.md", "LICENSE"]
)
def test_image_checks_run_when_shared_build_inputs_change(event, workflow, path):
    patterns = load(ROOT / ".github" / "workflows" / workflow)[True][event]["paths"]
    assert any(fnmatchcase(path, pattern) for pattern in patterns), (workflow, event, path)
