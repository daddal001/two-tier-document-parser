"""Workflow hygiene that does not need the network: pinned actions, least privilege."""

import re
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


def test_scheduled_scans_exist_for_dependencies_code_and_images():
    security = load(ROOT / ".github" / "workflows" / "security.yml")

    assert set(security["jobs"]) == {"dependencies", "code", "fast-image"}
    assert "schedule" in security[True]  # YAML reads the key "on" as True


def test_a_failed_audit_does_not_hide_the_next_one():
    security = load(ROOT / ".github" / "workflows" / "security.yml")
    audits = [
        s for s in security["jobs"]["dependencies"]["steps"] if "pip-audit" in s.get("run", "")
    ]

    assert len(audits) == 2
    # the second runs even when the first fails, so both sets are always reported
    assert "cancelled" in audits[1].get("if", "")


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
