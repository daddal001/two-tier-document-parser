"""The accurate CPU image installs pinned versions; the pins must cover and satisfy every dependency."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS = ROOT / "deploy" / "constraints" / "accurate-cpu.txt"
MINERU_PYPROJECT = ROOT / "MinerU" / "pyproject.toml"
IMAGE_EXTRAS = ("telemetry", "telemetry-sdk")  # what Dockerfile.accurate.cpu installs

# What the CPU image runs on: Debian-based Python 3.12 on either CPU architecture.
ENVIRONMENTS = [
    {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "os_name": "posix",
        "platform_machine": machine,
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "extra": "",
    }
    for machine in ("x86_64", "aarch64")
]


def load_pins():
    """Return the pins as (name, version, marker) for every requirement line."""
    pins = []
    for line in CONSTRAINTS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line.split(" #")[0].strip())
        (specifier,) = requirement.specifier
        assert specifier.operator == "==", f"not an exact pin: {line}"
        pins.append((canonicalize_name(requirement.name), specifier.version, requirement.marker))
    return pins


def applies(marker, environment):
    return marker is None or marker.evaluate(environment)


def pins_for(environment):
    chosen = {}
    for name, version, marker in load_pins():
        if applies(marker, environment):
            assert name not in chosen, f"{name} is pinned twice for one environment"
            chosen[name] = version
    return chosen


def declared(pyproject_path, extras):
    data = tomllib.loads(pyproject_path.read_text())["project"]
    names = list(data.get("dependencies", []))
    for extra in extras:
        names += data.get("optional-dependencies", {}).get(extra, [])
    return [Requirement(text) for text in names]


@pytest.mark.parametrize("environment", ENVIRONMENTS, ids=lambda env: env["platform_machine"])
@pytest.mark.parametrize(
    ("source", "extras"),
    [
        pytest.param(ROOT / "pyproject.toml", IMAGE_EXTRAS, id="this-package"),
        pytest.param(
            MINERU_PYPROJECT,
            (),
            id="mineru",
            marks=pytest.mark.skipif(
                not MINERU_PYPROJECT.exists(), reason="the MinerU submodule is not checked out"
            ),
        ),
    ],
)
def test_every_declared_dependency_is_pinned_to_a_satisfying_version(source, extras, environment):
    pinned = pins_for(environment)
    for requirement in declared(source, extras):
        if not applies(requirement.marker, environment):
            continue
        name = canonicalize_name(requirement.name)
        assert (
            name in pinned
        ), f"{requirement.name} is declared but not pinned; run make constraints"
        assert requirement.specifier.contains(
            pinned[name], prereleases=True
        ), f"{requirement.name}=={pinned[name]} does not satisfy {requirement.specifier}"


@pytest.mark.parametrize("environment", ENVIRONMENTS, ids=lambda env: env["platform_machine"])
def test_the_cpu_image_never_pulls_a_cuda_stack(environment):
    pinned = pins_for(environment)
    forbidden = {"torch", "torchvision", "triton", "vllm"}
    assert not (forbidden & set(pinned))
    assert not [name for name in pinned if name.startswith("nvidia-")]


def test_the_cpu_dockerfile_installs_with_the_pins():
    text = (ROOT / "deploy" / "Dockerfile.accurate.cpu").read_text().replace("\\\n", " ")
    installs = [line for line in text.splitlines() if "pip install" in line]
    assert installs
    for line in installs:
        assert "-c /tmp/constraints.txt" in line
    assert "COPY deploy/constraints/accurate-cpu.txt /tmp/constraints.txt" in text


COMPAT = ROOT / "deploy" / "constraints" / "mineru-compat.txt"


def compat_requirements():
    lines = [line for line in COMPAT.read_text().splitlines() if line.strip() and line[0] != "#"]
    return [Requirement(line) for line in lines]


def test_the_compiled_pins_respect_the_compatibility_limits():
    pinned = pins_for(ENVIRONMENTS[0])
    for requirement in compat_requirements():
        name = canonicalize_name(requirement.name)
        assert name in pinned, f"{requirement.name} is limited but not pinned"
        assert requirement.specifier.contains(
            pinned[name]
        ), f"{requirement.name}=={pinned[name]} is outside the limit {requirement.specifier}"


def test_the_cpu_dockerfile_copies_the_compatibility_limits_too():
    # The CPU image pins everything, so the limits are already inside accurate-cpu.txt; the
    # GPU image installs unpinned dependencies and relies on this file directly.
    gpu = (ROOT / "deploy" / "Dockerfile.accurate").read_text()
    assert "COPY deploy/constraints/mineru-compat.txt /opt/mineru-compat.txt" in gpu
