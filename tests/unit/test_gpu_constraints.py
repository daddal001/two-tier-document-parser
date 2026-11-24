"""CUDA dependency resolution stays pinned while the base image owns its native stack."""

import re
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "deploy/constraints/accurate-gpu.txt"


def test_gpu_requirements_are_exact_pins_and_match_the_image():
    pins = {}
    for line in PINS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        (specifier,) = requirement.specifier
        assert specifier.operator == "==", line
        pins[requirement.name] = specifier.version
    dockerfile = (ROOT / "deploy/Dockerfile.accurate").read_text()
    inputs = (ROOT / "deploy/constraints/accurate-gpu.in").read_text()
    assert (
        re.search(r"vllm==([\d.]+)", inputs)[1]
        == re.search(r"FROM vllm/vllm-openai:v([\d.]+)", dockerfile)[1]
    )
    assert int(pins["pillow"].split(".")[0]) >= 13
    assert {"transformers", "docvortex", "fastapi", "gradio"} <= pins.keys()
    assert not ({"torch", "torchvision", "torchaudio", "triton", "vllm"} & pins.keys())
    assert not [name for name in pins if name.startswith(("nvidia-", "cuda-"))]


def test_gpu_installation_uses_the_frozen_set_without_resolving_ranges():
    dockerfile = (ROOT / "deploy/Dockerfile.accurate").read_text().replace("\\\n", " ")
    assert "COPY deploy/constraints/accurate-gpu.txt /opt/accurate-gpu.txt" in dockerfile
    for line in dockerfile.splitlines():
        if "pip install" in line:
            assert "--no-deps" in line
            assert "-c /opt/base-constraints.txt" in line
    assert "-r /opt/accurate-gpu.txt" in dockerfile


BASE = ROOT / "deploy/constraints/vllm-base.txt"
# Packages the CUDA pins are allowed to move away from the vLLM base image, and why. Anything
# else drifting means the compile started from something other than the base, which puts the
# image on versions vLLM was never run with.
INTENTIONAL_CHANGES = {
    "modelscope": "MinerU 4.2 needs >=1.39.1",
    "openai": "MinerU 4.2 needs <3; the base ships 3.x, and vLLM accepts >=2.25",
    "pillow": "security: the base's 12.x has published advisories, 13.0.0 fixes them",
    "certifi": "security: the base's 2026.7.22 bundles compromised root certificates; fixed in 2026.10.18",
    "dnspython": "security: the base's 2.8.0 has a denial-of-service advisory; fixed in 2.9.2",
    "aiohttp": "security: the base's 3.14.3 has eleven high-severity advisories; fixed in 3.14.4",
    "filelock": "security: the base's 4.0.1 has published advisories; fixed in 4.2.1",
    "pyjwt": "security: the base's 2.14.0 has published advisories; fixed in 2.15.0",
    **{
        name: "security floor: the base's 1.40 line has denial-of-service advisories (fixed in "
        "1.45.2), and the rest of OpenTelemetry has to move to the same release line with it"
        for name in (
            "opentelemetry-api",
            "opentelemetry-sdk",
            "opentelemetry-proto",
            "opentelemetry-semantic-conventions",
            "opentelemetry-exporter-otlp",
            "opentelemetry-exporter-otlp-proto-common",
            "opentelemetry-exporter-otlp-proto-grpc",
            "opentelemetry-exporter-otlp-proto-http",
        )
    },
}


def read_pins(path):
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split(";")[0].strip()
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            (specifier,) = requirement.specifier
            pins[requirement.name.lower().replace("_", "-")] = specifier.version
    return pins


def test_the_pins_change_only_what_the_base_image_does_not_already_satisfy():
    base, pins = read_pins(BASE), read_pins(PINS)

    drift = {
        name: (base[name], version)
        for name, version in pins.items()
        if name in base and base[name] != version
    }

    unexpected = {name: change for name, change in drift.items() if name not in INTENTIONAL_CHANGES}
    assert (
        not unexpected
    ), f"pins moved away from the vLLM base image without a recorded reason: {unexpected}"
    # A reason for a pin that no longer differs from the base is a stale claim: drop it.
    stale = sorted(set(INTENTIONAL_CHANGES) - set(drift))
    assert not stale, f"reasons recorded for pins that match the base image: {stale}"


def test_the_base_freeze_holds_no_native_packages_and_names_its_image():
    text = BASE.read_text()
    dockerfile = (ROOT / "deploy/Dockerfile.accurate").read_text()
    image = re.search(r"FROM (vllm/vllm-openai:\S+)", dockerfile)[1]

    assert image in text.splitlines()[0]
    assert not re.search(
        r"^(torch|torchvision|torchaudio|triton|vllm)==|^(nvidia-|cuda-)", text, re.M
    )
    assert not LOCAL_BUILD.search(text), "a locally built package belongs to the base image"


# A version with a local tag (+cu130) is a build from the base image's own index, such as
# torchcodec, which vLLM requires. PyPI's plain build of the same version must not replace it.
LOCAL_BUILD = re.compile(r"^[A-Za-z0-9_.-]+==[^\s;]*\+", re.M)


def test_no_package_the_base_built_for_cuda_is_replaced_by_the_pins():
    assert not LOCAL_BUILD.search((ROOT / "deploy/constraints/accurate-gpu.txt").read_text())


def test_the_image_freezes_every_package_the_base_built_for_cuda():
    dockerfile = (ROOT / "deploy/Dockerfile.accurate").read_text()
    freeze = re.search(r"grep -E '([^']+)'", dockerfile)[1]  # what the build keeps as constraints

    for line in ("torch==2.13.0+cu130", "torchcodec==0.17.0+cu130", "nvidia-nccl-cu13==2.31.2"):
        assert re.search(freeze, line), f"{line} is not frozen by the Dockerfile"
    for line in ("numpy==2.2.6", "fastapi==0.136.3"):
        assert not re.search(freeze, line), f"{line} should stay installable by the pins"


NATIVE = ROOT / "deploy/constraints/vllm-base-native.txt"


def native_names():
    lines = [line.strip() for line in NATIVE.read_text().splitlines()]
    return {line for line in lines if line and not line.startswith("#")}


def test_the_pins_name_no_package_the_base_image_owns():
    # The Dockerfile freezes the base's native stack and its CUDA builds. A pin for one of them,
    # at whatever version the resolver picked, can contradict that freeze and fail the build
    # (the 0.30.0 base ships torchcodec 0.16.0+cu130; the resolver chose 0.17.0).
    owned = native_names()

    assert {"torch", "vllm"} <= owned, "the native list should name torch and vllm"
    named = sorted(owned & set(read_pins(PINS)))
    assert not named, f"pinned although the base image owns them: {named}"
