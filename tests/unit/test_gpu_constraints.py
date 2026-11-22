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
    "pillow": "security: the base's 12.x has published advisories, 13.0.0 fixes them",
    "transformers": "MinerU 4 needs >=5.10.1",
    "tokenizers": "required by the newer transformers",
    "safetensors": "required by the newer transformers",
    "urllib3": "the service imports it directly and sets a security floor (pyproject.toml)",
    "starlette": "required by the newer web-UI and API dependencies MinerU installs",
    "prometheus-fastapi-instrumentator": "required with the newer starlette",
    "modelscope": "MinerU 4.2 needs >=1.39.1",
    "openai": "MinerU 4.2 needs <3; the base ships 3.x, and vLLM accepts >=2.25",
    **{
        name: "vLLM 0.34.0 requires >=1.45.1, but its base image ships 1.40.0 (the base's own "
        "pip check fails on it); the pins follow vLLM, and the tracing extras need the same line"
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


def test_the_base_freeze_holds_no_native_packages_and_names_its_image():
    text = BASE.read_text()
    dockerfile = (ROOT / "deploy/Dockerfile.accurate").read_text()
    image = re.search(r"FROM (vllm/vllm-openai:\S+)", dockerfile)[1]

    assert image in text.splitlines()[0]
    assert not re.search(
        r"^(torch|torchvision|torchaudio|triton|vllm)==|^(nvidia-|cuda-)", text, re.M
    )
