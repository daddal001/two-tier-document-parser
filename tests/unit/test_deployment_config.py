"""Guard the repository's CUDA and CPU Compose profiles."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_compose(filename):
    """Load a Compose file from the repository deployment directory."""
    path = ROOT / "deploy" / filename
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_cuda_profile_uses_the_repository_gpu_image():
    compose = read_compose("docker-compose.yml")

    assert compose["services"]["fast-parser"]["build"]["dockerfile"] == ("deploy/Dockerfile.fast")
    accurate = compose["services"]["accurate-parser-cuda"]
    assert accurate["profiles"] == ["cuda"]
    assert accurate["build"]["dockerfile"] == ("deploy/Dockerfile.accurate")
    assert accurate["runtime"] == "nvidia"
    devices = accurate["deploy"]["resources"]["reservations"]["devices"]
    assert any("gpu" in device["capabilities"] for device in devices)
    assert compose["services"]["fast-parser"]["ports"] == ["8004:8004"]
    assert accurate["ports"] == ["8005:8005"]


def test_cpu_profile_uses_cpu_image_and_does_not_request_nvidia():
    compose = read_compose("docker-compose.yml")
    accurate = compose["services"]["accurate-parser-cpu"]

    assert accurate["profiles"] == ["cpu"]
    assert accurate["build"]["dockerfile"] == "deploy/Dockerfile.accurate.cpu"
    assert accurate["environment"]["MINERU_DEVICE_MODE"] == "cpu"
    assert accurate["environment"]["MINERU_MODEL_SMALL_BACKEND"] == "onnx"
    assert "runtime" not in accurate
    assert "deploy" not in accurate
