"""Guard the repository's CUDA and CPU Compose profiles."""

from pathlib import Path

import pytest
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


ACCURATE_DOCKERFILES = ["Dockerfile.accurate", "Dockerfile.accurate.cpu"]


def read_dockerfile(filename):
    """Return a Dockerfile's instructions with line continuations joined and comments dropped."""
    text = (ROOT / "deploy" / filename).read_text(encoding="utf-8").replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if line.strip() and line[0] != "#"]


@pytest.mark.parametrize("filename", ACCURATE_DOCKERFILES)
def test_accurate_images_run_unprivileged_with_a_home_the_user_owns(filename):
    lines = read_dockerfile(filename)

    assert "USER 1000" in lines  # the platform forces this user; the image must work as it
    env = " ".join(line for line in lines if line.startswith("ENV "))
    assert "MINERU_HOME=/opt/mineru" in env
    assert "HOME=/opt/mineru" in env  # an unknown UID has no home, and ~ would stay literal
    chown = [line for line in lines if line.startswith("RUN") and "chown" in line]
    assert any("/opt/mineru" in line and "1000:1000" in line for line in chown)
    assert not any("/root" in line for line in lines), "nothing may live under /root"


def test_the_cpu_image_models_directory_is_writable_by_that_user():
    lines = read_dockerfile("Dockerfile.accurate.cpu")

    chown = [line for line in lines if line.startswith("RUN") and "chown" in line]
    assert any("/models" in line and "1000:1000" in line for line in chown)
    # The user switch comes after the root-only steps, and the entrypoint stays executable.
    user = lines.index("USER 1000")
    assert all(i < user for i, line in enumerate(lines) if "chmod +x /entrypoint.sh" in line)
    assert all(i < user for i, line in enumerate(lines) if "pip install" in line)


def test_the_gpu_image_cannot_replace_the_base_images_cuda_stack():
    lines = read_dockerfile("Dockerfile.accurate")

    freeze = [line for line in lines if "pip freeze" in line]
    assert freeze, "the base image's torch stack must be frozen into a constraints file"
    assert all("test -s" in line for line in freeze)  # an empty freeze must fail the build
    installs = [line for line in lines if "pip install" in line]
    assert installs
    for line in installs:
        assert "-c /opt/base-constraints.txt" in line, f"no base constraints: {line}"
        assert "--no-deps" in line, f"runtime dependencies must come from the compiled set: {line}"
    assert any("-r /opt/accurate-gpu.txt" in line for line in installs)


def test_the_gpu_image_bakes_models_into_a_directory_the_user_can_read():
    lines = read_dockerfile("Dockerfile.accurate")

    user = lines.index("USER 1000")
    download = [i for i, line in enumerate(lines) if "accurate.prepare_models" in line]
    assert download and all(i > user for i in download)  # downloaded files belong to that user
    assert not any(
        "mineru-kit models download" in line for line in lines
    ), "MinerU's own downloader cannot pin revisions; the build must use prepare_models"


def test_the_cuda_profile_does_not_hide_the_baked_models_behind_a_volume():
    compose = read_compose("docker-compose.yml")
    accurate = compose["services"]["accurate-parser-cuda"]

    assert not accurate.get(
        "volumes"
    ), "a named volume would shadow the models baked into the image"
    assert "mineru-models" not in (compose.get("volumes") or {})


def test_the_fast_image_has_no_compiler_and_installs_a_pinned_pip():
    lines = read_dockerfile("Dockerfile.fast")
    text = "\n".join(lines)

    assert "build-essential" not in text  # every dependency installs from a wheel
    assert "libffi-dev" not in text
    upgrade = [line for line in lines if "pip install" in line and "--upgrade pip" in line]
    assert upgrade and all("pip==" in line for line in upgrade)  # the one unpinned tool


def test_compose_does_not_carry_the_obsolete_version_key():
    assert "version" not in read_compose("docker-compose.yml")


def test_the_handlers_use_the_running_loop():
    source = (ROOT / "src" / "two_tier_parser" / "fast" / "main.py").read_text()

    assert (
        "get_event_loop" not in source
    )  # deprecated inside a coroutine; get_running_loop is exact


def test_stray_session_files_cannot_be_committed():
    ignored = (ROOT / ".gitignore").read_text().splitlines()

    assert ":memory:.ses" in ignored


def test_the_fast_image_leaves_the_installer_out_of_the_final_layer():
    lines = read_dockerfile("Dockerfile.fast")
    install = max(i for i, line in enumerate(lines) if "pip install" in line)
    removal = [i for i, line in enumerate(lines) if "pip uninstall" in line]

    assert removal, "pip and its vendored libraries are reported by every image scanner"
    assert removal[0] > install  # after everything that needs it
    assert all(name in lines[removal[0]] for name in ("pip", "setuptools"))


def test_fast_image_installs_the_redis_client_for_cancellation():
    """Cancellation turns on with REDIS_URL; the image must hold the client it needs."""
    dockerfile = (ROOT / "deploy" / "Dockerfile.fast").read_text(encoding="utf-8")

    assert "--extra cancellation" in dockerfile  # the pins come from uv.lock
    assert '".[fast,cancellation,telemetry,telemetry-sdk]"' in dockerfile
