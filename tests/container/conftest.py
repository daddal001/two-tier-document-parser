"""Fixtures that run the fast image the way a locked-down production pod does.

These tests need a Docker daemon and a network, so they carry the ``container`` marker and
are left out of the default run. Select them with ``pytest -m container``.

Environment:
    FAST_IMAGE        Test this prebuilt image instead of building ``deploy/Dockerfile.fast``.
    BASELINE_IMAGE    An older image to compare responses against (the comparison test skips
                      without it).
    MINIO_TEST_IMAGE  Object store image for the claim-check tests.
    MINIO_CLIENT_IMAGE  Client image used to create users and buckets.
"""

import contextlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SAMPLE_PDF = ROOT / "examples" / "data" / "sample.pdf"

MINIO_IMAGE = os.getenv("MINIO_TEST_IMAGE", "cgr.dev/chainguard/minio:latest")
MC_IMAGE = os.getenv("MINIO_CLIENT_IMAGE", "cgr.dev/chainguard/minio-client:latest")
ROOT_USER, ROOT_PASSWORD = "gate-admin", "gate-admin-secret"

# Flags that mirror a hardened Kubernetes pod: unprivileged user, read-only root, a writable
# /tmp, no capabilities, bounded memory and CPU.
POD_FLAGS = [
    "--user",
    "1000:1000",
    "--read-only",
    "--tmpfs",
    "/tmp",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--memory",
    "3g",
    "--cpus",
    "4",
]
POD_ENV = {
    "WORKERS": "2",
    "PAGES_WORKERS": "1",
    "MAX_INFLIGHT_DOCUMENTS": "4",
    "WORKER_MAX_TASKS": "20",
    "OTEL_SDK_DISABLED": "true",
}


def docker(*args: str, check: bool = True, timeout: int = 900) -> str:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and result.returncode:
        raise RuntimeError(f"docker {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


@pytest.fixture(scope="session", autouse=True)
def _require_docker():
    try:
        docker("info", "--format", "{{.ServerVersion}}", timeout=30)
    except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("a Docker daemon is required")


@pytest.fixture(scope="session")
def fast_image() -> str:
    prebuilt = os.getenv("FAST_IMAGE")
    if prebuilt:
        return prebuilt
    tag = "two-tier-parser-fast:container-tests"
    docker("build", "-q", "-f", "deploy/Dockerfile.fast", "-t", tag, str(ROOT), timeout=1800)
    return tag


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return SAMPLE_PDF.read_bytes()


@pytest.fixture(scope="session")
def network():
    name = f"gate-{uuid.uuid4().hex[:8]}"
    docker("network", "create", name)
    yield name
    docker("network", "rm", name, check=False)


class Service:
    """A running container and the host URL that reaches it."""

    def __init__(self, container: str, url: str):
        self.container = container
        self.url = url

    def request(self, method: str, path: str, *, data=None, headers=None, timeout=120):
        request = urllib.request.Request(
            self.url + path, data=data, method=method, headers=headers or {}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def json(self, method: str, path: str, **kwargs):
        status, headers, body = self.request(method, path, **kwargs)
        return status, headers, json.loads(body) if body else None

    def upload(self, path: str, pdf: bytes, name: str = "sample.pdf"):
        boundary = uuid.uuid4().hex
        body = (
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=file; "
                f'filename="{name}"\r\nContent-Type: application/pdf\r\n\r\n'
            ).encode()
            + pdf
            + f"\r\n--{boundary}--\r\n".encode()
        )
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        return self.json("POST", path, data=body, headers=headers)

    def post_json(self, path: str, payload: dict):
        return self.json(
            "POST",
            path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

    def logs(self) -> list[str]:
        result = subprocess.run(
            ["docker", "logs", self.container], capture_output=True, text=True, check=False
        )
        return [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]


@contextlib.contextmanager
def run_pod(image: str, *, env: dict | None = None, network: str | None = None, flags=()):
    """Start ``image`` with the pod flags, wait for ``/health``, and remove it afterwards."""
    name = f"gate-fast-{uuid.uuid4().hex[:8]}"
    args = ["run", "-d", "--name", name, "-p", "127.0.0.1::8004", *POD_FLAGS, *flags]
    if network:
        args += ["--network", network]
    for key, value in {**POD_ENV, **(env or {})}.items():
        args += ["-e", f"{key}={value}"]
    docker(*args, image)
    try:
        host_port = docker("port", name, "8004/tcp").splitlines()[0].rsplit(":", 1)[1]
        service = Service(name, f"http://127.0.0.1:{host_port}")
        deadline = time.monotonic() + 90
        while True:
            try:
                if service.request("GET", "/health", timeout=3)[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("container never became healthy:\n" + "\n".join(service.logs()))
            time.sleep(1)
        yield service
    finally:
        docker("rm", "-f", name, check=False)


@pytest.fixture(scope="module")
def standalone(fast_image):
    with run_pod(fast_image) as service:
        yield service


class ObjectStore:
    """A throwaway S3-compatible server on the test network."""

    def __init__(self, network: str, name: str, host_port: str):
        self.network = network
        self.name = name
        self.endpoint = f"{name}:9000"
        self.host_endpoint = f"127.0.0.1:{host_port}"

    def mc(self, *args: str) -> str:
        script = (
            f"mc alias set local http://{self.endpoint} {ROOT_USER} {ROOT_PASSWORD} >/dev/null && "
            "mc " + " ".join(args)
        )
        return docker(
            "run",
            "--rm",
            "--network",
            self.network,
            "--entrypoint",
            "/bin/sh",
            MC_IMAGE,
            "-c",
            script,
        )

    def client(self, access_key: str = ROOT_USER, secret_key: str = ROOT_PASSWORD):
        from minio import Minio

        return Minio(self.host_endpoint, access_key=access_key, secret_key=secret_key, secure=False)


@pytest.fixture(scope="module")
def object_store(network):
    name = f"gate-store-{uuid.uuid4().hex[:8]}"
    docker(
        "run",
        "-d",
        "--name",
        name,
        "--network",
        network,
        "-p",
        "127.0.0.1::9000",
        "-e",
        f"MINIO_ROOT_USER={ROOT_USER}",
        "-e",
        f"MINIO_ROOT_PASSWORD={ROOT_PASSWORD}",
        MINIO_IMAGE,
        "server",
        "/data",
    )
    try:
        host_port = docker("port", name, "9000/tcp").splitlines()[0].rsplit(":", 1)[1]
        store = ObjectStore(network, name, host_port)
        deadline = time.monotonic() + 60
        while True:
            try:
                store.client().list_buckets()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("object store never became ready")
                time.sleep(1)
        yield store
    finally:
        docker("rm", "-f", name, check=False)
