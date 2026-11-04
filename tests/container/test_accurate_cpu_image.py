"""The accurate CPU image under the constraints its production pod applies.

The pod forces an unprivileged user, drops every capability and sets no HOME. The first start
downloads the models into a volume, which needs the network and several minutes, so these
tests carry the ``models`` marker as well: the quick container workflow leaves them out and
you run them with ``pytest -m "container and models" --run-models``.

Environment:
    ACCURATE_IMAGE  Test this prebuilt image instead of building ``Dockerfile.accurate.cpu``.
    ACCURATE_MODELS_VOLUME  A Docker volume that keeps the downloaded models between runs.
"""

import io
import json
import os
import re
import socket
import subprocess
import threading
import time

import pytest
from pypdf import PdfReader, PdfWriter

from .conftest import ROOT, SAMPLE_PDF, docker, run_pod

pytestmark = [pytest.mark.container, pytest.mark.models]

VOLUME = os.getenv("ACCURATE_MODELS_VOLUME", "two-tier-gate-cpu-models")
# The accurate pod keeps a writable root, so the read-only flag of the fast pod is left out.
ACCURATE_FLAGS = [
    "--user",
    "1000:1000",
    "--tmpfs",
    "/tmp",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--memory",
    "6g",
    "--cpus",
    "4",
    "-v",
    f"{VOLUME}:/models",
]
ACCURATE_ENV = {"WORKERS": "1", "MAX_INFLIGHT_DOCUMENTS": "1", "OTEL_SDK_DISABLED": "true"}
# Model downloads show progress bars and the layout runtime prints native lines; neither is
# the service's own logging, which starts once the server is up.
# One native library prints this to stderr itself; it is not the service's logging.
NATIVE_LINE = re.compile(r"\[[A-Z]:onnxruntime:|^Object \d+ \d+ not defined")
SERVER_STARTED = re.compile(r'"message": ?"[^"]*(Application startup|Started)', re.IGNORECASE)


@pytest.fixture(scope="module")
def accurate_image() -> str:
    prebuilt = os.getenv("ACCURATE_IMAGE")
    if prebuilt:
        return prebuilt
    tag = "two-tier-parser-accurate-cpu:container-tests"
    docker(
        "build",
        "-q",
        "-f",
        "deploy/Dockerfile.accurate.cpu",
        "-t",
        tag,
        str(ROOT),
        timeout=3600,
    )
    return tag


@pytest.fixture(scope="module")
def accurate(accurate_image):
    docker("volume", "create", VOLUME)
    with run_pod(
        accurate_image,
        flags=ACCURATE_FLAGS,
        base_env=ACCURATE_ENV,
        port=8005,
        health_timeout=1800,  # the first start downloads the models
    ) as service:
        yield service


@pytest.fixture(scope="module")
def two_page_pdf() -> bytes:
    writer = PdfWriter()
    for page in PdfReader(str(SAMPLE_PDF)).pages[:2]:
        writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_runs_as_the_pod_user_with_a_home_it_owns(accurate):
    assert docker("exec", accurate.container, "id", "-u") == "1000"
    assert docker("exec", accurate.container, "sh", "-c", "echo $HOME") == "/opt/mineru"
    docker("exec", accurate.container, "sh", "-c", "touch $HOME/probe && touch /models/probe")


def test_health_reports_a_cpu_service(accurate):
    status, _, body = accurate.json("GET", "/health")
    assert status == 200
    assert body["status"] == "healthy"
    assert body["gpu_available"] is False


def test_a_parse_returns_markdown_from_the_pinned_mineru(accurate, two_page_pdf):
    status, _, body = accurate.upload("/parse", two_page_pdf, name="two-pages.pdf")
    assert status == 200
    assert body["markdown"].strip()
    assert body["metadata"]["pages"] == 2
    assert body["metadata"]["version"].startswith("4.")
    assert set(body) >= {"markdown", "images", "tables", "formulas", "metadata"}


def test_a_second_document_gets_503_with_retry_after_while_health_stays_up(accurate, two_page_pdf):
    first = {}

    def parse():
        first["result"] = accurate.upload("/parse", two_page_pdf, name="busy.pdf")

    worker = threading.Thread(target=parse)
    worker.start()
    time.sleep(2)  # the first request holds the only admission slot while it parses
    status, headers, body = accurate.upload("/parse", two_page_pdf, name="rejected.pdf")
    assert status == 503
    assert headers["retry-after"] == "5"
    assert body == {"detail": "document_admission_saturated"}
    assert accurate.request("GET", "/health")[0] == 200  # liveness is not starved by the parse
    worker.join(300)
    assert first["result"][0] == 200


def test_an_oversized_declared_upload_is_refused_without_reading_it(accurate):
    port = int(accurate.url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        connection.sendall(
            b"POST /parse HTTP/1.1\r\nHost: gate\r\n"
            b"Content-Type: multipart/form-data; boundary=x\r\n"
            b"Content-Length: " + str(600 * 1024 * 1024).encode() + b"\r\nConnection: close\r\n\r\n"
        )
        status_line = connection.makefile("rb").readline()
    assert b" 413 " in status_line


def test_the_service_logs_json_once_it_is_serving(accurate):
    lines = accurate.logs()
    serving = [i for i, line in enumerate(lines) if SERVER_STARTED.search(line)]
    assert serving, "the service never logged that it started"
    for line in lines[serving[0] :]:
        if NATIVE_LINE.search(line):
            continue
        json.loads(line)


def run_in_image(image, *args, env=None, mounts=(), check=True):
    """Run a one-off command in the image as the pod user and return its output."""
    flags = ["--rm", "--user", "1000:1000", "--entrypoint", args[0]]
    for key, value in {"MINERU_HOME": "/tmp/mineru", "HOME": "/tmp/mineru", **(env or {})}.items():
        flags += ["-e", f"{key}={value}"]
    for host, inside in mounts:
        flags += ["-v", f"{host}:{inside}:ro"]
    return docker("run", *flags, image, *args[1:], check=check)


def test_the_model_manifest_covers_every_repository_mineru_registers(accurate_image):
    registered = run_in_image(
        accurate_image,
        "python",
        "-c",
        "import json; from mineru.model.registry import MODEL_REPOS; "
        "print(json.dumps([r.repos['huggingface'] for r in MODEL_REPOS]))",
    )
    manifest = run_in_image(
        accurate_image,
        "python",
        "-c",
        "import json; from two_tier_parser.accurate.prepare_models import load_revisions; "
        "print(json.dumps(load_revisions()))",
    )
    assert set(json.loads(registered)) <= set(json.loads(manifest))


def test_a_download_asks_for_the_pinned_commit_not_the_newest(accurate_image, tmp_path):
    bogus = tmp_path / "revisions.json"
    pins = json.loads(
        run_in_image(
            accurate_image,
            "python",
            "-c",
            "import json; from two_tier_parser.accurate.prepare_models import load_revisions; "
            "print(json.dumps(load_revisions()))",
        )
    )
    pins["opendatalab/MinerU-4_models_onnx"] = "0" * 40  # a commit that does not exist
    bogus.write_text(json.dumps({"huggingface": pins}))
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "1000:1000",
            "--entrypoint",
            "python",
            "-v",
            f"{bogus}:/revisions.json:ro",
            "-e",
            "MINERU_MODEL_REVISIONS=/revisions.json",
            "-e",
            "MINERU_HOME=/tmp/mineru",
            "-e",
            "MINERU_MODEL_BASE_DIR=/tmp/empty-models",  # nothing downloaded, so it must fetch
            "-e",
            "MINERU_MODEL_SOURCE=huggingface",
            "-e",
            "MINERU_DOWNLOAD_MODELS=pipeline",
            accurate_image,
            "-m",
            "two_tier_parser.accurate.prepare_models",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "0" * 40 in result.stderr or "Revision Not Found" in result.stderr


def test_the_models_on_disk_are_the_pinned_commits(accurate):
    pinned = json.loads(
        docker(
            "exec",
            accurate.container,
            "python",
            "-c",
            "import json; from two_tier_parser.accurate.prepare_models import load_revisions; "
            "print(json.dumps(load_revisions()))",
        )
    )["opendatalab/MinerU-4_models_onnx"]
    commits = docker(
        "exec",
        accurate.container,
        "sh",
        "-c",
        "find /models/MinerU-4_models_onnx -name '*.metadata' -exec head -qn1 {} + | sort -u",
    ).split()
    assert commits == [pinned]
