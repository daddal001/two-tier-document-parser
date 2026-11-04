"""The fast image under production-like constraints.

Every test runs the real image as an unprivileged user on a read-only root with a small
writable /tmp and no Linux capabilities. They guard the things a unit test cannot see:
whether the image starts, parses an upload that spills to disk, reads from an object store,
logs only JSON, and agrees with an earlier image.
"""

import io
import json
import os
import re
import socket
import subprocess

import pytest

from .conftest import ROOT_PASSWORD, ROOT_USER, docker, run_pod

pytestmark = pytest.mark.container

# The layout model prints one native line to stderr on CPU-restricted hosts. It does not come
# from this service, so it is the one tolerated non-JSON line.
NATIVE_LIBRARY_LINE = re.compile(r"\[[A-Z]:onnxruntime:")
PARSE_KEYS = {"markdown", "metadata", "images"}
HEALTH_KEYS = {
    "status",
    "workers",
    "no_gil",
    "parser",
    "version",
    "minio_connected",
    "storage_configured",
    "layout_enabled",
}


def non_json_lines(lines: list[str]) -> list[str]:
    bad = []
    for line in lines:
        if NATIVE_LIBRARY_LINE.search(line):
            continue
        try:
            json.loads(line)
        except ValueError:
            bad.append(line)
    return bad


def test_container_is_unprivileged_with_a_read_only_root(standalone):
    assert docker("exec", standalone.container, "id", "-u") == "1000"
    root_write = subprocess.run(
        ["docker", "exec", standalone.container, "sh", "-c", "touch /root-probe"],
        capture_output=True,
        check=False,
    )
    assert root_write.returncode != 0
    docker("exec", standalone.container, "sh", "-c", "touch /tmp/probe")  # /tmp stays writable


def test_health_reports_a_healthy_standalone_service(standalone):
    status, _, body = standalone.json("GET", "/health")
    assert status == 200
    assert set(body) == HEALTH_KEYS
    assert body["status"] == "healthy"
    assert body["storage_configured"] is False


def test_upload_larger_than_the_spool_threshold_parses(standalone, sample_pdf):
    assert len(sample_pdf) > 1024 * 1024  # forces the multipart body onto the tmpfs
    status, _, body = standalone.upload("/parse", sample_pdf)
    assert status == 200
    assert set(body) == PARSE_KEYS
    assert body["metadata"]["pages"] > 0 and body["markdown"]


def test_page_range_parses(standalone, sample_pdf):
    status, _, body = standalone.upload("/parse-pages?start_page=0&end_page=1", sample_pdf)
    assert status == 200
    assert body["metadata"]["pages_parsed"] == 1


def test_oversized_json_body_is_413(standalone):
    # The service refuses on the declared length alone, before any of the body is sent, so the
    # request is made over a raw socket rather than a client that insists on sending the body.
    port = int(standalone.url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        connection.sendall(
            b"POST /parse HTTP/1.1\r\nHost: gate\r\nContent-Type: application/json\r\n"
            b"Content-Length: 2097152\r\nConnection: close\r\n\r\n"
        )
        status_line = connection.makefile("rb").readline()
    assert b" 413 " in status_line


def test_claim_check_without_a_store_is_501(standalone):
    status, _, _ = standalone.post_json("/parse", {"bucket": "b", "key": "k.pdf"})
    assert status == 501


def test_every_log_line_is_json(standalone, sample_pdf):
    standalone.upload("/parse", sample_pdf)
    standalone.request("GET", "/health")
    lines = standalone.logs()
    assert lines
    assert non_json_lines(lines) == []


@pytest.fixture(scope="module")
def stored_sample(object_store, sample_pdf):
    object_store.client().make_bucket("documents")
    object_store.client().put_object(
        "documents", "user/sample.pdf", io.BytesIO(sample_pdf), len(sample_pdf)
    )
    return "documents", "user/sample.pdf"


def store_env(object_store, user=ROOT_USER, password=ROOT_PASSWORD):
    # The names an existing deployment sets; the image must keep honouring them.
    return {
        "MINIO_ENDPOINT": object_store.endpoint,
        "MINIO_ACCESS_KEY": user,
        "MINIO_SECRET_KEY": password,
    }


def test_claim_check_reads_from_the_object_store(
    fast_image, network, object_store, stored_sample, sample_pdf
):
    bucket, key = stored_sample
    with run_pod(fast_image, env=store_env(object_store), network=network) as service:
        _, _, uploaded = service.upload("/parse", sample_pdf)
        status, _, claimed = service.post_json(
            "/parse", {"bucket": bucket, "key": key, "filename": "sample.pdf"}
        )
        assert status == 200
        assert claimed["markdown"] == uploaded["markdown"]
        assert [i["image_id"] for i in claimed["images"]] == [
            i["image_id"] for i in uploaded["images"]
        ]

        status, _, body = service.post_json(
            "/parse", {"bucket": bucket, "key": "user/missing.pdf", "filename": "missing.pdf"}
        )
        assert status == 502
        assert "missing" not in json.dumps(body)  # storage details stay out of the response

        status, _, health = service.json("GET", "/health")
        assert health["storage_configured"] is True and health["minio_connected"] is True
        assert non_json_lines(service.logs()) == []


@pytest.mark.xfail(
    strict=True,
    reason="the health probe lists every bucket, so credentials that may only read objects "
    "report degraded; fixed with the Step 3 health change",
)
def test_health_with_read_only_credentials(fast_image, network, object_store, stored_sample):
    object_store.mc("admin user add local reader reader-secret-1")
    object_store.mc("admin policy attach local readonly --user reader")
    env = store_env(object_store, "reader", "reader-secret-1")
    with run_pod(fast_image, env=env, network=network) as service:
        _, _, health = service.json("GET", "/health")
        assert health["status"] == "healthy"


def test_responses_match_the_baseline_image(standalone, sample_pdf):
    baseline = os.getenv("BASELINE_IMAGE")
    if not baseline:
        pytest.skip("set BASELINE_IMAGE to compare against an earlier image")
    _, _, current = standalone.upload("/parse", sample_pdf)
    with run_pod(baseline) as old:
        _, _, previous = old.upload("/parse", sample_pdf)
        assert current["markdown"] == previous["markdown"]
        assert [i["image_id"] for i in current["images"]] == [
            i["image_id"] for i in previous["images"]
        ]
        assert current["metadata"]["pages"] == previous["metadata"]["pages"]
        assert set(previous["metadata"]) <= set(current["metadata"])  # keys are only ever added
        _, _, new_health = standalone.json("GET", "/health")
        _, _, old_health = old.json("GET", "/health")
        assert set(old_health) <= set(new_health)
