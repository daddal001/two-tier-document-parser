"""Response contract of the fast service.

These tests pin the status codes, keys and types that API callers depend on. They check
shape, not volatile values (timings, versions, counts), so a harmless change does not fail
them but a removed key, a renamed field or a changed status code does.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from two_tier_parser.fast import main as fast_main

PARSE_METADATA_KEYS = {"pages", "processing_time_ms", "parser", "version", "filename"}
PAGES_METADATA_KEYS = {
    "filename",
    "total_pages",
    "pages_parsed",
    "start_page",
    "end_page",
    "processing_time_ms",
    "parser",
    "version",
}
IMAGE_KEYS = {"image_id", "image_base64", "page", "bbox", "pages", "mime", "width", "height"}
HEALTH_TYPES = {
    "status": str,
    "workers": int,
    "no_gil": bool,
    "parser": str,
    "version": str,
    "minio_connected": bool,
    "storage_configured": bool,
    "layout_enabled": bool,
}


@pytest.fixture
def pdf_bytes() -> bytes:
    doc = pymupdf.open()
    for number in range(3):
        doc.new_page().insert_text((72, 72), f"Contract page {number}")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(monkeypatch):
    """A client whose parser work runs in threads, so the contract tests stay fast."""
    monkeypatch.setattr(fast_main, "_inflight_documents", 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(fast_main, "executor", pool)
        monkeypatch.setattr(fast_main, "pages_executor", pool)
        yield TestClient(fast_main.app, raise_server_exceptions=False)


def _upload(pdf: bytes, name: str = "doc.pdf"):
    return {"file": (name, pdf, "application/pdf")}


def test_health_shape(client):
    body = client.get("/health").json()
    assert set(body) == set(HEALTH_TYPES)
    for key, expected in HEALTH_TYPES.items():
        assert isinstance(body[key], expected), key
    assert body["status"] in {"healthy", "degraded"}


def test_metrics_is_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "executor_queue_depth" in response.text


def test_parse_upload_shape(client, pdf_bytes):
    response = client.post("/parse", files=_upload(pdf_bytes))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"markdown", "metadata", "images"}
    assert isinstance(body["markdown"], str) and "Contract page 0" in body["markdown"]
    assert PARSE_METADATA_KEYS <= set(body["metadata"])
    assert body["metadata"]["pages"] == 3
    assert body["metadata"]["parser"] == "pymupdf4llm"
    assert body["images"] == []


def test_parse_pages_shape(client, pdf_bytes):
    response = client.post("/parse-pages?start_page=0&end_page=2", files=_upload(pdf_bytes))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"markdown", "metadata", "images"}
    assert PAGES_METADATA_KEYS <= set(body["metadata"])
    assert (body["metadata"]["start_page"], body["metadata"]["end_page"]) == (0, 2)
    assert body["metadata"]["pages_parsed"] == 2


def test_image_record_shape(client):
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(72, 72, 172, 172), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    images = client.post("/parse", files=_upload(data)).json()["images"]
    assert len(images) == 1
    assert set(images[0]) == IMAGE_KEYS


def test_preemption_is_409_with_resume_details(client, monkeypatch):
    def preempted(*_args):
        return {
            "markdown": "partial",
            "images": [],
            "metadata": {"pages": 10, "preempted": True, "last_page_parsed": 3},
        }

    monkeypatch.setattr(fast_main, "parse_pdf", preempted)
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr(fast_main, "read_object", lambda *a: b"%PDF-1.4")
    response = client.post(
        "/parse",
        json={"bucket": "b", "key": "k.pdf", "filename": "k.pdf", "task_id": "t-1"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "reason": "preempted",
        "last_page_parsed": 3,
        "pages_total": 10,
        "markdown_prefix_length": 7,
    }


@pytest.mark.parametrize("path", ["/parse", "/parse-pages"])
def test_saturated_admission_is_503_with_retry_after(client, monkeypatch, path, pdf_bytes):
    monkeypatch.setattr(fast_main, "_inflight_documents", fast_main.MAX_INFLIGHT_DOCUMENTS)
    response = client.post(path, files=_upload(pdf_bytes))
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json() == {"detail": "document_admission_saturated"}


def test_full_pool_queue_is_503(client, monkeypatch, pdf_bytes):
    monkeypatch.setattr(fast_main, "_executor_queue_depth", lambda: 10_000)
    response = client.post("/parse", files=_upload(pdf_bytes))
    assert response.status_code == 503
    assert response.json() == {"detail": "parse_pool_saturated"}


def test_oversized_upload_is_413(client, monkeypatch, pdf_bytes):
    monkeypatch.setattr(fast_main, "MAX_PDF_BYTES", 10)
    response = client.post("/parse", files=_upload(pdf_bytes))
    assert response.status_code == 413
    assert response.json() == {"detail": "File too large (max 100MB)"}


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"bucket": "b", "key": "../x.pdf"}, 400),
        ({"bucket": "b", "key": "x.txt"}, 400),
        ({"bucket": "", "key": "x.pdf"}, 400),
        ({"key": "x.pdf"}, 400),
        ({"bucket": "b", "key": "x.pdf", "filename": ""}, 400),
        ({"bucket": "b", "key": "x.pdf", "task_id": 5}, 400),
        ([1, 2], 400),
    ],
)
def test_claim_check_validation_is_400(client, payload, status):
    assert client.post("/parse", json=payload).status_code == status


def test_unsupported_content_type_is_415(client):
    response = client.post("/parse", content=b"x", headers={"content-type": "text/plain"})
    assert response.status_code == 415


def test_non_pdf_upload_is_400(client):
    response = client.post("/parse", files={"file": ("x.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_storage_failure_is_502(client, monkeypatch):
    def broken(*_args):
        raise ConnectionError("unreachable")

    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr(fast_main, "read_object", broken)
    response = client.post("/parse", json={"bucket": "b", "key": "x.pdf", "filename": "x.pdf"})
    assert response.status_code == 502


def test_invalid_page_range_is_422(client, pdf_bytes):
    response = client.post("/parse-pages?start_page=-1", files=_upload(pdf_bytes))
    assert response.status_code == 422


def test_corrupt_pdf_is_generic_500(client):
    response = client.post("/parse", files=_upload(b"not a pdf"))
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Parsing failed (correlation_id=")
    assert "pdf" not in detail.lower().replace("parsing", "")
