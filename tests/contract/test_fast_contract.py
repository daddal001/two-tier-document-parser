"""Response contract of the fast service.

These tests pin the status codes, keys and types that API callers depend on. They check
shape, not volatile values (timings, versions, counts), so a harmless change does not fail
them but a removed key, a renamed field or a changed status code does.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
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


def preempted_claim_check(client, monkeypatch, result):
    monkeypatch.setattr(fast_main, "parse_pdf", lambda *_args: result)
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr(fast_main, "read_object", lambda *a: b"%PDF-1.4")
    return client.post(
        "/parse",
        json={"bucket": "b", "key": "k.pdf", "filename": "k.pdf", "task_id": "t-1"},
    )


def test_preemption_is_409_with_resume_details(client, monkeypatch):
    response = preempted_claim_check(
        client,
        monkeypatch,
        {
            "markdown": "partial",
            "images": [],
            "metadata": {"pages": 10, "preempted": True, "last_page_parsed": 3},
        },
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    # the fields callers already read keep their names and meaning
    assert detail["reason"] == "preempted"
    assert detail["last_page_parsed"] == 3
    assert detail["pages_total"] == 10
    assert detail["markdown_prefix_length"] == 7


def test_the_409_carries_the_finished_pages_so_a_resume_loses_nothing(client, monkeypatch):
    image = {
        "image_id": "abc",
        "image_base64": "aGk=",
        "page": 1,
        "bbox": None,
        "pages": [1],
        "mime": "image/png",
        "width": 1,
        "height": 1,
    }
    response = preempted_claim_check(
        client,
        monkeypatch,
        {
            "markdown": "pages two to four",
            "images": [image],
            "metadata": {
                "pages": 10,
                "preempted": True,
                "last_page_parsed": 3,
                "start_page": 1,
                "skipped_pages": [3],
                "warning": "Skipped 1 page(s) due to parsing errors",
                "images_truncated": "bytes",
            },
        },
    )

    detail = response.json()["detail"]
    assert detail["start_page"] == 1  # the first page the prefix covers
    assert detail["markdown_prefix"] == "pages two to four"
    assert detail["images"] == [image]
    assert detail["skipped_pages"] == [3]
    assert detail["images_truncated"] == "bytes"


def test_a_409_without_skips_or_truncation_has_no_such_keys(client, monkeypatch):
    response = preempted_claim_check(
        client,
        monkeypatch,
        {
            "markdown": "x",
            "images": [],
            "metadata": {"pages": 2, "preempted": True, "last_page_parsed": 0, "start_page": 0},
        },
    )

    assert set(response.json()["detail"]) == {
        "reason",
        "last_page_parsed",
        "pages_total",
        "markdown_prefix_length",
        "start_page",
        "markdown_prefix",
        "images",
    }


def numbered_pages_with_images(count):
    """A document with a distinct line of text and a distinct image on every page."""
    doc = pymupdf.open()
    for number in range(count):
        page = doc.new_page()
        page.insert_text((72, 72), f"Heading for page {number + 1}", fontsize=20)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 30, 30), False)
        pix.set_rect(pix.irect, (20 + 40 * number, 90, 200 - 30 * number))
        page.insert_image(pymupdf.Rect(72, 120, 172, 220), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def test_cancelling_and_resuming_returns_every_page_and_image_exactly_once(client, monkeypatch):
    """The whole protocol with the real converter: interrupt, resume after the cursor, assemble."""
    from two_tier_parser.fast import service

    pdf = numbered_pages_with_images(5)
    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr(fast_main, "read_object", lambda *a: pdf)
    payload = {"bucket": "b", "key": "k.pdf", "filename": "k.pdf", "task_id": "t-1"}

    uninterrupted = client.post("/parse", json=payload)
    assert uninterrupted.status_code == 200
    whole = uninterrupted.json()

    asked = iter([False, False, True])  # cancelled before the third page
    monkeypatch.setattr(service, "_is_cancelled", lambda task_id: next(asked, False))
    first = client.post("/parse", json=payload)
    assert first.status_code == 409
    prefix = first.json()["detail"]
    assert prefix["last_page_parsed"] == 1 and prefix["start_page"] == 0

    monkeypatch.setattr(service, "_is_cancelled", lambda task_id: False)
    second = client.post("/parse", json={**payload, "start_page": prefix["last_page_parsed"] + 1})
    assert second.status_code == 200
    rest = second.json()

    assembled = prefix["markdown_prefix"] + "\n\n" + rest["markdown"]
    assert assembled == whole["markdown"]
    for number in range(1, 6):
        assert assembled.count(f"Heading for page {number}") == 1
    assembled_ids = [i["image_id"] for i in prefix["images"] + rest["images"]]
    assert sorted(assembled_ids) == sorted(i["image_id"] for i in whole["images"])
    assert len(assembled_ids) == len(set(assembled_ids)) == 5


@pytest.mark.parametrize("path", ["/parse", "/parse-pages"])
def test_saturated_admission_is_503_with_retry_after(client, monkeypatch, path, pdf_bytes):
    monkeypatch.setattr(fast_main, "_inflight_documents", fast_main.MAX_INFLIGHT_DOCUMENTS)
    response = client.post(path, files=_upload(pdf_bytes))
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json() == {"detail": "document_admission_saturated"}


POOL_SATURATION = [
    ("/parse", "_executor_queue_depth", "parse_pool_saturated"),
    ("/parse-pages", "_pages_executor_queue_depth", "parse_pages_pool_saturated"),
]


@pytest.mark.parametrize(("path", "queue_depth", "detail"), POOL_SATURATION)
def test_pool_saturation_is_503(client, monkeypatch, pdf_bytes, path, queue_depth, detail):
    monkeypatch.setattr(fast_main, queue_depth, lambda: 10_000)
    response = client.post(path, files=_upload(pdf_bytes))
    assert response.status_code == 503
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(("path", "queue_depth", "detail"), POOL_SATURATION)
def test_pool_saturation_advertises_retry_after(
    client, monkeypatch, pdf_bytes, path, queue_depth, detail
):
    monkeypatch.setattr(fast_main, queue_depth, lambda: 10_000)
    response = client.post(path, files=_upload(pdf_bytes))
    assert response.status_code == 503
    assert response.headers.get("retry-after") == "2"


class _BrokenPool:
    """Stands in for a process pool whose worker died."""

    def submit(self, *_args, **_kwargs):
        raise BrokenProcessPool("worker exited")

    def shutdown(self, *_args, **_kwargs):
        pass


@pytest.mark.parametrize(
    ("path", "pool"), [("/parse", "executor"), ("/parse-pages", "pages_executor")]
)
def test_broken_pool_is_503_with_retry_after_and_recovers(
    client, monkeypatch, pdf_bytes, path, pool
):
    replacements = []

    def replacement(workers):
        replacements.append(workers)
        return ThreadPoolExecutor(max_workers=1)

    monkeypatch.setattr(fast_main, pool, _BrokenPool())
    monkeypatch.setattr(fast_main, "_new_pool", replacement)

    first = client.post(path, files=_upload(pdf_bytes))
    assert first.status_code == 503
    assert first.headers["retry-after"] == "2"
    assert first.json() == {"detail": "parser_worker_unavailable"}

    second = client.post(path, files=_upload(pdf_bytes))  # the replacement pool serves it
    assert second.status_code == 200
    assert len(replacements) == 1


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


def test_streamed_oversized_upload_is_413(client, monkeypatch, pdf_bytes):
    monkeypatch.setattr(fast_main, "MAX_PDF_BYTES", 10)
    monkeypatch.setattr(fast_main, "_MULTIPART_OVERHEAD_BYTES", 10)

    def chunks():
        yield b"--b\r\n"
        yield b'Content-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n'
        yield pdf_bytes

    response = client.post(
        "/parse", content=chunks(), headers={"content-type": "multipart/form-data; boundary=b"}
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "File too large (max 100MB)"}


def test_content_type_is_case_insensitive(client):
    response = client.post("/parse", content=b"{}", headers={"content-type": "Application/JSON"})
    assert response.status_code == 400  # parsed as JSON, then rejected for missing fields


def test_claim_check_without_a_store_is_501(client):
    response = client.post("/parse", json={"bucket": "b", "key": "x.pdf", "filename": "x.pdf"})
    assert response.status_code == 501
    assert "multipart" in response.json()["detail"]


def test_disallowed_bucket_is_403(client, monkeypatch):
    monkeypatch.setenv("PARSER_ALLOWED_BUCKETS", "allowed")
    response = client.post("/parse", json={"bucket": "other", "key": "x.pdf", "filename": "x.pdf"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Bucket is not permitted"}


def test_a_range_with_a_bad_page_is_200_with_the_skipped_pages_listed(
    client, monkeypatch, pdf_bytes
):
    real = fast_main.parse_pdf_page_range

    def damaged(pdf, name, start, end, *args, **kwargs):
        calls = []

        def to_markdown(document, pages=None, **_):
            calls.append(list(pages))
            if len(pages) > 1 or pages == [1]:
                raise AttributeError("'NoneType' object has no attribute 'tables'")
            return f"page {pages[0] + 1}"

        monkeypatch.setattr("two_tier_parser.fast.service.pymupdf4llm.to_markdown", to_markdown)
        return real(pdf, name, start, end, *args, **kwargs)

    monkeypatch.setattr(fast_main, "parse_pdf_page_range", damaged)
    response = client.post("/parse-pages?start_page=0&end_page=3", files=_upload(pdf_bytes))

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["skipped_pages"] == [2]
    assert body["metadata"]["warning"] == "Skipped 1 page(s) due to parsing errors"
    assert "skipped due to parsing error" in body["markdown"]


def test_a_clean_range_keeps_exactly_the_documented_metadata_keys(client, pdf_bytes):
    body = client.post("/parse-pages?start_page=0&end_page=2", files=_upload(pdf_bytes)).json()

    assert set(body["metadata"]) == PAGES_METADATA_KEYS  # no new null-valued keys


def test_a_range_reports_that_images_were_left_out_by_a_limit(client, pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes)
    page = doc[0]
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(50, 100, 150, 200), pixmap=pix)
    data = doc.tobytes()
    doc.close()

    response = client.post(
        "/parse-pages?start_page=0&end_page=1",
        files={"file": ("doc.pdf", data, "application/pdf")},
    )
    # an ordinary request keeps the image and carries no marker
    assert response.status_code == 200
    assert "images_truncated" not in response.json()["metadata"]
    assert len(response.json()["images"]) == 1


def test_a_range_marks_truncation_when_a_limit_drops_images(client, monkeypatch, pdf_bytes):
    real = fast_main.parse_pdf_page_range

    def limited(*args, **kwargs):
        result = real(*args, **kwargs)
        result["metadata"]["images_truncated"] = "bytes"
        return result

    monkeypatch.setattr(fast_main, "parse_pdf_page_range", limited)
    body = client.post("/parse-pages?start_page=0&end_page=1", files=_upload(pdf_bytes)).json()

    assert body["metadata"]["images_truncated"] == "bytes"


def test_range_image_records_keep_every_documented_key_even_when_null(client, monkeypatch):
    """Leaving unset fields out must not remove a key the parser set to null (a masked image)."""
    from two_tier_parser.fast import service

    real = service.extract_images

    def masked(doc, **kwargs):
        result = real(doc, **kwargs)
        for image in result["images"]:
            image["bbox"] = None
        return result

    monkeypatch.setattr(service, "extract_images", masked)
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (10, 120, 200))
    page.insert_image(pymupdf.Rect(50, 100, 150, 200), pixmap=pix)
    data = doc.tobytes()
    doc.close()

    body = client.post(
        "/parse-pages?start_page=0&end_page=1",
        files={"file": ("doc.pdf", data, "application/pdf")},
    ).json()

    assert len(body["images"]) == 1
    assert set(body["images"][0]) == IMAGE_KEYS
    assert body["images"][0]["bbox"] is None


@pytest.mark.parametrize("path", ["/parse", "/parse-pages"])
def test_a_storage_timeout_is_502_and_frees_its_admission_slot(client, monkeypatch, path):
    from two_tier_parser.fast.storage import StorageTimeout

    def too_slow(*_args):
        raise StorageTimeout("object read exceeded its 120 second budget")

    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "store.invalid")
    monkeypatch.setattr(fast_main, "read_object", too_slow)

    response = client.post(path, json={"bucket": "b", "key": "k.pdf", "filename": "k.pdf"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Failed to read file from storage: StorageTimeout"
    assert fast_main._inflight_documents == 0  # a stalled read no longer occupies a slot
