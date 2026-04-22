"""ADR-0049 unit tests for fast-parser image extraction.

Covers:
- SHA-256 content dedup (same xref on multiple pages → one ImageData, pages=[...])
- OWASP ASVS V12 caps: max_count, max_bytes, max_pixels
- Bbox fallback for masked / zero-bbox images (pymupdf raises or returns empty rect)
- Mime mapped from pymupdf's `info["ext"]` (RFC 2046 canonical subtype)

The tests exercise ``extract_images()`` directly with a stubbed pymupdf
Document so they run offline without the real pymupdf install. The hot-path
``parse_pdf`` integration is covered by tests/perf/bench_parse.py and the
existing integration tests.
"""
from __future__ import annotations

import base64
import hashlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _install_pymupdf_stub() -> None:
    if "pymupdf" not in sys.modules:
        stub_pymupdf = types.ModuleType("pymupdf")
        stub_pymupdf.open = MagicMock()
        sys.modules["pymupdf"] = stub_pymupdf
    if "pymupdf4llm" not in sys.modules:
        stub_pymupdf4llm = types.ModuleType("pymupdf4llm")
        stub_pymupdf4llm.to_markdown = MagicMock(return_value="")
        sys.modules["pymupdf4llm"] = stub_pymupdf4llm


_install_pymupdf_stub()

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from two_tier_parser.fast.service import extract_images  # noqa: E402


# ------------------------- doc stub machinery --------------------------


class _StubRect:
    def __init__(self, x0=0.0, y0=0.0, x1=0.0, y1=0.0, *, empty=False, infinite=False):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.is_empty = empty
        self.is_infinite = infinite


class _StubPage:
    def __init__(self, images: list[dict]):
        # `images` is a list of {xref, bytes, ext, width, height, bbox?, raise_bbox?}
        self._images = images

    def get_images(self, full: bool = True):
        # pymupdf returns tuples where position 0 is xref.
        return [(img["xref"], 0, 0, 0, 0, 0, "", "", "", 0) for img in self._images]

    def get_image_bbox(self, img_tuple_or_xref):
        # Accept either the original tuple from get_images or a bare xref.
        xref = img_tuple_or_xref[0] if isinstance(img_tuple_or_xref, tuple) else img_tuple_or_xref
        for img in self._images:
            if img["xref"] == xref:
                if img.get("raise_bbox"):
                    raise RuntimeError("no bbox for masked image")
                if "bbox" in img:
                    return _StubRect(*img["bbox"])
                return _StubRect(empty=True)
        raise LookupError(xref)


class _StubDoc:
    def __init__(self, pages: list[list[dict]]):
        # pages[i] is a list of image dicts visible on page i (same dict can
        # appear on multiple pages to simulate xref reuse).
        self._pages = [_StubPage(imgs) for imgs in pages]
        # Flattened map of xref -> extract_image() result
        self._xref_data: dict[int, dict] = {}
        for page_imgs in pages:
            for img in page_imgs:
                self._xref_data.setdefault(
                    img["xref"],
                    {
                        "image": img["bytes"],
                        "ext": img.get("ext", "png"),
                        "width": img.get("width", 100),
                        "height": img.get("height", 100),
                    },
                )

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, i):
        return self._pages[i]

    def extract_image(self, xref):
        return self._xref_data[xref]


# --------------------------------- tests -------------------------------


def test_dedup_by_sha256_across_pages():
    shared_bytes = b"logo-on-every-page" * 256
    unique_bytes = b"unique-image" * 512
    page0 = [
        {"xref": 1, "bytes": shared_bytes, "ext": "png", "bbox": (10, 20, 30, 40)},
        {"xref": 2, "bytes": unique_bytes, "ext": "jpeg", "bbox": (50, 60, 70, 80)},
    ]
    page1 = [
        # Same xref+bytes → should dedup.
        {"xref": 1, "bytes": shared_bytes, "ext": "png", "bbox": (15, 25, 35, 45)},
    ]
    page2 = [
        {"xref": 1, "bytes": shared_bytes, "ext": "png"},
    ]
    doc = _StubDoc([page0, page1, page2])

    result = extract_images(doc, max_count=10, max_bytes=10**8, max_pixels=10**8)
    imgs = result["images"]

    assert result["truncated"] is None
    assert len(imgs) == 2

    shared_id = hashlib.sha256(shared_bytes).hexdigest()[:16]
    unique_id = hashlib.sha256(unique_bytes).hexdigest()[:16]
    by_id = {im["image_id"]: im for im in imgs}

    assert shared_id in by_id
    assert by_id[shared_id]["pages"] == [0, 1, 2]
    assert by_id[shared_id]["mime"] == "image/png"
    assert base64.b64decode(by_id[shared_id]["image_base64"]) == shared_bytes
    # bbox captured from the first occurrence (page 0)
    assert by_id[shared_id]["bbox"] == [10.0, 20.0, 30.0, 40.0]

    assert unique_id in by_id
    assert by_id[unique_id]["pages"] == [0]
    assert by_id[unique_id]["mime"] == "image/jpeg"


def test_max_count_cap_truncates_with_reason():
    # 5 unique images; cap at 2.
    pages = [[{"xref": i, "bytes": f"img-{i}".encode() * 100, "ext": "png"} for i in range(5)]]
    doc = _StubDoc(pages)

    result = extract_images(doc, max_count=2, max_bytes=10**8, max_pixels=10**8)

    assert result["truncated"] == "count"
    assert len(result["images"]) == 2


def test_max_bytes_cap_truncates_with_reason():
    # Content must be distinct — same bytes dedup by SHA-256 hash.
    big1 = b"a" * (1024 * 1024)
    big2 = b"b" * (1024 * 1024)
    big3 = b"c" * (1024 * 1024)
    pages = [[
        {"xref": 1, "bytes": big1, "ext": "png"},
        {"xref": 2, "bytes": big2, "ext": "png"},
        {"xref": 3, "bytes": big3, "ext": "png"},
    ]]
    doc = _StubDoc(pages)

    # Cap at 2 MB + 1 — first two fit (2 MB exactly), third would break cap.
    result = extract_images(doc, max_count=100, max_bytes=2 * 1024 * 1024 + 1, max_pixels=10**8)

    assert result["truncated"] == "bytes"
    assert len(result["images"]) == 2


def test_max_pixels_cap_triggers_pixel_flood_protection():
    # OWASP ASVS V12.2 / ASVS#1740 — pixel-flood DoS mitigation.
    pages = [[{
        "xref": 1,
        "bytes": b"tiny-but-claims-big",
        "ext": "png",
        "width": 100_000,
        "height": 100_000,  # 10 billion pixels
    }]]
    doc = _StubDoc(pages)

    result = extract_images(doc, max_count=10, max_bytes=10**8, max_pixels=50_000_000)

    assert result["truncated"] == "pixels"
    assert len(result["images"]) == 0


def test_bbox_fallback_when_get_image_bbox_raises():
    pages = [[{
        "xref": 1,
        "bytes": b"masked-image" * 50,
        "ext": "png",
        "width": 100,
        "height": 100,
        "raise_bbox": True,
    }]]
    doc = _StubDoc(pages)

    result = extract_images(doc)
    assert len(result["images"]) == 1
    assert result["images"][0]["bbox"] is None


def test_bbox_empty_falls_back_to_none():
    # No bbox key → _StubPage returns empty Rect → extract_images sets bbox=None
    pages = [[{"xref": 1, "bytes": b"blob" * 100, "ext": "png"}]]
    doc = _StubDoc(pages)
    result = extract_images(doc)
    assert result["images"][0]["bbox"] is None


def test_page_range_subset():
    pages = [
        [{"xref": 1, "bytes": b"p0" * 100, "ext": "png"}],
        [{"xref": 2, "bytes": b"p1" * 100, "ext": "png"}],
        [{"xref": 3, "bytes": b"p2" * 100, "ext": "png"}],
    ]
    doc = _StubDoc(pages)
    # Stage 1 only looks at pages 0–1
    result = extract_images(doc, page_range=range(0, 2))
    by_page = {im["page"] for im in result["images"]}
    assert by_page == {0, 1}


def test_respects_mime_extension():
    pages = [[
        {"xref": 1, "bytes": b"jp" * 100, "ext": "jpeg"},
        {"xref": 2, "bytes": b"ti" * 100, "ext": "tiff"},
    ]]
    doc = _StubDoc(pages)
    result = extract_images(doc)
    mimes = {im["mime"] for im in result["images"]}
    assert mimes == {"image/jpeg", "image/tiff"}
