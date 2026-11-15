"""Image limits: the server's ceilings, and checks that run before the expensive work."""

import pytest

from two_tier_parser.fast import main as fast_main
from two_tier_parser.fast import service

MAX_PDF = 100 * 1024 * 1024


def options(payload, stage1=False):
    return fast_main._image_opts_from_payload(payload, stage1=stage1)


# ---- ceilings on what a request may ask for ----


@pytest.mark.parametrize("stage1", [False, True])
def test_an_absurd_request_is_clamped_to_the_server_ceiling(stage1):
    got = options(
        {"image_max_count": 10**15, "image_max_bytes": 10**15, "image_max_pixels": 10**15}, stage1
    )

    assert got["image_max_count"] == fast_main.IMAGE_MAX_COUNT_CEILING
    assert got["image_max_bytes"] == fast_main.IMAGE_MAX_BYTES_CEILING
    assert got["image_max_pixels"] == fast_main.IMAGE_MAX_PIXELS_CEILING


def test_a_request_below_the_ceiling_is_kept_exactly():
    got = options({"image_max_count": 7, "image_max_bytes": 1234, "image_max_pixels": 99})

    assert (got["image_max_count"], got["image_max_bytes"], got["image_max_pixels"]) == (
        7,
        1234,
        99,
    )


def test_the_budgets_a_caller_sends_for_previews_are_never_clamped():
    """The largest preview budget a caller computes: count 100,000, ten times a 100 MiB file."""
    got = options(
        {
            "image_max_count": 100_000,
            "image_max_bytes": 10 * MAX_PDF,
            "image_max_pixels": 10_000_000_000,
        },
        stage1=True,
    )

    assert got["image_max_count"] == 100_000
    assert got["image_max_bytes"] == 10 * MAX_PDF
    assert got["image_max_pixels"] == 10_000_000_000


def test_the_one_gigabyte_fallback_budget_is_not_clamped():
    assert (
        options({"image_max_bytes": 1_000_000_000}, stage1=True)["image_max_bytes"] == 1_000_000_000
    )


def test_defaults_without_a_request_are_unchanged():
    assert options({}, stage1=True)["image_max_count"] == 50
    assert options({}, stage1=False)["image_max_count"] == 500
    assert options({}, stage1=True)["image_max_bytes"] == 10 * 1024 * 1024
    assert options({}, stage1=False)["image_max_bytes"] == 50 * 1024 * 1024


def test_negative_and_malformed_values_still_become_safe_numbers():
    got = options({"image_max_count": -5, "image_max_bytes": "lots", "image_max_pixels": None})

    assert got["image_max_count"] == 0
    assert got["image_max_bytes"] == 50 * 1024 * 1024
    assert got["image_max_pixels"] == 50_000_000


def test_an_operator_can_lower_a_ceiling(monkeypatch):
    monkeypatch.setattr(fast_main, "IMAGE_MAX_BYTES_CEILING", 1_000_000)

    assert options({"image_max_bytes": 10**9})["image_max_bytes"] == 1_000_000


def test_the_default_ceilings_cover_the_largest_expected_budget():
    assert fast_main.IMAGE_MAX_COUNT_CEILING >= 100_000
    assert fast_main.IMAGE_MAX_BYTES_CEILING >= 10 * MAX_PDF
    assert fast_main.IMAGE_MAX_PIXELS_CEILING >= 10_000_000_000


# ---- checks before extraction ----


class FakePage:
    def __init__(self, images):
        self.images = images

    def get_images(self, full=True):
        return self.images

    def get_image_bbox(self, info):
        import pymupdf

        return pymupdf.Rect(0, 0, 10, 10)


class FakeDocument:
    """Counts how often the expensive extraction runs."""

    def __init__(self, pages, data=b"image-bytes", size=(10, 10)):
        self.pages = pages
        self.data = data
        self.size = size
        self.extractions = []

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def extract_image(self, xref):
        self.extractions.append(xref)
        return {
            "image": self.data + str(xref).encode(),
            "width": self.size[0],
            "height": self.size[1],
            "ext": "png",
        }


def info(xref, width, height):
    # the tuple PyMuPDF's get_images(full=True) returns: xref, smask, width, height, ...
    return (xref, 0, width, height, 8, "DeviceRGB", "", f"Im{xref}", "FlateDecode", 0)


def test_an_oversized_image_is_refused_before_it_is_extracted():
    doc = FakeDocument([FakePage([info(7, 100_000, 100_000)])])

    result = service.extract_images(doc, max_pixels=1_000_000)

    assert result["truncated"] == "pixels"
    assert result["images"] == []
    assert doc.extractions == []  # nothing was decoded to find that out


def test_an_image_within_the_pixel_limit_is_extracted():
    doc = FakeDocument([FakePage([info(7, 100, 100)])])

    result = service.extract_images(doc, max_pixels=1_000_000)

    assert result["truncated"] is None and len(result["images"]) == 1
    assert doc.extractions == [7]


def test_the_pixel_limit_stops_extraction_of_the_images_after_it():
    doc = FakeDocument([FakePage([info(1, 10, 10), info(2, 50_000, 50_000), info(3, 10, 10)])])

    result = service.extract_images(doc, max_pixels=1_000_000)

    assert [i["image_id"] for i in result["images"]] and len(result["images"]) == 1
    assert doc.extractions == [1]  # the oversized one and everything after it are never extracted


def test_an_image_used_on_every_page_is_extracted_once():
    page_images = [info(5, 10, 10)]
    doc = FakeDocument([FakePage(page_images) for _ in range(4)])

    result = service.extract_images(doc)

    assert doc.extractions == [5]
    assert len(result["images"]) == 1
    assert result["images"][0]["pages"] == [0, 1, 2, 3]


def test_the_count_limit_is_not_spent_on_repeats_of_one_image():
    doc = FakeDocument([FakePage([info(5, 10, 10)]) for _ in range(6)])

    result = service.extract_images(doc, max_count=1)

    assert result["truncated"] is None  # six uses of one image are one image
    assert len(result["images"]) == 1


def test_the_count_limit_still_stops_distinct_images():
    doc = FakeDocument([FakePage([info(n, 10, 10) for n in range(1, 6)])])

    result = service.extract_images(doc, max_count=2)

    assert result["truncated"] == "count" and len(result["images"]) == 2
    assert len(doc.extractions) <= 3  # it stops at the first image over the limit
