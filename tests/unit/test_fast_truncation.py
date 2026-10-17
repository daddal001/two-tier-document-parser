"""A parse stopped by the wall-clock deadline is marked, never silently partial."""

import sys
import time
from pathlib import Path

import pymupdf
import pymupdf4llm
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from two_tier_parser.fast import service


def _pdf(pages: int) -> bytes:
    doc = pymupdf.open()
    for number in range(pages):
        doc.new_page().insert_text((72, 72), f"page {number}")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def slow_pages(monkeypatch):
    def to_markdown(doc, pages=None, **_kw):
        time.sleep(0.02)
        return f"page-{pages[0]}"

    monkeypatch.setattr(pymupdf4llm, "to_markdown", to_markdown)


def test_deadline_stop_is_marked_truncated(monkeypatch, slow_pages):
    monkeypatch.setattr(service, "FAST_PARSER_WALL_DEADLINE_S", 0.03)
    result = service.parse_pdf(_pdf(10), "doc.pdf", task_id="t-1", extract_images_flag=False)
    metadata = result["metadata"]
    assert metadata["truncated"] == "deadline"
    assert 0 <= metadata["last_page_parsed"] < 9
    assert metadata["pages"] == 10
    assert metadata["preempted"] is False


def test_complete_parse_has_no_truncated_key(monkeypatch, slow_pages):
    monkeypatch.setattr(service, "FAST_PARSER_WALL_DEADLINE_S", 0)
    result = service.parse_pdf(_pdf(3), "doc.pdf", task_id="t-1", extract_images_flag=False)
    assert "truncated" not in result["metadata"]
    assert result["metadata"]["last_page_parsed"] == 2


def test_full_document_path_never_reports_deadline(monkeypatch):
    monkeypatch.setattr(service, "FAST_PARSER_WALL_DEADLINE_S", 0.0001)
    result = service.parse_pdf(_pdf(2), "doc.pdf", extract_images_flag=False)
    assert "truncated" not in result["metadata"]
