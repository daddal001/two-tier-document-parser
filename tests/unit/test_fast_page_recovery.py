"""How the fast parser recovers from a page it cannot convert, and how it stops early.

These pin the behaviour of the full-document path, the task-aware path (cancellation, resume,
deadline) and the page-range path, so the shared recovery logic can be refactored safely.
"""

import time

import pymupdf
import pytest

from two_tier_parser.fast import service

TABLES_ERROR = AttributeError("'NoneType' object has no attribute 'tables'")
TEXTPAGE_ERROR = ValueError("not a textpage of this page")
SKIPPED = "[Page {n} skipped due to parsing error]"


def pdf_with_pages(count=3):
    """A small document whose pages are easy to tell apart."""
    document = pymupdf.open()
    for number in range(count):
        document.new_page().insert_text((40, 40), f"Page {number + 1}")
    data = document.tobytes()
    document.close()
    return data


class Converter:
    """Stands in for ``pymupdf4llm.to_markdown`` and records how it was called."""

    def __init__(self, fail_on=(), whole_document_error=None, error=TABLES_ERROR, delay=0.0):
        self.calls = []
        self.fail_on = set(fail_on)
        self.whole_document_error = whole_document_error
        self.error = error
        self.delay = delay

    def __call__(self, document, pages=None, **_kwargs):
        self.calls.append(None if pages is None else list(pages))
        if self.delay:
            time.sleep(self.delay)
        if pages is None:
            if self.whole_document_error:
                raise self.whole_document_error
            return "whole document"
        if self.fail_on & set(pages):
            raise self.error
        return "".join(f"Recovered page {page + 1}. " for page in pages)


@pytest.fixture
def convert(monkeypatch):
    def install(**kwargs):
        converter = Converter(**kwargs)
        monkeypatch.setattr(service.pymupdf4llm, "to_markdown", converter)
        return converter

    return install


def parse(pdf=None, **kwargs):
    kwargs.setdefault("extract_images_flag", False)
    return service.parse_pdf(pdf or pdf_with_pages(), "doc.pdf", **kwargs)


# ---- full-document path (no task id) ----


def test_a_document_that_converts_whole_is_returned_as_it_is(convert):
    converter = convert()

    result = parse()

    assert result["markdown"] == "whole document"
    assert converter.calls == [None]
    assert "skipped_pages" not in result["metadata"]


@pytest.mark.parametrize("error", [TABLES_ERROR, TEXTPAGE_ERROR])
def test_a_recognized_error_falls_back_to_page_by_page_and_skips_the_bad_page(convert, error):
    converter = convert(whole_document_error=error, fail_on={1}, error=error)

    result = parse()

    assert converter.calls == [None, [0], [1], [2]]
    assert result["metadata"]["skipped_pages"] == [2]  # one-based, for people
    assert result["metadata"]["warning"] == "Skipped 1 page(s) due to parsing errors"
    assert "Recovered page 1." in result["markdown"]
    assert "Recovered page 3." in result["markdown"]
    assert SKIPPED.format(n=2) in result["markdown"]
    assert result["metadata"]["pages"] == 3


def test_every_page_failing_still_returns_a_document_of_placeholders(convert):
    convert(whole_document_error=TABLES_ERROR, fail_on={0, 1, 2})

    result = parse()

    assert result["metadata"]["skipped_pages"] == [1, 2, 3]
    assert all(SKIPPED.format(n=n) in result["markdown"] for n in (1, 2, 3))


@pytest.mark.parametrize(
    "error", [RuntimeError("boom"), AttributeError("something else"), ValueError("bad value")]
)
def test_an_unrecognized_whole_document_error_propagates(convert, error):
    convert(whole_document_error=error)

    with pytest.raises(type(error)):
        parse()


def test_an_unrecognized_error_on_one_page_propagates_during_the_fallback(convert):
    convert(whole_document_error=TABLES_ERROR, fail_on={1}, error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        parse()


def test_a_corrupt_pdf_raises_instead_of_returning_a_result(convert):
    convert()

    with pytest.raises(Exception):
        parse(b"this is not a pdf")


# ---- task-aware path (task id): page by page, cancellable, resumable ----


def test_a_task_aware_parse_converts_one_page_at_a_time(convert):
    converter = convert()

    result = parse(task_id="task-1")

    assert converter.calls == [[0], [1], [2]]
    assert result["metadata"]["preempted"] is False
    assert result["metadata"]["last_page_parsed"] == 2
    assert result["metadata"]["start_page"] == 0


def test_a_task_aware_parse_skips_a_bad_page_the_same_way(convert):
    convert(fail_on={1})

    result = parse(task_id="task-1")

    assert result["metadata"]["skipped_pages"] == [2]
    assert SKIPPED.format(n=2) in result["markdown"]
    assert result["metadata"]["last_page_parsed"] == 2


def test_a_task_aware_parse_propagates_an_unrecognized_page_error(convert):
    convert(fail_on={0}, error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        parse(task_id="task-1")


def test_resuming_starts_at_the_cursor_and_reports_it(convert):
    converter = convert()

    result = parse(task_id="task-1", start_page=1)

    assert converter.calls == [[1], [2]]
    assert result["metadata"]["start_page"] == 1
    assert "Recovered page 1." not in result["markdown"]
    assert result["metadata"]["pages"] == 3  # the document's page count, not the pages returned


def test_a_cursor_past_the_end_converts_nothing(convert):
    converter = convert()

    result = parse(task_id="task-1", start_page=9)

    assert converter.calls == []
    assert result["markdown"] == ""
    assert result["metadata"]["last_page_parsed"] == -1


def test_a_cancel_flag_stops_the_parse_at_a_page_boundary(convert, monkeypatch):
    converter = convert()
    flags = iter([False, True])  # not cancelled before page 1, cancelled before page 2
    monkeypatch.setattr(service, "_is_cancelled", lambda task_id: next(flags))

    result = parse(task_id="task-1", progress_ceiling=0.9)

    assert converter.calls == [[0]]
    assert result["metadata"]["preempted"] is True
    assert result["metadata"]["last_page_parsed"] == 0
    assert result["markdown"].strip() == "Recovered page 1."


def test_work_past_the_progress_ceiling_ignores_a_cancel_flag(convert, monkeypatch):
    converter = convert()
    monkeypatch.setattr(service, "_is_cancelled", lambda task_id: True)

    result = parse(task_id="task-1", progress_ceiling=0.0)  # every page is "nearly finished"

    assert converter.calls == [[0], [1], [2]]
    assert result["metadata"]["preempted"] is False


def test_the_wall_clock_deadline_returns_partial_markdown_marked_truncated(convert, monkeypatch):
    convert(delay=0.2)
    monkeypatch.setattr(service, "FAST_PARSER_WALL_DEADLINE_S", 0.1)

    result = parse(task_id="task-1")

    assert result["metadata"]["truncated"] == "deadline"
    assert result["metadata"]["last_page_parsed"] == 0  # the page in progress finished
    assert result["metadata"]["preempted"] is False  # a deadline is not a preemption


def test_a_non_positive_deadline_means_no_deadline(convert, monkeypatch):
    convert(delay=0.05)
    monkeypatch.setattr(service, "FAST_PARSER_WALL_DEADLINE_S", 0)

    result = parse(task_id="task-1")

    assert "truncated" not in result["metadata"]
    assert result["metadata"]["last_page_parsed"] == 2


# ---- page-range path (/parse-pages) ----


def test_a_page_range_converts_exactly_those_pages(convert):
    converter = convert()

    result = service.parse_pdf_page_range(
        pdf_with_pages(), "doc.pdf", 1, 3, extract_images_flag=False
    )

    assert converter.calls == [[1, 2]]
    assert result["metadata"]["pages_parsed"] == 2
    assert result["metadata"]["start_page"] == 1 and result["metadata"]["end_page"] == 3


def test_a_page_range_is_clamped_to_the_document(convert):
    converter = convert()

    result = service.parse_pdf_page_range(
        pdf_with_pages(), "doc.pdf", 2, 99, extract_images_flag=False
    )

    assert converter.calls == [[2]]
    assert result["metadata"]["end_page"] == 3


def test_an_empty_page_range_converts_nothing(convert):
    converter = convert()

    result = service.parse_pdf_page_range(
        pdf_with_pages(), "doc.pdf", 5, None, extract_images_flag=False
    )

    assert converter.calls == []
    assert result["markdown"] == "" and result["metadata"]["pages_parsed"] == 0


# ---- page-range path: the same recovery as the full document ----


def parse_range(start, end):
    return service.parse_pdf_page_range(
        pdf_with_pages(5), "doc.pdf", start, end, extract_images_flag=False
    )


@pytest.mark.parametrize("error", [TABLES_ERROR, TEXTPAGE_ERROR])
def test_a_range_with_a_bad_page_is_recovered_page_by_page(convert, error):
    converter = convert(fail_on={2}, error=error, whole_document_error=None)
    converter.fail_on = {2}  # pages=[1, 2, 3] as a whole fails, as the library does

    result = parse_range(1, 4)

    assert converter.calls == [[1, 2, 3], [1], [2], [3]]
    assert result["metadata"]["skipped_pages"] == [3]  # one-based and relative to the document
    assert result["metadata"]["warning"] == "Skipped 1 page(s) due to parsing errors"
    assert SKIPPED.format(n=3) in result["markdown"]
    assert "Recovered page 2." in result["markdown"] and "Recovered page 4." in result["markdown"]
    assert result["metadata"]["pages_parsed"] == 3  # the size of the range, skipped pages included


def test_a_clean_range_has_no_skipped_page_keys(convert):
    convert()

    metadata = parse_range(0, 2)["metadata"]

    assert "skipped_pages" not in metadata and "warning" not in metadata


def test_a_range_where_every_page_fails_still_returns_placeholders(convert):
    convert(fail_on={0, 1})

    result = parse_range(0, 2)

    assert result["metadata"]["skipped_pages"] == [1, 2]
    assert SKIPPED.format(n=1) in result["markdown"] and SKIPPED.format(n=2) in result["markdown"]


def test_an_unrecognized_error_in_a_range_propagates(convert):
    convert(fail_on={1}, error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        parse_range(0, 3)


def test_an_unrecognized_error_on_one_page_of_a_range_propagates_during_recovery(monkeypatch):
    class Mixed(Converter):
        def __call__(self, document, pages=None, **kwargs):
            if pages == [0, 1]:
                raise TABLES_ERROR  # the whole range fails with a recognized error...
            if pages == [1]:
                raise RuntimeError("boom")  # ...then one page fails with a different one
            return super().__call__(document, pages, **kwargs)

    converter = Mixed()
    monkeypatch.setattr(service.pymupdf4llm, "to_markdown", converter)

    with pytest.raises(RuntimeError, match="boom"):
        parse_range(0, 2)


# ---- a cancelled parse keeps its finished pages' images ----


@pytest.fixture
def spy_on_images(monkeypatch):
    calls = []

    def extract(doc, **kwargs):
        calls.append(list(kwargs["page_range"]))
        return {"images": [], "truncated": None, "total_bytes": 0}

    monkeypatch.setattr(service, "extract_images", extract)
    return calls


def test_a_cancelled_parse_returns_the_images_of_the_pages_it_finished(
    convert, monkeypatch, spy_on_images
):
    convert()
    flags = iter([False, True])
    monkeypatch.setattr(service, "_is_cancelled", lambda task_id: next(flags))

    result = parse(task_id="task-1", progress_ceiling=0.9, extract_images_flag=True)

    assert result["metadata"]["preempted"] is True
    # The 409 hands these back with the Markdown, so a resume loses nothing of the first pages.
    assert spy_on_images == [[0]]
    assert result["images"] == []


def test_a_completed_task_aware_parse_still_extracts_the_images_of_its_pages(
    convert, spy_on_images
):
    convert()

    parse(task_id="task-1", extract_images_flag=True)

    assert spy_on_images == [[0, 1, 2]]


def test_a_deadline_truncated_parse_still_extracts_images_for_the_pages_it_returns(
    convert, monkeypatch, spy_on_images
):
    convert(delay=0.2)
    monkeypatch.setattr(service, "FAST_PARSER_WALL_DEADLINE_S", 0.1)

    result = parse(task_id="task-1", extract_images_flag=True)

    assert result["metadata"]["truncated"] == "deadline"
    assert spy_on_images == [[0]]  # a deadline returns a 200, so its pages keep their images
