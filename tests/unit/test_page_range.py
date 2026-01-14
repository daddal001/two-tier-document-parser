"""Unit tests for page-range parsing functionality."""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from two_tier_parser.fast.service import parse_pdf_page_range
from two_tier_parser.fast.models import PageRangeParseResponse, PageRangeMetadata


def test_parse_first_two_pages(sample_pdf_path):
    """Parse only first 2 pages of a multi-page PDF."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=0, end_page=2)

    assert "markdown" in result
    assert "metadata" in result

    metadata = result["metadata"]
    assert metadata["start_page"] == 0
    assert metadata["end_page"] == 2
    # pages_parsed should be min(2, total_pages)
    assert metadata["pages_parsed"] == min(2, metadata["total_pages"])
    assert metadata["total_pages"] > 0


def test_parse_middle_pages(sample_pdf_path):
    """Parse pages 1-3 (middle pages) of a PDF."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=1, end_page=4)

    metadata = result["metadata"]
    assert metadata["start_page"] == 1
    # end_page is clamped to total_pages if document is shorter
    expected_pages = min(3, max(0, metadata["total_pages"] - 1))
    assert metadata["pages_parsed"] == expected_pages


def test_parse_to_end(sample_pdf_path):
    """Parse from page 1 to end (no end_page specified)."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=1, end_page=None)

    metadata = result["metadata"]
    assert metadata["start_page"] == 1
    assert metadata["end_page"] == metadata["total_pages"]
    assert metadata["pages_parsed"] == metadata["total_pages"] - 1


def test_invalid_range_clamped(sample_pdf_path):
    """Invalid page range (start > total_pages) is clamped to valid bounds."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=100, end_page=200)

    metadata = result["metadata"]
    assert metadata["pages_parsed"] == 0
    assert result["markdown"] == ""
    # start_page should be clamped to total_pages
    assert metadata["start_page"] == metadata["total_pages"]


def test_single_page_document(single_page_pdf_path):
    """Handle single-page document with range request."""
    with open(single_page_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "single.pdf", start_page=0, end_page=10)

    metadata = result["metadata"]
    assert metadata["pages_parsed"] == 1
    assert metadata["total_pages"] == 1
    assert metadata["end_page"] == 1  # Clamped to total_pages
    assert len(result["markdown"]) > 0


def test_metadata_structure(sample_pdf_path):
    """Verify metadata contains all required fields."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=0, end_page=1)

    metadata = result["metadata"]
    # All required fields must be present
    assert "filename" in metadata
    assert "total_pages" in metadata
    assert "pages_parsed" in metadata
    assert "start_page" in metadata
    assert "end_page" in metadata
    assert "processing_time_ms" in metadata
    assert "parser" in metadata
    assert "version" in metadata

    # Validate types
    assert isinstance(metadata["total_pages"], int)
    assert isinstance(metadata["pages_parsed"], int)
    assert isinstance(metadata["processing_time_ms"], int)
    assert metadata["parser"] == "pymupdf4llm"


def test_page_range_response_model(sample_pdf_path):
    """Test that result matches PageRangeParseResponse model."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=0, end_page=2)

    # Should be able to create PageRangeParseResponse from result
    response = PageRangeParseResponse(**result)
    assert response.markdown is not None
    assert response.metadata is not None
    assert isinstance(response.metadata, PageRangeMetadata)


def test_empty_range(sample_pdf_path):
    """Test that start_page == end_page returns empty result."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Start at page 2, end at page 2 (empty range)
    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=2, end_page=2)

    metadata = result["metadata"]
    assert metadata["pages_parsed"] == 0
    assert result["markdown"] == ""


def test_parse_all_pages(sample_pdf_path):
    """Test parsing all pages (start=0, end=None)."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = parse_pdf_page_range(pdf_bytes, "test.pdf", start_page=0, end_page=None)

    metadata = result["metadata"]
    assert metadata["pages_parsed"] == metadata["total_pages"]
    assert metadata["start_page"] == 0
    assert metadata["end_page"] == metadata["total_pages"]
    assert len(result["markdown"]) > 0
