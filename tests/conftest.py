"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"

# Project root (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_pdf_path():
    """Path to sample PDF for testing."""
    # Use absolute path relative to project root
    pdf_path = PROJECT_ROOT / "examples" / "data" / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}")
    return pdf_path


@pytest.fixture
def single_page_pdf_path(sample_pdf_path):
    """Create a single-page PDF from the sample PDF for faster testing."""
    try:
        import pypdf
    except ImportError:
        try:
            import PyPDF2 as pypdf
        except ImportError:
            pytest.skip("pypdf or PyPDF2 required for single-page PDF creation")

    # Create temporary file for single-page PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        # Read original PDF and extract first page
        reader = pypdf.PdfReader(sample_pdf_path)
        if len(reader.pages) == 0:
            pytest.skip("Sample PDF has no pages")

        # Create new PDF with only first page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[0])

        # Write to temporary file
        with open(temp_path, "wb") as f:
            writer.write(f)

        yield temp_path
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def fast_parser_url():
    """Fast parser service URL."""
    return "http://localhost:8004"


@pytest.fixture
def accurate_parser_url():
    """Accurate parser service URL."""
    return "http://localhost:8005"
