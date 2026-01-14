"""Unit tests for fast parser service."""
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from two_tier_parser.fast.service import parse_pdf
from two_tier_parser.fast.models import ParseResponse


def test_parse_pdf_basic(sample_pdf_path):
    """Test basic PDF parsing."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = parse_pdf(pdf_bytes, sample_pdf_path.name)
    
    assert "markdown" in result
    assert "metadata" in result
    assert isinstance(result["markdown"], str)
    assert len(result["markdown"]) > 0


def test_parse_pdf_metadata(sample_pdf_path):
    """Test that metadata is correctly structured."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = parse_pdf(pdf_bytes, sample_pdf_path.name)
    
    metadata = result["metadata"]
    assert "pages" in metadata
    assert "processing_time_ms" in metadata
    assert isinstance(metadata["pages"], int)
    assert metadata["pages"] > 0


def test_parse_response_model(sample_pdf_path):
    """Test that result matches ParseResponse model."""
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = parse_pdf(pdf_bytes, sample_pdf_path.name)
    
    # Should be able to create ParseResponse from result
    response = ParseResponse(**result)
    assert response.markdown is not None
    assert response.metadata is not None


def test_parse_empty_pdf():
    """Test handling of empty PDF."""
    # Create minimal PDF bytes (this is a placeholder - actual test would need valid PDF)
    empty_bytes = b""
    
    # This should raise an error or return empty result
    # Adjust based on actual implementation
    with pytest.raises(Exception):
        parse_pdf(empty_bytes, "empty.pdf")

