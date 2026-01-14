"""Unit tests for accurate parser service."""
import importlib.util
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from two_tier_parser.accurate.models import (
    FormulaData,
    ImageData,
    ParseResponse,
    TableData,
)
from two_tier_parser.accurate.service import parse_pdf

# Check if mineru is available (required for accurate parser)
# MinerU is a git submodule, so check both installed package and submodule path
MINERU_AVAILABLE = importlib.util.find_spec("mineru") is not None
if not MINERU_AVAILABLE:
    # Try importing from submodule directory
    mineru_path = Path(__file__).parent.parent.parent.parent / "MinerU"
    if mineru_path.exists():
        sys.path.insert(0, str(mineru_path))
        MINERU_AVAILABLE = importlib.util.find_spec("mineru") is not None


@pytest.mark.slow
def test_parse_pdf_basic(single_page_pdf_path):
    """Test basic PDF parsing with accurate parser (single page for faster testing)."""
    if not MINERU_AVAILABLE:
        pytest.skip("MinerU not installed. Install with: pip install -e '.[fast,accurate,dev]'")
    
    with open(single_page_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = parse_pdf(pdf_bytes, single_page_pdf_path.name)
    
    # Handle GPU out of memory errors gracefully
    if "error" in result:
        if "CUDA out of memory" in result.get("error", ""):
            pytest.skip(f"GPU out of memory - restart container to free GPU memory: {result['error']}")
        elif "No module named 'mineru'" in result.get("error", ""):
            pytest.skip("MinerU not installed. Install with: pip install -e '.[fast,accurate,dev]'")
        else:
            pytest.fail(f"Parsing failed: {result.get('error', 'Unknown error')}")
    
    assert "markdown" in result
    assert "metadata" in result
    assert "images" in result
    assert "tables" in result
    assert "formulas" in result
    assert isinstance(result["markdown"], str)
    assert isinstance(result["images"], list)
    assert isinstance(result["tables"], list)
    assert isinstance(result["formulas"], list)
    # Verify it's a single page
    assert result["metadata"]["pages"] == 1


@pytest.mark.slow
def test_parse_pdf_metadata(single_page_pdf_path):
    """Test that metadata is correctly structured (single page for faster testing)."""
    if not MINERU_AVAILABLE:
        pytest.skip("MinerU not installed. Install with: pip install -e '.[fast,accurate,dev]'")
    
    with open(single_page_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = parse_pdf(pdf_bytes, single_page_pdf_path.name)
    
    # Handle GPU out of memory errors gracefully
    if "error" in result:
        if "CUDA out of memory" in result.get("error", ""):
            pytest.skip(f"GPU out of memory - restart container to free GPU memory: {result['error']}")
        elif "No module named 'mineru'" in result.get("error", ""):
            pytest.skip("MinerU not installed. Install with: pip install -e '.[fast,accurate,dev]'")
        else:
            pytest.fail(f"Parsing failed: {result.get('error', 'Unknown error')}")
    
    metadata = result["metadata"]
    assert "pages" in metadata
    assert "processing_time_ms" in metadata
    assert isinstance(metadata["pages"], int)
    assert metadata["pages"] == 1  # Should be exactly 1 page
    assert "backend" in metadata
    assert "gpu_used" in metadata
    assert "accuracy_tier" in metadata


@pytest.mark.slow
def test_parse_response_model(single_page_pdf_path):
    """Test that result matches ParseResponse model (single page for faster testing)."""
    if not MINERU_AVAILABLE:
        pytest.skip("MinerU not installed. Install with: pip install -e '.[fast,accurate,dev]'")
    
    with open(single_page_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result = parse_pdf(pdf_bytes, single_page_pdf_path.name)
    
    # Handle GPU out of memory errors gracefully
    if "error" in result:
        if "CUDA out of memory" in result.get("error", ""):
            pytest.skip(f"GPU out of memory - restart container to free GPU memory: {result['error']}")
        elif "No module named 'mineru'" in result.get("error", ""):
            pytest.skip("MinerU not installed. Install with: pip install -e '.[fast,accurate,dev]'")
        else:
            pytest.fail(f"Parsing failed: {result.get('error', 'Unknown error')}")
    
    # Should be able to create ParseResponse from result
    response = ParseResponse(**result)
    assert response.markdown is not None
    assert response.metadata is not None
    assert isinstance(response.images, list)
    assert isinstance(response.tables, list)
    assert isinstance(response.formulas, list)
    # Verify it's a single page (metadata is a Pydantic model, use attribute access)
    assert response.metadata.pages == 1


def test_image_data_model():
    """Test ImageData model validation."""
    image_data = ImageData(
        image_id="test_img_1",
        image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        page=1,
        bbox=[0.0, 0.0, 100.0, 100.0]
    )
    assert image_data.image_id == "test_img_1"
    assert image_data.page == 1


def test_table_data_model():
    """Test TableData model validation."""
    table_data = TableData(
        table_id="test_table_1",
        markdown="| Col1 | Col2 |\n|------|------|\n| Val1 | Val2 |",
        page=1,
        bbox=[0.0, 0.0, 200.0, 100.0]
    )
    assert table_data.table_id == "test_table_1"
    assert "Col1" in table_data.markdown


def test_formula_data_model():
    """Test FormulaData model validation."""
    formula_data = FormulaData(
        formula_id="test_formula_1",
        latex="E = mc^2",
        page=1,
        bbox=[0.0, 0.0, 50.0, 50.0]
    )
    assert formula_data.formula_id == "test_formula_1"
    assert formula_data.latex == "E = mc^2"

