"""Tests for Pydantic models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from two_tier_parser.fast.models import ParseResponse as FastParseResponse, HealthResponse as FastHealthResponse
from two_tier_parser.accurate.models import (
    ParseResponse as AccurateParseResponse,
    HealthResponse as AccurateHealthResponse,
    ImageData,
    TableData,
    FormulaData
)


def test_fast_parse_response():
    """Test Fast ParseResponse model."""
    response = FastParseResponse(
        markdown="# Test\n\nContent",
        metadata={
            "pages": 1,
            "processing_time_ms": 100,
            "parser": "pymupdf4llm",
            "version": "1.0.0"
        }
    )
    assert response.markdown == "# Test\n\nContent"
    assert response.metadata["pages"] == 1


def test_fast_health_response():
    """Test Fast HealthResponse model."""
    response = FastHealthResponse(
        status="healthy",
        workers=4,
        no_gil=True
    )
    assert response.status == "healthy"
    assert response.workers == 4
    assert response.no_gil is True


def test_accurate_parse_response():
    """Test Accurate ParseResponse model."""
    response = AccurateParseResponse(
        markdown="# Test\n\nContent",
        metadata={
            "pages": 1,
            "processing_time_ms": 1000,
            "parser": "mineru",
            "backend": "transformers",
            "gpu_used": True,
            "accuracy_tier": "very-high",
            "version": "1.0.0",
            "filename": "test.pdf"
        },
        images=[],
        tables=[],
        formulas=[]
    )
    assert response.markdown == "# Test\n\nContent"
    assert len(response.images) == 0
    assert len(response.tables) == 0
    assert len(response.formulas) == 0
    assert response.metadata.backend == "transformers"
    assert response.metadata.gpu_used is True
    assert response.metadata.accuracy_tier == "very-high"
    assert response.metadata.filename == "test.pdf"


def test_accurate_health_response():
    """Test Accurate HealthResponse model."""
    response = AccurateHealthResponse(
        status="healthy",
        workers=2,
        gpu_available=True
    )
    assert response.status == "healthy"
    assert response.workers == 2
    assert response.gpu_available is True


def test_image_data():
    """Test ImageData model."""
    image = ImageData(
        image_id="img_1",
        image_base64="base64data",
        page=1,
        bbox=[0.0, 0.0, 100.0, 100.0]
    )
    assert image.image_id == "img_1"
    assert image.page == 1
    assert len(image.bbox) == 4


def test_table_data():
    """Test TableData model."""
    table = TableData(
        table_id="table_1",
        markdown="| Col |\n|-----|\n| Val |",
        page=1,
        bbox=[0.0, 0.0, 200.0, 100.0]
    )
    assert table.table_id == "table_1"
    assert "Col" in table.markdown


def test_formula_data():
    """Test FormulaData model."""
    formula = FormulaData(
        formula_id="formula_1",
        latex="x^2 + y^2 = z^2",
        page=1,
        bbox=[0.0, 0.0, 50.0, 50.0]
    )
    assert formula.formula_id == "formula_1"
    assert "^2" in formula.latex

