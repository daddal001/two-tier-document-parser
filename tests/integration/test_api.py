"""Integration tests for API endpoints."""
import pytest
import requests


@pytest.mark.integration
def test_fast_parser_health(fast_parser_url):
    """Test fast parser health endpoint."""
    response = requests.get(f"{fast_parser_url}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "workers" in data
    assert "parser" in data


@pytest.mark.integration
def test_accurate_parser_health(accurate_parser_url):
    """Test accurate parser health endpoint."""
    response = requests.get(f"{accurate_parser_url}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "workers" in data
    assert "gpu_available" in data


@pytest.mark.integration
def test_fast_parser_parse(sample_pdf_path, fast_parser_url):
    """Test fast parser parse endpoint."""
    with open(sample_pdf_path, "rb") as f:
        files = {"file": (sample_pdf_path.name, f, "application/pdf")}
        response = requests.post(f"{fast_parser_url}/parse", files=files, timeout=30)
    
    assert response.status_code == 200
    data = response.json()
    assert "markdown" in data
    assert "metadata" in data
    assert len(data["markdown"]) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_accurate_parser_parse(single_page_pdf_path, accurate_parser_url):
    """Test accurate parser parse endpoint with single page for faster testing."""
    with open(single_page_pdf_path, "rb") as f:
        files = {"file": (single_page_pdf_path.name, f, "application/pdf")}
        response = requests.post(
            f"{accurate_parser_url}/parse", 
            files=files, 
            timeout=300  # Reduced timeout since we're only testing 1 page
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "markdown" in data
    assert "metadata" in data
    assert "images" in data
    assert "tables" in data
    assert "formulas" in data
    # Verify it's a single page
    assert data["metadata"]["pages"] == 1


@pytest.mark.integration
def test_fast_parser_invalid_file(fast_parser_url):
    """Test fast parser with invalid file."""
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = requests.post(f"{fast_parser_url}/parse", files=files, timeout=10)
    assert response.status_code == 400


@pytest.mark.integration
def test_accurate_parser_invalid_file(accurate_parser_url):
    """Test accurate parser with invalid file."""
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = requests.post(f"{accurate_parser_url}/parse", files=files, timeout=10)
    assert response.status_code == 400

