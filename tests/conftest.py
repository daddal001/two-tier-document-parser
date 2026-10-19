"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

# Settings that change service behavior. A developer shell or CI job that exports any of them
# must not change test results.
_AMBIENT_SETTINGS = (
    "OBJECT_STORE_ENDPOINT",
    "OBJECT_STORE_ACCESS_KEY",
    "OBJECT_STORE_SECRET_KEY",
    "OBJECT_STORE_SECURE",
    "OBJECT_STORE_REGION",
    "OBJECT_STORE_HEALTH_TIMEOUT_SECONDS",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_SECURE",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    "MINIO_HEALTH_TIMEOUT_SECONDS",
    "PARSER_ALLOWED_BUCKETS",
    "SERVICE_NAME",
    "OTEL_SERVICE_NAME",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "LOG_FORMAT",
)


@pytest.fixture(autouse=True)
def _isolate_ambient_settings(monkeypatch):
    for name in _AMBIENT_SETTINGS:
        monkeypatch.delenv(name, raising=False)


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
        pytest.fail(f"Required sample PDF not found at {pdf_path}")
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
            pytest.fail("Install the dev extra: pypdf is required for single-page PDF creation")

    # Create temporary file for single-page PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        # Read original PDF and extract first page
        reader = pypdf.PdfReader(sample_pdf_path)
        if len(reader.pages) == 0:
            pytest.fail("Required sample PDF has no pages")

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


def pytest_addoption(parser):
    """Register explicit opt-ins for benchmarks, model inference, and live services."""
    parser.addoption("--run-bench", action="store_true", help="Run local performance harness")
    parser.addoption("--run-models", action="store_true", help="Run required real MinerU tests")
    parser.addoption(
        "--run-integration", action="store_true", help="Run tests against live services"
    )


def pytest_collection_modifyitems(config, items):
    """Skip model and live-service tests unless the corresponding opt-in is set."""
    for item in items:
        for marker, option in [("models", "--run-models"), ("integration", "--run-integration")]:
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"Explicit opt-in required: {option}"))
