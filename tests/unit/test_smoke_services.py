"""Tests for the real-document validation harness."""

import base64
import subprocess
from unittest.mock import Mock

import pytest

from scripts.smoke_services import (
    deployment_profiles,
    docker_gpu_available,
    render_html,
    validate_result,
)


def test_smoke_always_runs_cpu_and_adds_cuda_when_available():
    assert deployment_profiles(False) == ["cpu"]
    assert deployment_profiles(True) == ["cpu", "cuda"]


def test_cuda_profile_requires_a_visible_nvidia_gpu(monkeypatch):
    monkeypatch.setattr("scripts.smoke_services.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "scripts.smoke_services.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=0, stdout="GPU 0: Test accelerator", stderr=""
        ),
    )

    assert docker_gpu_available() is True


def test_cuda_profile_is_skipped_when_nvidia_smi_is_missing(monkeypatch):
    run = Mock()
    monkeypatch.setattr("scripts.smoke_services.shutil.which", lambda _: None)
    monkeypatch.setattr("scripts.smoke_services.subprocess.run", run)

    assert docker_gpu_available() is False
    run.assert_not_called()


def fast_result():
    """Return a minimal valid fast-parser response with a zero-based image page."""
    return {
        "markdown": "# Sample heading\n![chart](data:image/png;base64,aW1hZ2UtZGF0YQ==)",
        "metadata": {
            "pages": 1,
            "processing_time_ms": 12,
            "parser": "pymupdf4llm",
            "version": "test",
            "filename": "sample.pdf",
        },
        "images": [
            {
                "image_id": "chart-1",
                "image_base64": "data:image/png;base64,aW1hZ2UtZGF0YQ==",
                "page": 0,
            }
        ],
    }


def test_validate_fast_response_and_real_page_index():
    summary, decoded_images = validate_result(
        fast_result(), "sample.pdf", ["Sample heading"], 1, "fast", 0.5
    )

    assert summary["metadata"]["pages"] == 1
    assert summary["images"] == 1
    assert summary["tables"] is None
    assert summary["formulas"] is None
    assert decoded_images == [b"image-data"]


def test_validate_accurate_response_includes_multimodal_fields():
    result = fast_result()
    result["metadata"].update(
        parser="mineru", backend="pipeline", gpu_used=False, accuracy_tier="high"
    )
    result["tables"] = [{"table_id": "table-1", "page": 0, "markdown": "| A |"}]
    result["formulas"] = [{"formula_id": "formula-1", "page": 0, "latex": "x^2"}]

    summary, _ = validate_result(result, "sample.pdf", [], 1, "accurate", 1.0)

    assert summary["tables"] == 1
    assert summary["formulas"] == 1


def test_validate_result_rejects_page_count_mismatch():
    with pytest.raises(AssertionError, match="expected 2"):
        validate_result(fast_result(), "sample.pdf", [], 2, "fast", 0.1)


def test_html_report_shows_images_and_avoids_inline_base64_in_markdown(tmp_path):
    result = fast_result()
    summary, _ = validate_result(result, "sample.pdf", [], 1, "fast", 0.5)
    output = tmp_path / "report.html"
    report = {
        "fixture": "sample.pdf",
        "fixture_sha256": "test-hash",
        "page_count": 1,
        "transport": "docker-compose-http",
        "health": {},
        "results": {"fast": summary},
    }

    render_html(report, {"fast": result}, output)
    rendered = output.read_text(encoding="utf-8")

    assert "Sample heading" in rendered
    assert "embedded image omitted here" in rendered
    assert "Not reported separately by the fast API" in rendered
    assert "N/A" in rendered
    assert "docker-compose-http" in rendered
    assert f"src=\"data:image/png;base64,{base64.b64encode(b'image-data').decode()}\"" in rendered
