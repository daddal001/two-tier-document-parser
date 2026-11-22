"""The Colab notebook stays valid, tests the path the CUDA image runs, and its checks work."""

import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "notebooks" / "accurate_t4_check.ipynb"


@pytest.fixture(scope="module")
def cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def code_cells(cells):
    return [source for kind, source in cells if kind == "code"]


def test_every_code_cell_compiles(cells):  # saved outputs are the recorded run and stay
    for source in code_cells(cells):
        ast.parse(source)


def test_it_tests_the_tier_and_engine_the_cuda_image_serves(cells):
    text = "\n".join(code_cells(cells))

    assert 'TIER = "standard"' in text  # on CUDA the service always selects the vision model
    assert '"MINERU_MODEL_VLM_ENGINE": "vllm"' in text
    assert "deploy/constraints/accurate-gpu.txt" in text
    # vLLM's dependencies stay at the base image's versions (vllm-base.txt) ...
    assert "deploy/constraints/vllm-base.txt" in text
    assert "-c {WORK}/base-constraints.txt" in text
    # ... except OpenTelemetry: vLLM 0.34.0 needs >=1.45.1 while the base image ships 1.40.0, so
    # constraining to the base makes the install unsatisfiable (it failed on a Colab T4).
    assert 'not line.startswith("opentelemetry")' in text


def test_it_installs_the_same_fixed_pillow_as_the_cuda_image(cells):
    text = "\n".join(code_cells(cells))
    dockerfile = (ROOT / "deploy" / "Dockerfile.accurate").read_text()

    assert "deploy/constraints/accurate-gpu.txt" in text and "accurate-gpu.txt" in dockerfile


def test_the_checks_import_only_what_the_notebook_kernel_has_installed(cells):
    text = "\n".join(code_cells(cells))

    # The checks import pypdf in the notebook's own kernel, not in the isolated environment.
    assert "from pypdf import" in text
    assert 'sh("pip install -q uv pypdf")' in text


def test_the_vllm_release_is_read_from_the_dockerfile_it_must_match(cells):
    text = "\n".join(code_cells(cells))
    dockerfile = (ROOT / "deploy" / "Dockerfile.accurate").read_text()
    pattern = re.search(r're\.search\(r"(.+?)", dockerfile', text).group(1)

    assert re.search(pattern, dockerfile, re.M), "the notebook cannot find the vLLM release"


def test_the_notebook_checks_pass_against_the_real_app_with_a_fake_parser(cells, tmp_path):
    helpers = next(source for source in code_cells(cells) if "def run_checks" in source)
    namespace = {}
    exec("import time, urllib.request, urllib.error\n" + helpers, namespace)  # noqa: S102

    port = 18900 + os.getpid() % 500
    log_path = tmp_path / "service.log"
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}",
        "WORKERS": "1",
        "MAX_INFLIGHT_DOCUMENTS": "1",
        "OTEL_SDK_DISABLED": "true",
    }
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "tests.fake_accurate_app:app", "--port", str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                assert server.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline
                time.sleep(0.3)
        results = namespace["run_checks"](
            base,
            str(ROOT / "examples" / "data" / "sample.pdf"),
            expect_gpu=False,
            log_text=log_path.read_text(),
        )
    finally:
        server.terminate()
        server.wait(15)

    failed = {name: detail for name, (passed, detail) in results.items() if not passed}
    assert not failed, failed
    assert len(results) == 5


def helper_namespace(cells):
    helpers = next(source for source in code_cells(cells) if "def run_checks" in source)
    namespace = {}
    exec("import time, urllib.request, urllib.error\n" + helpers, namespace)  # noqa: S102
    return namespace


def test_the_markdown_pictures_are_resolved_for_display_by_image_id(cells):
    inline = helper_namespace(cells)["inline_images"]
    images = [{"image_id": "page_0_img_0", "image_base64": "QUJD"}]

    shown = inline("a ![fig](page_0_img_0) b ![](page_9_img_9)", images)

    assert shown == "a ![fig](data:image/png;base64,QUJD) b ![](page_9_img_9)"


def test_a_failed_check_prints_the_service_log_for_its_correlation_id(cells, tmp_path, capsys):
    show = helper_namespace(cells)["show_failure_log"]
    log = tmp_path / "service.log"
    log.write_text(
        '{"message": "started"}\n'
        '{"message": "Parse failed", "correlation_id": "abc123def456", "exception": "Traceback ... boom"}\n'
        '{"message": "other"}\n'
    )

    show(
        str(log),
        {
            "a parse": (
                False,
                "AssertionError: (500, {'detail': 'Parsing failed (correlation_id=abc123def456)'})",
            )
        },
    )

    output = capsys.readouterr().out
    assert "correlation_id abc123def456" in output
    assert "Traceback ... boom" in output
    assert "last 60 lines" in output


def test_it_parses_and_shows_the_paper_the_visualization_notebook_uses(cells):
    text = "\n".join(code_cells(cells))
    visualization = (ROOT / "examples" / "notebooks" / "parser_visualization.ipynb").read_text()

    assert "examples/data/sample.pdf" in text
    assert "../data/sample.pdf" in visualization  # the same file
    for display in (
        "inline_images(result",
        'result["images"]',
        'result["tables"]',
        'result["formulas"]',
    ):
        assert display in text
    assert "CC BY 4.0" in "\n".join(source for kind, source in cells if kind == "markdown")


def test_the_service_runs_with_the_venv_tools_on_path_and_without_the_flashinfer_compile(cells):
    """vLLM's FlashInfer sampler compiles a kernel on first use and needs ``ninja`` on PATH.

    The service runs from the venv without activating it, so the venv's bin directory must be
    on PATH; the sampler is also turned off, because a Colab venv has none of the prebuilt
    kernels the CUDA image ships and the compile needs a CUDA toolkit that matches torch.
    """
    start = next(s for s in code_cells(cells) if "uvicorn" in s and "Popen" in s)

    assert 'f"{WORK}/venv/bin:"' in start and "PATH" in start
    assert '"VLLM_USE_FLASHINFER_SAMPLER": "0"' in start
