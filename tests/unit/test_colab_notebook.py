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


def test_every_code_cell_compiles_and_nothing_is_saved_in_it(cells):
    notebook = json.loads(NOTEBOOK.read_text())
    for source in code_cells(cells):
        ast.parse(source)
    assert all(c.get("outputs", []) == [] for c in notebook["cells"] if c["cell_type"] == "code")


def test_it_tests_the_tier_and_engine_the_cuda_image_serves(cells):
    text = "\n".join(code_cells(cells))

    assert 'TIER = "standard"' in text  # on CUDA the service always selects the vision model
    assert '"MINERU_MODEL_VLM_ENGINE": "vllm"' in text
    assert "deploy/constraints/mineru-compat.txt" in text


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
