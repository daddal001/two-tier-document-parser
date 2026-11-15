"""The build context holds what the Dockerfiles copy and nothing else."""

import re
from pathlib import Path

import pytest

from tests.dockerignore import excluded, parse

ROOT = Path(__file__).resolve().parents[2]
RULES = parse((ROOT / ".dockerignore").read_text()) if (ROOT / ".dockerignore").exists() else []


def copy_sources():
    """Every (Dockerfile, source) pair from the COPY instructions that read the build context."""
    pairs = []
    for dockerfile in sorted((ROOT / "deploy").glob("Dockerfile*")):
        text = dockerfile.read_text().replace("\\\n", " ")
        for line in text.splitlines():
            if not line.startswith("COPY ") or "--from=" in line:
                continue
            sources = [w for w in line.split()[1:] if not w.startswith("--")][
                :-1
            ]  # last is the destination
            pairs.extend((dockerfile.name, source) for source in sources)
    return pairs


def files_under(source):
    path = ROOT / source
    if path.is_dir():
        return [str(p.relative_to(ROOT)) for p in path.rglob("*") if p.is_file()]
    return [source]


def test_the_file_exists():
    assert (ROOT / ".dockerignore").exists()


@pytest.mark.parametrize(("dockerfile", "source"), copy_sources())
def test_everything_a_dockerfile_copies_survives_the_ignore_rules(dockerfile, source):
    assert (ROOT / source).exists(), f"{dockerfile} copies {source}, which does not exist"
    if not files_under(source):
        pytest.skip(f"{source} is an empty submodule checkout; the image builds check it")
    kept = [name for name in files_under(source) if not excluded(name, RULES)]
    assert kept, f"{dockerfile}: nothing of {source} is left in the build context"
    dropped = [
        name
        for name in files_under(source)
        if excluded(name, RULES) and "__pycache__" not in name and not name.endswith(".pyc")
    ]
    assert dropped == [], f"{dockerfile}: {source} loses {dropped[:3]}"


@pytest.mark.parametrize(
    "path",
    [
        ".git",
        ".git/config",
        ".codeql-cache/2.27.2/bundle",
        ".venv/lib/python3.12/site.py",
        "parser_results/out.json",
        "dist/pkg.whl",
        "tests/unit/test_x.py",
        "docs/API.md",
        "examples/data/sample.pdf",
        "styles/config/x.txt",
        ".github/workflows/ci.yml",
        ".claude/settings.local.json",
        ".env",
        ".env.local",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "src/two_tier_parser/__pycache__/x.cpython-312.pyc",
        ".pytest_cache/v/cache",
        ".mypy_cache/3.12/x.json",
    ],
)
def test_development_files_stay_out_of_the_build_context(path):
    assert excluded(path, RULES), f"{path} would be sent to the Docker daemon"


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "src/two_tier_parser/fast/main.py",
        "deploy/Dockerfile.fast",
        "deploy/entrypoint-accurate.sh",
        "deploy/constraints/accurate-cpu.txt",
        "deploy/constraints/accurate-gpu.txt",
        "MinerU/pyproject.toml",
        "MinerU/LICENSE.md",  # MinerU's build reads its license file, so markdown there is kept
        "MinerU/README.md",
        "MinerU/mineru/__init__.py",
    ],
)
def test_what_the_images_need_is_kept(path):
    assert not excluded(path, RULES), f"{path} would be missing from the build context"


def test_markdown_is_excluded_only_at_the_top_level():
    text = (ROOT / ".dockerignore").read_text()

    assert not re.search(r"^\*\*/\*\.md", text, re.M), "a recursive rule would break MinerU's build"


@pytest.mark.parametrize(
    ("patterns", "path", "expected"),
    [
        ("docs", "docs/a/b.md", True),
        ("docs", "other/docs", False),
        ("*.md", "README.md", True),
        ("*.md", "sub/README.md", False),  # * stays inside one segment
        ("*.md\n!README.md", "README.md", False),
        ("*.md\n!README.md", "NOTES.md", True),
        ("**/__pycache__", "a/b/__pycache__/x.pyc", True),
        ("**/__pycache__", "__pycache__/x.pyc", True),
        (".env*", ".env.local", True),
        ("# comment\n\n", "anything", False),
    ],
)
def test_the_matcher_follows_dockers_rules(patterns, path, expected):
    assert excluded(path, parse(patterns)) is expected
