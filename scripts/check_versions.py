#!/usr/bin/env python3
"""Fail when the package, its metadata and the changelog name different releases.

Checks that pyproject.toml's version, ``__version__`` in the package and the newest released
section of CHANGELOG.md agree. ``API_VERSION`` is the HTTP interface version and may differ.
The README's front-matter version is a document revision, not the package version, and is not
compared.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)


def read_versions(root):
    """Return the three versions as found, with None for any that is missing."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())["project"].get("version")
    init = (root / "src" / "two_tier_parser" / "__init__.py").read_text()
    package = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.MULTILINE)
    headings = [
        m["version"]
        for m in HEADING.finditer((root / "CHANGELOG.md").read_text())
        if m["version"].lower() != "unreleased"
    ]
    return {
        "pyproject": pyproject,
        "package": package[1] if package else None,
        "changelog": headings[0] if headings else None,
    }


def problems(root):
    """Return a human-readable line for each disagreement; empty when all agree."""
    found = read_versions(root)
    messages = []
    if found["changelog"] is None:
        messages.append("CHANGELOG.md has no released version heading (## [x.y.z])")
    if found["pyproject"] is None:
        messages.append("pyproject.toml declares no version")
    if found["package"] is None:
        messages.append("src/two_tier_parser/__init__.py defines no __version__")
    if messages:
        return messages
    expected = found["pyproject"]
    if found["package"] != expected:
        messages.append(f"__version__ is {found['package']} but pyproject.toml says {expected}")
    if found["changelog"] != expected:
        messages.append(
            f"the newest CHANGELOG release is {found['changelog']} but pyproject.toml says {expected}"
        )
    return messages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    messages = problems(args.root)
    for message in messages:
        print(f"version mismatch: {message}")
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
