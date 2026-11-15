"""Update the pinned CodeQL manifest from validated official release metadata."""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "codeql-bundle.json"
RELEASES_API = "https://api.github.com/repos/github/codeql-action/releases?per_page=100"
ASSETS = (
    "codeql-bundle-linux-arm64.tar.gz",
    "codeql-bundle-linux64.tar.gz",
    "codeql-bundle-osx64.tar.gz",
    "codeql-bundle-win64.tar.gz",
)


def manifest_from_release(release, current_version):
    """Build a pin manifest from stable official GitHub release metadata.

    Args:
        release: GitHub releases API response with bundle assets and digests.
        current_version: Existing version used to reject accidental downgrades.

    Returns:
        Version and verified-origin URLs with published SHA-256 digests.

    Raises:
        ValueError: Metadata describes an unstable, older, incomplete, or
            unexpected release. This validates metadata, not archive contents;
            the installer verifies the actual downloaded bytes.
    """
    tag = release.get("tag_name", "")
    match = re.fullmatch(r"codeql-bundle-v(\d+\.\d+\.\d+)", tag)
    if release.get("draft") or release.get("prerelease") or not match:
        raise ValueError("Expected a stable official CodeQL bundle release")
    version = match[1]
    if tuple(map(int, version.split("."))) < tuple(map(int, current_version.split("."))):
        raise ValueError("Refusing to downgrade the pinned CodeQL version")
    assets = {}
    prefix = f"https://github.com/github/codeql-action/releases/download/{tag}/"
    for asset in release.get("assets", []):
        name = asset.get("name")
        if name not in ASSETS:
            continue
        if name in assets or asset.get("browser_download_url") != prefix + name:
            raise ValueError(f"Invalid or duplicate official asset: {name}")
        digest = asset.get("digest") or ""
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"Missing valid published SHA-256 for {name}")
        assets[name] = {"url": prefix + name, "sha256": digest.removeprefix("sha256:")}
    if set(assets) != set(ASSETS):
        raise ValueError("Release is missing a required platform bundle")
    return {"version": version, "assets": {name: assets[name] for name in ASSETS}}


def select_release(releases):
    """Return the newest stable bundle release from the repository's release list.

    The repository also publishes the CodeQL action's own releases (``v4.x``), which carry no
    bundle, so GitHub's "latest release" is usually one of those and never the bundle.
    """
    candidates = []
    for release in releases:
        match = re.fullmatch(r"codeql-bundle-v(\d+)\.(\d+)\.(\d+)", release.get("tag_name", ""))
        if match and not release.get("draft") and not release.get("prerelease"):
            candidates.append((tuple(map(int, match.groups())), release))
    if not candidates:
        raise ValueError("No stable CodeQL bundle release found")
    return max(candidates, key=lambda candidate: candidate[0])[1]


def fetch_release():
    """Fetch the newest stable bundle release, optionally using GH_TOKEN for API limits."""
    headers = {
        "User-Agent": "two-tier-parser-codeql-updater",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(RELEASES_API, headers=headers), timeout=30) as response:
        return select_release(json.load(response))


def main():
    """Check or atomically update the CodeQL pin; return a nonzero status on an available check update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check for updates without writing")
    args = parser.parse_args()
    current = json.loads(MANIFEST.read_text())
    candidate = manifest_from_release(fetch_release(), current["version"])
    if candidate == current:
        print(f"CodeQL {current['version']} is current; published checksums match.")
        return 0
    if args.check:
        print(f"CodeQL manifest update available: {current['version']} -> {candidate['version']}")
        return 1
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=MANIFEST.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(candidate, indent=2) + "\n")
        temporary.replace(MANIFEST)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Updated CodeQL manifest: {current['version']} -> {candidate['version']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"CodeQL update failed: {error}", file=sys.stderr)
        sys.exit(1)
