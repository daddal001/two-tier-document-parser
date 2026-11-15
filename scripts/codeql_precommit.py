"""Install a pinned CodeQL bundle and block commits on Python or Actions findings."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "codeql-bundle.json"
CONFIG = ROOT / ".github" / "codeql" / "codeql-config.yml"


def bundle_name():
    """Select a supported official archive for the current operating system and CPU."""
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "x86_64"}:
        return "codeql-bundle-osx64.tar.gz"
    if system == "Linux" and machine in {"aarch64", "arm64"}:
        return "codeql-bundle-linux-arm64.tar.gz"
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return "codeql-bundle-linux64.tar.gz"
    if system == "Windows" and machine in {"x86_64", "amd64"}:
        return "codeql-bundle-win64.tar.gz"
    raise RuntimeError(f"No CodeQL bundle configured for {system}/{machine}")


def install_codeql():
    """Return the cached pinned CodeQL executable, downloading it when absent.

    Checks the downloaded archive against the committed SHA-256 manifest and
    publishes the cache only after extraction completes. Network, checksum,
    or archive failures propagate and block the commit.
    """
    manifest = json.loads(MANIFEST.read_text())
    name = bundle_name()
    cache = (
        Path(os.environ.get("CODEQL_CACHE_DIR", str(ROOT / ".codeql-cache"))).expanduser().resolve()
    )
    destination = cache / manifest["version"] / name.removesuffix(".tar.gz")
    executable = (
        destination / "codeql" / ("codeql.exe" if platform.system() == "Windows" else "codeql")
    )
    if executable.is_file():
        return executable

    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = manifest["assets"][name]
    print(f"Installing CodeQL {manifest['version']} in {destination}", flush=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=destination.parent) as directory:
        staging = Path(directory)
        archive = staging / name
        digest = hashlib.sha256()
        request = Request(asset["url"], headers={"User-Agent": "two-tier-parser-codeql-hook"})
        with urlopen(request, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != asset["sha256"]:
            raise RuntimeError("CodeQL bundle SHA-256 verification failed")
        unpacked = staging / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(unpacked, filter="data")
        # Publish only complete, verified installations. Parallel repositories can
        # share this cache without seeing a partially extracted executable.
        try:
            unpacked.rename(destination)
        except OSError:
            if not executable.is_file():
                raise
    if not executable.is_file():
        raise RuntimeError("CodeQL bundle is missing its executable")
    return executable


def snapshot_sources(destination):
    """Copy tracked and new first-party source files into an isolated scan directory."""
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    )
    for filename in set(files.decode().split("\0")):
        if not filename or not filename.startswith(("src/", "tests/", "scripts/", ".github/")):
            continue
        source = ROOT / filename
        if source.is_file() and not source.is_symlink():
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def sarif_findings(path):
    """Return all SARIF findings; reject missing runs or unsuccessful analysis."""
    runs = json.loads(path.read_text())["runs"]
    if not runs:
        raise RuntimeError(f"CodeQL produced no analysis runs in {path}")
    findings = []
    for run in runs:
        invocations = run.get("invocations", [])
        if not invocations:
            raise RuntimeError(f"CodeQL produced no execution status in {path}")
        for invocation in invocations:
            if invocation.get("executionSuccessful") is not True:
                raise RuntimeError(f"CodeQL reported an unsuccessful analysis in {path}")
        findings.extend(run.get("results", []))
    return findings


def run_codeql(command, source, log):
    """Keep verbose extractor output in ignored logs; show failures concisely."""
    with log.open("w") as output:
        result = subprocess.run(command, cwd=source, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(log.read_text(errors="replace").splitlines()[-30:]))
        raise RuntimeError(f"CodeQL command failed ({result.returncode}); see {log}")


def main():
    """Scan Python and Actions with the verified bundle; fail on any finding or tool error."""
    codeql = install_codeql()
    threads = os.environ.get("CODEQL_THREADS", "2")
    ram = os.environ.get("CODEQL_RAM", "4096")
    reports = ROOT / "parser_results" / "codeql"
    reports.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(prefix="parser-codeql-") as directory:
        working = Path(directory)
        source = working / "source"
        source.mkdir()
        snapshot_sources(source)
        for language in ("python", "actions"):
            database = working / language
            report = reports / f"{language}.sarif"
            report.unlink(missing_ok=True)
            print(f"Scanning {language} with CodeQL", flush=True)
            run_codeql(
                [
                    str(codeql),
                    "database",
                    "create",
                    str(database),
                    f"--language={language}",
                    f"--source-root={source}",
                    f"--codescanning-config={CONFIG}",
                    f"--threads={threads}",
                    f"--ram={ram}",
                ],
                source,
                reports / f"{language}-create.log",
            )
            run_codeql(
                [
                    str(codeql),
                    "database",
                    "analyze",
                    str(database),
                    "--format=sarif-latest",
                    f"--output={report}",
                    f"--threads={threads}",
                    f"--ram={ram}",
                ],
                source,
                reports / f"{language}-analyze.log",
            )
            findings = sarif_findings(report)
            count += len(findings)
            for finding in findings:
                location = finding.get("locations", [{}])[0].get("physicalLocation", {})
                filename = location.get("artifactLocation", {}).get("uri", "unknown")
                line = location.get("region", {}).get("startLine", "?")
                print(f"{filename}:{line}: {finding['ruleId']}: {finding['message']['text']}")
    if count:
        raise RuntimeError(f"CodeQL found {count} issue(s). Reports: {reports}")
    print("CodeQL Python and Actions scans passed with no findings.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"CodeQL hook failed: {error}", file=sys.stderr)
        sys.exit(1)
