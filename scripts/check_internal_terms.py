"""Block commits and commit messages that contain terms from a private deny-list.

The term list is never stored in the repository. It is read from, in order:

1. the file named by ``LEAK_GUARD_TERMS_FILE``;
2. ``leak-guard-terms.txt`` inside the shared git directory (``git rev-parse --git-common-dir``,
   so one list serves subdirectories and linked worktrees, and it cannot be committed).

One term per line; blank lines and lines starting with ``#`` are ignored; matching is
case-insensitive. When no list exists the check passes silently, so contributors and CI
without the list are unaffected.

Usage:
    python scripts/check_internal_terms.py              # scan the staged diff
    python scripts/check_internal_terms.py MSG_FILE     # scan a commit message file
"""

import os
import subprocess
import sys
from pathlib import Path


def load_terms(path):
    """Return the normalized (lowercase) terms from ``path``, or an empty list when missing."""
    if path is None or not Path(path).is_file():
        return []
    terms = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line.lower())
    return terms


def find_hits(lines, terms):
    """Return ``(line_number, term)`` for every term found in ``lines``, one hit per pair."""
    hits = []
    for number, line in enumerate(lines, start=1):
        lowered = line.lower()
        for term in terms:
            if term in lowered:
                hits.append((number, term))
    return hits


def _git(*args):
    """Run git with quoting of unusual path names disabled and return its stdout."""
    return subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def added_lines_from_diff(diff):
    """Return ``(file, text)`` for every line a unified diff adds.

    A ``+++`` file header only appears before the first hunk of a file, so a content line
    that happens to start with ``++`` is still reported.
    """
    current = ""
    in_hunk = False
    added = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            added.append((current, line[1:]))
        elif not in_hunk and line.startswith("+++ b/"):
            current = line[6:]
    return added


def staged_added_lines():
    """Return ``(label, text)`` for every line added in the staged diff, including file names."""
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
    diff = _git("diff", "--cached", "-U0", "--no-color", "--diff-filter=ACMR")
    return [(f"{name} (file name)", name) for name in names] + added_lines_from_diff(diff)


def resolve_terms_path():
    """Return the configured deny-list path, if any."""
    configured = os.environ.get("LEAK_GUARD_TERMS_FILE")
    if configured:
        return Path(configured)
    try:
        common_dir = _git("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(common_dir) / "leak-guard-terms.txt"


def main(argv):
    """Scan the staged diff or a commit message; return 1 when a deny-listed term is found."""
    terms = load_terms(resolve_terms_path())
    if not terms:
        return 0
    if argv:
        lines = Path(argv[0]).read_text(encoding="utf-8", errors="replace").splitlines()
        findings = [(f"{argv[0]} line {n}", term) for n, term in find_hits(lines, terms)]
    else:
        findings = [
            (label, term)
            for label, text in staged_added_lines()
            for _, term in find_hits([text], terms)
        ]
    for label, term in findings:
        # Report the location and the matching term, never the surrounding text.
        print(f"internal term {term!r} found in {label}", file=sys.stderr)
    if findings:
        print("Remove the internal reference before committing.", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
