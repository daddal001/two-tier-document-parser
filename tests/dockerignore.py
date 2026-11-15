"""A small implementation of Docker's .dockerignore matching, enough for this repository's rules.

Docker matches each pattern against the path and against each of its parent directories; a
``!`` pattern re-includes what an earlier pattern excluded. ``*`` stays within one path segment,
``**`` crosses segments. Tests check this against a real Docker build as well.
"""

import re
from pathlib import Path


def _compile(pattern):
    regex = ""
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            regex += "(?:.*/)?"
            index += 3
        elif pattern.startswith("**", index):
            regex += ".*"
            index += 2
        elif pattern[index] == "*":
            regex += "[^/]*"
            index += 1
        elif pattern[index] == "?":
            regex += "[^/]"
            index += 1
        else:
            regex += re.escape(pattern[index])
            index += 1
    return re.compile(regex + r"\Z")


def parse(text):
    """Return ``[(negated, compiled_pattern)]`` for the non-comment lines of a .dockerignore."""
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        rules.append((negated, _compile(line.lstrip("!").strip("/"))))
    return rules


def excluded(path, rules):
    """True when ``path`` (relative to the context, with forward slashes) is left out."""
    path = path.strip("/")
    candidates = [path] + [str(parent) for parent in Path(path).parents if str(parent) != "."]
    matched = False
    for negated, pattern in rules:
        if negated != matched:
            continue  # an exclusion only matters for kept paths, a re-inclusion for excluded ones
        if any(pattern.match(candidate) for candidate in candidates):
            matched = not negated
    return matched
