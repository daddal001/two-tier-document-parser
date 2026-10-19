"""Tests for the private deny-list commit guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_internal_terms as guard

SCRIPT = Path(guard.__file__)


@pytest.fixture(autouse=True)
def isolate_from_outer_git(monkeypatch):
    """Drop GIT_* variables so these tests never touch the repository that is running them.

    During a real ``git commit`` git exports GIT_INDEX_FILE and friends to hooks; subprocesses
    started here would otherwise operate on the outer repository's index.
    """
    for key in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(key)


def run_script(cwd, terms_file, *args):
    env = {"LEAK_GUARD_TERMS_FILE": str(terms_file), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def test_load_terms_ignores_comments_blanks_and_case(tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("# comment\n\n  Acme-Corp  \nSecret-Host\n")

    assert guard.load_terms(terms) == ["acme-corp", "secret-host"]


def test_load_terms_missing_file_means_no_terms(tmp_path):
    assert guard.load_terms(tmp_path / "absent.txt") == []
    assert guard.load_terms(None) == []


def test_find_hits_is_case_insensitive_and_reports_line_numbers():
    lines = ["nothing here", "Talks to ACME-CORP internally"]

    assert guard.find_hits(lines, ["acme-corp"]) == [(2, "acme-corp")]
    assert guard.find_hits(lines, ["other"]) == []


def test_message_with_term_fails_without_echoing_the_line(tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("acme-corp\n")
    message = tmp_path / "MSG"
    message.write_text("fix: handle retries\n\nSee ACME-CORP runbook for the details\n")

    result = run_script(tmp_path, terms, str(message))

    assert result.returncode == 1
    assert "'acme-corp'" in result.stderr
    assert "runbook" not in result.stderr


def test_clean_message_passes(tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("acme-corp\n")
    message = tmp_path / "MSG"
    message.write_text("fix: handle retries\n")

    assert run_script(tmp_path, terms, str(message)).returncode == 0


def test_missing_term_list_passes_silently(tmp_path):
    message = tmp_path / "MSG"
    message.write_text("mentions acme-corp\n")

    result = run_script(tmp_path, tmp_path / "absent.txt", str(message))

    assert result.returncode == 0
    assert result.stderr == ""


@pytest.fixture
def staged_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


def test_staged_diff_scan_flags_added_lines_and_file_names(staged_repo, tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("acme-corp\n")
    (staged_repo / "notes.md").write_text("hello\nconnect to Acme-Corp now\n")
    (staged_repo / "acme-corp-notes.txt").write_text("harmless text\n")
    subprocess.run(["git", "add", "."], cwd=staged_repo, check=True)

    result = run_script(staged_repo, terms)

    assert result.returncode == 1
    assert "notes.md" in result.stderr
    assert "acme-corp-notes.txt (file name)" in result.stderr


def test_staged_diff_scan_ignores_removed_lines(staged_repo, tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("acme-corp\n")
    target = staged_repo / "notes.md"
    target.write_text("mention of acme-corp\n")
    subprocess.run(["git", "add", "."], cwd=staged_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed", "--no-verify"], cwd=staged_repo, check=True
    )
    target.write_text("cleaned\n")
    subprocess.run(["git", "add", "."], cwd=staged_repo, check=True)

    assert run_script(staged_repo, terms).returncode == 0


def test_load_terms_tolerates_a_byte_order_mark(tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_bytes("﻿acme-corp\nsecond\n".encode("utf-8"))

    assert guard.load_terms(terms) == ["acme-corp", "second"]


def test_added_lines_from_diff_keeps_content_that_starts_with_plus_signs():
    diff = (
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+++ looks like a header\n"
        "+plain\n"
        "diff --git a/b.txt b/b.txt\n"
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    assert guard.added_lines_from_diff(diff) == [
        ("a.txt", "++ looks like a header"),
        ("a.txt", "plain"),
        ("b.txt", "new"),
    ]


def run_without_override(cwd, *args):
    """Run the guard with no LEAK_GUARD_TERMS_FILE, as a git hook would."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def test_default_term_list_is_found_from_a_subdirectory(staged_repo):
    (staged_repo / ".git" / "leak-guard-terms.txt").write_text("acme-corp\n")
    sub = staged_repo / "pkg"
    sub.mkdir()
    (sub / "notes.md").write_text("talk to Acme-Corp\n")
    subprocess.run(["git", "add", "."], cwd=staged_repo, check=True)

    result = run_without_override(sub)

    assert result.returncode == 1
    assert "pkg/notes.md" in result.stderr


def test_default_term_list_is_found_in_a_linked_worktree(staged_repo):
    (staged_repo / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "."], cwd=staged_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed", "--no-verify"], cwd=staged_repo, check=True
    )
    worktree = staged_repo.parent / "linked"
    subprocess.run(["git", "worktree", "add", "-q", str(worktree)], cwd=staged_repo, check=True)
    common_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (worktree / common_dir / "leak-guard-terms.txt").resolve().write_text("acme-corp\n")
    (worktree / "notes.md").write_text("Acme-Corp\n")
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)

    assert (worktree / ".git").is_file()
    assert run_without_override(worktree).returncode == 1
