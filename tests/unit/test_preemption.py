"""Unit tests for ADR-0047 cooperative preemption in fast-parser.

These tests exercise service.parse_pdf with a mocked Redis client and a
monkey-patched pymupdf4llm.to_markdown so we don't need a real PDF pipeline.
They verify:
- When ``task_id`` is None, the fast full-document path is used (no cancel check).
- When ``task_id`` is set and parse_cancel:{task_id} is present, parsing
  yields at the next page boundary and returns ``preempted=True`` with the
  ``last_page_parsed`` cursor.
- progress_ceiling disables preemption once enough pages are done.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _install_pymupdf_stub() -> None:
    """Stub pymupdf + pymupdf4llm before importing the service module.

    The service imports both at module load, so we inject fake modules here
    to keep this unit test runnable outside the fast-parser container.
    """
    if "pymupdf" not in sys.modules:
        stub_pymupdf = types.ModuleType("pymupdf")
        stub_pymupdf.open = MagicMock()
        sys.modules["pymupdf"] = stub_pymupdf

    if "pymupdf4llm" not in sys.modules:
        stub_pymupdf4llm = types.ModuleType("pymupdf4llm")
        stub_pymupdf4llm.to_markdown = MagicMock(return_value="")
        sys.modules["pymupdf4llm"] = stub_pymupdf4llm


_install_pymupdf_stub()

from two_tier_parser.fast import service as fast_service


@pytest.fixture
def stub_pymupdf(monkeypatch):
    """Stub pymupdf + pymupdf4llm so parse_pdf doesn't touch a real PDF.

    Two call sites in parse_pdf:
      ``with pymupdf.open(tmp_path) as doc: total_pages = len(doc)``
      ``doc = pymupdf.open(tmp_path); for page in range(total_pages): ...``
    Both must return the same stub document that supports ``len()`` and the
    context-manager protocol.
    """
    stub_doc = MagicMock()
    stub_doc.__len__ = MagicMock(return_value=10)
    stub_doc.__enter__ = MagicMock(return_value=stub_doc)
    stub_doc.__exit__ = MagicMock(return_value=False)
    stub_doc.close = MagicMock()

    stub_mod = types.ModuleType("pymupdf")
    stub_mod.open = MagicMock(return_value=stub_doc)

    monkeypatch.setitem(sys.modules, "pymupdf", stub_mod)
    monkeypatch.setattr(
        fast_service.pymupdf4llm,
        "to_markdown",
        lambda _doc, **kw: f"page-{kw.get('pages', ['all'])[0]}-md\n",
    )
    return stub_doc


@pytest.fixture(autouse=True)
def reset_redis_stub():
    fast_service._redis_client = None
    yield
    fast_service._redis_client = None


class _FakeRedis:
    def __init__(self, cancel_at_call: int | None = None):
        self._cancel_at = cancel_at_call
        self._calls = 0

    def exists(self, key: str) -> int:
        self._calls += 1
        if self._cancel_at is not None and self._calls >= self._cancel_at:
            return 1
        return 0


class TestPreemptionDisabled:
    def test_no_task_id_uses_fast_path(self, stub_pymupdf, monkeypatch):
        """With task_id=None the happy-path full-document call runs, no cancel
        check is made, and no preempted/last_page_parsed is reported."""
        monkeypatch.setattr(
            fast_service, "_get_preempt_redis", lambda: _FakeRedis(cancel_at_call=1)
        )
        result = fast_service.parse_pdf(b"%PDF-1.4\n", "doc.pdf", task_id=None)
        assert "preempted" not in result["metadata"]


class TestPreemptionActive:
    def test_cancels_at_page_boundary(self, stub_pymupdf, monkeypatch):
        """With task_id set and cancel flag appearing on call #2, the parse
        should return after parsing page 0 (last_page_parsed == 0)."""
        fake = _FakeRedis(cancel_at_call=2)
        monkeypatch.setattr(fast_service, "_get_preempt_redis", lambda: fake)

        result = fast_service.parse_pdf(
            b"%PDF-1.4\n",
            "doc.pdf",
            task_id="stage2-tid",
            start_page=0,
            progress_ceiling=0.99,
        )

        meta = result["metadata"]
        assert meta["preempted"] is True
        assert meta["last_page_parsed"] == 0
        assert meta["start_page"] == 0

    def test_progress_ceiling_ignores_cancel_when_nearly_done(
        self, stub_pymupdf, monkeypatch
    ):
        """When progress_ceiling is very low, cancel is ignored after the
        first few pages and parsing runs to completion."""
        # Cancel flag is permanent from first call.
        monkeypatch.setattr(
            fast_service, "_get_preempt_redis", lambda: _FakeRedis(cancel_at_call=1)
        )
        result = fast_service.parse_pdf(
            b"%PDF-1.4\n",
            "doc.pdf",
            task_id="stage2-tid",
            start_page=0,
            progress_ceiling=0.0,  # any progress disables cancel
        )
        meta = result["metadata"]
        # Loop broke on the first iteration because progress 0.0 is not < 0.0.
        # With ceiling 0, every page is "past the ceiling" so cancel is ignored.
        assert meta["preempted"] is False
        assert meta["last_page_parsed"] == 9  # all 10 pages
