"""Progress bars must not write to stderr, where they corrupt the JSON log stream."""

import io
import sys
import types

import pytest

from two_tier_parser.common import progress_bars


class FakeTqdm:
    """Just enough of tqdm to see whether a bar was asked to print."""

    def __init__(self, iterable=None, total=None, disable=False, file=None, desc=None):
        self.iterable = iterable
        self.disable = disable
        self.file = file
        if not disable and file is not None:
            file.write(f"{desc or 'bar'} 0%\r")

    def __iter__(self):
        return iter(self.iterable)

    def update(self, n=1):
        if not self.disable and self.file is not None:
            self.file.write("tick\r")


@pytest.fixture
def fake_tqdm(monkeypatch):
    module = types.ModuleType("tqdm")
    module.tqdm = FakeTqdm
    monkeypatch.setitem(sys.modules, "tqdm", module)
    monkeypatch.setattr(FakeTqdm, "__init__", FakeTqdm.__init__)  # restored after the test
    monkeypatch.setattr(progress_bars, "_patched", False)
    return FakeTqdm


def test_a_bar_that_asks_to_be_shown_is_disabled(fake_tqdm):
    progress_bars.disable_progress_bars()
    sink = io.StringIO()

    bar = fake_tqdm(total=3, disable=False, file=sink)  # how MinerU's OCR calls ask for a bar
    bar.update(3)

    assert bar.disable
    assert sink.getvalue() == ""


def test_iterating_a_disabled_bar_still_yields_every_item(fake_tqdm):
    progress_bars.disable_progress_bars()

    assert list(fake_tqdm(range(5), desc="work", disable=False, file=io.StringIO())) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_disabling_twice_does_not_stack_wrappers(fake_tqdm):
    progress_bars.disable_progress_bars()
    once = fake_tqdm.__init__
    progress_bars.disable_progress_bars()

    assert fake_tqdm.__init__ is once


def test_bars_print_until_disabled(fake_tqdm):
    sink = io.StringIO()
    fake_tqdm(total=1, file=sink, desc="before")

    assert sink.getvalue() == "before 0%\r"


def test_a_missing_tqdm_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tqdm", None)  # makes `import tqdm` raise ImportError
    monkeypatch.setattr(progress_bars, "_patched", False)

    progress_bars.disable_progress_bars()  # MinerU is not installed (for example the fast image)


def test_the_real_tqdm_is_silenced_too(monkeypatch):
    real = pytest.importorskip("tqdm").tqdm
    original = real.__init__
    monkeypatch.setattr(progress_bars, "_patched", False)
    try:
        progress_bars.disable_progress_bars()
        sink = io.StringIO()
        assert list(real(range(3), disable=False, file=sink)) == [0, 1, 2]
        assert sink.getvalue() == ""
    finally:
        real.__init__ = original
