"""Switch off tqdm progress bars in a service that logs JSON.

MinerU turns its bars on at the call site (``tqdm_enable=True``), so the ``TQDM_DISABLE``
environment variable cannot silence them. A bar redraws one stderr line with carriage
returns and no newline, so the next log record starts in the middle of it and cannot be
parsed. A disabled bar still iterates and counts normally, it just prints nothing.
"""

_patched = False


def disable_progress_bars() -> None:
    """Force every tqdm bar in this process to be disabled. Safe to call more than once."""
    global _patched
    if _patched:
        return
    try:
        from tqdm import tqdm
    except ImportError:  # tqdm arrives with MinerU; without it there are no bars to silence
        return

    original = tqdm.__init__

    def init(self, *args, **kwargs):
        kwargs["disable"] = True
        original(self, *args, **kwargs)

    tqdm.__init__ = init
    _patched = True
