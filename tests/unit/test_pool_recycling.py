"""Recycled pool workers must be replaced, or work queued behind them waits for nothing.

The parser pools retire each worker after ``WORKER_MAX_TASKS`` tasks. Before Python 3.13 (see
CPython issue 115634, fixed in 3.13, 3.14 and 3.15 but not backported to 3.12) the pool could
retire its only worker and start no replacement, so tasks queued behind it were never run until
some later request arrived. A live request then waited for minutes with no error.

The check runs in its own process with a deadline, so a pool that does hang fails this test
instead of holding up the whole run.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="CPython issue 115634: Python 3.12 can strand queued work after a worker is recycled",
)

CHILD = """
import time
from concurrent.futures import wait

if __name__ == "__main__":
    from two_tier_parser.fast.main import _new_pool

    pool = _new_pool(1)  # one worker, retired after WORKER_MAX_TASKS tasks
    futures = [pool.submit(time.sleep, 0.2) for _ in range(6)]  # all queued at once
    done, pending = wait(futures, timeout={deadline})
    print(f"finished {{len(done)}} of {{len(futures)}}", flush=True)
    pool.shutdown(wait=False, cancel_futures=True)
"""


def test_work_queued_behind_a_recycled_worker_still_runs(tmp_path):
    script = tmp_path / "recycle_probe.py"
    script.write_text(CHILD.format(deadline=30))
    environment = {
        **os.environ,
        "WORKERS": "1",
        "PAGES_WORKERS": "1",
        "WORKER_MAX_TASKS": "2",  # six tasks on one worker: two recycles
        "OTEL_SDK_DISABLED": "true",
    }

    result = subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,  # never hang the test run, whatever the pool does
        check=False,
    )

    assert "finished 6 of 6" in result.stdout, result.stdout + result.stderr
