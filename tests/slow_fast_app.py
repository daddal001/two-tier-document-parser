"""The fast app with a slow parser running in threads, to test shutdown with a request in flight.

Run as ``uvicorn tests.slow_fast_app:app``. ``SLOW_PARSE_SECONDS`` sets how long each parse takes.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

from two_tier_parser.fast import main, service


def slow_parse(*args, **kwargs):
    time.sleep(float(os.getenv("SLOW_PARSE_SECONDS", "4")))
    return service.parse_pdf(*args, **kwargs)


main.parse_pdf = slow_parse
main.executor = ThreadPoolExecutor(max_workers=2)
main.pages_executor = ThreadPoolExecutor(max_workers=1)
app = main.app
