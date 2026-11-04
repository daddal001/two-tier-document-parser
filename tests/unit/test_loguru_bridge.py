"""MinerU logs through loguru; its records must reach the service's JSON log stream."""

import json
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from two_tier_parser.common import json_logging
from two_tier_parser.common.loguru_bridge import forward_loguru


class FakeLoguru:
    """Records what the bridge asks of loguru, and lets a test emit a record into the sink."""

    def __init__(self):
        self.removed = False
        self.sinks = []

    def remove(self, *_args):
        self.removed = True

    def add(self, sink, **kwargs):
        self.sinks.append((sink, kwargs))
        return 1


def record(message, level_no=20, name="mineru.backend.analyze", exception=None):
    return SimpleNamespace(
        record={
            "message": message,
            "name": name,
            "level": SimpleNamespace(no=level_no),
            "exception": exception,
        }
    )


def test_loguru_default_sink_is_replaced_by_one_that_uses_standard_logging():
    fake = FakeLoguru()
    forward_loguru(fake)

    assert fake.removed, "the plain-text stderr sink must go"
    assert len(fake.sinks) == 1
    assert fake.sinks[0][1]["level"] == 0  # the standard logger's own level decides what passes


def test_records_keep_their_message_logger_name_and_level(caplog):
    fake = FakeLoguru()
    forward_loguru(fake)
    sink = fake.sinks[0][0]

    with caplog.at_level(logging.INFO):
        sink(record("model_list infer finished", 20))
        sink(record("slow page", 30, name="mineru.backend.window"))
        sink(record("bad page", 40))

    seen = [(r.name, r.levelno, r.getMessage()) for r in caplog.records]
    assert ("mineru.backend.analyze", logging.INFO, "model_list infer finished") in seen
    assert ("mineru.backend.window", logging.WARNING, "slow page") in seen
    assert ("mineru.backend.analyze", logging.ERROR, "bad page") in seen


def test_braces_in_a_message_are_not_treated_as_a_format_string(caplog):
    fake = FakeLoguru()
    forward_loguru(fake)
    with caplog.at_level(logging.INFO):
        fake.sinks[0][0](record("page {0} %s done"))
    assert caplog.records[0].getMessage() == "page {0} %s done"


def test_the_exception_text_is_carried_to_the_log(caplog):
    fake = FakeLoguru()
    forward_loguru(fake)
    error = ValueError("broken")
    exception = SimpleNamespace(type=ValueError, value=error, traceback=error.__traceback__)
    with caplog.at_level(logging.ERROR):
        fake.sinks[0][0](record("failed", 40, exception=exception))
    assert caplog.records[0].exc_info[1] is error


def test_a_missing_loguru_is_not_an_error():
    forward_loguru(None)  # MinerU is not installed (for example in the fast image)


def test_the_accurate_service_sets_up_json_output_for_mineru():
    """In a fresh interpreter: loguru lines come out as JSON and progress bars are off."""
    program = (
        "import json, os, sys\n"
        "from types import ModuleType\n"
        "fake = ModuleType('loguru')\n"
        "calls = {}\n"
        "class L:\n"
        "    def remove(self, *a): calls['removed'] = True\n"
        "    def add(self, sink, **kw): calls['sink'] = sink\n"
        "fake.logger = L()\n"
        "sys.modules['loguru'] = fake\n"
        "tq = ModuleType('tqdm')\n"
        "class tqdm:\n"
        "    def __init__(self, total=None, disable=False, file=None):\n"
        "        self.disable = disable\n"
        "tq.tqdm = tqdm\n"
        "sys.modules['tqdm'] = tq\n"
        "import two_tier_parser.accurate.main\n"
        "bar = tq.tqdm(total=1, disable=False)\n"
        "print(json.dumps({'removed': calls.get('removed'), 'sink': 'sink' in calls,\n"
        "                  'bars_off': bool(bar.disable)}))\n"
    )
    env = {**os.environ, "OTEL_SDK_DISABLED": "true", "WORKERS": "1"}
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"removed": True, "sink": True, "bars_off": True}


def test_json_formatter_is_what_the_forwarded_record_goes_through():
    handler = logging.StreamHandler()
    handler.setFormatter(json_logging.JsonFormatter())
    line = handler.format(
        logging.LogRecord("mineru.x", logging.INFO, "", 0, "page 3 of 9", (), None)
    )
    assert json.loads(line)["message"] == "page 3 of 9"


@pytest.mark.parametrize("name", ["", None])
def test_a_record_without_a_name_still_has_a_logger(caplog, name):
    fake = FakeLoguru()
    forward_loguru(fake)
    with caplog.at_level(logging.INFO):
        fake.sinks[0][0](record("anonymous", name=name))
    assert caplog.records[0].name == "mineru"
