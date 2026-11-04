"""The accurate app with a fake parser, so the notebook's checks can run without MinerU.

Run as ``uvicorn tests.fake_accurate_app:app``.
"""

import time

from two_tier_parser.accurate import main


def fake_parse(pdf_bytes, filename):
    time.sleep(2)  # long enough for a second request to find the only slot taken
    return {
        "markdown": "# fake document",
        "images": [],
        "tables": [],
        "formulas": [],
        "metadata": {
            "pages": 2,
            "processing_time_ms": 2000,
            "parser": "mineru",
            "backend": "pipeline",
            "gpu_used": main.GPU_AVAILABLE,
            "accuracy_tier": "high",
            "version": "4.0.12",
            "filename": filename,
        },
    }


main.parse_pdf = fake_parse
app = main.app
