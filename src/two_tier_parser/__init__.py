"""Two-tier PDF extraction for RAG pipelines and document processing.

The fast tier uses PyMuPDF4LLM for CPU text and image extraction. The accurate
tier uses MinerU for Markdown, images, tables, and formulas, selecting the
Transformers VLM backend when CUDA is available and the pipeline backend on CPU.
The fast tier retries recognized table/text-page conversion errors page by page,
keeps successful pages, and reports skipped page numbers and placeholders.
Task-aware parsing adds optional cancellation and resume support. The HTTP
service also replaces broken fast-parser process pools and returns a retryable
503. The caller chooses when to escalate a document between services.

The fast parser runs in a production deployment. Install the dependencies for the
tier you intend to run.

Examples:
    Parse PDF bytes with the fast tier after installing the ``fast`` extra::

        from pathlib import Path
        from two_tier_parser.fast.service import parse_pdf

        pdf_bytes = Path("document.pdf").read_bytes()
        result = parse_pdf(pdf_bytes, "document.pdf")
        print(result["markdown"])

    Use the accurate tier after installing the pinned MinerU submodule, the
    ``accurate`` extra, and the required models::

        from two_tier_parser.accurate.service import parse_pdf

        result = parse_pdf(pdf_bytes, "document.pdf")
        print(result["tables"])

Logging stays self-contained. This package installs a NullHandler for library
use; HTTP entry points configure local JSON logging. OpenTelemetry API spans
do not export traces unless a deployment explicitly configures the optional SDK
and exporter. Importing this package alone does not load parser models.

Attributes:
    __version__: Installed distribution version, matching ``pyproject.toml``.
    API_VERSION: Legacy HTTP interface version used by both FastAPI applications.

See ``README.md`` for installation and backend selection, ``docs/API.md`` for
the HTTP contract, and ``docs/TESTING.md`` for reproducible validation commands.
"""

import logging

__version__ = "1.0.0"
API_VERSION = "1.0.0"
__all__ = ["fast", "accurate", "__version__", "API_VERSION"]

# Library best practice: NullHandler prevents "No handlers found" warnings
# Applications can configure their own handlers as needed
# See: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(logging.NullHandler())
