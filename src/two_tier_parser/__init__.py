"""Two Tier Document Parser - Fast and Accurate PDF parsing for RAG pipelines.

A production-ready, two-tier document parsing system:
- Fast tier: PyMuPDF4LLM for quick markdown extraction (~1s/page)
- Accurate tier: MinerU for detailed structure extraction with GPU support

This library follows OpenTelemetry best practices for instrumentation:
- Uses API-only pattern (no SDK dependency)
- Zero overhead when tracing SDK is not installed
- Fully portable for external OSS users

Example:
    # Fast parsing (CPU, ~1s/page)
    from two_tier_parser.fast.service import parse_pdf
    result = parse_pdf(pdf_bytes, "document.pdf")

    # Accurate parsing (GPU recommended, 10-30s/page)
    from two_tier_parser.accurate.service import parse_pdf
    result = parse_pdf(pdf_bytes, "document.pdf")
"""
import logging

__version__ = "1.0.0"
__all__ = ["fast", "accurate"]

# Library best practice: NullHandler prevents "No handlers found" warnings
# Applications can configure their own handlers as needed
# See: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(logging.NullHandler())
