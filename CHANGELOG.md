# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- OpenTelemetry API-only instrumentation for distributed tracing (74+ span points)
- NullHandler logging pattern for library portability
- Structured logging with JSON format support
- `core/telemetry.py` module with `traced_operation` context manager
- GPU detection tracing spans
- Per-page parsing spans for debugging
- MinerU operation wrapping with detailed attributes

### Changed
- **BREAKING:** Renamed entry points from `app.py` → `main.py` (industry standard)
- Added backwards-compatible `app.py` wrappers with deprecation warnings
- Updated Dockerfiles to use `main.py` entry points
- Raised FastAPI, `python-multipart`, and `transformers` dependency floors to remove
  known vulnerable versions from parser service manifests
- Enhanced error handling with structured logging

### Deprecated
- `two_tier_parser.fast.app` module (use `two_tier_parser.fast.main` instead)
- `two_tier_parser.accurate.app` module (use `two_tier_parser.accurate.main` instead)

## [1.0.0] - 2026-01-15

### Added
- Initial release with two-tier architecture
- **Fast Parser** (Port 8004)
  - PyMuPDF4LLM-based text extraction
  - CPU-only operation
  - Sub-second per-page processing
  - Page-by-page fallback for complex PDFs
- **Accurate Parser** (Port 8005)
  - MinerU-based deep extraction
  - Automatic GPU detection and fallback
  - VLM backend for 95%+ accuracy (GPU)
  - Pipeline backend for 80-85% accuracy (CPU)
  - Multimodal extraction (images, tables, formulas)
- Docker Compose deployment configuration
- Health check endpoints
- Comprehensive API documentation
- Python package installation support

### Infrastructure
- FastAPI-based REST APIs
- ProcessPoolExecutor for parallel parsing
- Temporary file management with cleanup
- Configurable timeouts and resource limits

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01-15 | Initial two-tier architecture release |
| (Unreleased) | - | OpenTelemetry instrumentation, `main.py` rename |
