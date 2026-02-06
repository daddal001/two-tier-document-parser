# Contributing to Two-Tier Document Parser

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful and inclusive. We welcome contributions from everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/daddal001/two_tier_document_parser/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Docker version, GPU model if applicable)
   - Relevant logs or error messages

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear use case
   - Expected behavior
   - Potential implementation approach (optional)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Commit with clear messages
7. Push to your fork
8. Create a Pull Request

#### Code Standards

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for public functions/classes
- Keep functions focused and testable
- Add tests for new features

#### Testing

- Write unit tests for new functionality
- Ensure integration tests pass
- Test with both parsers (fast and accurate)
- Document any new dependencies

## Development Setup

```bash
# Clone your fork
git clone https://github.com/daddal001/two-tier-document-parser.git
cd two-tier-document-parser

# Install development dependencies
pip install -e .[dev]

# Run tests
make test

# Format code
make format

# Check linting
make lint
```

## Project Structure

```
two_tier_document_parser/
├── src/
│   └── two_tier_parser/          # Main package
│       ├── __init__.py          # Package init with NullHandler + version
│       ├── fast/                 # Fast parser service (PyMuPDF4LLM)
│       │   ├── main.py          # FastAPI application (entry point)
│       │   ├── app.py           # Backwards-compatible wrapper (deprecated)
│       │   ├── service.py       # Parser implementation
│       │   ├── models.py        # Pydantic schemas
│       │   └── core/            # Telemetry utilities
│       │       └── telemetry.py # OpenTelemetry (API-only pattern)
│       ├── accurate/             # Accurate parser service (MinerU)
│       │   ├── main.py          # FastAPI application (entry point)
│       │   ├── app.py           # Backwards-compatible wrapper (deprecated)
│       │   ├── service.py       # Parser implementation with tracing
│       │   ├── models.py        # Pydantic schemas
│       │   └── core/            # Telemetry utilities
│       │       └── telemetry.py # OpenTelemetry (API-only pattern)
│       └── common/               # Shared utilities
│           ├── config.py        # Configuration management
│           ├── logger.py        # Logging setup
│           └── utils.py         # Common helpers
├── deploy/
│   ├── docker-compose.yml       # Docker Compose orchestration
│   ├── Dockerfile.fast          # Fast parser image
│   ├── Dockerfile.accurate      # Accurate parser image
│   └── magic-pdf.json          # MinerU configuration
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── fixtures/                # Test data
│   └── conftest.py             # Pytest configuration
├── examples/
│   ├── demo_client.py          # CLI demo client
│   ├── notebooks/              # Jupyter notebooks
│   └── data/                   # Sample data
├── docs/
│   ├── API.md                  # API reference
│   ├── DOCKER_SETUP.md         # Docker setup guide
│   ├── TESTING.md              # Testing guide
│   └── GIT_SUBMODULES.md       # Submodule guide
├── MinerU/                      # Git submodule (MinerU library)
├── pyproject.toml              # Project metadata & dependencies
├── Makefile                    # Development automation
├── LICENSE                     # AGPL-3.0 license
├── NOTICE                      # Third-party licenses
└── README.md                   # Project overview
```

## Questions?

Feel free to open an issue for discussion or questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.

