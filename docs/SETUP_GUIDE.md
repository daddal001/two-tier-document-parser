# Setup Guide

## Directory Structure

```
two_tier_document_parser/
├── src/                            # Source code
│   └── two_tier_parser/
│       ├── common/                 # Shared utilities
│       │   ├── config.py          # Configuration management
│       │   └── logger.py          # Logging setup
│       ├── fast/                  # Fast parser (PyMuPDF)
│       │   ├── app.py
│       │   ├── service.py
│       │   └── models.py
│       └── accurate/              # Accurate parser (MinerU)
│           ├── app.py
│           ├── service.py
│           └── models.py
├── deploy/                        # Deployment configs
│   ├── docker-compose.yml
│   ├── Dockerfile.fast
│   ├── Dockerfile.accurate
│   └── magic-pdf.json
├── examples/                      # Examples and demos
│   ├── data/                      # Sample data
│   │   └── sample.pdf
│   ├── notebooks/                 # Jupyter notebooks
│   │   └── parser_visualization.ipynb
│   └── demo_client.py            # CLI demo
├── docs/                          # Documentation
├── tests/                         # Test files
├── pyproject.toml                # Python project config
├── Makefile                      # Build automation
└── README.md
```

## Quick Start

### Option 1: Using Make (Recommended)
```bash
# Build images
make build

# Start services
make up

# View logs
make logs

# Stop services
make down

# Clean everything
make clean
```

### Option 2: Using Docker Compose Directly
```bash
# Build and start
docker-compose -f deploy/docker-compose.yml up --build -d

# View logs
docker-compose -f deploy/docker-compose.yml logs -f

# Stop
docker-compose -f deploy/docker-compose.yml down
```

### Option 3: Install as Python Package (Development)

Use Python 3.11+ for the fast service. Accurate-parser's model and native
dependencies are provided by the Docker setup above.

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dependencies
pip install -e '.[fast,telemetry,dev]'

# Start the fast service
python -m uvicorn two_tier_parser.fast.main:app --host 127.0.0.1 --port 8004
```

In another terminal with the environment activated, run
`python examples/demo_client.py examples/data/sample.pdf --mode fast`.

## Testing the Services

### Health Check
```bash
# Fast parser
curl http://localhost:8004/health

# Accurate parser
curl http://localhost:8005/health
```

### Parse a Document
```bash
# Using cURL
curl -X POST -F "file=@examples/data/sample.pdf" http://localhost:8004/parse > output.json

# Using the demo client
python examples/demo_client.py examples/data/sample.pdf --mode accurate
```

## Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter notebook ipykernel

# Start Jupyter
cd examples/notebooks
jupyter notebook parser_visualization.ipynb
```

**Note**: Update the `PDF_FILE_PATH` variable in the notebook to point to `../data/sample.pdf`.

## Environment Variables

### Fast Parser
- `PYTHON_GIL`: Enable/disable GIL (0 = disabled for true parallelism)
- `WORKERS`: Number of thread pool workers (default: 4)
- `LOG_LEVEL`: Logging level (default: INFO)

### Accurate Parser
- `WORKERS`: Number of thread pool workers (default: 2)
- `LOG_LEVEL`: Logging level (default: INFO)
- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: 0)
- `MINERU_VIRTUAL_VRAM_SIZE`: Virtual VRAM size in GB (default: 15)
- `UVICORN_TIMEOUT_KEEP_ALIVE`: Keep-alive timeout in seconds (default: 600)
- `TOKENIZERS_PARALLELISM`: Disable tokenizers parallelism warnings (default: false)

## Troubleshooting

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed troubleshooting.

## License

This project is licensed under AGPL-3.0. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE) for details.
