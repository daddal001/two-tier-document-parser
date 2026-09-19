# Docker Setup Guide

This guide explains how to run the two-tier document parser using Docker Compose and visualize results with Jupyter notebooks.

Performance and accuracy figures below are historical examples without a
reproducible benchmark attached to this guide. They are not service guarantees;
measure your corpus and hardware before setting capacity or timeout budgets.

## Prerequisites

- Docker Desktop (with Docker Compose)
- **Optional**: NVIDIA GPU with drivers + nvidia-docker (for highest accuracy VLM mode)
  - **The accurate parser automatically detects hardware** and uses:
    - GPU available → VLM backend (95%+ accuracy)
    - No GPU → Pipeline backend (80-85% accuracy, CPU-only)
  - **No configuration changes needed** - detection is automatic
- Python 3.8+ (for Jupyter notebook)
- At least 8GB RAM (16GB recommended for accurate parser)

## Quick Start

### 1. Build and Start Services

```bash
# Build and start both parser services
docker-compose up --build
sudo docker-compose up --build

# Remove containers and volumes
sudo docker-compose down -v
sudo docker system prune -a --volumes -f
df -h

# Rebuild and start
sudo docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

This will start:
- **Fast Parser** on `http://localhost:8004`
- **Accurate Parser** on `http://localhost:8005`

### 2. Verify Services are Running

```bash
# Check fast parser health
curl http://localhost:8004/health

# Check accurate parser health
curl http://localhost:8005/health
```

Expected output:
```json
{
  "status": "healthy",
  "workers": 4,
  "no_gil": true,
  "parser": "pymupdf4llm",
  "version": "1.0.0"
}
```

### 3. Test Parsing

```bash
# Parse a PDF with fast parser
curl -X POST http://localhost:8004/parse \
  -F "file=@/path/to/your/document.pdf" \
  > fast_result.json

# Parse a PDF with accurate parser (takes 10-11 minutes for VLM processing)
# Set a client deadline appropriate to your document and hardware
curl -X POST http://localhost:8005/parse \
  -F "file=@/path/to/your/document.pdf" \
  > accurate_result.json
```

The demo client's default request timeout is 600 seconds and can be overridden
with `--timeout`. This would be insufficient for an 11-minute parse. Configure
client and proxy deadlines for measured processing time. The Compose setting
`UVICORN_TIMEOUT_KEEP_ALIVE=600` is not a processing-time budget.

## 🔄 Automatic GPU Detection & Fallback

The **accurate parser automatically detects GPU availability** at startup and selects the optimal backend - **no configuration changes needed!**

### 🚀 With GPU (Transformers Backend)
- Uses Transformers backend running MinerU VLM model
- **Accuracy**: 95%+ (highest quality)
- **Speed**: ~15-60 seconds per page on T4/A10
- **Requirements**: NVIDIA GPU with CUDA support
- **Logs**: `🚀 Using VLM backend for highest accuracy (GPU-accelerated)`

### 💻 Without GPU (Pipeline Backend)
- Automatically falls back to MinerU pipeline mode
- **Accuracy**: 80-85% (still very good!)
- **Speed**: ~5-15 seconds per page on CPU
- **Requirements**: CPU only (no GPU needed)
- **Logs**: `💻 No GPU detected - using pipeline backend (CPU-only, 80-85% accuracy)`

### 🔍 How to Check Which Mode is Running

**In Docker logs:**
```bash
docker-compose logs accurate-parser | grep "Using"
# GPU mode: "🚀 Using VLM backend for highest accuracy" (transformers backend)
# CPU mode: "💻 No GPU detected - using pipeline backend"
```

**In API response metadata:**
```json
{
  "metadata": {
    "backend": "transformers",  // or "pipeline"
    "gpu_used": true,            // or false
    "accuracy_tier": "very-high" // or "high"
  }
}
```

### 🖥️ CPU-Only Deployment

**The service automatically detects and adapts** - no changes needed! However, if Docker fails to start with a "nvidia runtime not found" error on CPU-only machines:

1. Edit `deploy/docker-compose.yml` and comment out:
   ```yaml
   # runtime: nvidia  # Comment this line
   ```
2. And comment out the devices section:
   ```yaml
   # devices:  # Comment out this entire section
   #   - driver: nvidia
   #     count: 1
   #     capabilities: [gpu]
   ```
3. Restart: `docker-compose -f deploy/docker-compose.yml up --build`

The Python service will detect no GPU and automatically use pipeline backend.

---

## 🎛️ GPU-Specific Configuration Reference

While the service automatically detects GPU availability, you can optimize Docker settings for your specific hardware. This table shows recommended configurations for different GPU types:

### Configuration Matrix

| Component | Tesla T4 / Turing (CC 7.5) | Ampere+ (A10/A100/RTX 3090+) | CPU Only |
|:----------|:---------------------------|:-----------------------------|:---------|
| **Backend** | Transformers (MinerU VLM) | Transformers (MinerU VLM) | Pipeline (CPU) |
| **Accuracy** | 95%+ | 95%+ | 80-85% |
| **Speed** | ~20-30s/page | ~15-20s/page | ~5-15s/page |
| **Docker Runtime** | `runtime: nvidia` | `runtime: nvidia` | Comment out |
| **GPU Devices** | `count: 1` | `count: 1` | Comment out |
| **CUDA_VISIBLE_DEVICES** | `0` | `0` | Remove/ignore |
| **MINERU_VIRTUAL_VRAM_SIZE** | `15` (GB) | `20-24` (GB) | N/A |
| **TOKENIZERS_PARALLELISM** | `false` | `false` | `false` |
| **Memory Limit** | `16G` | `24G` | `8G` |
| **Memory Reservation** | `8G` | `12G` | `4G` |
| **CPU Limit** | `2` | `2` | `2` |
| **Workers** | `2` | `2` | `2-4` |
| **Timeout (Keep-Alive)** | `600s` | `600s` | `300s` |

### Detailed Setup Instructions

#### 🔧 Tesla T4 / Turing Architecture (Compute Capability 7.5)

**Your current configuration is already optimized!**

```yaml
accurate-parser:
  runtime: nvidia
  environment:
    - WORKERS=2
    - LOG_LEVEL=INFO
    - CUDA_VISIBLE_DEVICES=0
    - MINERU_VIRTUAL_VRAM_SIZE=15      # Optimal for 16GB VRAM
    - UVICORN_TIMEOUT_KEEP_ALIVE=600
    - TOKENIZERS_PARALLELISM=false
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 16G
      reservations:
        cpus: '1'
        memory: 8G
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Why these settings:**
- `MINERU_VIRTUAL_VRAM_SIZE=15`: Leaves 1GB buffer for system operations on 16GB T4
- `memory: 16G`: Sufficient for Transformers backend running MinerU VLM
- Transformers backend: Most stable for Turing (CC 7.5) architecture
- Note: Backend name in code is `transformers` (not `vlm`)

**Expected Performance:**
- Processing: ~20-30 seconds per page
- 11-page document: ~8-10 minutes
- Batch size: 4 (automatically set based on VRAM)
- Backend: `transformers` (using MinerU VLM model)

---

#### ⚡ Ampere+ Architecture (A10, A100, RTX 3090+, Compute Capability 8.0+)

**Optional optimizations for better performance:**

```yaml
accurate-parser:
  runtime: nvidia
  environment:
    - WORKERS=2
    - LOG_LEVEL=INFO
    - CUDA_VISIBLE_DEVICES=0
    - MINERU_VIRTUAL_VRAM_SIZE=20      # Increase for 24GB+ GPUs
    - UVICORN_TIMEOUT_KEEP_ALIVE=600
    - TOKENIZERS_PARALLELISM=false
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 24G                     # Increase from 16G
      reservations:
        cpus: '1'
        memory: 12G                     # Increase from 8G
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Why these settings:**
- `MINERU_VIRTUAL_VRAM_SIZE=20-24`: Utilize larger VRAM for bigger batches
- `memory: 24G`: More headroom for concurrent operations
- Ampere supports FlashAttention and newer CUDA features

**Expected Performance:**
- Processing: ~15-20 seconds per page
- 11-page document: ~5-8 minutes
- Batch size: 6-8 (automatically scales with VRAM)
- Backend: `transformers` (using MinerU VLM model)

**Advanced Options:**
For A100/H100 with 40GB+ VRAM, you can further increase:
```yaml
- MINERU_VIRTUAL_VRAM_SIZE=32
memory: 32G
```

---

#### 💻 CPU-Only Configuration (No GPU)

**Service automatically uses pipeline backend. Only Docker config needs changes:**

```yaml
accurate-parser:
  # runtime: nvidia  # ← COMMENT OUT THIS LINE
  environment:
    - WORKERS=2                        # Can increase to 4 for more concurrency
    - LOG_LEVEL=INFO
    # CUDA_VISIBLE_DEVICES not needed
    # MINERU_VIRTUAL_VRAM_SIZE not needed
    - UVICORN_TIMEOUT_KEEP_ALIVE=300   # Reduced timeout (CPU is faster)
    - TOKENIZERS_PARALLELISM=false
  deploy:
    resources:
      limits:
        cpus: '2'                      # Can increase based on CPU cores
        memory: 8G                     # Reduced from 16G
      reservations:
        cpus: '1'
        memory: 4G
        # devices:  # ← COMMENT OUT THIS ENTIRE SECTION
        #   - driver: nvidia
        #     count: 1
        #     capabilities: [gpu]
```

**Why these settings:**
- No GPU-specific environment variables needed
- Lower memory requirements (8GB vs 16GB)
- Can increase CPU/workers for better throughput
- Shorter timeout since CPU processing is faster

**Expected Performance:**
- Processing: ~5-15 seconds per page (depends on CPU)
- 11-page document: ~2-3 minutes
- Uses traditional CV models (OCR, layout detection)

**Multi-core Optimization:**
For servers with 8+ CPU cores:
```yaml
- WORKERS=4
limits:
  cpus: '4'
  memory: 12G
```

---

### Environment Variables Reference

| Variable | Purpose | Default | When to Change |
|:---------|:--------|:--------|:---------------|
| `WORKERS` | Number of Uvicorn workers | `2` | Increase for high concurrency |
| `LOG_LEVEL` | Logging verbosity | `INFO` | Use `DEBUG` for troubleshooting |
| `CUDA_VISIBLE_DEVICES` | Which GPU to use | `0` | Multi-GPU setups (e.g., `0,1`) |
| `MINERU_VIRTUAL_VRAM_SIZE` | Virtual VRAM in GB | `15` | Match to your GPU VRAM (leave 1-2GB buffer) |
| `UVICORN_TIMEOUT_KEEP_ALIVE` | Connection timeout (seconds) | `600` | Increase for very large documents |
| `TOKENIZERS_PARALLELISM` | HuggingFace tokenizer threading | `false` | Keep `false` to avoid warnings |
| `MINERU_DEVICE_MODE` | Force device mode | auto-detect | Override: `cuda`, `cpu`, `mps` (Apple) |

### Troubleshooting by Hardware

**Tesla T4 Issues:**
- ❌ Error: "too many resources requested for launch"
  - ✅ Solution: Using transformers backend (already configured)
  - Don't use vLLM backend on T4
  
**Ampere+ Issues:**
- ⚠️ Slow processing despite good GPU
  - ✅ Increase `MINERU_VIRTUAL_VRAM_SIZE` to match GPU VRAM
  - ✅ Check GPU utilization: `nvidia-smi`

**CPU-Only Issues:**
- ❌ Error: "nvidia runtime not found"
  - ✅ Comment out `runtime: nvidia` in docker-compose.yml
  - ✅ Comment out `devices` section
- ⚠️ Slow processing
  - ✅ Increase `WORKERS` to match CPU cores
  - ✅ Use faster CPU with more cores

---

## Using the Jupyter Notebook

### 1. Install Notebook Dependencies

### Create venv

```bash
python3 -m venv venv
```

### activate venv
```bash
source venv/bin/activate
or
.\venv\Scripts\Activate.ps1
```

### Install Notebook Dependencies
```bash
pip install -r notebook_requirements.txt
```

### 2. Launch Jupyter Lab

```bash
jupyter lab
```

### 3. Open the Visualization Notebook

1. Navigate to `parser_visualization.ipynb`
2. Update the `PDF_FILE_PATH` variable with your PDF file path
3. Run all cells to see comprehensive visualizations

The notebook will:
- ✅ Check health of both services
- ✅ Parse your PDF with both parsers
- ✅ Display metadata, markdown, images, tables, and formulas
- ✅ Compare performance metrics
- ✅ Export results to JSON and markdown files

## Docker Compose Configuration

### Service: fast-parser

- **Port**: 8004
- **Base Image**: python:3.13-slim
- **CPU**: 4 cores
- **Memory**: 2-4GB
- **No-GIL Mode**: Enabled for true thread parallelism
- **Workers**: 4 concurrent threads

### Service: accurate-parser

- **Port**: 8005
- **Base Image**: vllm/vllm-openai:v0.10.2 (contains PyTorch + transformers)
- **Backend**: VLM with transformers (vLLM not compatible with Tesla T4)
- **CPU**: 2 cores
- **Memory**: 8-16GB
- **GPU**: Required for VLM backend (NVIDIA GPU with CC >= 7.0)
- **Workers**: 2 concurrent threads

## Enabling GPU Support (Accurate Parser)

### 1. Install nvidia-docker

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Uncomment GPU Configuration in docker-compose.yml

Edit `docker-compose.yml` and uncomment:

```yaml
accurate-parser:
  # ...
  runtime: nvidia
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### 3. Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

### 4. GPU Architecture Compatibility

**Important:** The accurate-parser uses the **transformers backend** (not vLLM) for compatibility with Tesla T4.

**vLLM Compatibility Status:**
- ❌ **Tesla T4 (Turing, CC 7.5)**: vLLM's FlashInfer sampling fails with `RuntimeError: too many resources requested for launch`
- ✅ **Ampere+ GPUs (CC >= 8.0)**: vLLM works (A10, A100, RTX 30xx/40xx)

**Why vLLM Fails on Tesla T4:**
- vLLM v0.10.2's V1 engine uses FlashInfer sampling kernels optimized for Ampere architecture
- These kernels request more GPU threads/blocks than T4's Turing architecture can provide
- All documented successful vLLM deployments use 40-48GB VRAM GPUs (A6000/A100)
- Tesla T4 has only 15GB VRAM, insufficient for vLLM's memory profiling requirements

**Current Configuration:**
- **Backend**: `transformers` (stable on Tesla T4)
- **Base Image**: `vllm/vllm-openai:v0.10.2` (used only for PyTorch/transformers dependencies)
- **Performance**: ~10-11 minutes per 11-page document
- **Accuracy**: 90%+ (identical to vLLM)

**Check Your GPU Compute Capability:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**Note:** If you upgrade to an Ampere+ GPU, you can modify `accurate/parser.py` to use `backend='vllm-engine'` for 3x speedup, but this requires significant code changes to add vLLM-specific parameters.

## Troubleshooting

### Docker Compose Issues

**Issue**: `KeyError: 'ContainerConfig'` when running docker-compose up
```bash
ERROR: for fast-parser  'ContainerConfig'
ERROR: for accurate-parser  'ContainerConfig'
KeyError: 'ContainerConfig'
```

**Cause**: This error occurs with older docker-compose versions (v1.29.2 or earlier) when trying to recreate containers with GPU configurations.

**Solution**:
```bash
# Option 1: Clean up existing containers and restart
docker-compose down
docker-compose up --build

# Option 2: If Option 1 fails, remove volumes too
docker-compose down -v
docker-compose up --build

# Option 3: Upgrade to Docker Compose v2 (recommended)
sudo apt-get update
sudo apt-get install docker-compose-plugin
# Then use 'docker compose' (without hyphen) instead of 'docker-compose'
docker compose up --build
```

**Prevention**: Use Docker Compose v2 (2.x) instead of v1.29.2. Check your version with:
```bash
docker-compose --version  # v1.x (old)
docker compose version     # v2.x (new, recommended)
```

### Fast Parser Issues

**Issue**: "PYTHON_GIL not disabled"
```bash
# Verify environment variable in container
docker exec fast-parser env | grep PYTHON_GIL
# Should output: PYTHON_GIL=0
```

**Issue**: Port 8004 already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8014:8004"  # Use port 8014 instead
```

### Accurate Parser Issues

**Issue**: Out of memory
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G  # Increase to 16GB or 32GB
```

**Issue**: CUDA not available
```bash
# Check if GPU is accessible
docker exec accurate-parser python -c "import torch; print(torch.cuda.is_available())"
```

**Issue**: MinerU model download failed
```bash
# Manually download models in container
docker exec accurate-parser mineru-models-download -s huggingface -m all
```

**Issue**: Request timeout when parsing large documents
```bash
# Error: ReadTimeout: HTTPConnectionPool(host='localhost', port=8005): Read timed out
```

**Cause**: VLM transformers backend takes 10-11 minutes for high-accuracy processing (90+ accuracy). Default client timeout may be too short.

**Solution**:
1. Server timeout is already configured to 600 seconds (10 minutes) in the Dockerfile
2. Increase client timeout in your code:

```python
# Python requests
response = requests.post(
    "http://localhost:8005/parse",
    files=files,
    timeout=600  # 10 minutes
)

# cURL
curl -X POST http://localhost:8005/parse \
  -F "file=@document.pdf" \
  --max-time 600 \
  > result.json
```

3. For very large documents (100+ pages), increase timeout further:
```yaml
# In docker-compose.yml
environment:
  - UVICORN_TIMEOUT_KEEP_ALIVE=900  # 15 minutes
```

**Performance Notes**:
- Fast parser (pipeline, 82+ accuracy): 5-10 seconds per document
- Accurate parser (VLM transformers, 90+ accuracy): 10-11 minutes per 11-page document
- Transformers backend (used for Tesla T4) is slower but stable - this is expected for maximum accuracy on Turing GPUs
- vLLM backend would be 3x faster (~3 minutes) but requires Ampere+ GPUs

### General Issues

**Issue**: No space left on device during build
```bash
# Error: No space left on device (os error 28)
```

**Cause**: Docker images, containers, and build cache consume disk space. MinerU models alone require 20+ GB.

**Solution**:
```bash
# Clean up all unused Docker data (containers, images, volumes, build cache)
sudo docker system prune -a --volumes -f

# Check available disk space
df -h

# Then rebuild
docker-compose build --no-cache
docker-compose up -d
```

**Warning**: This removes ALL unused Docker data. Make sure you don't have other important containers/volumes.

**Issue**: Containers won't start
```bash
# Check logs
docker-compose logs fast-parser
docker-compose logs accurate-parser

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

**Issue**: Slow parsing performance
```bash
# Check resource usage
docker stats

# Ensure enough resources allocated in Docker Desktop settings
# Recommended: 4+ CPUs, 8+ GB RAM
```

## Stopping Services

```bash
# Stop services (preserve containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v

# Stop and remove everything including images
docker-compose down --rmi all -v
```

## Development Mode

For development with auto-reload:

```bash
# Fast parser
cd fast/
PYTHON_GIL=0 uvicorn app:app --host 0.0.0.0 --port 8004 --reload

# Accurate parser
cd accurate/
uvicorn app:app --host 0.0.0.0 --port 8005 --reload
```

## Production Deployment

See the private `document_agent_v0.2` repository for:
- Kubernetes manifests
- Horizontal Pod Autoscaling configuration
- Network policies
- Monitoring setup
- CI/CD workflows

## API Documentation

Once services are running, interactive API docs are available at:

- Fast Parser: http://localhost:8004/docs
- Accurate Parser: http://localhost:8005/docs

## Performance Benchmarks

### Fast Parser (PyMuPDF4LLM)

- **Latency**: 100-500ms per document
- **Throughput**: ~33 documents/second per pod (4 workers)
- **Concurrency**: 4 simultaneous parses (no-GIL mode)
- **Best for**: Quick text extraction, high-volume processing

### Accurate Parser (MinerU)

- **Latency**: 1-3 minutes per document
- **Throughput**: 1.70-2.12 pages/second
- **Concurrency**: 2 workers (GPU bottleneck)
- **Best for**: Documents with images, tables, formulas
- **Base**: vLLM/OpenAI (MinerU 2.6.4+)

## License

AGPL-3.0 - See LICENSE file

Source code: https://github.com/daddal001/two_tier_document_parser

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review logs: `docker-compose logs`
3. Open an issue on GitHub
