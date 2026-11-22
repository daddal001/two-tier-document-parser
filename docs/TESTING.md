# Testing Guide

This guide covers testing approaches for the two-tier document parser project.

## Table of Contents
- [Quick Start](#quick-start)
- [Testing Philosophy](#testing-philosophy)
- [Installation Requirements](#installation-requirements)
- [Container-Based Testing (Recommended)](#container-based-testing-recommended)
- [Local pytest Testing (Optional)](#local-pytest-testing-optional)
- [GCP GPU VM Testing](#gcp-gpu-vm-testing)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies (choose based on what you're testing)

# For integration tests (recommended, no MinerU needed):
pip install -e ".[dev]"

# For fast parser unit tests:
pip install -e ".[fast,dev]"

# For accurate parser unit tests:
# Note: MinerU is a git submodule - install it first
cd MinerU && pip install -e .[core] && cd ..
pip install -e ".[fast,accurate,dev]"

# 2. Start services (for integration tests)
make up

# 3. Run tests
pytest tests/integration/ -v  # Test HTTP APIs (recommended)
pytest tests/unit/ -v          # Test parser code directly (needs [fast] or [accurate])
```

**Key Points:**
- **Integration tests** call HTTP APIs → only need `[dev]` (test tools)
- **Unit tests** import parser code → need `[fast,dev]` or `[fast,accurate,dev]`
- Integration tests use **single-page PDFs** for faster execution (~1-3 min vs 10+ min)

---

## Testing Philosophy

**This project uses a container-first approach.** Since all services run in Docker containers in production, testing should primarily be done against the containerized services.

### Testing Approaches

| Approach | Use Case | Setup Required | Speed | Dependencies |
|:---------|:---------|:---------------|:------|:-------------|
| **Integration Tests** | Primary testing method | Docker + `[dev]` | Medium | `pip install -e ".[dev]"` |
| **Unit Tests (Fast)** | Fast parser code testing | Docker + `[fast,dev]` | Fast | `pip install -e ".[fast,dev]"` |
| **Unit Tests (Accurate)** | Accurate parser code testing | Docker + `[fast,accurate,dev]` + MinerU submodule | Slow | `pip install -e MinerU[core] && pip install -e ".[fast,accurate,dev]"` |
| **GCP VM Testing** | GPU validation, production-like | GCP account + GPU quota | Slow | N/A |

---

## Installation Requirements

**⚠️ Important:** Different test types require different dependencies.

### For Integration Tests (Recommended)

Integration tests call HTTP APIs and don't import parser code directly:

```bash
pip install -e ".[dev]"  # Includes pytest, pypdf, and test utilities
```

**What's included:**
- `pytest` - Test framework
- `pypdf` - For creating single-page PDFs in tests
- `pytest-cov` - Coverage reporting
- Other test utilities

**What's NOT needed:**
- Parser code dependencies (`pymupdf4llm`, `mineru`, etc.)
- Services run in containers, so parser code isn't imported

### For Unit Tests

Unit tests import parser code directly, so they need the parser dependencies:

```bash
# Fast parser unit tests only
pip install -e ".[fast,dev]"

# All unit tests (requires MinerU from submodule):
# 1. First install MinerU from submodule (REQUIRED)
cd MinerU && pip install -e .[core] && cd ..

# 2. Then install project dependencies
pip install -e ".[fast,accurate,dev]"
```

**What's included:**
- All `[dev]` dependencies (pytest, pypdf, etc.)
- `[fast]` - `pymupdf4llm`, `pymupdf` (for fast parser)
- `[accurate]` - `torch`, `transformers` (for accurate parser)
- **MinerU** - Must be installed separately from submodule: `pip install -e MinerU[core]`

**⚠️ Important:** The `[accurate]` extra does NOT include MinerU. MinerU is a git submodule and must be installed separately:
```bash
cd MinerU
pip install -e .[core]
cd ..
```

**Note:** Accurate parser unit tests require GPU access and may fail with OOM errors if GPU memory is already in use (see [GPU Memory Management](#gpu-memory-management-for-accurate-parser-tests)).

---

## Container-Based Testing (Recommended)

**This is the primary way to test the project** since it tests the actual Docker images that will be deployed.

### Quick Start

```bash
# 1. Install test dependencies (integration tests only need [dev])
pip install -e ".[dev]"

# 2. Build and start containers
make build
make up

# 3. Run integration tests (tests HTTP APIs)
pytest tests/integration/ -v

# 4. Or test manually with curl
curl http://localhost:8004/health | jq
curl http://localhost:8005/health | jq
curl -X POST http://localhost:8004/parse -F "file=@examples/data/sample.pdf" -o result.json

# 5. Stop containers
make down
```

### Integration Tests

**Integration tests are the recommended testing method.** They test the actual running services via HTTP APIs, which matches production behavior.

**Setup:**
```bash
# 1. Install dependencies (integration tests only need [dev])
pip install -e ".[dev]"  # Includes pytest, pypdf, and test utilities
# Note: Integration tests call HTTP APIs, so parser code dependencies not needed

# 2. Start services
make up

# 3. Wait for services to be healthy (check logs)
docker logs accurate-parser | grep "Application startup complete"

# 4. Run integration tests
pytest tests/integration/ -v
```

**What integration tests do:**
- Test HTTP endpoints (`/health`, `/parse`)
- Verify response structure and status codes
- Test error handling (invalid files, etc.)
- Use **single-page PDFs** for faster execution

**Single-page testing:**
- `test_accurate_parser_parse` extracts the first page from `sample.pdf`
- Creates a temporary single-page PDF using `pypdf`
- Makes tests run in ~1-3 minutes instead of 10+ minutes
- Verifies: `assert data["metadata"]["pages"] == 1`

**Test files:**
- `tests/integration/test_api.py` - All integration tests
- `tests/conftest.py` - Fixtures including `single_page_pdf_path`

# Or manually test APIs
curl http://localhost:8004/health | jq
curl http://localhost:8005/health | jq

# Test with a PDF
curl -X POST http://localhost:8004/parse \
  -F "file=@examples/data/sample.pdf" \
  -o fast_result.json

curl -X POST http://localhost:8005/parse \
  -F "file=@examples/data/sample.pdf" \
  -o accurate_result.json


### Unit Tests Inside Containers (Not Recommended)

**⚠️ Important:**

- **pytest is NOT installed** in production containers (to keep images lean)
- **GPU OOM errors** can occur when running tests in a running container

**Why OOM happens:**

- The MinerU VLM model is loaded and cached in GPU memory when the service starts
- The model uses ~6-7GB of GPU memory (out of 14.56GB on T4)
- When unit tests call `parse_pdf()` directly, they use the same loaded model
- Processing a page needs additional memory (~458MB) for activations
- With only ~180MB free, allocation fails → OOM error

**Solution:** Run unit tests from your **host machine** (Option 2 below). This avoids GPU memory conflicts and doesn't require installing pytest in containers.

If you must run tests inside containers, you have two options:

#### Option 1: Copy test files into container (temporary)

```bash
# IMPORTANT: Restart container first to clear GPU memory
docker restart accurate-parser
# Wait for container to be healthy before running tests

# Install pytest in containers (temporary, lost on restart)
docker exec fast-parser pip install pytest
docker exec accurate-parser pip install pytest

# Create examples directory structure in containers
docker exec fast-parser mkdir -p /app/examples/data
docker exec accurate-parser mkdir -p /app/examples/data

# Copy test files into the containers
docker cp tests/ fast-parser:/app/tests/
docker cp tests/ accurate-parser:/app/tests/

# Copy sample PDF file directly (more reliable than copying directory)
docker cp examples/data/sample.pdf fast-parser:/app/examples/data/sample.pdf
docker cp examples/data/sample.pdf accurate-parser:/app/examples/data/sample.pdf

# Run tests inside container (use python3 -m pytest if pytest command not found)
docker exec -it fast-parser python3 -m pytest /app/tests/unit/test_fast_parser.py -v
docker exec -it accurate-parser python3 -m pytest /app/tests/unit/test_accurate_parser.py -v

# Note: Files and installed packages are lost when container is removed/recreated
# Note: If you see OOM errors, restart the container again to free GPU memory
```

**Note:** Tests require the sample PDF (`examples/data/sample.pdf`). The `single_page_pdf_path` fixture creates a temporary single-page PDF from this file. Without the sample PDF, tests that require it will be skipped.

#### Option 2: Run tests from host machine (Recommended)

```bash
# Install dependencies on host
pip install -e ".[dev]"

# Run tests against running containers (integration tests)
pytest tests/integration/ -v

# Or run unit tests locally
pytest tests/unit/ -v
```

**Recommendation:** Run tests from your host machine. Integration tests verify the containerized services work correctly via their HTTP APIs, which is more valuable than running unit tests inside containers. The test files are already on your host machine, so there's no need to copy them into containers.

---

## Local pytest Testing (Optional)

**Local pytest is OPTIONAL** and only useful for quick development feedback without rebuilding containers.

### When to Use Local pytest

✅ **Use for:**
- Quick feedback during development
- Testing models and utilities
- Running unit tests without containers
- Testing parser code logic directly

❌ **Don't use for:**
- Final validation (use integration tests instead)
- Production-like testing (use containers)
- GPU testing (use containers with GPU access)

### Setup for Local Testing

**⚠️ Important: You must install the correct dependencies based on what you're testing.**

```bash
# 1. Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies based on what you're testing

# For fast parser unit tests:
pip install -e ".[fast,dev]"  # Includes pymupdf4llm, pytest, pypdf

# For accurate parser unit tests:
# 1. First install MinerU from submodule (required)
cd MinerU
pip install -e .[core]
cd ..

# 2. Then install project dependencies
pip install -e ".[fast,accurate,dev]"

# For integration tests (tests HTTP APIs, services must be running):
pip install -e ".[dev]"  # Just test tools, parser code not needed

# 3. Verify installation
pytest --version
python3 -c "import pypdf; print('pypdf installed')"
python3 -c "import pymupdf4llm; print('pymupdf4llm installed')"  # If [fast] installed

# 4. Run tests
pytest tests/unit/test_fast_parser.py -v      # Fast parser unit tests
pytest tests/unit/test_accurate_parser.py -v  # Accurate parser unit tests (needs GPU)
pytest tests/test_models.py -v                # Model validation tests
pytest tests/ --cov=src/two_tier_parser --cov-report=html  # With coverage
```

**Dependency Summary:**
- `[dev]` - Test tools (pytest, pypdf, pytest-cov, etc.)
- `[fast]` - Fast parser dependencies (pymupdf4llm, pymupdf)
- `[accurate]` - Accurate parser dependencies (torch, transformers)
- **MinerU** - Must be installed separately from git submodule: `pip install -e MinerU[core]`

**⚠️ Note:** MinerU is NOT included in `[accurate]` because it's a git submodule. You must install it separately before running accurate parser unit tests.

### GPU Memory Management for Accurate Parser Tests

**⚠️ Important:** Accurate parser unit tests may fail with `torch.OutOfMemoryError: CUDA out of memory` if the GPU memory is already in use.

**Why OOM happens:**
- The MinerU VLM model is loaded into GPU memory when the `accurate-parser` service starts
- Model uses ~6-7GB of GPU memory (out of 14.56GB on Tesla T4)
- When unit tests call `parse_pdf()` directly, they use the same loaded model
- Processing a page needs additional memory (~458MB) for activations
- With only ~180MB free, allocation fails → OOM error

**Solutions:**

**Option 1: Restart container before running tests (Recommended)**
```bash
# Restart the accurate-parser container to free GPU memory
docker restart accurate-parser

# Wait for it to be healthy (check logs)
docker logs accurate-parser | grep "Application startup complete"

# Then run tests again
pytest tests/unit/test_accurate_parser.py -v
```

**Option 2: Use integration tests instead (Recommended)**
Integration tests call HTTP APIs and don't have GPU memory conflicts:
```bash
# Integration tests don't import parser code, so no OOM issues
pytest tests/integration/ -v
```

**Option 3: Tests automatically skip on OOM**
Unit tests will automatically skip with a helpful message if GPU memory is unavailable:
```
SKIPPED [1] tests/unit/test_accurate_parser.py: Skipping due to CUDA OOM: ...
Restart accurate-parser container to free GPU memory.
```

This is expected behavior when the GPU is already processing requests.

### Available Test Commands

```bash
# All tests (requires services running for integration tests)
make test

# Unit tests only (requires [fast,dev] or [fast,accurate,dev])
make test-unit

# Integration tests only (requires services running + [dev])
make test-integration

# Coverage report (requires services running)
make test-coverage

# Or use pytest directly
pytest tests/integration/ -v  # Integration tests
pytest tests/unit/ -v          # Unit tests
pytest tests/ -v              # All tests
```

### Test Dependency Quick Reference

| Test Type | Dependencies | Services Running? | GPU Needed? |
|:----------|:-------------|:------------------|:------------|
| **Integration Tests** | `pip install -e ".[dev]"` | ✅ Yes | ❌ No (tests HTTP APIs) |
| **Fast Parser Unit Tests** | `pip install -e ".[fast,dev]"` | ❌ No | ❌ No |
| **Accurate Parser Unit Tests** | `pip install -e ".[fast,accurate,dev]"` | ❌ No | ✅ Yes (or will skip) |
| **All Tests** | `pip install -e ".[fast,accurate,dev]"` | ✅ For integration | ✅ For accurate unit |

**Recommendation:** Start with integration tests (`pip install -e ".[dev]"` + `make up` + `pytest tests/integration/ -v`). They're the most reliable and don't require parser dependencies.

---

## GCP GPU VM Testing

---

## Quick Start

**Prerequisites:**
- GCP account with billing enabled
- GPU quota (1x T4 GPU in your region)
- `gcloud` CLI installed and authenticated
- Terraform (optional, for automated setup)

**Total Setup Time:** ~10 minutes (driver installation takes 5-7 minutes)

---

## Setup Methods

### Method 1: Terraform (Recommended)

#### 1. Install Terraform

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Windows
choco install terraform
```

#### 2. Initialize Terraform

```bash
cd terraform/  # Or wherever you saved the .tf files

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
gcp_project_id = "your-project-id"
gcp_zone       = "us-central1-a"
instance_name  = "parser-gpu-test"
ssh_user       = "$(whoami)"
EOF

# Initialize
terraform init
```

#### 3. Review Plan

```bash
terraform plan
```

Expected resources:
- 1x Compute instance (n1-standard-4 + T4 GPU)
- 2x Firewall rules (SSH, parser ports)
- 1x 100GB SSD boot disk

**Estimated cost:** ~$0.54/hour (~$13/day if left running)

#### 4. Deploy

```bash
terraform apply
# Type 'yes' to confirm

# Save the output
# VM External IP: x.x.x.x
# SSH Command: gcloud compute ssh parser-gpu-test --zone=us-central1-a
```

#### 5. Connect

```bash
# Wait 2-3 minutes for startup script to complete
sleep 180

# SSH into VM
gcloud compute ssh parser-gpu-test --zone=us-central1-a

# Verify setup
nvidia-smi                          # GPU drivers
docker --version                    # Docker installed
docker run --rm --gpus all vllm/vllm-openai:v0.10.1.1 nvidia-smi  # GPU access
```

---

### Method 2: Manual gcloud

If you prefer not to use Terraform:

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create VM
gcloud compute instances create parser-gpu-test \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True \
  --tags=parser-test

# Allow SSH
gcloud compute firewall-rules create ssh-parser-test \
  --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=parser-test

# Allow parser ports (optional)
gcloud compute firewall-rules create parser-ports \
  --allow=tcp:8004,tcp:8005 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=parser-test

# SSH
gcloud compute ssh parser-gpu-test --zone=us-central1-a

# Install dependencies manually
sudo apt-get update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit docker-compose git
sudo systemctl restart docker

# Re-login for docker group
exit
gcloud compute ssh parser-gpu-test --zone=us-central1-a
```

---

## Testing Procedures

### 1. Clone Repository

```bash
# Clone with submodules (includes MinerU)
git clone --recurse-submodules https://github.com/YOUR_ORG/two_tier_document_parser.git
cd two_tier_document_parser
```

### 2. Option A: Test with Docker Compose (Both Services)

```bash
# Build and run both parsers
docker-compose up --build

# In another terminal (new SSH session)
gcloud compute ssh parser-gpu-test --zone=us-central1-a

# Get test PDF
curl -o test.pdf https://arxiv.org/pdf/2301.00000.pdf

# Test fast parser (CPU-only)
curl -X POST http://localhost:8004/parse \
  -F "file=@test.pdf" \
  -o fast_result.json

# Test accurate parser (GPU)
curl -X POST http://localhost:8005/parse \
  -F "file=@test.pdf" \
  -o accurate_result.json

# Compare results
cat fast_result.json | python3 -m json.tool
cat accurate_result.json | python3 -m json.tool
```

### 2. Option B: Test Accurate Parser Only

```bash
cd accurate/

# Build Docker image
docker build -t accurate-parser:test .

# Run with GPU
docker run --gpus all \
  -p 8005:8005 \
  -e LOG_LEVEL=DEBUG \
  --name accurate-parser \
  accurate-parser:test

# In another terminal, test
curl -o test.pdf https://arxiv.org/pdf/2301.00000.pdf

curl -X POST http://localhost:8005/parse \
  -F "file=@test.pdf" \
  -o result.json

cat result.json | python3 -m json.tool
```

### 3. Health Checks

```bash
# Fast parser
curl http://localhost:8004/health | python3 -m json.tool

# Expected:
# {
#   "status": "healthy",
#   "workers": 4,
#   "no_gil": true,
#   "parser": "pymupdf4llm",
#   "version": "0.0.17+"
# }

# Accurate parser
curl http://localhost:8005/health | python3 -m json.tool

# Expected:
# {
#   "status": "healthy",
#   "workers": 2,
#   "gpu_available": true,
#   "parser": "mineru",
#   "version": "2.6.4+"
# }
```

### 4. Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# While parsing, you should see:
# - GPU Memory Usage: 2-4GB
# - GPU Utilization: 60-90%
# - Temperature: 50-70°C
```

### 5. Concurrency Testing

**Fast Parser (4 concurrent threads):**

```bash
# Create test script
cat > test_concurrency.sh <<'EOF'
#!/bin/bash
for i in {1..4}; do
  echo "Starting request $i..."
  curl -X POST http://localhost:8004/parse \
    -F "file=@test.pdf" \
    -o "result_$i.json" \
    -w "\nRequest $i: %{time_total}s\n" &
done
wait
echo "All requests complete"
EOF

chmod +x test_concurrency.sh
./test_concurrency.sh
```

**Accurate Parser (GPU queue):**

```bash
# Test 3 simultaneous requests (2 workers + 1 queued)
for i in {1..3}; do
  curl -X POST http://localhost:8005/parse \
    -F "file=@test.pdf" \
    -o "gpu_result_$i.json" \
    -w "\nGPU Request $i: %{time_total}s\n" &
done
wait
```

### 6. Performance Benchmarking

```bash
# Install Apache Bench
sudo apt-get install -y apache2-utils

# Fast parser benchmark (100 requests, 4 concurrent)
ab -n 100 -c 4 -p test.pdf -T 'multipart/form-data' \
  http://localhost:8004/parse

# Accurate parser benchmark (10 requests, 2 concurrent)
ab -n 10 -c 2 -p test.pdf -T 'multipart/form-data' \
  http://localhost:8005/parse
```

### 7. Memory Leak Testing

```bash
# Run same PDF 1000x
for i in {1..1000}; do
  curl -X POST http://localhost:8004/parse \
    -F "file=@test.pdf" > /dev/null

  if [ $((i % 100)) -eq 0 ]; then
    echo "Processed $i documents"
    docker stats --no-stream
  fi
done
```

### 8. Error Handling Tests

```bash
# Invalid file type (should return 400)
echo "not a pdf" > fake.pdf
curl -X POST http://localhost:8004/parse -F "file=@fake.pdf"

# Empty file (should return 400)
touch empty.pdf
curl -X POST http://localhost:8004/parse -F "file=@empty.pdf"

# Large file (should handle or return 413)
curl -o large.pdf https://example.com/100mb.pdf
curl -X POST http://localhost:8004/parse -F "file=@large.pdf"
```

---

## Performance Benchmarks

### Expected Results

**Fast Parser (PyMuPDF4LLM):**
- Single document: <1 second (target: 0.12s)
- 100 documents (4 concurrent): <30 seconds
- CPU utilization: 400% (4 cores fully utilized)
- Memory usage: 2-4GB

**Accurate Parser (MinerU):**
- Single document: 1-3 minutes (depends on page count)
- Processing speed: 1.70-2.12 pages/second
- GPU utilization: 60-90%
- GPU memory: 2-4GB
- System memory: 8-16GB

### Benchmark Script

```bash
#!/bin/bash

echo "=== Fast Parser Benchmark ==="
time for i in {1..10}; do
  curl -X POST http://localhost:8004/parse \
    -F "file=@test.pdf" > /dev/null 2>&1
done

echo -e "\n=== Accurate Parser Benchmark ==="
time curl -X POST http://localhost:8005/parse \
  -F "file=@test.pdf" > /dev/null 2>&1

echo -e "\n=== GPU Metrics ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

---

## Troubleshooting

### Common Test Errors

#### `ModuleNotFoundError: No module named 'pymupdf4llm'`

**Problem:** Running unit tests without installing `[fast]` extra.

**Solution:**
```bash
# Install fast parser dependencies
pip install -e ".[fast,dev]"

# Then run tests again
pytest tests/unit/test_fast_parser.py -v
```

**Why:** Unit tests import parser code directly, so they need parser dependencies. Integration tests only need `[dev]`.

#### `TypeError: 'ParsingMetadata' object is not subscriptable`

**Problem:** Accessing Pydantic model attributes with dictionary syntax.

**Solution:** Use attribute access instead of dictionary access:
```python
# ❌ Wrong
response.metadata["pages"]

# ✅ Correct
response.metadata.pages
```

#### `torch.OutOfMemoryError: CUDA out of memory`

**Problem:** GPU memory already in use by the running service.

**Solution:**
```bash
# Restart container to free GPU memory
docker restart accurate-parser

# Wait for healthy status, then run tests
pytest tests/unit/test_accurate_parser.py -v
```

**Better:** Use integration tests instead (they don't have GPU memory conflicts):
```bash
pytest tests/integration/ -v
```

#### `pytest: executable file not found in $PATH`

**Problem:** Trying to run `pytest` inside containers where it's not installed.

**Solution:** Run tests from your host machine:
```bash
# Install dependencies on host
pip install -e ".[dev]"  # For integration tests
pip install -e ".[fast,dev]"  # For unit tests

# Run from host
pytest tests/integration/ -v
```

**Why:** Production containers don't include `pytest` to keep images lean.

#### `SKIPPED (pypdf or PyPDF2 required for single-page PDF creation)`

**Problem:** `pypdf` not installed (required for single-page PDF fixture).

**Solution:**
```bash
# Install dev dependencies (includes pypdf)
pip install -e ".[dev]"
```

#### `ReadTimeoutError: Read timed out. (read timeout=600)`

**Problem:** Accurate parser taking longer than client timeout.

**Solution:** Integration tests use single-page PDFs to avoid this. If you see this:
- Check if services are running: `docker ps`
- Check service logs: `docker logs accurate-parser`
- Verify GPU is available: `docker exec accurate-parser nvidia-smi`

---

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# If fails, manually install
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot

# After reboot
nvidia-smi
```

### Docker Cannot Access GPU

```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get purge -y nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all vllm/vllm-openai:v0.10.1.1 nvidia-smi
```

### Out of Memory (OOM)

```bash
# Check memory usage
free -h
docker stats

# Increase VM memory (requires VM stop)
gcloud compute instances stop parser-gpu-test --zone=us-central1-a
gcloud compute instances set-machine-type parser-gpu-test \
  --machine-type=n1-standard-8 \
  --zone=us-central1-a
gcloud compute instances start parser-gpu-test --zone=us-central1-a
```

### Port Already in Use

```bash
# Check what's using the port
sudo lsof -i :8004
sudo lsof -i :8005

# Kill process
sudo kill -9 <PID>

# Or stop Docker containers
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
```

### Slow Parsing Performance

```bash
# Check GPU utilization
nvidia-smi -l 1

# If GPU util is low (<50%), check:
# 1. CUDA version matches Dockerfile
docker run --rm --gpus all accurate-parser:test nvidia-smi

# 2. MinerU models downloaded
docker exec -it accurate-parser ls /root/.cache/huggingface/hub

# 3. Disk I/O (should use SSD)
sudo iostat -xz 1
```

---

## Cost Management

### Estimated Costs (us-central1-a)

| Resource | Cost/Hour | Cost/Day |
|----------|-----------|----------|
| n1-standard-4 | $0.19 | $4.56 |
| NVIDIA T4 GPU | $0.35 | $8.40 |
| 100GB SSD | $0.02 | $0.48 |
| **Total** | **$0.56** | **$13.44** |

### Cost Optimization

**1. Stop VM when not testing:**

```bash
# Stop (keeps disk, stops compute billing)
gcloud compute instances stop parser-gpu-test --zone=us-central1-a

# Start when needed
gcloud compute instances start parser-gpu-test --zone=us-central1-a
```

**2. Use preemptible instances (60-80% discount):**

```bash
# Add to Terraform or gcloud create command
--preemptible
```

⚠️ **Warning:** Preemptible VMs can be shut down at any time (max 24h runtime)

**3. Delete when testing complete:**

```bash
# Terraform
terraform destroy

# Or manual
gcloud compute instances delete parser-gpu-test --zone=us-central1-a
```

### Monitoring Costs

```bash
# View current month's costs
gcloud billing accounts list
gcloud beta billing projects describe YOUR_PROJECT_ID

# Set budget alerts in GCP Console:
# Billing → Budgets & Alerts → Create Budget
# - Set threshold: $50/month
# - Alert at 50%, 90%, 100%
```

---

## Testing Checklist

Before considering testing complete, verify:

- [ ] GPU drivers installed (`nvidia-smi` works)
- [ ] Docker has GPU access (`docker run --gpus all vllm/vllm-openai:v0.10.1.1 nvidia-smi`)
- [ ] Fast parser health check returns `"status": "healthy"`
- [ ] Accurate parser health check returns `"gpu_available": true`
- [ ] Fast parser completes single document in <1s
- [ ] Accurate parser completes single document in 1-3min
- [ ] Fast parser handles 4 concurrent requests without blocking
- [ ] Accurate parser utilizes GPU (>60% during parsing)
- [ ] No memory leaks (1000 document test shows stable memory)
- [ ] Error handling works (invalid file returns 400)
- [ ] Markdown output quality verified (manual inspection)
- [ ] Images extracted correctly (accurate parser only)
- [ ] Temporary files cleaned up (check `/tmp/`)

---

## Advanced Testing

### External Access Testing

If you want to test from your local machine (not SSH):

```bash
# Get VM external IP
gcloud compute instances describe parser-gpu-test \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Test from local machine
VM_IP=<external-ip>
curl -X POST http://$VM_IP:8004/parse -F "file=@local-test.pdf"
curl -X POST http://$VM_IP:8005/parse -F "file=@local-test.pdf"
```

### Load Testing

```bash
# Install hey (HTTP load testing tool)
sudo apt-get install -y hey

# Fast parser load test (100 requests, 10 concurrent)
hey -n 100 -c 10 -m POST \
  -T "multipart/form-data" \
  -D test.pdf \
  http://localhost:8004/parse

# Accurate parser load test (20 requests, 2 concurrent)
hey -n 20 -c 2 -m POST \
  -T "multipart/form-data" \
  -D test.pdf \
  http://localhost:8005/parse
```

### Profiling

```bash
# Install profiling tools
pip install py-spy memory_profiler

# CPU profiling (fast parser)
py-spy record --native -o fast_profile.svg -- python fast/app.py

# Memory profiling (accurate parser)
mprof run accurate/app.py
mprof plot
```

---

## Next Steps

After successful testing:

1. **Push images to GCR:**
   ```bash
   docker tag accurate-parser:test gcr.io/YOUR_PROJECT/accurate-parser:v1.0.0
   docker push gcr.io/YOUR_PROJECT/accurate-parser:v1.0.0
   ```

2. **Update the image tag in your Kubernetes manifests**

3. **Deploy to GKE** (see "Production Deployment" in `docs/DOCKER_SETUP.md`)

4. **Set up monitoring** (Prometheus, Grafana)

5. **Configure autoscaling** (HPA for both services)

---

**Last Updated:** November 2025
**Maintainer:** Daddal
**Related:** [DOCKER_SETUP.md](DOCKER_SETUP.md)
