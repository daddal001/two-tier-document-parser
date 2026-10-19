.PHONY: help install dev build build-accurate up down logs clean clean-volumes test test-unit test-integration test-coverage lint format type-check coverage check hooks

.DEFAULT_GOAL := help

COMPOSE_FILE := deploy/docker-compose.yml
# Increase timeout for model downloads during build (10-20 minutes)
export COMPOSE_HTTP_TIMEOUT := 1800

# Docker commands
build:
	@echo "🔨 Building Docker images..."
	@echo "⏳ Note: accurate-parser build includes model download (~10-20 min)"
	docker-compose -f $(COMPOSE_FILE) build

build-accurate:
	@echo "🔨 Building accurate-parser (includes model download, ~10-20 min)..."
	docker-compose -f $(COMPOSE_FILE) build accurate-parser

up:
	docker-compose -f $(COMPOSE_FILE) up --build -d

down:
	docker-compose -f $(COMPOSE_FILE) down

logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

# Development commands
test:
	pytest tests/ -v -m "not integration and not models"

test-unit:
	pytest tests/unit/ -v -m "not models"

test-integration:
	pytest tests/integration/ -v -m integration --run-integration

test-coverage:
	pytest tests/ -m "not integration and not models" --cov=src/two_tier_parser --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ examples/ scripts/
	@echo "✓ Linting complete"

format:
	black src/ tests/ examples/ scripts/ --line-length=100
	isort src/ tests/ examples/ scripts/ --profile=black --line-length=100
	@echo "✓ Formatting complete"

type-check:
	mypy src/
	@echo "✓ Type checking complete"

coverage: test-coverage
	@echo "Coverage report generated in htmlcov/index.html"

# Only this project's containers and Python caches; volumes need an explicit command.
clean:
	docker-compose -f $(COMPOSE_FILE) down --remove-orphans
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name "htmlcov" \) -not -path "./MinerU/*" -not -path "./.git/*" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -not -path "./MinerU/*" -prune -exec rm -rf {} +

clean-volumes:
	docker-compose -f $(COMPOSE_FILE) down --volumes --remove-orphans

check:
	black --check src/ tests/ examples/ scripts/
	isort --check-only src/ tests/ examples/ scripts/ --profile black
	flake8 src/ tests/ examples/ scripts/
	mypy src/

# Install the local git hooks (commit and commit-message stages)
hooks:
	pre-commit install --hook-type pre-commit --hook-type commit-msg

# Environment setup
# Installs exactly the versions in uv.lock into ./.venv (requires uv). Activate it with
# `source .venv/bin/activate` so the pre-commit hooks and make targets find the tools.
install:
	uv sync --extra fast --extra dev --extra cancellation --extra telemetry --extra telemetry-sdk
	@echo "Activate the environment: source .venv/bin/activate"

dev: install hooks

# Help
help:
	@echo "Setup"
	@echo "  make install          - Create ./.venv from uv.lock (fast, dev and cancellation extras)"
	@echo "  make hooks            - Install the git pre-commit and commit-msg hooks"
	@echo "  make dev              - install + hooks"
	@echo ""
	@echo "Docker (COMPOSE_HTTP_TIMEOUT is 1800s for model downloads)"
	@echo "  make build            - Build all Docker images (accurate-parser: ~10-20 min)"
	@echo "  make build-accurate   - Build only accurate-parser (includes model download)"
	@echo "  make up               - Start services (builds if needed)"
	@echo "  make down             - Stop services"
	@echo "  make logs             - View service logs"
	@echo ""
	@echo "Quality"
	@echo "  make test             - Unit and contract tests (no live services, no models)"
	@echo "  make test-unit        - Unit tests only"
	@echo "  make test-integration - Tests against running services"
	@echo "  make test-coverage    - Tests with coverage report"
	@echo "  make check            - black, isort, flake8 and mypy (CI also runs every pre-commit hook)"
	@echo "  make lint             - flake8"
	@echo "  make format           - Format code with black and isort"
	@echo "  make type-check       - mypy"
	@echo "  make clean            - Stop this project's containers and delete Python caches"
	@echo "  make clean-volumes    - Also delete this project's Docker volumes (model caches)"
	@echo "  make help             - Show this help message"
