# LM Watermark Lab - Makefile

.PHONY: help install install-dev test test-cov lint format type-check security clean build docs serve dev docker-build docker-run

# Default target
help:
	@echo "LM Watermark Lab - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install           Install package in production mode"
	@echo "  install-dev       Install package in development mode with all dependencies"
	@echo "  clean             Clean build artifacts and cache files"
	@echo ""
	@echo "Development:"
	@echo "  dev               Start development server with hot reload"
	@echo "  serve             Start production server"
	@echo "  format            Format code with black and isort"
	@echo "  lint              Run linting with flake8"
	@echo "  type-check        Run type checking with mypy"
	@echo "  security          Run security checks with bandit and safety"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-cov          Run tests with coverage report"
	@echo "  test-fast         Run tests excluding slow ones"
	@echo "  test-integration  Run integration tests only"
	@echo ""
	@echo "Documentation:"
	@echo "  docs              Build documentation"
	@echo "  docs-serve        Serve documentation locally"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  build             Build Python package"
	@echo "  docker-build      Build Docker image"
	@echo "  docker-run        Run Docker container"
	@echo ""
	@echo "Data & Models:"
	@echo "  download-models   Download required model weights"
	@echo "  prepare-data      Prepare evaluation datasets"
	@echo ""

# Python and package management
PYTHON := python3
PIP := pip
PACKAGE_NAME := lm-watermark-lab

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

# Development server
dev:
	uvicorn src.watermark_lab.api.main:app --reload --host 0.0.0.0 --port 8080

serve:
	gunicorn src.watermark_lab.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080

# Code quality
format:
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

lint:
	flake8 $(SRC_DIR) $(TEST_DIR)

type-check:
	mypy $(SRC_DIR)

security:
	bandit -r $(SRC_DIR)
	safety check

quality: format lint type-check security

# Testing
test:
	pytest $(TEST_DIR) -v

test-cov:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR)/watermark_lab --cov-report=html --cov-report=term

test-fast:
	pytest $(TEST_DIR) -v -m "not slow"

test-integration:
	pytest $(TEST_DIR) -v -m "integration"

test-api:
	pytest $(TEST_DIR) -v -m "api"

test-cli:
	pytest $(TEST_DIR) -v -m "cli"

# Documentation
docs:
	cd $(DOCS_DIR) && make html

docs-serve:
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	cd $(DOCS_DIR) && make clean

# Build and distribution
build: clean
	$(PYTHON) -m build

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -delete
	find . -type d -name .mypy_cache -delete
	rm -rf .coverage htmlcov/ .tox/

# Docker
docker-build:
	docker build -t $(PACKAGE_NAME):latest .

docker-run:
	docker run -it --rm -p 8080:8080 -v $(PWD)/data:/app/data $(PACKAGE_NAME):latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Data preparation
download-models:
	$(PYTHON) scripts/download_models.py

prepare-data:
	$(PYTHON) scripts/prepare_datasets.py

# Database (if using)
migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

# Benchmarking
benchmark:
	$(PYTHON) benchmarks/run_benchmark.py

benchmark-quick:
	$(PYTHON) benchmarks/run_benchmark.py --quick

# Jupyter
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Environment setup
setup-env:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# CI/CD helpers
ci-install:
	$(PIP) install -e ".[test]"

ci-test:
	pytest $(TEST_DIR) --junitxml=test-results.xml --cov=$(SRC_DIR)/watermark_lab --cov-report=xml

ci-quality:
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check $(SRC_DIR) $(TEST_DIR)
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)
	bandit -r $(SRC_DIR) -f json -o bandit-report.json

# Performance profiling
profile:
	$(PYTHON) -m cProfile -o profile.stats scripts/profile_generation.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	mprof run scripts/profile_memory.py
	mprof plot

# Release preparation
version-bump:
	semantic-release version

release: clean build
	twine upload $(DIST_DIR)/*

# Health checks
health-check:
	curl -f http://localhost:8080/health || exit 1

# Load testing
load-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8080

# Configuration validation
validate-config:
	$(PYTHON) scripts/validate_configs.py

# Dependency updates
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Security audit
audit:
	safety check
	bandit -r $(SRC_DIR)
	pip-audit

# All checks (used in CI)
all-checks: ci-quality ci-test security

# Development workflow
dev-setup: install-dev setup-env download-models prepare-data
	@echo "Development environment ready!"

# Quick development checks
quick-check: format lint test-fast

# Full validation (before PR)
validate: quality test-cov docs build