# Multi-stage Dockerfile for LM Watermark Lab
# Optimized for security, performance, and minimal size

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add labels for metadata
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.url="https://github.com/terragon-labs/lm-watermark-lab" \
      org.opencontainers.image.source="https://github.com/terragon-labs/lm-watermark-lab" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.title="LM Watermark Lab" \
      org.opencontainers.image.description="Comprehensive toolkit for watermarking, detecting, and attacking LLM-generated text" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels dist/*.whl

# Production stage
FROM python:3.11-slim as production

# Security: Create non-root user
RUN groupadd -r watermarklab && useradd -r -g watermarklab -d /app -s /bin/bash watermarklab

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory and ownership
WORKDIR /app
RUN chown -R watermarklab:watermarklab /app

# Copy wheels from builder
COPY --from=builder /build/wheels /tmp/wheels
COPY --from=builder /build/dist/*.whl /tmp/

# Install the application
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/wheels /tmp/*.whl /root/.cache

# Copy application files
COPY --chown=watermarklab:watermarklab . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R watermarklab:watermarklab /app/data /app/logs /app/cache

# Switch to non-root user
USER watermarklab

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_PATH=/app/data/models \
    DATA_ROOT=/app/data \
    LOG_FILE=/app/logs/watermark_lab.log

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["uvicorn", "watermark_lab.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER watermarklab

# Development command with reload
CMD ["uvicorn", "watermark_lab.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# Testing stage
FROM development as testing

# Switch to root for test setup
USER root

# Copy test files
COPY --chown=watermarklab:watermarklab tests/ /app/tests/

# Install test dependencies
RUN pip install --no-cache-dir -e ".[test]"

# Switch back to app user
USER watermarklab

# Test command
CMD ["pytest", "tests/", "-v", "--cov=src/watermark_lab"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Security: Create non-root user
RUN groupadd -r watermarklab && useradd -r -g watermarklab -d /app -s /bin/bash watermarklab

# Set work directory
WORKDIR /app
RUN chown -R watermarklab:watermarklab /app

# Copy wheels from builder
COPY --from=builder /build/wheels /tmp/wheels
COPY --from=builder /build/dist/*.whl /tmp/

# Install the application with GPU support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/wheels /tmp/*.whl /root/.cache

# Copy application files
COPY --chown=watermarklab:watermarklab . .

# Create directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R watermarklab:watermarklab /app/data /app/logs /app/cache

# Switch to non-root user
USER watermarklab

# Set GPU environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" \
    MODEL_CACHE_PATH=/app/data/models \
    DATA_ROOT=/app/data

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# GPU command
CMD ["uvicorn", "watermark_lab.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]