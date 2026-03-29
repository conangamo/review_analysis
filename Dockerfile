# Dockerfile for Product Review Analyzer
# NOTE: This is for FUTURE use when deploying to production
# For development, use virtual environment instead (see docs/DOCKER_DECISION.md)

# Multi-stage build for smaller final image

# ============================================
# Stage 1: Builder (install dependencies)
# ============================================
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Runtime (final image)
# ============================================
FROM python:3.9-slim

# Metadata
LABEL maintainer="your.email@example.com"
LABEL description="Product Review Analyzer - AI-powered sentiment analysis"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY .env.example .env

# Create necessary directories
RUN mkdir -p data/raw data/processed data/cache logs

# Make PATH include local packages
ENV PATH=/root/.local/bin:$PATH

# Set Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501')"

# Default command (run Streamlit UI)
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================
# Usage:
# ============================================
# Build:
#   docker build -t product-review-analyzer .
#
# Run UI only:
#   docker run -p 8501:8501 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/config:/app/config \
#     product-review-analyzer
#
# Run with custom command:
#   docker run product-review-analyzer \
#     python scripts/setup_database.py
#
# ============================================
# GPU Support (for AI processing):
# ============================================
# NOTE: GPU support in Docker is complex. 
# For development, run AI processing natively instead.
#
# If you really need GPU in Docker:
# 1. Install NVIDIA Container Toolkit
# 2. Use nvidia/cuda base image instead
# 3. Run with: docker run --gpus all ...
#
# See: docs/DOCKER_DECISION.md for details
# ============================================
