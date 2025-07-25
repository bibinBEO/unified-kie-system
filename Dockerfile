# Multi-stage build for optimized production image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies with retry logic
RUN apt-get clean && \
    apt-get update --allow-releaseinfo-change || apt-get update --allow-releaseinfo-change || true && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Create non-root user
RUN useradd -m -s /bin/bash kie-user && \
    mkdir -p /app && \
    chown -R kie-user:kie-user /app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements without flash-attn first
RUN sed '/flash-attn/d' requirements.txt > requirements_temp.txt && \
    pip3 install --no-cache-dir -r requirements_temp.txt

# Install flash-attention separately with proper build dependencies
RUN pip3 install --no-cache-dir packaging ninja && \
    pip3 install --no-cache-dir flash-attn --no-build-isolation

# Copy application code
COPY --chown=kie-user:kie-user . .

# Create necessary directories
RUN mkdir -p uploads results models logs schemas && \
    chown -R kie-user:kie-user uploads results models logs schemas

# Switch to non-root user
USER kie-user

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "app.py"]