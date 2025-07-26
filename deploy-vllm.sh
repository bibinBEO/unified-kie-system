#!/bin/bash

# Deployment script for Unified KIE System with vLLM GPU support
# Optimized for mira.beo-software.de with NVIDIA RTX 4000 SFF Ada Generation

set -e

echo "🚀 Deploying Unified KIE System with vLLM GPU support..."
echo "🎯 Target: mira.beo-software.de"
echo "🖥️  GPU: NVIDIA RTX 4000 SFF Ada Generation"
echo "⚡ vLLM: nanonets-ocr-s model"

# Check if Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA Docker runtime not available. Please install nvidia-container-toolkit."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
sudo docker stop unified-kie-vllm 2>/dev/null || true
sudo docker rm unified-kie-vllm 2>/dev/null || true

# Build the Docker image if it doesn't exist or if forced rebuild
if [[ "$1" == "--rebuild" ]] || ! sudo docker images unified-kie-vllm:latest --format "table {{.Repository}}" | grep -q unified-kie-vllm; then
    echo "🔨 Building Docker image with vLLM support..."
    sudo docker build -f Dockerfile.vllm -t unified-kie-vllm:latest .
fi

# Create necessary directories
echo "📁 Creating directories..."
sudo mkdir -p /opt/unified-kie/{uploads,results,models,logs,schemas}
sudo chown -R 1000:1000 /opt/unified-kie

# Start the container
echo "🐳 Starting Docker container..."
sudo docker run -d \
    --name unified-kie-vllm \
    --gpus all \
    --restart unless-stopped \
    -p 8000:8000 \
    -v /opt/unified-kie/uploads:/app/uploads:rw \
    -v /opt/unified-kie/results:/app/results:rw \
    -v /opt/unified-kie/models:/app/models:rw \
    -v /opt/unified-kie/logs:/app/logs:rw \
    -v /opt/unified-kie/schemas:/app/schemas:rw \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e TORCH_CUDA_ARCH_LIST="8.9" \
    -e VLLM_MODEL="nanonets/Nanonets-OCR-s" \
    -e MAX_MODEL_LEN=8192 \
    -e GPU_MEMORY_UTILIZATION=0.9 \
    -e TRUST_REMOTE_CODE=true \
    unified-kie-vllm:latest

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 10

# Check container status
if sudo docker ps | grep -q unified-kie-vllm; then
    echo "✅ Container is running!"
    
    # Test GPU access
    echo "🧪 Testing GPU access..."
    sudo docker exec unified-kie-vllm python3 -c "
import torch
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'🖥️  GPU: {torch.cuda.get_device_name(0)}')
print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
    
    # Test vLLM import
    echo "🧪 Testing vLLM..."
    sudo docker exec unified-kie-vllm python3 -c "
from vllm import LLM, SamplingParams
print('✅ vLLM is working correctly!')
"
    
    # Show container logs
    echo "📋 Container logs (last 20 lines):"
    sudo docker logs --tail 20 unified-kie-vllm
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "🌐 Access the application at: http://mira.beo-software.de:8000"
    echo "📊 Health check: http://mira.beo-software.de:8000/health"
    echo ""
    echo "📝 Useful commands:"
    echo "  View logs:    sudo docker logs -f unified-kie-vllm"
    echo "  Stop:         sudo docker stop unified-kie-vllm"
    echo "  Restart:      sudo docker restart unified-kie-vllm"
    echo "  Shell:        sudo docker exec -it unified-kie-vllm bash"
    echo ""
    echo "🔧 GPU Model: nanonets-ocr-s with vLLM acceleration"
    echo "🎯 Optimized for NVIDIA RTX 4000 SFF Ada Generation"
    
else
    echo "❌ Container failed to start. Check logs:"
    sudo docker logs unified-kie-vllm
    exit 1
fi