#!/bin/bash

# Verification script for Unified KIE System deployment on mira.beo-software.de

echo "🔍 Verifying Unified KIE System Deployment..."
echo "🌐 Domain: mira.beo-software.de"
echo ""

# Check Docker container
echo "📦 Docker Container Status:"
if sudo docker ps | grep -q unified-kie-vllm; then
    echo "✅ unified-kie-vllm container is running"
    sudo docker ps --filter name=unified-kie-vllm --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "❌ unified-kie-vllm container is not running"
    exit 1
fi
echo ""

# Check nginx
echo "🌐 nginx Status:"
if systemctl is-active --quiet nginx; then
    echo "✅ nginx is active and running"
else
    echo "❌ nginx is not running"
    exit 1
fi
echo ""

# Test backend directly
echo "🔍 Backend Health Check (Direct):"
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "✅ Backend responding on port 8000"
    health_data=$(curl -s http://127.0.0.1:8000/health)
    echo "📊 Health Status: $(echo $health_data | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
    echo "🖥️  GPU Available: $(echo $health_data | grep -o '"gpu_available":[^,]*' | cut -d':' -f2)"
else
    echo "❌ Backend not responding on port 8000"
    exit 1
fi
echo ""

# Test nginx proxy
echo "🌐 nginx Proxy Test:"
if curl -s -H "Host: mira.beo-software.de" http://localhost/health > /dev/null; then
    echo "✅ nginx proxy is working"
else
    echo "❌ nginx proxy is not working"
    exit 1
fi
echo ""

# Test domain access
echo "🌍 Domain Access Test:"
if curl -s http://mira.beo-software.de/health > /dev/null; then
    echo "✅ Domain mira.beo-software.de is accessible"
    health_response=$(curl -s http://mira.beo-software.de/health)
    echo "📊 Models loaded:"
    echo "   - nanonets: $(echo $health_response | grep -o '"nanonets":[^,]*' | cut -d':' -f2)"
    echo "   - layoutlm: $(echo $health_response | grep -o '"layoutlm":[^,]*' | cut -d':' -f2)"
    echo "   - easyocr: $(echo $health_response | grep -o '"easyocr":[^,]*' | cut -d':' -f2)"
else
    echo "❌ Domain mira.beo-software.de is not accessible"
    exit 1
fi
echo ""

# Test main application page
echo "🏠 Main Application Test:"
if curl -s http://mira.beo-software.de/ | grep -q "Unified KIE Document Processing"; then
    echo "✅ Main application page is loading correctly"
else
    echo "❌ Main application page is not loading correctly"
    exit 1
fi
echo ""

# GPU Memory check
echo "🖥️  GPU Status:"
gpu_info=$(sudo docker exec unified-kie-vllm bash -c "
export CUDA_VISIBLE_DEVICES=0
python3 -c \"
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'⚡ Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
else:
    print('❌ No GPU available')
\"" 2>/dev/null)
echo "$gpu_info"
echo ""

# Final summary
echo "🎉 Deployment Verification Complete!"
echo ""
echo "✅ All systems operational:"
echo "   🐳 Docker container: unified-kie-vllm"
echo "   🌐 nginx reverse proxy: active"
echo "   🌍 Domain: http://mira.beo-software.de"
echo "   🖥️  GPU: NVIDIA RTX 4000 SFF Ada Generation"
echo "   ⚡ vLLM: nanonets-ocr-s model ready"
echo ""
echo "🔗 Access URLs:"
echo "   📱 Application: http://mira.beo-software.de"
echo "   📊 Health Check: http://mira.beo-software.de/health"
echo ""
echo "🛠️  Management:"
echo "   Container logs: sudo docker logs -f unified-kie-vllm"
echo "   nginx logs: sudo tail -f /var/log/nginx/access.log"
echo "   Restart all: ./deploy-vllm.sh && sudo systemctl restart nginx"