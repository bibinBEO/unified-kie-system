#!/bin/bash

# Setup nginx for Unified KIE System on mira.beo-software.de
# This script configures nginx to proxy requests to the vLLM Docker container

set -e

echo "🌐 Setting up nginx for mira.beo-software.de..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script needs to be run as root or with sudo"
    exit 1
fi

# Install nginx if not installed
if ! command -v nginx &> /dev/null; then
    echo "📦 Installing nginx..."
    apt update
    apt install -y nginx
fi

# Stop nginx if running
echo "🛑 Stopping nginx..."
systemctl stop nginx 2>/dev/null || true

# Backup existing nginx configuration
if [ -f /etc/nginx/nginx.conf ]; then
    echo "💾 Backing up existing nginx configuration..."
    cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup.$(date +%Y%m%d_%H%M%S)
fi

# Copy our nginx configuration
echo "📝 Installing new nginx configuration..."
cp /home/bibin.wilson/unified-kie-system/deployment/nginx.conf /etc/nginx/nginx.conf

# Test nginx configuration
echo "🧪 Testing nginx configuration..."
if ! nginx -t; then
    echo "❌ nginx configuration test failed!"
    if [ -f /etc/nginx/nginx.conf.backup.* ]; then
        echo "🔄 Restoring backup configuration..."
        cp /etc/nginx/nginx.conf.backup.* /etc/nginx/nginx.conf
    fi
    exit 1
fi

# Check if unified-kie-vllm container is running
if ! docker ps | grep -q unified-kie-vllm; then
    echo "⚠️  Warning: unified-kie-vllm container is not running!"
    echo "🚀 Starting the container..."
    cd /home/bibin.wilson/unified-kie-system
    ./deploy-vllm.sh
    echo "⏳ Waiting for container to be ready..."
    sleep 30
fi

# Verify the backend is responding
echo "🔍 Checking backend health..."
max_attempts=10
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "✅ Backend is healthy!"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts: Backend not ready, waiting..."
        sleep 5
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Backend failed to respond after $max_attempts attempts"
    echo "📋 Container logs:"
    docker logs --tail 20 unified-kie-vllm
    exit 1
fi

# Start nginx
echo "🚀 Starting nginx..."
systemctl start nginx
systemctl enable nginx

# Check nginx status
if systemctl is-active --quiet nginx; then
    echo "✅ nginx is running!"
else
    echo "❌ Failed to start nginx"
    systemctl status nginx
    exit 1
fi

# Test the full setup
echo "🧪 Testing complete setup..."
sleep 2

# Test HTTP request
if curl -s -H "Host: mira.beo-software.de" http://localhost/health > /dev/null; then
    echo "✅ nginx proxy is working!"
else
    echo "❌ nginx proxy test failed"
    echo "📋 nginx error log:"
    tail -10 /var/log/nginx/error.log
fi

echo ""
echo "🎉 nginx setup completed successfully!"
echo "🌐 Application is now available at: http://mira.beo-software.de"
echo "📊 Health check: http://mira.beo-software.de/health"
echo ""
echo "📝 nginx management commands:"
echo "  Reload config:   sudo systemctl reload nginx"
echo "  Restart:         sudo systemctl restart nginx"
echo "  View logs:       sudo tail -f /var/log/nginx/access.log"
echo "  Error logs:      sudo tail -f /var/log/nginx/error.log"
echo ""
echo "🔧 Configuration file: /etc/nginx/nginx.conf"
echo "🐳 Backend container: unified-kie-vllm (port 8000)"
echo ""

# Display current status
echo "📊 Current status:"
echo "  nginx:           $(systemctl is-active nginx)"
echo "  Docker container: $(docker ps --filter name=unified-kie-vllm --format 'table {{.Status}}')"
echo ""
echo "✅ Setup complete! Your Unified KIE System with vLLM is now accessible via nginx."