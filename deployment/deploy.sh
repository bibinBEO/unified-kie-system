#!/bin/bash

# Unified KIE System Deployment Script
# Usage: ./deploy.sh [local|server] [GPU_COUNT]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TYPE=${1:-server}
GPU_COUNT=${2:-1}
SERVER_HOST=${3:-"your-server-ip"}
SSH_USER=${4:-"ubuntu"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check requirements
check_requirements() {
    log "Checking deployment requirements..."
    
    if [ "$DEPLOYMENT_TYPE" = "local" ]; then
        # Check Docker
        if ! command -v docker &> /dev/null; then
            error "Docker is not installed. Please install Docker first."
        fi
        
        # Check Docker Compose
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            error "Docker Compose is not installed. Please install Docker Compose first."
        fi
        
        # Check NVIDIA Docker (for GPU support)
        if [ "$GPU_COUNT" -gt 0 ]; then
            if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
                warn "NVIDIA Docker runtime not available. Will deploy in CPU-only mode."
                GPU_COUNT=0
            fi
        fi
    else
        # Check SSH access for server deployment
        if ! command -v ssh &> /dev/null; then
            error "SSH is not available. Cannot deploy to remote server."
        fi
        
        if ! command -v scp &> /dev/null; then
            error "SCP is not available. Cannot copy files to remote server."
        fi
    fi
}

# Prepare deployment files
prepare_deployment() {
    log "Preparing deployment files..."
    
    cd "$PROJECT_DIR"
    
    # Create deployment package
    DEPLOY_PACKAGE="/tmp/unified-kie-system-$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # Exclude unnecessary files
    tar --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='uploads/*' \
        --exclude='results/*' \
        --exclude='models/*' \
        --exclude='logs/*' \
        --exclude='.pytest_cache' \
        --exclude='*.log' \
        -czf "$DEPLOY_PACKAGE" .
    
    info "Deployment package created: $DEPLOY_PACKAGE"
    echo "$DEPLOY_PACKAGE"
}

# Deploy locally
deploy_local() {
    log "Starting local deployment..."
    
    cd "$PROJECT_DIR"
    
    # Update environment configuration
    if [ "$GPU_COUNT" -eq 0 ]; then
        export FORCE_CPU=true
        warn "Deploying in CPU-only mode"
    else
        export FORCE_CPU=false
        info "Deploying with GPU support ($GPU_COUNT GPUs)"
    fi
    
    # Stop existing containers
    log "Stopping existing containers..."
    docker-compose down --remove-orphans || true
    
    # Build and start services
    log "Building and starting services..."
    if [ "$GPU_COUNT" -gt 0 ]; then
        docker-compose up --build -d
    else
        # Use CPU-only configuration
        COMPOSE_FILE=docker-compose.cpu.yml
        if [ ! -f "$COMPOSE_FILE" ]; then
            # Create CPU-only compose file
            sed 's/devices:/# devices:/g; s/- driver: nvidia/# - driver: nvidia/g; s/count: 1/# count: 1/g; s/capabilities: \[gpu\]/# capabilities: [gpu]/g' docker-compose.yml > "$COMPOSE_FILE"
        fi
        docker-compose -f "$COMPOSE_FILE" up --build -d
    fi
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
    
    # Check health
    check_health "localhost:8000"
    
    log "Local deployment completed successfully!"
    info "Access the system at: http://localhost:8000"
}

# Deploy to remote server
deploy_server() {
    log "Starting server deployment to $SERVER_HOST..."
    
    # Prepare deployment package
    DEPLOY_PACKAGE=$(prepare_deployment)
    
    # Copy files to server
    log "Copying files to server..."
    scp "$DEPLOY_PACKAGE" "$SSH_USER@$SERVER_HOST:/tmp/"
    
    # Remote deployment script
    REMOTE_SCRIPT=$(cat << 'EOF'
#!/bin/bash
set -e

PACKAGE_FILE="$1"
GPU_COUNT="$2"

# Extract package
cd /opt
sudo rm -rf unified-kie-system-old || true
sudo mv unified-kie-system unified-kie-system-old || true
sudo mkdir -p unified-kie-system
sudo tar -xzf "$PACKAGE_FILE" -C unified-kie-system
sudo chown -R $USER:$USER unified-kie-system
cd unified-kie-system

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Install NVIDIA Docker if GPU support needed
if [ "$GPU_COUNT" -gt 0 ]; then
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        echo "Installing NVIDIA Docker..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
fi

# Set environment variables
if [ "$GPU_COUNT" -eq 0 ]; then
    export FORCE_CPU=true
    echo "Deploying in CPU-only mode"
else
    export FORCE_CPU=false
    echo "Deploying with GPU support"
fi

# Stop existing containers
docker-compose down --remove-orphans || true

# Start services
if [ "$GPU_COUNT" -gt 0 ]; then
    docker-compose up --build -d
else
    # Create CPU-only compose file if needed
    if [ ! -f docker-compose.cpu.yml ]; then
        sed 's/devices:/# devices:/g; s/- driver: nvidia/# - driver: nvidia/g; s/count: 1/# count: 1/g; s/capabilities: \[gpu\]/# capabilities: [gpu]/g' docker-compose.yml > docker-compose.cpu.yml
    fi
    docker-compose -f docker-compose.cpu.yml up --build -d
fi

# Clean up
rm -f "$PACKAGE_FILE"

echo "Server deployment completed!"
EOF
    )
    
    # Execute remote deployment
    log "Executing remote deployment..."
    ssh "$SSH_USER@$SERVER_HOST" "bash -s" -- "$DEPLOY_PACKAGE" "$GPU_COUNT" <<< "$REMOTE_SCRIPT"
    
    # Clean up local package
    rm -f "$DEPLOY_PACKAGE"
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 60
    
    # Check health
    check_health "$SERVER_HOST:8000"
    
    log "Server deployment completed successfully!"
    info "Access the system at: http://$SERVER_HOST:8000"
}

# Check service health
check_health() {
    local endpoint="$1"
    local max_attempts=12
    local attempt=1
    
    log "Checking service health at $endpoint..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "http://$endpoint/health" &> /dev/null; then
            log "Service is healthy!"
            return 0
        fi
        
        info "Health check attempt $attempt/$max_attempts failed, waiting..."
        sleep 10
        ((attempt++))
    done
    
    error "Service health check failed after $max_attempts attempts"
}

# Show system status
show_status() {
    local endpoint="$1"
    
    info "System Status:"
    echo "=============="
    
    # Get health status
    HEALTH=$(curl -s "http://$endpoint/health" 2>/dev/null || echo "Service unavailable")
    echo "Health: $HEALTH"
    
    # Get stats
    STATS=$(curl -s "http://$endpoint/stats" 2>/dev/null || echo "Stats unavailable")
    echo "Stats: $STATS"
    
    echo ""
    info "Available endpoints:"
    echo "- Web Interface: http://$endpoint/"
    echo "- Health Check: http://$endpoint/health"
    echo "- API Documentation: http://$endpoint/docs"
    echo "- Statistics: http://$endpoint/stats"
}

# Main deployment logic
main() {
    log "Starting Unified KIE System Deployment"
    info "Deployment Type: $DEPLOYMENT_TYPE"
    info "GPU Count: $GPU_COUNT"
    
    check_requirements
    
    case "$DEPLOYMENT_TYPE" in
        "local")
            deploy_local
            show_status "localhost:8000"
            ;;
        "server")
            if [ "$SERVER_HOST" = "your-server-ip" ]; then
                error "Please specify server host as third argument: ./deploy.sh server 1 your-server-ip"
            fi
            deploy_server
            show_status "$SERVER_HOST:8000"
            ;;
        *)
            error "Invalid deployment type. Use 'local' or 'server'"
            ;;
    esac
    
    log "Deployment completed successfully!"
}

# Script usage
usage() {
    echo "Usage: $0 [local|server] [GPU_COUNT] [SERVER_HOST] [SSH_USER]"
    echo ""
    echo "Examples:"
    echo "  $0 local 1                    # Deploy locally with 1 GPU"
    echo "  $0 local 0                    # Deploy locally CPU-only"
    echo "  $0 server 1 192.168.1.100    # Deploy to server with 1 GPU"
    echo "  $0 server 0 192.168.1.100 ubuntu  # Deploy to server CPU-only with custom SSH user"
    echo ""
    echo "Parameters:"
    echo "  Deployment Type: local or server (default: server)"
    echo "  GPU Count: Number of GPUs to use, 0 for CPU-only (default: 1)"
    echo "  Server Host: IP address or hostname for server deployment"
    echo "  SSH User: SSH username for server deployment (default: ubuntu)"
}

# Handle help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"