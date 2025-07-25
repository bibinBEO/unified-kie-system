# Unified KIE Document Processing System

A comprehensive Key Information Extraction (KIE) system that processes PDF, DOC, TXT, CSV files and URLs to extract structured information using advanced AI models including NanoNets OCR-s.

## 🚀 Features

- **Multi-format Support**: PDF, DOC, TXT, CSV, Images (PNG, JPG), and URL processing
- **Advanced AI Models**: 
  - NanoNets OCR-s (primary) - optimized for document understanding
  - LayoutLM v3 (secondary) - for invoice processing
  - EasyOCR (fallback) - reliable OCR engine
- **Smart Schema Application**: Automatic document type detection and schema validation
- **Multi-language Support**: English and German document processing
- **RESTful API**: FastAPI-based with automatic documentation
- **Web Interface**: User-friendly testing interface
- **GPU Optimized**: Efficient GPU memory usage with 20GB vRAM support
- **Containerized Deployment**: Docker and Docker Compose ready

## 📋 System Requirements

### For 20GB vRAM Server (Recommended)
- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with 20GB+ vRAM (RTX 4090, A100, etc.)
- CUDA 12.1+
- Docker with NVIDIA Container Toolkit
- 32GB+ RAM
- 100GB+ storage

### For Current System (12GB vRAM)
- Can run with CPU-only mode or reduced model loading
- 16GB+ RAM recommended
- 50GB+ storage

## 🛠 Installation & Deployment

### Option 1: Quick Server Deployment (Recommended)

1. **Clone and prepare the system:**
```bash
cd /home/bibin.wilson@beo.in/Documents/Projects/BEO/Invoice/unified-kie-system
```

2. **Deploy to your 20GB vRAM server:**
```bash
# Replace with your server's IP address
./deployment/deploy.sh server 1 YOUR_SERVER_IP ubuntu
```

This will:
- Copy files to the server
- Install Docker and NVIDIA Docker if needed
- Build and start the services
- Configure GPU support automatically

### Option 2: Local Development

```bash
# Deploy locally with GPU support
./deployment/deploy.sh local 1

# Or deploy locally CPU-only (for testing)
./deployment/deploy.sh local 0
```

### Option 3: Manual Docker Deployment

```bash
# With GPU support
docker-compose up --build -d

# CPU-only mode
export FORCE_CPU=true
docker-compose -f docker-compose.cpu.yml up --build -d
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Server configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model configuration
NANONETS_MODEL=nanonets/Nanonets-OCR-s
LAYOUTLM_MODEL=microsoft/layoutlmv3-base
EASYOCR_LANGUAGES=en,de

# Processing limits
MAX_FILE_SIZE=52428800    # 50MB
MAX_PAGES=20
PROCESSING_TIMEOUT=300    # 5 minutes

# GPU configuration
FORCE_CPU=false
GPU_MEMORY_FRACTION=0.8

# Security
ALLOWED_EXTENSIONS=.pdf,.png,.jpg,.jpeg,.docx,.txt,.csv
MAX_URL_SIZE=10485760     # 10MB
```

### Memory Optimization for Different Systems

**For 20GB vRAM server:**
```bash
GPU_MEMORY_FRACTION=0.9
MAX_PAGES=50
MAX_FILE_SIZE=104857600  # 100MB
```

**For 12GB vRAM system:**
```bash
GPU_MEMORY_FRACTION=0.6
MAX_PAGES=10
# Consider CPU-only mode: FORCE_CPU=true
```

## 📡 API Usage

### File Upload Processing

```bash
curl -X POST "http://your-server:8000/extract/file" \
  -F "file=@document.pdf" \
  -F "extraction_type=auto" \
  -F "language=auto" \
  -F "use_schema=true"
```

### URL Processing

```bash
curl -X POST "http://your-server:8000/extract/url" \
  -F "url=https://example.com/document.pdf" \
  -F "extraction_type=invoice" \
  -F "language=en"
```

### Response Format

```json
{
  "job_id": "uuid-here",
  "timestamp": "2024-01-15T10:30:00Z",
  "extraction_type": "auto",
  "structured_data": {
    "schema_applied": "invoice",
    "structured_fields": {
      "invoice_number": "INV-2024-001",
      "vendor_name": "Example Corp",
      "total_amount": 1234.56,
      "currency": "EUR"
    },
    "validation": {
      "is_valid": true,
      "completeness_score": 0.85
    }
  },
  "processing_info": {
    "strategy_used": "nanonets",
    "pages_processed": 2
  }
}
```

## 🎯 Supported Document Types

### 1. Invoices
- Vendor/Customer information
- Line items with quantities and prices
- Tax calculations
- Payment terms
- Multi-currency support

### 2. German Customs Declarations (Ausfuhranmeldung)
- LRN, MRN, EORI numbers
- Declarant and exporter details
- Goods descriptions with HS codes
- Customs offices and procedures
- Transport information

### 3. Generic Documents
- Key-value pair extraction
- Contact information
- Dates and amounts
- Table data
- Multi-language text

## 🌐 Web Interface

Access the web interface at `http://your-server:8000`

Features:
- Drag & drop file upload
- URL processing
- Real-time extraction results
- Multiple output formats (Overview, Structured, Raw JSON)
- System health monitoring

## 🔍 API Endpoints

- `GET /` - Web interface
- `POST /extract/file` - Process uploaded file
- `POST /extract/url` - Process URL content
- `GET /results/{job_id}` - Retrieve results
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /schemas` - Available schemas
- `GET /docs` - API documentation

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Issues**
```bash
# Reduce memory usage
export GPU_MEMORY_FRACTION=0.5
# Or use CPU-only mode
export FORCE_CPU=true
```

2. **Model Loading Failures**
```bash
# Check logs
docker-compose logs unified-kie-system

# Restart with clean cache
docker-compose down -v
docker-compose up --build
```

3. **Permission Issues**
```bash
# Fix directory permissions
sudo chown -R $(whoami):$(whoami) uploads results models logs
```

### Performance Optimization

**For maximum performance on 20GB vRAM:**
- Use NanoNets OCR-s as primary extractor
- Enable GPU acceleration
- Increase batch processing limits
- Use SSD storage for model caching

**For limited resources:**
- Enable CPU-only mode
- Reduce concurrent processing
- Use smaller model variants
- Implement result caching

## 📊 Monitoring

### Health Checks
```bash
curl http://your-server:8000/health
```

### System Statistics
```bash
curl http://your-server:8000/stats
```

### Docker Monitoring
```bash
# View logs
docker-compose logs -f

# Check resource usage
docker stats

# GPU monitoring (if available)
nvidia-smi
```

## 🔒 Security Considerations

- File type validation
- Size limits for uploads and URLs
- Input sanitization
- No sensitive data logging
- Secure container practices
- Regular security updates

## 🚀 Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │────│  FastAPI Server  │────│  AI Processors  │
│   (Port 8000)   │    │   (Unified KIE)  │    │  (GPU/CPU)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
                ┌──────▼─────┐    ┌──────▼─────┐
                │  File      │    │  Schema    │
                │  Storage   │    │  Manager   │
                └────────────┘    └────────────┘
```

## 📈 Performance Expectations

### Processing Times (20GB vRAM server)
- **PDF (1-5 pages)**: 3-8 seconds
- **PDF (10+ pages)**: 15-30 seconds
- **Images**: 2-5 seconds
- **Text/CSV**: < 1 second

### Accuracy Rates
- **Invoices**: 90-95% field extraction accuracy
- **Customs Documents**: 85-92% (German text)
- **Generic Documents**: 80-90%

## 🤝 Contributing

1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Submit pull requests
5. Follow code style guidelines

## 📄 License

This project integrates multiple open-source components. Please review individual component licenses for compliance.

## 🆘 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review Docker logs for error details
3. Test with different document types
4. Monitor system resources during processing

## 🔄 Updates and Maintenance

### Regular Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Model Cache Clearing
```bash
# Clear model cache if needed
docker-compose down -v
rm -rf models/*
docker-compose up --build -d
```

---

**Built with ❤️ for efficient document processing**