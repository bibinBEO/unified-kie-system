"""
Simplified FastAPI application focusing on optimized vLLM performance
"""

import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import os
from pathlib import Path
import time
from PIL import Image

# Import only the optimized vLLM extractor
from extractors.nanonets_vllm_extractor import NanoNetsVLLMExtractor
from utils.json_encoder import safe_json_dumps

# Global processor
processor = None

app = FastAPI(
    title="Unified KIE System - Optimized",
    description="High-performance document processing with optimized vLLM",
    version="2.0.0-optimized"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the optimized vLLM extractor"""
    global processor
    
    print("🚀 Starting simplified optimized KIE system...")
    
    # Simple config
    config = {
        "model_cache_dir": "./models",
        "temp_dir": "./temp"
    }
    
    try:
        processor = NanoNetsVLLMExtractor(config)
        await processor.initialize()
        print("✅ Optimized vLLM extractor ready for high-performance processing!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        processor = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        "status": "healthy" if processor else "initializing",
        "timestamp": time.time(),
        "processor_ready": processor is not None,
        "version": "2.0.0-optimized"
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {
            "service": "Unified KIE System - Optimized",
            "status": "ready" if processor else "initializing",
            "version": "2.0.0-optimized",
            "message": "Web interface not found. API available at /extract/file",
            "endpoints": {
                "health": "/health",
                "extract": "/extract/file"
            }
        }

@app.post("/extract/file")
async def extract_from_file_optimized(
    file: UploadFile = File(...),
    extraction_type: str = Form("auto"),
    language: str = Form("auto")
):
    """Optimized document extraction"""
    
    if not processor:
        raise HTTPException(status_code=503, detail="Extractor not ready yet")
    
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.png', '.jpg', '.jpeg']:
            raise HTTPException(status_code=400, detail=f"Only image files supported: {file_ext}")
        
        # Read and process image
        import io
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Extract using optimized vLLM
        result = await processor.extract(image, language)
        
        processing_time = time.time() - start_time
        
        # Add metadata
        result["processing_metadata"] = {
            "file_size_mb": len(content) / (1024 * 1024),
            "processing_time": processing_time,
            "extraction_type": extraction_type,
            "optimized": True
        }
        
        return JSONResponse(
            content=result,
            headers={
                "X-Processing-Time": str(processing_time)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Extraction error: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "processing_time": processing_time
            }
        )

if __name__ == "__main__":
    uvicorn.run(
        "app_simple:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        access_log=False,
        log_level="info"
    )