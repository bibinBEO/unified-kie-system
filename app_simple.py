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

# Import extractors with fallback capability
from extractors.nanonets_vllm_extractor import NanoNetsVLLMExtractor
from extractors.nanonets_extractor import NanoNetsExtractor
from utils.json_encoder import safe_json_dumps

# Global processors
vllm_processor = None
fallback_processor = None

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
    """Initialize extractors with fallback capability"""
    global vllm_processor, fallback_processor
    
    print("🚀 Starting optimized KIE system with fallback...")
    
    # Simple config
    config = {
        "model_cache_dir": "./models",
        "temp_dir": "./temp"
    }
    
    # Try to initialize optimized vLLM extractor
    try:
        print("🔄 Initializing optimized vLLM extractor...")
        vllm_processor = NanoNetsVLLMExtractor(config)
        await vllm_processor.initialize()
        print("✅ Optimized vLLM extractor ready!")
        
    except Exception as e:
        print(f"⚠️ vLLM initialization failed: {e}")
        vllm_processor = None
    
    # Initialize reliable fallback extractor
    try:
        print("🔄 Initializing fallback extractor...")
        fallback_processor = NanoNetsExtractor(config)
        await fallback_processor.initialize()
        print("✅ Fallback extractor ready!")
        
    except Exception as e:
        print(f"❌ Fallback initialization failed: {e}")
        fallback_processor = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        "status": "healthy" if (vllm_processor or fallback_processor) else "initializing",
        "timestamp": time.time(),
        "vllm_ready": vllm_processor is not None,
        "fallback_ready": fallback_processor is not None,
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
    """Document extraction with intelligent fallback"""
    
    if not (vllm_processor or fallback_processor):
        raise HTTPException(status_code=503, detail="No extractors ready yet")
    
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
        
        # Try optimized vLLM first, then fallback
        result = None
        method_used = "unknown"
        
        if vllm_processor:
            try:
                print("🚀 Trying optimized vLLM extraction...")
                result = await vllm_processor.extract(image, language)
                if result and not result.get("raw_text", "").startswith("Extraction failed"):
                    method_used = "vllm_optimized"
                    print("✅ vLLM extraction successful")
                else:
                    raise Exception("vLLM extraction returned error")
                    
            except Exception as e:
                print(f"⚠️ vLLM extraction failed: {e}")
                result = None
        
        # Fallback to standard extractor if vLLM failed
        if not result and fallback_processor:
            try:
                print("🔄 Using fallback extractor...")
                result = await fallback_processor.extract(image, language)
                method_used = "nanonets_standard"
                print("✅ Fallback extraction successful")
                
            except Exception as e:
                print(f"❌ Fallback extraction failed: {e}")
                raise HTTPException(status_code=500, detail=f"All extraction methods failed: {e}")
        
        if not result:
            raise HTTPException(status_code=500, detail="No extractors available")
        
        processing_time = time.time() - start_time
        
        # Add metadata
        result["processing_metadata"] = {
            "file_size_mb": len(content) / (1024 * 1024),
            "processing_time": processing_time,
            "extraction_type": extraction_type,
            "method_used": method_used,
            "vllm_available": vllm_processor is not None,
            "fallback_used": method_used == "nanonets_standard"
        }
        
        return JSONResponse(
            content=result,
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Method-Used": method_used
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