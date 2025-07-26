"""
Optimized FastAPI application with performance enhancements
"""

import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time
import uuid
from contextlib import asynccontextmanager

# Import optimized components
from document_processor import UnifiedDocumentProcessor
from utils.request_queue import SmartRequestQueue, RequestPriority
from utils.gpu_memory_manager import GPUMemoryManager, MemoryStrategy, ModelPreloader
from utils.image_optimizer import ImageOptimizer
from extractors.nanonets_ocr_s_extractor_optimized import NanoNetsOCRSExtractorOptimized
from utils.json_encoder import safe_json_dumps

# Global instances
request_queue = None
memory_manager = None
image_optimizer = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with optimized startup and shutdown"""
    
    global request_queue, memory_manager, image_optimizer, processor
    
    print("🚀 Starting optimized unified KIE system...")
    
    # Initialize components
    memory_manager = GPUMemoryManager(MemoryStrategy.BALANCED)
    await memory_manager.optimize_for_inference()
    
    image_optimizer = ImageOptimizer()
    
    request_queue = SmartRequestQueue(max_concurrent_requests=6)
    await request_queue.start()
    
    # Initialize document processor
    config = {
        "model_cache_dir": "./models",
        "temp_dir": "./temp",
        "log_level": "WARNING"
    }
    
    processor = UnifiedDocumentProcessor(config)
    await processor.initialize()
    
    # Preload models for faster first request
    print("🔄 Preloading models for optimal performance...")
    try:
        # Warm up the system with a dummy request
        await _warm_up_system()
        print("✅ System warmed up and ready for high-performance processing")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")
    
    print("✅ Optimized system startup complete")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down optimized system...")
    
    if request_queue:
        await request_queue.stop()
    
    if hasattr(processor, 'nanonets_ocr_s_extractor') and processor.nanonets_ocr_s_extractor:
        if hasattr(processor.nanonets_ocr_s_extractor, 'close'):
            await processor.nanonets_ocr_s_extractor.close()
    
    print("✅ Optimized system shutdown complete")

# Create FastAPI app with optimizations
app = FastAPI(
    title="Unified KIE System - Optimized",
    description="High-performance document processing with vLLM acceleration",
    version="2.0.0-optimized",
    lifespan=lifespan
)

# Add performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Optimized health check with detailed system status"""
    
    try:
        # Get comprehensive system status
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "gpu_available": processor.gpu_available if processor else False,
            "system_ready": all([request_queue, memory_manager, image_optimizer, processor]),
            "version": "2.0.0-optimized"
        }
        
        if processor:
            status["models_loaded"] = await processor.get_model_status()
            status["memory_usage"] = await processor.get_memory_usage()
        
        if memory_manager:
            status["gpu_memory"] = memory_manager.get_memory_stats()
        
        if request_queue:
            status["queue_status"] = request_queue.get_status()
        
        return status
        
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}

@app.get("/stats")
async def get_stats():
    """Get detailed performance statistics"""
    
    stats = {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - (processor.start_time.timestamp() if processor else 0)
    }
    
    if request_queue:
        stats["queue"] = request_queue.get_status()
    
    if memory_manager:
        stats["memory"] = memory_manager.get_memory_stats()
    
    if processor:
        stats["models"] = await processor.get_model_status()
        stats["processing"] = await processor.get_memory_usage()
    
    return stats

@app.post("/extract/file")
async def extract_from_file_optimized(
    file: UploadFile = File(...),
    extraction_type: str = Form("auto"),
    language: str = Form("auto"),
    use_schema: bool = Form(True),
    priority: str = Form("normal")  # low, normal, high, urgent
):
    """Optimized document extraction with performance enhancements"""
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Parse priority
        priority_mapping = {
            "low": RequestPriority.LOW,
            "normal": RequestPriority.NORMAL,
            "high": RequestPriority.HIGH,
            "urgent": RequestPriority.URGENT
        }
        req_priority = priority_mapping.get(priority.lower(), RequestPriority.NORMAL)
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.docx', '.txt', '.csv']:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Submit to request queue for processing
        result_id = await request_queue.submit_request(
            processing_func=_process_document_optimized,
            payload={
                "file_path": tmp_file_path,
                "extraction_type": extraction_type,
                "language": language,
                "use_schema": use_schema,
                "request_id": request_id
            },
            priority=req_priority,
            timeout=180.0  # 3 minutes timeout
        )
        
        # Wait for result
        result = await request_queue.get_result(result_id, wait=True)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        processing_time = time.time() - start_time
        
        # Add metadata
        final_result = result["result"]
        final_result["request_metadata"] = {
            "request_id": request_id,
            "total_processing_time": processing_time,
            "priority": priority,
            "file_size_mb": len(content) / (1024 * 1024),
            "optimized_processing": True
        }
        
        # Clean up temp file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        return JSONResponse(
            content=final_result,
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Request-ID": request_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Extraction error for request {request_id}: {e}")
        
        # Clean up temp file on error
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "request_id": request_id,
                "processing_time": time.time() - start_time
            }
        )

async def _process_document_optimized(
    file_path: str,
    extraction_type: str,
    language: str,
    use_schema: bool,
    request_id: str
) -> Dict[str, Any]:
    """Optimized document processing function"""
    
    print(f"🚀 Processing document {request_id} with optimizations...")
    
    start_time = time.time()
    
    try:
        # Use optimized processing
        result = await processor.process_document(
            file_path=file_path,
            extraction_type=extraction_type,
            language=language,
            use_schema=use_schema
        )
        
        processing_time = time.time() - start_time
        result["optimized_processing_time"] = processing_time
        
        print(f"✅ Optimized processing completed for {request_id} in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Optimized processing failed for {request_id} after {processing_time:.2f}s: {e}")
        raise e

async def _warm_up_system():
    """Warm up the system for optimal first-request performance"""
    
    print("🔥 Warming up system components...")
    
    # Create a small test image
    from PIL import Image
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Warm up image optimizer
    await image_optimizer.optimize_for_vllm(test_image)
    
    # Warm up OCR extractor if available
    if hasattr(processor, 'nanonets_ocr_s_extractor') and processor.nanonets_ocr_s_extractor:
        try:
            await processor.nanonets_ocr_s_extractor.extract(test_image, "en")
        except:
            pass  # Warm-up can fail, that's ok
    
    print("✅ System warm-up completed")

if __name__ == "__main__":
    # Optimized server configuration
    uvicorn.run(
        "app_optimized:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU optimization
        loop="asyncio",
        access_log=False,  # Disable access logs for performance
        log_level="warning",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )