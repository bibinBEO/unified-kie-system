from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import os
import uuid
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json
import requests
import tempfile
from pathlib import Path

from document_processor import UnifiedDocumentProcessor
from schema_manager import SchemaManager
from deployment.config import Config

app = FastAPI(
    title="Unified KIE Document Processing API", 
    version="2.0.0",
    description="Extract key information from PDF/DOC/TXT/CSV/URL using multiple AI strategies"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
processor = UnifiedDocumentProcessor(config)
schema_manager = SchemaManager()

# Setup directories
for dir_path in [config.UPLOAD_DIR, config.RESULTS_DIR, config.MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize all models and processors"""
    await processor.initialize()
    print("✅ Unified KIE System Ready")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the web interface"""
    async with aiofiles.open("templates/index.html", "r") as f:
        content = await f.read()
    return HTMLResponse(content=content)

@app.post("/extract/file", response_model=Dict[str, Any])
async def extract_from_file(
    file: UploadFile = File(...),
    extraction_type: str = Form("auto"),  # auto, invoice, customs, generic
    language: str = Form("auto"),  # auto, en, de
    use_schema: bool = Form(True)
):
    """Extract key information from uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.docx', '.txt', '.csv'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    job_id = str(uuid.uuid4())
    file_path = os.path.join(config.UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    # Save uploaded file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        # Process document
        result = await processor.process_document(
            file_path=file_path,
            extraction_type=extraction_type,
            language=language,
            use_schema=use_schema
        )
        
        # Add job metadata
        result.update({
            "job_id": job_id,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "extraction_type": extraction_type,
            "language": language,
            "file_type": file_extension
        })
        
        # Save results
        result_file = os.path.join(config.RESULTS_DIR, f"{job_id}_results.json")
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/extract/url", response_model=Dict[str, Any])
async def extract_from_url(
    url: str = Form(...),
    extraction_type: str = Form("auto"),
    language: str = Form("auto"),
    use_schema: bool = Form(True)
):
    """Extract key information from URL"""
    job_id = str(uuid.uuid4())
    
    try:
        # Download content from URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type from content-type or URL
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            file_extension = '.pdf'
        elif 'image' in content_type:
            file_extension = '.jpg'
        else:
            file_extension = '.txt'
        
        # Save content temporarily
        temp_file = os.path.join(config.UPLOAD_DIR, f"{job_id}_url_content{file_extension}")
        
        if file_extension == '.txt':
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write(response.text)
        else:
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(response.content)
        
        # Process document
        result = await processor.process_document(
            file_path=temp_file,
            extraction_type=extraction_type,
            language=language,
            use_schema=use_schema
        )
        
        # Add job metadata
        result.update({
            "job_id": job_id,
            "source_url": url,
            "timestamp": datetime.now().isoformat(),
            "extraction_type": extraction_type,
            "language": language,
            "content_type": content_type
        })
        
        # Save results
        result_file = os.path.join(config.RESULTS_DIR, f"{job_id}_results.json")
        async with aiofiles.open(result_file, 'w') as f:
            await f.write(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)

@app.get("/results/{job_id}", response_model=Dict[str, Any])
async def get_results(job_id: str):
    """Retrieve extraction results by job ID"""
    result_file = os.path.join(config.RESULTS_DIR, f"{job_id}_results.json")
    
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Results not found")
    
    async with aiofiles.open(result_file, 'r') as f:
        content = await f.read()
        return json.loads(content)

@app.get("/schemas", response_model=Dict[str, Any])
async def get_available_schemas():
    """Get all available extraction schemas"""
    return {
        "schemas": schema_manager.get_available_schemas(),
        "default_schema": schema_manager.get_default_schema_name()
    }

@app.get("/schema/{schema_name}", response_model=Dict[str, Any])
async def get_schema(schema_name: str):
    """Get specific schema definition"""
    schema = schema_manager.get_schema(schema_name)
    if not schema:
        raise HTTPException(status_code=404, detail="Schema not found")
    return schema

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": await processor.get_model_status(),
        "gpu_available": processor.gpu_available,
        "memory_usage": await processor.get_memory_usage()
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_processed": len([f for f in os.listdir(config.RESULTS_DIR) if f.endswith('_results.json')]),
        "models_loaded": await processor.get_model_status(),
        "uptime": (datetime.now() - processor.start_time).total_seconds(),
        "gpu_memory": await processor.get_gpu_memory_info() if processor.gpu_available else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        workers=1,  # Single worker for GPU models
        log_level="info"
    )