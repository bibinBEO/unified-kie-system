import torch
import asyncio
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from PIL import Image
import tempfile
import os
from pathlib import Path
import fitz  # PyMuPDF
import docx2txt
import pandas as pd
import io

from extractors.nanonets_extractor import NanoNetsExtractor
from extractors.layoutlm_extractor import LayoutLMExtractor
from extractors.easyocr_extractor import EasyOCRExtractor
from schema_manager import SchemaManager
from utils.text_processing import TextProcessor
from utils.file_converter import FileConverter

class UnifiedDocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.schema_manager = SchemaManager()
        self.text_processor = TextProcessor()
        self.file_converter = FileConverter()
        
        # Extractors
        self.nanonets_extractor = None
        self.layoutlm_extractor = None
        self.easyocr_extractor = None
        
        self.gpu_available = torch.cuda.is_available()
        self.device = "cuda" if self.gpu_available else "cpu"
        self.start_time = datetime.now()
        
        print(f"🚀 Initializing Unified Document Processor on {self.device}")
        if self.gpu_available:
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    async def initialize(self):
        """Initialize all extractors based on available resources"""
        try:
            # Always initialize NanoNets (best performance)
            self.nanonets_extractor = NanoNetsExtractor(self.config)
            await self.nanonets_extractor.initialize()
            print("✅ NanoNets OCR-s initialized")
            
            # Initialize LayoutLM if enough GPU memory
            if self.gpu_available:
                try:
                    self.layoutlm_extractor = LayoutLMExtractor(self.config)
                    await self.layoutlm_extractor.initialize()
                    print("✅ LayoutLM initialized")
                except Exception as e:
                    print(f"⚠️  LayoutLM initialization failed: {e}")
            
            # Initialize EasyOCR as fallback
            self.easyocr_extractor = EasyOCRExtractor(self.config)
            await self.easyocr_extractor.initialize()
            print("✅ EasyOCR initialized")
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise

    async def process_document(
        self,
        file_path: str,
        extraction_type: str = "auto",
        language: str = "auto",
        use_schema: bool = True
    ) -> Dict[str, Any]:
        """Process document and extract key information"""
        
        # Convert file to processable format
        processed_files = await self._prepare_document(file_path)
        
        # Determine optimal extraction strategy
        strategy = self._determine_strategy(file_path, extraction_type)
        
        # Extract from all pages/images
        all_extractions = []
        for file_info in processed_files:
            extraction = await self._extract_with_strategy(
                file_info, strategy, language
            )
            all_extractions.append(extraction)
        
        # Merge and post-process results
        merged_result = self._merge_extractions(all_extractions)
        
        # Apply schema validation if requested
        if use_schema:
            schema_result = self._apply_schema(merged_result, extraction_type)
            merged_result["structured_data"] = schema_result
        
        # Add processing metadata
        merged_result["processing_info"] = {
            "strategy_used": strategy,
            "extraction_type": extraction_type,
            "language": language,
            "pages_processed": len(processed_files),
            "schema_applied": use_schema,
            "timestamp": datetime.now().isoformat()
        }
        
        return merged_result

    async def _prepare_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Convert document to processable format(s)"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return await self._process_pdf(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            return [{"type": "image", "path": file_path, "page": 1}]
        elif file_ext == '.docx':
            return await self._process_docx(file_path)
        elif file_ext == '.txt':
            return await self._process_text(file_path)
        elif file_ext == '.csv':
            return await self._process_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    async def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Convert PDF to images for processing"""
        def convert_pdf():
            doc = fitz.open(file_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert to image with high DPI for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x scaling
                pix = page.get_pixmap(matrix=mat)
                
                # Save as temporary image
                temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                pix.save(temp_img.name)
                
                images.append({
                    "type": "image",
                    "path": temp_img.name,
                    "page": page_num + 1,
                    "temp": True
                })
            
            doc.close()
            return images
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, convert_pdf)

    async def _process_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from DOCX"""
        def extract_text():
            text = docx2txt.process(file_path)
            return [{"type": "text", "content": text, "page": 1}]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_text)

    async def _process_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Process plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return [{"type": "text", "content": content, "page": 1}]

    async def _process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file"""
        def read_csv():
            df = pd.read_csv(file_path)
            # Convert to structured format
            return [{
                "type": "structured",
                "content": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "page": 1
            }]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, read_csv)

    def _determine_strategy(self, file_path: str, extraction_type: str) -> str:
        """Determine optimal extraction strategy"""
        file_ext = Path(file_path).suffix.lower()
        
        # For structured data, use direct processing
        if file_ext in ['.csv', '.txt']:
            return "direct"
        
        # For document images, use best available OCR
        if extraction_type == "customs" and self.nanonets_extractor:
            return "nanonets"  # Best for German customs docs
        elif extraction_type == "invoice" and self.layoutlm_extractor:
            return "layoutlm"  # Best for invoices
        elif self.nanonets_extractor:
            return "nanonets"  # General best performance
        else:
            return "easyocr"   # Fallback

    async def _extract_with_strategy(
        self,
        file_info: Dict[str, Any],
        strategy: str,
        language: str
    ) -> Dict[str, Any]:
        """Extract using specified strategy"""
        
        if file_info["type"] == "text":
            return {
                "raw_text": file_info["content"],
                "page": file_info["page"],
                "extraction_method": "direct_text"
            }
        
        elif file_info["type"] == "structured":
            return {
                "structured_content": file_info["content"],
                "columns": file_info["columns"],
                "page": file_info["page"],
                "extraction_method": "direct_csv"
            }
        
        elif file_info["type"] == "image":
            image = Image.open(file_info["path"])
            
            result = None
            extraction_errors = []
            
            # Try extractors in order of preference with automatic fallback
            extractors_to_try = []
            
            if strategy == "nanonets" and self.nanonets_extractor:
                extractors_to_try = [
                    ("nanonets", self.nanonets_extractor),
                    ("layoutlm", self.layoutlm_extractor) if self.layoutlm_extractor else None,
                    ("easyocr", self.easyocr_extractor)
                ]
            elif strategy == "layoutlm" and self.layoutlm_extractor:
                extractors_to_try = [
                    ("layoutlm", self.layoutlm_extractor),
                    ("easyocr", self.easyocr_extractor)
                ]
            else:
                extractors_to_try = [("easyocr", self.easyocr_extractor)]
            
            # Remove None entries
            extractors_to_try = [ext for ext in extractors_to_try if ext is not None]
            
            for extractor_name, extractor in extractors_to_try:
                try:
                    print(f"Trying {extractor_name} extractor...")
                    result = await extractor.extract(image, language)
                    result["extraction_method"] = f"{extractor_name}_ocr"
                    print(f"✅ {extractor_name} extraction successful")
                    break
                except Exception as e:
                    error_msg = str(e)
                    extraction_errors.append(f"{extractor_name}: {error_msg}")
                    print(f"❌ {extractor_name} failed: {error_msg}")
                    continue
            
            if result is None:
                # All extractors failed, return error info
                result = {
                    "raw_text": f"All extraction methods failed. Errors: {'; '.join(extraction_errors)}",
                    "extraction_method": "extraction_failed",
                    "errors": extraction_errors
                }
            
            # Cleanup temp files
            if file_info.get("temp") and os.path.exists(file_info["path"]):
                os.unlink(file_info["path"])
            
            result["page"] = file_info["page"]
            return result
        
        else:
            raise ValueError(f"Unknown file type: {file_info['type']}")

    def _merge_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple pages/sources"""
        if len(extractions) == 1:
            return extractions[0]
        
        merged = {
            "pages": extractions,
            "combined_text": "",
            "all_key_values": {},
            "extraction_methods": []
        }
        
        # Combine text from all pages
        for ext in extractions:
            if "raw_text" in ext:
                merged["combined_text"] += ext["raw_text"] + "\n\n"
            if "extraction_method" in ext:
                merged["extraction_methods"].append(ext["extraction_method"])
            if "key_values" in ext:
                merged["all_key_values"].update(ext["key_values"])
        
        # Use text processing to extract unified key-value pairs
        if merged["combined_text"]:
            merged["unified_extraction"] = self.text_processor.extract_key_values(
                merged["combined_text"]
            )
        
        return merged

    def _apply_schema(self, extraction_result: Dict[str, Any], extraction_type: str) -> Dict[str, Any]:
        """Apply appropriate schema to structure the data"""
        
        # Determine schema to use
        if extraction_type == "customs":
            schema_name = "customs_declaration"
        elif extraction_type == "invoice":
            schema_name = "invoice"
        else:
            schema_name = self.schema_manager.detect_document_type(
                extraction_result.get("combined_text", "")
            )
        
        # Apply schema
        return self.schema_manager.apply_schema(extraction_result, schema_name)

    async def get_model_status(self) -> Dict[str, bool]:
        """Get status of all loaded models"""
        return {
            "nanonets": self.nanonets_extractor is not None,
            "layoutlm": self.layoutlm_extractor is not None,
            "easyocr": self.easyocr_extractor is not None
        }

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            "ram_usage_mb": memory_info.rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent()
        }
        
        if self.gpu_available:
            result["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return result

    async def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get detailed GPU memory information"""
        if not self.gpu_available:
            return None
        
        props = torch.cuda.get_device_properties(0)
        return {
            "total_memory_gb": props.total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "free_gb": (props.total_memory - torch.cuda.memory_reserved()) / 1024**3
        }