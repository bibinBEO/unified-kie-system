"""
NanoNets OCR-s Extractor - Enhanced with docext integration
CUDA error handling for RTX 4000 ADA GPU compatibility
"""
import os
import sys
import torch
import asyncio
import base64
import io
import json
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from PIL import Image
import requests
import tempfile

# Add docext to path
docext_path = os.path.join(os.path.dirname(__file__), '..', 'docext')
if docext_path not in sys.path:
    sys.path.insert(0, docext_path)

try:
    import json_repair
    from litellm import completion
    from loguru import logger
    from tenacity import retry, stop_after_attempt, wait_exponential
    DOCEXT_DEPS_AVAILABLE = True
except ImportError:
    DOCEXT_DEPS_AVAILABLE = False
    print("Warning: docext dependencies not available")

# Import docext if available
try:
    from docext.core.extract import DocExtractor
    from docext.core.config import Config as DocExtConfig
    from docext.core.vllm import VLLMServer as DocExtVLLMServer
    DOCEXT_AVAILABLE = True
except ImportError:
    DOCEXT_AVAILABLE = False
    print("Warning: docext package not available")

from .nanonets_extractor import NanoNetsExtractor

# Try to import BaseExtractor - create a minimal one if not available
try:
    from .base_extractor import BaseExtractor
except ImportError:
    # Create minimal BaseExtractor if not available
    class BaseExtractor:
        def __init__(self, config):
            self.config = config
            self.name = "base"
            self.supports_multiple_pages = False


class VLLMServer:
    """vLLM Server management for NanoNets OCR-s"""
    
    def __init__(
        self,
        model_name: str = "nanonets/Nanonets-OCR-s",
        host: str = "127.0.0.1",
        port: int = 8001,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        max_num_imgs: int = 5,
        vllm_start_timeout: int = 300,
        dtype: str = "bfloat16",
    ):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_imgs = max_num_imgs
        self.server_process = None
        self.url = f"http://{self.host}:{self.port}/v1/models"
        self.completion_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        self.vllm_start_timeout = vllm_start_timeout
        self.dtype = dtype
        self._server_ready = False
        
    def start_server(self):
        """Start the vLLM server."""
        if self._is_server_running():
            print(f"✅ vLLM server already running on {self.host}:{self.port}")
            self._server_ready = True
            return
            
        print(f"🚀 Starting vLLM server for {self.model_name}...")
        
        command = [
            "vllm", "serve", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--limit-mm-per-prompt", f"image={self.max_num_imgs},video=0",
            "--served-model-name", self.model_name,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--enforce-eager",  # More stable for complex models
            "--disable-log-stats",
            "--trust-remote-code",
        ]
        
        try:
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            print(f"📡 vLLM server started with PID: {self.server_process.pid}")
        except Exception as e:
            print(f"❌ Failed to start vLLM server: {e}")
            raise
    
    def _is_server_running(self) -> bool:
        """Check if server is already running."""
        try:
            response = requests.get(self.url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server(self, timeout: int = 300) -> bool:
        """Wait until the vLLM server is ready."""
        print("⏳ Waiting for vLLM server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ vLLM server ready on {self.host}:{self.port}")
                    self._server_ready = True
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        
        print(f"❌ vLLM server did not start within {timeout} seconds")
        return False
    
    def stop_server(self):
        """Stop the vLLM server gracefully."""
        if self.server_process:
            print("🛑 Stopping vLLM server...")
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
            finally:
                self.server_process = None
                self._server_ready = False
            print("✅ vLLM server stopped")
    
    def is_ready(self) -> bool:
        """Check if server is ready."""
        return self._server_ready and self._is_server_running()


class NanoNetsOCRSExtractor(BaseExtractor):
    """
    NanoNets OCR-s Extractor with docext integration
    Enhanced CUDA error handling for RTX 4000 ADA compatibility
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "nanonets_ocr_s"
        self.supports_multiple_pages = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # DocExt extractor
        self.docext_extractor = None
        
        # vLLM server for advanced usage
        self.vllm_server = None
        
        # Fallback extractor
        self.fallback_extractor = None
        
        # Model configuration
        self.model_name = "nanonets/Nanonets-OCR-s"
        self.host = "127.0.0.1"
        self.port = 8001
        
        print(f"🚀 NanoNets OCR-s initializing on {self.device.upper()}")
        
        # Set CUDA environment for stability
        if self.device == "cuda":
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        
    async def initialize(self):
        """Initialize DocExt extractor with comprehensive error handling."""
        
        # Try DocExt first (most efficient)
        if DOCEXT_AVAILABLE and self.device == "cuda":
            try:
                await self._init_docext()
                if self.docext_extractor:
                    print("✅ DocExt extractor initialized successfully")
                    return
            except Exception as e:
                print(f"⚠️  DocExt initialization failed: {e}")
        
        # Try vLLM server as secondary option
        if DOCEXT_DEPS_AVAILABLE and self.device == "cuda":
            try:
                await self._init_vllm_server()
                if self.vllm_server and self.vllm_server.is_ready():
                    print("✅ vLLM server initialized successfully")
                    return
            except Exception as e:
                print(f"⚠️  vLLM server initialization failed: {e}")
        
        # Fallback to standard extractor
        print("🔄 Falling back to standard NanoNets extractor")
        await self._init_fallback()
    
    async def _init_docext(self):
        """Initialize DocExt extractor."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Configure DocExt
            config = DocExtConfig()
            config.model_name = self.model_name
            config.max_new_tokens = 2048
            config.temperature = 0.1
            config.top_p = 0.9
            
            # Initialize extractor
            self.docext_extractor = DocExtractor(config)
            print("✅ DocExt core extractor initialized")
            
        except Exception as e:
            print(f"❌ DocExt initialization error: {e}")
            self.docext_extractor = None
            raise
    
    async def _init_vllm_server(self):
        """Initialize vLLM server."""
        try:
            self.vllm_server = VLLMServer(
                model_name=self.model_name,
                host=self.host,
                port=self.port,
                max_model_len=8192,
                gpu_memory_utilization=0.8,
                max_num_imgs=5,
                dtype="bfloat16"
            )
            
            # Start server in background thread
            server_thread = threading.Thread(target=self._start_server_thread, daemon=True)
            server_thread.start()
            
            # Wait for server to be ready
            if not self.vllm_server.wait_for_server(timeout=300):
                raise Exception("vLLM server failed to start")
                
        except Exception as e:
            print(f"❌ vLLM server initialization error: {e}")
            self.vllm_server = None
            raise
    
    def _start_server_thread(self):
        """Start vLLM server in thread."""
        try:
            self.vllm_server.start_server()
        except Exception as e:
            print(f"❌ Server thread error: {e}")
    
    async def _init_fallback(self):
        """Initialize fallback extractor."""
        if not self.fallback_extractor:
            self.fallback_extractor = NanoNetsExtractor(self.config)
            await self.fallback_extractor.initialize()
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _create_kie_messages(self, image: Image.Image, fields: List[Dict[str, str]]) -> List[Dict]:
        """Create messages for KIE extraction following docext pattern."""
        if not fields:
            # Default KIE fields for general documents
            fields = [
                {"name": "document_type", "description": "Type of document"},
                {"name": "date", "description": "Document date"},
                {"name": "amount", "description": "Total amount or value"},
                {"name": "invoice_number", "description": "Invoice or reference number"},
                {"name": "sender_name", "description": "Sender or issuer name"},
                {"name": "recipient_name", "description": "Recipient or customer name"},
            ]
        
        # Create field descriptions
        field_descriptions = "\n".join([
            f"{field['name'].replace(' ', '_').lower()}: {field.get('description', '')}"
            for field in fields
        ])
        
        # Create output format
        output_format = {
            field["name"].replace(" ", "_").lower(): "..."
            for field in fields
        }
        
        # Encode image
        image_b64 = self.encode_image(image)
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract the following fields from the document:\n{field_descriptions}."
                },
                {
                    "type": "text",
                    "text": "Document:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Return a JSON with the following format:\n{json.dumps(output_format, indent=2)}. If a field is not found, return '' for that field. Do not give any explanation."
                }
            ]
        }]
        
        return messages
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_vllm_request(self, messages: List[Dict]) -> Dict:
        """Make request to vLLM server with retry logic."""
        if not self.vllm_server or not self.vllm_server.is_ready():
            raise Exception("vLLM server not ready")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
            "n": 1
        }
        
        response = requests.post(
            self.vllm_server.completion_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"vLLM request failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def extract_key_value_pairs(
        self, 
        image_data: Union[bytes, str, Image.Image], 
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract key-value pairs using the best available method."""
        
        # Convert input to PIL Image
        image = self._prepare_image(image_data)
        if not image:
            return self._create_error_response("Failed to process image")
        
        # Convert fields to docext format if provided
        docext_fields = self._convert_fields_format(fields) if fields else None
        
        # Try DocExt extraction first
        if self.docext_extractor:
            try:
                return await self._extract_with_docext(image, docext_fields)
            except RuntimeError as e:
                if "device-side assert" in str(e) or "CUDA error" in str(e):
                    print(f"🔧 CUDA error in DocExt: {e}")
                    await self._handle_cuda_error()
                    return self._create_cuda_error_response(str(e))
                else:
                    print(f"⚠️  DocExt extraction failed: {e}")
        
        # Try vLLM extraction
        if self.vllm_server and self.vllm_server.is_ready():
            try:
                return await self._extract_with_vllm(image, docext_fields)
            except Exception as e:
                print(f"⚠️  vLLM extraction failed: {e}")
        
        # Fallback to standard extractor
        return await self._extract_with_fallback(image, kwargs.get('language', 'auto'))
    
    def _prepare_image(self, image_data: Union[bytes, str, Image.Image]) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image."""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # Handle base64 data URL
                    header, data = image_data.split(',', 1)
                    image_bytes = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # Handle file path
                    image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                return None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"❌ Image preparation failed: {e}")
            return None
    
    def _convert_fields_format(self, fields: List[str]) -> List[Dict[str, str]]:
        """Convert simple field list to docext format."""
        return [{"name": field, "description": f"Extract {field}"} for field in fields]
    
    async def _extract_with_docext(self, image: Image.Image, fields: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Extract using DocExt with CUDA error handling."""
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, 'PNG')
            temp_path = tmp_file.name
        
        try:
            # Extract using DocExt
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._docext_blocking_extract,
                temp_path,
                fields
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def _docext_blocking_extract(self, image_path: str, fields: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Blocking DocExt extraction (runs in thread pool)."""
        
        # Set deterministic behavior for stability
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Perform extraction
        with torch.cuda.device(0) if torch.cuda.is_available() else torch.no_grad():
            
            if fields:
                # Use custom fields
                field_names = [f["name"] for f in fields]
                extracted_data = self.docext_extractor.extract(image_path, fields=field_names)
            else:
                # Use default extraction
                extracted_data = self.docext_extractor.extract(image_path)
        
        # Process results
        return self._process_docext_results(extracted_data)
    
    def _process_docext_results(self, extracted_data: Any) -> Dict[str, Any]:
        """Process DocExt extraction results."""
        
        key_values = {}
        raw_text = ""
        
        if isinstance(extracted_data, dict):
            key_values = {k: str(v).strip() for k, v in extracted_data.items() if v and str(v).strip()}
            raw_text = "\n".join([f"{k}: {v}" for k, v in key_values.items()])
        elif isinstance(extracted_data, str):
            raw_text = extracted_data
            # Try to parse key-value pairs from text
            key_values = self._parse_text_to_kvp(raw_text)
        else:
            raw_text = str(extracted_data)
        
        return {
            "raw_text": raw_text,
            "key_values": key_values,
            "confidence_scores": {k: 0.9 for k in key_values.keys()},
            "extraction_method": "nanonets_ocr_s_docext",
            "model_info": {
                "name": "Nanonets-OCR-s",
                "version": "3B",
                "device": self.device,
                "provider": "DocExt"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_text_to_kvp(self, text: str) -> Dict[str, str]:
        """Parse text into key-value pairs."""
        key_values = {}
        lines = text.split('\n')
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    if value:
                        key_values[key] = value
        
        return key_values
    
    async def _handle_cuda_error(self):
        """Handle CUDA errors by clearing cache and resetting."""
        print("🔧 Handling CUDA device-side assertion error...")
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                print("✅ CUDA memory cleared and synchronized")
            except Exception as e:
                print(f"⚠️  CUDA cleanup warning: {e}")
    
    def _create_cuda_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create response for CUDA errors."""
        return {
            "raw_text": f"Extraction failed: CUDA error: device-side assert triggered\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n",
            "key_values": {
                "error": "CUDA error: device-side assert triggered\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n",
                "error_type": "RuntimeError",
                "device": self.device,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            },
            "confidence_scores": {},
            "extraction_method": "nanonets_ocr_s_cuda_error",
            "fallback_reason": "CUDA device-side assertion error"
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create generic error response."""
        return {
            "raw_text": f"Extraction failed: {error_message}",
            "key_values": {"error": error_message},
            "confidence_scores": {},
            "extraction_method": "nanonets_ocr_s_error",
            "fallback_reason": error_message
        }
    
    async def _extract_with_vllm(self, image: Image.Image, fields: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Extract using vLLM server."""
        
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create messages for KIE
        messages = self._create_kie_messages(image, fields)
        
        # Make request to vLLM
        response = await self._make_vllm_request(messages)
        
        # Parse response
        content = response["choices"][0]["message"]["content"]
        
        # Parse JSON using json_repair for robustness
        try:
            if DOCEXT_DEPS_AVAILABLE:
                extracted_data = json_repair.loads(content)
            else:
                extracted_data = json.loads(content)
        except Exception as e:
            print(f"⚠️  JSON parsing failed: {e}")
            extracted_data = {"raw_response": content}
        
        return {
            "raw_text": content,
            "key_values": extracted_data,
            "extraction_method": "nanonets_ocr_s_vllm",
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name,
            "fields_extracted": list(extracted_data.keys()) if isinstance(extracted_data, dict) else [],
            "confidence_score": 0.9  # vLLM typically has high confidence
        }
    
    async def _extract_with_fallback(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Extract using fallback extractor."""
        if not self.fallback_extractor:
            await self._init_fallback()
        
        result = await self.fallback_extractor.extract(image, language)
        
        # Mark as fallback method
        result["extraction_method"] = "nanonets_fallback_from_ocr_s"
        result["fallback_reason"] = "vLLM OCR-s unavailable"
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        if self.vllm_server:
            self.vllm_server.stop_server()
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.cleanup()
        except:
            pass


# Predefined KIE field templates for different document types
KIE_TEMPLATES = {
    "invoice": [
        {"name": "invoice_number", "description": "Invoice number"},
        {"name": "invoice_date", "description": "Invoice date"},
        {"name": "invoice_amount", "description": "Invoice total amount"},
        {"name": "invoice_currency", "description": "Invoice currency"},
        {"name": "seller_name", "description": "Seller or vendor name"},
        {"name": "buyer_name", "description": "Buyer or customer name"},
        {"name": "seller_address", "description": "Seller address"},
        {"name": "buyer_address", "description": "Buyer address"},
        {"name": "seller_tax_id", "description": "Seller tax ID or VAT number"},
        {"name": "buyer_tax_id", "description": "Buyer tax ID"},
    ],
    "receipt": [
        {"name": "date", "description": "Receipt date"},
        {"name": "receipt_number", "description": "Receipt or transaction number"},
        {"name": "seller_name", "description": "Store or merchant name"},
        {"name": "seller_address", "description": "Store address"},
        {"name": "total_amount", "description": "Total amount paid"},
        {"name": "tax_amount", "description": "Tax amount"},
        {"name": "payment_method", "description": "Payment method used"},
    ],
    "passport": [
        {"name": "full_name", "description": "Full name"},
        {"name": "date_of_birth", "description": "Date of birth in YYYY-MM-DD format"},
        {"name": "passport_number", "description": "Passport number"},
        {"name": "passport_type", "description": "Passport type"},
        {"name": "date_of_issue", "description": "Date of issue in YYYY-MM-DD format"},
        {"name": "date_of_expiry", "description": "Date of expiry in YYYY-MM-DD format"},
        {"name": "place_of_birth", "description": "Place of birth"},
        {"name": "nationality", "description": "Nationality"},
        {"name": "gender", "description": "Gender"},
    ],
    "customs": [
        {"name": "document_type", "description": "Type of customs document"},
        {"name": "reference_number", "description": "LRN, MRN, or reference number"},
        {"name": "declaration_date", "description": "Declaration date"},
        {"name": "exporter_name", "description": "Exporter company name"},
        {"name": "importer_name", "description": "Importer company name"},
        {"name": "goods_description", "description": "Description of goods"},
        {"name": "total_value", "description": "Total value of goods"},
        {"name": "currency", "description": "Currency of transaction"},
        {"name": "country_of_origin", "description": "Country of origin"},
        {"name": "destination_country", "description": "Destination country"},
    ]
}